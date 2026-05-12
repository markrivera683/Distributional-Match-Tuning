"""Lookahead helper for the prompts_dataloader used by ebft_trainer.

Goal
====
Let the trainer ask "which `doc_id`s — and therefore which raw teacher
prompt strings — will appear in the next `K` training batches on this rank?"
WITHOUT consuming the dataloader's iterator.

Why
===
``ebft_trainer.fit()`` runs a step like this::

    for sequence_ids, doc_ids, qa_masks in self.prompts_dataloader:   # ← step t
        rollout_samples = samples_generator.generate_samples(...)     # student rollout
        experiences = experience_maker.make_experience_batch(...)
            └─ inside: teacher_provider.sample_targets(...)           # ← teacher call (slow)
        ppo_train(...)                                                # GPU train (~10s)

If we know step `t+1, t+2, ..., t+K`'s prompts BEFORE entering them, we can
``schedule_prefetch`` them while ``ppo_train`` runs and have teacher answers
in mem-queue / SQLite by the time the next step's ``sample_targets`` fires.

The previous implementation in ``ebft_trainer.py`` mistakenly scheduled the
CURRENT step's prompts (which had already been queried), so the prefetch
mem-queue hit rate was ~0%. This module fixes that by exposing the next K
batches.

Determinism
===========
The prompts dataloader uses our local ``DistributedSampler`` whose ``__iter__``
is purely a function of ``(seed, epoch, len(dataset))``::

    indices = torch.randperm(len(dataset), generator=Generator(seed+epoch))
    indices = indices[:total_size]                  # truncate / pad
    indices = indices[rank::num_replicas]            # per-rank slice
    indices = indices[consumed_indicies:]            # skip resumed

We replicate the first three lines (skip the last, since we want to peek
positions BEYOND consumed) and read ``dataset.doc_ids[seq_idx]`` for each
peeked sequence index. ``dataset.prompts[doc_id]`` then gives the raw prompt
string for the teacher.

Out of scope
============
- Cross-epoch peeking. Near the end of an epoch we just return fewer than
  ``num_steps`` batches' worth of prompts; the trainer's ``schedule_prefetch``
  no-ops on empty input.
- Mid-epoch resume correctness. On checkpoint resume the dataloader's
  internal state restores via ``state_dict``, but our caller's
  ``_within_epoch_batch_idx`` resets to 0, so for the first few resumed steps
  ``peek_*`` will look at the wrong slice. This only causes a few wasted
  prefetches (cached but never consumed), not incorrectness — the slow path
  ``sample_targets()`` still works.

Author: 2026-04-29 (lookahead prefetch fix for the dead-code prefetch_depth).
"""
from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


class PromptLookahead:
    """Peek the next K batches' unique doc_ids / raw prompts on this rank.

    Args:
        dataset:    The prompts ``QADataset`` / ``SequenceDataset``. Must
                    expose ``.doc_ids: List[Tensor]`` (per-sequence doc-id
                    streams) and ``.prompts: List[str]`` (raw prompt text
                    indexed by doc-id).
        sampler:    The same ``DistributedSampler`` instance the
                    ``StatefulDataLoader`` uses. We read its
                    ``epoch / seed / shuffle / num_replicas / rank /
                    drop_last / total_size`` to replicate ``__iter__``.
        batch_size: ``args.rollout_batch_size`` — same as the dataloader's.
        max_doc_id_per_chunk: Optional cap to avoid pathological chunks.

    Notes:
        Re-builds the rank-local index list lazily, only when
        ``sampler.epoch`` changes. Within one epoch ``peek_*`` is
        ``O(num_steps * batch_size * doc_ids_per_chunk)``, dominated by
        the ``unique()`` scan over each chunk's doc-id tensor.
    """

    # Sentinel for "no padding doc" used by qa_dataset.pack_to_fixed_chunks
    _PAD_DOC_ID = -1

    # Attribute names this class reads from `sampler` to replicate
    # DistributedSampler.__iter__. If any are missing, the sampler is not
    # a DistributedSampler (most commonly torch.utils.data.RandomSampler,
    # which is what StatefulDataLoader picks when the trainer process did
    # not call torch.distributed.init_process_group — the OpenRLHF Ray
    # head-trainer architecture is exactly that case: the EBFTTrainer Ray
    # actor itself is a coordinator and never initializes torch.distributed,
    # so its `prompts_dataloader.sampler` ends up as RandomSampler).
    # We refuse construction in that case rather than crash later inside
    # `peek_*` with a confusing AttributeError mid-training.
    _REQUIRED_SAMPLER_ATTRS = (
        "epoch", "seed", "shuffle", "num_replicas", "rank", "drop_last", "total_size",
    )

    def __init__(
        self,
        dataset,
        sampler,
        batch_size: int,
        drop_last: bool = True,
    ):
        missing = [a for a in self._REQUIRED_SAMPLER_ATTRS if not hasattr(sampler, a)]
        if missing:
            raise TypeError(
                f"PromptLookahead requires a DistributedSampler-like sampler with "
                f"attributes {self._REQUIRED_SAMPLER_ATTRS}; got "
                f"{type(sampler).__name__} which is missing {missing}. "
                f"Caller should treat this as 'prefetch unavailable' and proceed."
            )
        self._dataset = dataset
        self._sampler = sampler
        self._batch_size = int(batch_size)
        self._drop_last = bool(drop_last)

        # Raw prompts indexed by doc_id. May be empty if the dataset path
        # didn't populate it (e.g. legacy SequenceDataset). In that case
        # peek_prompts() will return [].
        self._prompts: List[str] = list(getattr(dataset, "prompts", None) or [])

        # Cached deterministic index list for the current epoch on this rank.
        # Recomputed when the sampler's epoch changes.
        self._cached_epoch: int = -1
        self._cached_indices: List[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def total_prompts(self) -> int:
        """Number of raw prompts indexed by doc_id."""
        return len(self._prompts)

    def steps_per_epoch_per_rank(self) -> int:
        """How many batches this rank yields per epoch (drop_last=True)."""
        # NOTE: when sampler.drop_last=True, len(rank_indices) is exactly
        # num_samples; integer division gives the batch count.
        self._maybe_rebuild_indices()
        if self._drop_last:
            return len(self._cached_indices) // self._batch_size
        return math.ceil(len(self._cached_indices) / self._batch_size)

    def peek_doc_ids_at_offset(
        self,
        num_consumed_in_epoch_on_this_rank: int,
        num_steps: int,
    ) -> List[int]:
        """Return unique doc_ids that will appear in the next ``num_steps``
        batches starting AFTER ``num_consumed_in_epoch_on_this_rank``
        sequence indices on this rank.

        Args:
            num_consumed_in_epoch_on_this_rank: Index of the next sequence
                this rank's sampler will yield. Equal to
                ``batches_consumed_in_this_epoch * batch_size``.
            num_steps: How many future batches to peek.

        Returns:
            Sorted list of unique doc_ids in the peeked window. Skips the
            ``-1`` padding sentinel emitted by ``pack_to_fixed_chunks``.
            Returns ``[]`` if ``num_steps <= 0`` or we're past end-of-epoch.
        """
        if num_steps <= 0:
            return []
        self._maybe_rebuild_indices()
        if not self._cached_indices:
            return []

        start = max(0, int(num_consumed_in_epoch_on_this_rank))
        end = min(start + int(num_steps) * self._batch_size, len(self._cached_indices))
        if start >= end:
            return []

        future_seq_indices = self._cached_indices[start:end]

        unique_dids: Set[int] = set()
        doc_id_chunks = self._dataset.doc_ids  # List[Tensor], shape (seq_len,)
        for seq_idx in future_seq_indices:
            # Defensive bound check; sampler indices are guaranteed in-range
            # but the dataset could have been truncated by max_samples after
            # the sampler was constructed.
            if seq_idx < 0 or seq_idx >= len(doc_id_chunks):
                continue
            doc_id_tensor = doc_id_chunks[seq_idx]
            for did in doc_id_tensor.unique().tolist():
                if did != self._PAD_DOC_ID:
                    unique_dids.add(int(did))

        return sorted(unique_dids)

    def peek_prompts_at_offset(
        self,
        num_consumed_in_epoch_on_this_rank: int,
        num_steps: int,
    ) -> List[str]:
        """Like ``peek_doc_ids_at_offset`` but returns raw prompt strings.

        Returns ``[]`` if the dataset has no ``.prompts`` attribute (e.g.
        a pure SequenceDataset path).
        """
        if not self._prompts:
            return []
        dids = self.peek_doc_ids_at_offset(num_consumed_in_epoch_on_this_rank, num_steps)
        out: List[str] = []
        for did in dids:
            if 0 <= did < len(self._prompts):
                out.append(self._prompts[did])
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_rebuild_indices(self) -> None:
        """Recompute the rank-local index list whenever ``sampler.epoch``
        changes. Mirrors ``DistributedSampler.__iter__`` but does NOT skip
        ``consumed_indicies`` — we keep the full list so callers can
        slice arbitrary windows.
        """
        if self._cached_epoch == self._sampler.epoch:
            return

        # 1) deterministic permutation of [0, len(dataset))
        if self._sampler.shuffle:
            # Local import keeps openrlhf.datasets free of a torch top-level
            # import in cold-start paths that don't need this module.
            import torch
            g = torch.Generator()
            g.manual_seed(self._sampler.seed + self._sampler.epoch)
            indices = torch.randperm(len(self._dataset), generator=g).tolist()
        else:
            indices = list(range(len(self._dataset)))

        # 2) pad / truncate to total_size
        total_size = self._sampler.total_size
        if not self._sampler.drop_last:
            padding_size = total_size - len(indices)
            if padding_size > 0:
                if padding_size <= len(indices):
                    indices = indices + indices[:padding_size]
                else:
                    indices = (indices * math.ceil(padding_size / max(1, len(indices))))[:total_size]
        else:
            indices = indices[:total_size]

        # 3) per-rank stride (DO NOT apply consumed_indicies)
        rank_indices = indices[self._sampler.rank : total_size : self._sampler.num_replicas]

        self._cached_indices = rank_indices
        self._cached_epoch = self._sampler.epoch
        logger.debug(
            "[PromptLookahead] rebuilt indices for epoch=%d rank=%d num_replicas=%d → %d sequences",
            self._cached_epoch, self._sampler.rank, self._sampler.num_replicas,
            len(self._cached_indices),
        )


__all__ = ["PromptLookahead"]
