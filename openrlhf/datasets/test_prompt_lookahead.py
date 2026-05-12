"""Unit tests for PromptLookahead.

Verifies that ``PromptLookahead.peek_doc_ids_at_offset`` returns exactly
the same set of unique ``doc_ids`` as actually iterating the
``DataLoader`` for the same window. Without this guarantee, the
prefetched teacher prompts won't line up with what the trainer asks for
on the next step → mem-queue hit rate stays at 0%.

Run::
    cd /mnt/data/ebft-distribution-new/code
    /mnt/workspace/venvs/.venv/bin/python -m openrlhf.datasets.test_prompt_lookahead
"""
from __future__ import annotations

import sys
import unittest
from typing import List, Set

import torch
from torch.utils.data import DataLoader, Dataset

from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.datasets.prompt_lookahead import PromptLookahead


# ---------------------------------------------------------------------------
# Test fixture: a minimal QADataset-shaped object
# ---------------------------------------------------------------------------

class _FakeQADataset(Dataset):
    """Mimics the bits of QADataset that PromptLookahead relies on:
    - ``__len__()``  : number of packed sequences (chunks)
    - ``doc_ids``    : List[Tensor] with shape (seq_len,), each element a doc index
    - ``prompts``    : List[str] indexed by doc_id (raw question text)
    Sequences themselves are dummy tensors; they don't matter for the lookahead.
    """

    def __init__(self, num_chunks: int = 50, seq_len: int = 16, num_docs: int = 12):
        torch.manual_seed(0)
        self.seq_len = seq_len
        # Each chunk packs 1-3 docs from [0, num_docs); we generate a deterministic
        # but irregular pattern so unique() per chunk gives realistic variety.
        self.doc_ids: List[torch.Tensor] = []
        self.sequences: List[torch.Tensor] = []
        for c in range(num_chunks):
            # rotate doc IDs deterministically
            ids = torch.tensor(
                [(c + t) % num_docs for t in range(seq_len)],
                dtype=torch.long,
            )
            # add some -1 padding at the tail of half the chunks to test
            # the PAD_DOC_ID skip path
            if c % 2 == 0 and seq_len >= 4:
                ids[-2:] = -1
            self.doc_ids.append(ids)
            self.sequences.append(torch.zeros(seq_len, dtype=torch.long))

        self.prompts: List[str] = [f"<question_{d}>" for d in range(num_docs)]
        self.answer_masks: List[torch.Tensor] = [
            torch.zeros(seq_len, dtype=torch.long) for _ in range(num_chunks)
        ]

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __getitem__(self, idx):
        return self.sequences[idx], self.doc_ids[idx], self.answer_masks[idx]


def _collate_fn(batch):
    return list(zip(*batch))


def _truth_unique_doc_ids_for_loader(
    dataset: _FakeQADataset,
    sampler,
    batch_size: int,
    skip_batches: int,
    take_batches: int,
) -> List[int]:
    """Brute-force: actually iterate the dataloader, collect unique doc_ids
    in the take_batches window. This is the source of truth."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    seen: Set[int] = set()
    for batch_idx, (_, doc_id_chunks_batch, _) in enumerate(loader):
        if batch_idx < skip_batches:
            continue
        if batch_idx >= skip_batches + take_batches:
            break
        for doc_id_tensor in doc_id_chunks_batch:
            for did in doc_id_tensor.unique().tolist():
                if did != -1:
                    seen.add(int(did))
    return sorted(seen)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPromptLookahead(unittest.TestCase):

    BATCH_SIZE = 4
    EPOCH = 0
    SEED = 43
    NUM_REPLICAS = 1
    RANK = 0

    def _build(self, num_chunks=50):
        ds = _FakeQADataset(num_chunks=num_chunks)
        sampler = DistributedSampler(
            ds,
            num_replicas=self.NUM_REPLICAS,
            rank=self.RANK,
            shuffle=True,
            seed=self.SEED,
            drop_last=True,
        )
        sampler.set_epoch(self.EPOCH)
        la = PromptLookahead(ds, sampler, self.BATCH_SIZE, drop_last=True)
        return ds, sampler, la

    def test_peek_matches_dataloader_at_start(self):
        ds, sampler, la = self._build()
        truth = _truth_unique_doc_ids_for_loader(ds, sampler, self.BATCH_SIZE, 0, 3)

        sampler.set_epoch(self.EPOCH)  # reset; the truth iter consumed it
        la = PromptLookahead(ds, sampler, self.BATCH_SIZE, drop_last=True)
        peeked = la.peek_doc_ids_at_offset(num_consumed_in_epoch_on_this_rank=0, num_steps=3)
        self.assertEqual(truth, peeked, "peek must equal dataloader's actual doc_ids")

    def test_peek_matches_dataloader_mid_epoch(self):
        ds, sampler, la = self._build()
        # skip 4 batches, take next 3
        truth = _truth_unique_doc_ids_for_loader(ds, sampler, self.BATCH_SIZE, 4, 3)

        sampler.set_epoch(self.EPOCH)
        la = PromptLookahead(ds, sampler, self.BATCH_SIZE, drop_last=True)
        peeked = la.peek_doc_ids_at_offset(
            num_consumed_in_epoch_on_this_rank=4 * self.BATCH_SIZE, num_steps=3
        )
        self.assertEqual(truth, peeked, "mid-epoch peek must match")

    def test_peek_handles_end_of_epoch(self):
        ds, sampler, la = self._build(num_chunks=50)
        # 50 chunks / batch_size=4 / drop_last=True → 12 batches per epoch.
        # If we ask for 100 batches near the end, return only what's left.
        peeked_near_end = la.peek_doc_ids_at_offset(
            num_consumed_in_epoch_on_this_rank=11 * self.BATCH_SIZE,
            num_steps=100,
        )
        # Only batch idx 11 left (= 4 chunks). Verify nonempty and bounded.
        self.assertTrue(len(peeked_near_end) > 0)

        # Past end → empty
        peeked_past_end = la.peek_doc_ids_at_offset(
            num_consumed_in_epoch_on_this_rank=12 * self.BATCH_SIZE + 100,
            num_steps=4,
        )
        self.assertEqual(peeked_past_end, [])

    def test_peek_skips_padding_sentinel(self):
        """PAD_DOC_ID = -1 must never appear in peeked doc_ids."""
        ds, sampler, la = self._build()
        for offset in (0, 8, 16, 24):
            peeked = la.peek_doc_ids_at_offset(
                num_consumed_in_epoch_on_this_rank=offset, num_steps=2
            )
            self.assertNotIn(-1, peeked)

    def test_peek_prompts_maps_doc_ids_correctly(self):
        ds, sampler, la = self._build()
        peeked_dids = la.peek_doc_ids_at_offset(num_consumed_in_epoch_on_this_rank=0, num_steps=2)
        peeked_prompts = la.peek_prompts_at_offset(
            num_consumed_in_epoch_on_this_rank=0, num_steps=2
        )
        expected = [f"<question_{d}>" for d in peeked_dids]
        self.assertEqual(peeked_prompts, expected)

    def test_rebuild_on_epoch_change(self):
        """When sampler.epoch changes, the cached index permutation must be
        rebuilt (not stale). Verify by comparing the cached_indices lists
        rather than doc_ids — with only 12 docs even a 1-batch window can
        cover all of them, making doc-id-set comparison unreliable."""
        ds, sampler, la = self._build()
        la._maybe_rebuild_indices()
        indices_e0 = list(la._cached_indices)

        sampler.set_epoch(self.EPOCH + 1)
        la._maybe_rebuild_indices()
        indices_e1 = list(la._cached_indices)
        self.assertNotEqual(indices_e0, indices_e1,
                            "permutation must change across epochs (different seed+epoch)")

    def test_zero_steps_or_negative(self):
        _, _, la = self._build()
        self.assertEqual(la.peek_doc_ids_at_offset(0, 0), [])
        self.assertEqual(la.peek_doc_ids_at_offset(0, -3), [])

    def test_multi_rank_disjoint_with_overlap_rules(self):
        """Each rank gets a different stride; same epoch → different windows."""
        ds = _FakeQADataset(num_chunks=64)
        sampler_0 = DistributedSampler(ds, num_replicas=2, rank=0,
                                       shuffle=True, seed=self.SEED, drop_last=True)
        sampler_1 = DistributedSampler(ds, num_replicas=2, rank=1,
                                       shuffle=True, seed=self.SEED, drop_last=True)
        sampler_0.set_epoch(self.EPOCH)
        sampler_1.set_epoch(self.EPOCH)
        la_0 = PromptLookahead(ds, sampler_0, self.BATCH_SIZE, drop_last=True)
        la_1 = PromptLookahead(ds, sampler_1, self.BATCH_SIZE, drop_last=True)

        # Each rank's first batch should peek different sequences (different
        # stride). Doc-ID overlap is allowed, but the underlying sequence
        # indices must be disjoint.
        # We can't directly access the indices from outside; verify by
        # rebuilding and reading the cache.
        la_0._maybe_rebuild_indices()
        la_1._maybe_rebuild_indices()
        # No overlap in the per-rank index lists
        self.assertEqual(set(la_0._cached_indices) & set(la_1._cached_indices), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
