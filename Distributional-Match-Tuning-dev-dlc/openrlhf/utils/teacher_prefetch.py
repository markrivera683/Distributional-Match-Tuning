"""Teacher prefetch / pre-queue mechanism for EBFT training.

Design goals
============
- Zero-invasive: all new code lives here; existing providers / trainers
  are untouched unless ``--enable_teacher_prefetch`` is passed.
- Non-blocking: the training thread never waits on the background fetcher.
- Cache-first: SQLite cache is the primary store; in-memory queue is a
  secondary fast path for completions that finished ahead of the step.
- Deduplication: a prompt is fetched at most once (skip if already in
  SQLite or already in-flight).
- Fallback: if prefetch fails the caller falls back to the original
  synchronous ``sample_targets`` call transparently.

Class layout
============
    TeacherPrefetchQueue
        Thread-safe dict  prompt_hash -> List[List[str]]

    TeacherPrefetchScheduler
        Deduplicates, submits to ThreadPoolExecutor, writes queue+cache.

    PrefetchingTeacherProvider
        Wraps any BaseTeacherProvider with cache-first sample_targets()
        and non-blocking schedule_prefetch().
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Set

from openrlhf.utils.teacher_provider import BaseTeacherProvider, TeacherCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Thread-safe in-memory queue
# ---------------------------------------------------------------------------

class TeacherPrefetchQueue:
    """Thread-safe mapping  prompt_hash -> completed teacher completions."""

    def __init__(self, max_size: int = 2048):
        self._lock = threading.Lock()
        self._data: Dict[str, List[List[str]]] = {}
        self._max_size = max_size

    def put(self, prompt_hash: str, completions: List[List[str]]) -> None:
        with self._lock:
            if len(self._data) >= self._max_size:
                try:
                    oldest = next(iter(self._data))
                    del self._data[oldest]
                except StopIteration:
                    pass
            self._data[prompt_hash] = completions

    def get(self, prompt_hash: str) -> Optional[List[List[str]]]:
        """Return completions if available, or None (non-blocking)."""
        with self._lock:
            return self._data.get(prompt_hash)

    def pop(self, prompt_hash: str) -> Optional[List[List[str]]]:
        """Return and remove completions if available (frees memory)."""
        with self._lock:
            return self._data.pop(prompt_hash, None)

    def size(self) -> int:
        with self._lock:
            return len(self._data)


# ---------------------------------------------------------------------------
# 2. Prefetch scheduler
# ---------------------------------------------------------------------------

class TeacherPrefetchScheduler:
    """Background work scheduler for teacher prefetch.

    Accepts future prompt batches from the training thread and submits
    teacher_provider.sample_targets(...) calls to a bounded thread pool.
    Results are written to SQLite (via the provider cache layer) and into
    the in-memory TeacherPrefetchQueue.

    Args:
        provider:    Any BaseTeacherProvider with SQLite cache attached.
        queue:       TeacherPrefetchQueue to write completed results into.
        max_workers: Maximum concurrent background fetch threads.
        cache:       Optional explicit TeacherCache for pre-flight dedup.
    """

    def __init__(
        self,
        provider: BaseTeacherProvider,
        queue: TeacherPrefetchQueue,
        max_workers: int = 8,
        cache: Optional[TeacherCache] = None,
    ):
        self._provider = provider
        self._queue = queue
        self._cache = cache
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="teacher_prefetch",
        )
        self._lock = threading.Lock()
        self._in_flight: Set[str] = set()
        self._futures: List[Future] = []

    @staticmethod
    def _prompt_hash(prompt: str) -> str:
        canonical = TeacherCache._canonicalize(prompt)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _is_cached_in_sqlite(
        self,
        prompt: str,
        model_name: str,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        api_style: str = "completions",
        system_prompt_id: str = "",
    ) -> bool:
        if self._cache is None:
            return False
        return self._cache.get(
            prompt, model_name, n_samples, temperature, top_p, max_new_tokens,
            api_style=api_style, system_prompt_id=system_prompt_id,
        ) is not None

    def schedule(
        self,
        prompts: List[str],
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        model_name: str = "",
        api_style: str = "completions",
        system_prompt_id: str = "",
    ) -> int:
        """Schedule prefetch for a list of prompts. Returns count submitted."""
        submitted = 0
        for prompt in prompts:
            ph = self._prompt_hash(prompt)
            with self._lock:
                if ph in self._in_flight:
                    continue
            if self._queue.get(ph) is not None:
                continue
            if self._is_cached_in_sqlite(
                prompt, model_name, n_samples, temperature, top_p, max_new_tokens,
                api_style=api_style, system_prompt_id=system_prompt_id,
            ):
                logger.debug("[Prefetch] SQLite HIT (skip): hash=%s", ph[:8])
                continue
            with self._lock:
                self._in_flight.add(ph)
            fut = self._executor.submit(
                self._fetch_one, prompt, ph,
                n_samples, temperature, top_p, max_new_tokens,
            )
            with self._lock:
                self._futures = [f for f in self._futures if not f.done()]
                self._futures.append(fut)
            submitted += 1
        if submitted:
            logger.debug(
                "[Prefetch] Scheduled %d/%d prompts (in_flight=%d queue=%d)",
                submitted, len(prompts), self.pending_count(), self._queue.size(),
            )
        return submitted

    def _fetch_one(
        self,
        prompt: str,
        prompt_hash: str,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> None:
        """Fetch one prompt in a background thread; write result to queue."""
        try:
            completions = self._provider.sample_targets(
                [prompt],
                n_samples=n_samples,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            if completions and completions[0]:
                self._queue.put(prompt_hash, completions)
                logger.debug(
                    "[Prefetch] OK hash=%s n=%d",
                    prompt_hash[:8], len(completions[0]),
                )
        except Exception as exc:
            logger.warning(
                "[Prefetch] fetch failed hash=%s: %s (training will fall back)",
                prompt_hash[:8], exc,
            )
        finally:
            with self._lock:
                self._in_flight.discard(prompt_hash)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._in_flight)

    def queue_size(self) -> int:
        return self._queue.size()

    def shutdown(self, wait: bool = False) -> None:
        logger.info(
            "[Prefetch] Shutting down scheduler (pending=%d queue=%d)",
            self.pending_count(), self.queue_size(),
        )
        self._executor.shutdown(wait=wait)


# ---------------------------------------------------------------------------
# 3. PrefetchingTeacherProvider  (drop-in wrapper)
# ---------------------------------------------------------------------------

class PrefetchingTeacherProvider(BaseTeacherProvider):
    """Drop-in wrapper around any BaseTeacherProvider that adds prefetch.

    Read strategy for sample_targets() (cache-first):
        1. In-memory TeacherPrefetchQueue  (zero-latency if hit).
        2. SQLite cache via wrapped provider (fast disk read, no HTTP).
        3. Fallback -> synchronous wrapped.sample_targets(...).

    Prefetch scheduling is triggered via schedule_prefetch(prompts, ...)
    which the trainer calls at the end of step t with the prompts for
    steps t+1 .. t+prefetch_depth.  Runs in background threads only.

    Args:
        wrapped:        Underlying provider (Remote or MultiWorker).
        prefetch_depth: Future batches to schedule per call (--prefetch_depth).
        max_workers:    Background thread pool size.
        cache:          Explicit TeacherCache; inferred from wrapped if None.
    """

    def __init__(
        self,
        wrapped: BaseTeacherProvider,
        prefetch_depth: int = 2,
        max_workers: int = 8,
        cache: Optional[TeacherCache] = None,
    ):
        self._wrapped = wrapped
        self._prefetch_depth = prefetch_depth

        # Infer cache from wrapped provider if not given explicitly
        if cache is None:
            cache = getattr(wrapped, "cache", None)
            if cache is None:
                workers = getattr(wrapped, "_workers", None)
                if workers:
                    cache = getattr(workers[0], "cache", None)

        self._cache = cache
        self._queue = TeacherPrefetchQueue()
        self._scheduler = TeacherPrefetchScheduler(
            provider=wrapped,
            queue=self._queue,
            max_workers=max_workers,
            cache=cache,
        )
        self._gen_params: Optional[dict] = None
        self._hits_memory = 0
        self._hits_sqlite = 0
        self._hits_fallback = 0

        logger.info(
            "[PrefetchingTeacherProvider] Init: depth=%d workers=%d cache=%s",
            prefetch_depth, max_workers, "ON" if cache else "OFF",
        )

    # -----------------------------------------------------------------------
    # Core read path
    # -----------------------------------------------------------------------

    def sample_targets(
        self,
        prompts: List[str],
        n_samples: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> List[List[str]]:
        """Return teacher completions using cache-first strategy."""
        self._gen_params = dict(
            n_samples=n_samples, temperature=temperature,
            top_p=top_p, max_new_tokens=max_new_tokens,
        )

        results: List[Optional[List[str]]] = [None] * len(prompts)
        fallback_indices: List[int] = []
        fallback_prompts: List[str] = []
        t0 = time.time()
        mem_hits = sqlite_hits = 0

        model_name = getattr(self._wrapped, "model_name", "")
        api_style = getattr(self._wrapped, "api_style", "completions")
        system_prompt_id = getattr(self._wrapped, "system_prompt_id", "")

        for idx, prompt in enumerate(prompts):
            ph = TeacherPrefetchScheduler._prompt_hash(prompt)

            # 1. memory queue
            cached = self._queue.pop(ph)
            if cached is not None:
                results[idx] = cached[0]
                mem_hits += 1
                continue

            # 2. SQLite cache
            if self._cache is not None:
                sqlite_result = self._cache.get(
                    prompt, model_name, n_samples, temperature,
                    top_p, max_new_tokens,
                    api_style=api_style, system_prompt_id=system_prompt_id,
                )
                if sqlite_result is not None:
                    results[idx] = sqlite_result
                    sqlite_hits += 1
                    continue

            # 3. needs synchronous fetch
            fallback_indices.append(idx)
            fallback_prompts.append(prompt)

        self._hits_memory += mem_hits
        self._hits_sqlite += sqlite_hits

        if fallback_prompts:
            self._hits_fallback += len(fallback_prompts)
            logger.info(
                "[PrefetchingTeacherProvider] mem=%d sqlite=%d fallback=%d (%.1fms)",
                mem_hits, sqlite_hits, len(fallback_prompts),
                (time.time() - t0) * 1000,
            )
            fb = self._wrapped.sample_targets(
                fallback_prompts, n_samples=n_samples,
                temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            for i, fb_idx in enumerate(fallback_indices):
                results[fb_idx] = fb[i]
        else:
            logger.info(
                "[PrefetchingTeacherProvider] 100%% prefetch hit "
                "mem=%d sqlite=%d (%.1fms)",
                mem_hits, sqlite_hits, (time.time() - t0) * 1000,
            )

        total = self._hits_memory + self._hits_sqlite + self._hits_fallback
        if total > 0:
            logger.debug(
                "[PrefetchingTeacherProvider] cumul: mem=%.0f%% sqlite=%.0f%% "
                "fallback=%.0f%% n=%d",
                100.0 * self._hits_memory / total,
                100.0 * self._hits_sqlite / total,
                100.0 * self._hits_fallback / total,
                total,
            )
        return results  # type: ignore[return-value]

    # -----------------------------------------------------------------------
    # Prefetch scheduling API (called by trainer)
    # -----------------------------------------------------------------------

    def schedule_prefetch(
        self,
        prompts: List[str],
        n_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> int:
        """Schedule background prefetch for future prompts.

        Gen params are optional; falls back to last sample_targets() call.
        Returns number of prompts actually submitted to the thread pool.
        """
        gp = self._gen_params or {}
        n_samples = n_samples if n_samples is not None else gp.get("n_samples", 4)
        temperature = temperature if temperature is not None else gp.get("temperature", 0.7)
        top_p = top_p if top_p is not None else gp.get("top_p", 0.95)
        max_new_tokens = (
            max_new_tokens if max_new_tokens is not None
            else gp.get("max_new_tokens", 512)
        )
        model_name = getattr(self._wrapped, "model_name", "")
        api_style = getattr(self._wrapped, "api_style", "completions")
        system_prompt_id = getattr(self._wrapped, "system_prompt_id", "")
        return self._scheduler.schedule(
            prompts=prompts,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            model_name=model_name,
            api_style=api_style,
            system_prompt_id=system_prompt_id,
        )

    def prefetch_depth(self) -> int:
        return self._prefetch_depth

    def status(self) -> dict:
        """Snapshot of prefetch stats for logging."""
        total = self._hits_memory + self._hits_sqlite + self._hits_fallback
        return {
            "prefetch_mem_hits": self._hits_memory,
            "prefetch_sqlite_hits": self._hits_sqlite,
            "prefetch_fallback": self._hits_fallback,
            "prefetch_hit_rate": (
                (self._hits_memory + self._hits_sqlite) / total if total else 0.0
            ),
            "prefetch_queue_size": self._queue.size(),
            "prefetch_pending": self._scheduler.pending_count(),
        }

    def shutdown(self, wait: bool = False) -> None:
        self._scheduler.shutdown(wait=wait)

    def __getattr__(self, name: str):
        """Delegate attribute lookups to the wrapped provider."""
        return getattr(self._wrapped, name)


# ---------------------------------------------------------------------------
# 4. Factory helper
# ---------------------------------------------------------------------------

def wrap_provider_with_prefetch(
    provider: BaseTeacherProvider,
    args,
) -> BaseTeacherProvider:
    """Optionally wrap ``provider`` with PrefetchingTeacherProvider.

    Returns the wrapped provider when ``args.enable_teacher_prefetch`` is
    True and the provider is a remote/dataset provider.  Otherwise returns
    ``provider`` unchanged so existing behaviour is completely unaffected.

    Args:
        provider:  Result of ``build_teacher_provider(args)``.
        args:      Parsed CLI args.  Reads:
                     --enable_teacher_prefetch (bool)
                     --prefetch_depth          (int, default 2)
                     --prefetch_max_workers    (int, default 8)
    """
    if provider is None:
        return None
    if not getattr(args, "enable_teacher_prefetch", False):
        return provider
    # Only makes sense for remote / dataset backends
    backend = getattr(args, "teacher_backend", "local")
    if backend == "local":
        logger.info(
            "[Prefetch] enable_teacher_prefetch=True but teacher_backend=local "
            "-- prefetch disabled (local model does not benefit)."
        )
        return provider

    depth = int(getattr(args, "prefetch_depth", 2))
    workers = int(getattr(args, "prefetch_max_workers", 8))

    logger.info(
        "[Prefetch] Wrapping teacher provider with PrefetchingTeacherProvider "
        "(depth=%d workers=%d backend=%s)",
        depth, workers, backend,
    )
    return PrefetchingTeacherProvider(
        wrapped=provider,
        prefetch_depth=depth,
        max_workers=workers,
    )
