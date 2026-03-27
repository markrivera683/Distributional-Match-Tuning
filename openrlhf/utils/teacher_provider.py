"""Teacher target provider abstraction for EBFT training.

Supports two backends:
  - local: existing SamplesGenerator path (no provider needed; handled directly)
  - remote: HTTP client for OpenAI/vLLM-compatible teacher service

The remote provider returns completion *text*; tokenization and feature extraction
remain local (inside the experience maker's critic pipeline).
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)


class BaseTeacherProvider(ABC):
    """Unified interface for obtaining teacher target completions."""

    @abstractmethod
    def sample_targets(
        self,
        prompts: List[str],
        n_samples: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> List[List[str]]:
        """Generate teacher completions for a batch of prompts.

        Args:
            prompts: List of prompt texts.
            n_samples: Number of completions per prompt (M).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            max_new_tokens: Maximum new tokens per completion.

        Returns:
            Nested list where ``result[i][j]`` is the j-th completion for prompt i.
        """
        ...


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

class TeacherCache:
    """Lightweight SQLite cache keyed on (prompt, model, generation params)."""

    def __init__(self, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        self.db_path = os.path.join(cache_dir, "teacher_cache.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "  key TEXT PRIMARY KEY,"
                "  completions TEXT NOT NULL,"
                "  model_name TEXT,"
                "  n_samples INTEGER,"
                "  temperature REAL,"
                "  top_p REAL,"
                "  max_new_tokens INTEGER,"
                "  created_at REAL"
                ")"
            )

    @staticmethod
    def _canonicalize(prompt: str) -> str:
        """Normalize prompt text before hashing or cache lookup.

        Strips leading/trailing whitespace and normalizes all line endings
        to LF.  This prevents spurious cache misses caused by OS-level
        line-ending differences or accidental trailing spaces.
        """
        return prompt.strip().replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _make_key(
        prompt: str,
        model_name: str,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        api_style: str = "completions",
    ) -> str:
        # api_style is included because chat_completions wraps the prompt in a
        # role-message server-side, producing a different completion distribution
        # than the raw completions endpoint for the same prompt text.
        # Prompt is canonicalized before hashing to avoid spurious misses.
        canonical = TeacherCache._canonicalize(prompt)
        raw = json.dumps(
            {
                "prompt": canonical,
                "model": model_name,
                "n": n_samples,
                "temp": round(temperature, 4),
                "top_p": round(top_p, 4),
                "max_tokens": max_new_tokens,
                "api_style": api_style,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self, prompt, model_name, n_samples, temperature, top_p, max_new_tokens,
        api_style: str = "completions",
    ) -> Optional[List[str]]:
        key = self._make_key(prompt, model_name, n_samples, temperature, top_p, max_new_tokens, api_style)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT completions FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(
        self, prompt, model_name, n_samples, temperature, top_p, max_new_tokens, completions,
        api_style: str = "completions",
    ):
        key = self._make_key(prompt, model_name, n_samples, temperature, top_p, max_new_tokens, api_style)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache"
                " (key, completions, model_name, n_samples, temperature, top_p, max_new_tokens, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    json.dumps(completions, ensure_ascii=False),
                    model_name,
                    n_samples,
                    temperature,
                    top_p,
                    max_new_tokens,
                    time.time(),
                ),
            )


# ---------------------------------------------------------------------------
# Dataset provider (pre-exported HF dataset, no API / SQLite at train time)
# ---------------------------------------------------------------------------

class DatasetTeacherProvider(BaseTeacherProvider):
    """Look up teacher completions from a pre-exported HuggingFace Dataset.

    Supports two dataset layouts:

    1. **question-keyed** (has ``question`` column):
       Lookup by exact question text match.  Generation params are ignored.

    2. **hash-keyed** (has ``cache_key`` column):
       Lookup by SHA-256 cache key, computed from (prompt, model, n,
       temperature, top_p, max_tokens) — identical to ``TeacherCache._make_key``.
       Requires ``model_name`` to be set (read from the first row's
       ``model_name`` column).

    Both layouts must have a ``teacher_completions`` column (List[str]).
    """

    def __init__(self, dataset_path: str):
        from datasets import load_from_disk
        ds = load_from_disk(dataset_path)

        cols = set(ds.column_names)
        self._hash_mode = "cache_key" in cols and "question" not in cols
        self._lookup: dict[str, list[str]] = {}

        if self._hash_mode:
            self.model_name = ds[0]["model_name"]
            for row in ds:
                self._lookup[row["cache_key"]] = row["teacher_completions"]
            logger.info(
                "[DatasetTeacher] Loaded %d entries (hash-keyed, model=%s) from %s",
                len(self._lookup), self.model_name, dataset_path,
            )
        else:
            self.model_name = None
            for row in ds:
                self._lookup[row["question"]] = row["teacher_completions"]
            logger.info(
                "[DatasetTeacher] Loaded %d questions (text-keyed) from %s",
                len(self._lookup), dataset_path,
            )

    def sample_targets(
        self,
        prompts: List[str],
        n_samples: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> List[List[str]]:
        results: List[List[str]] = []
        miss = 0
        short = 0
        for prompt in prompts:
            # Canonicalize before lookup so trailing whitespace/CR differences
            # between online-warmup and offline-training do not cause spurious misses.
            canonical = TeacherCache._canonicalize(prompt)
            if self._hash_mode:
                key = TeacherCache._make_key(
                    canonical, self.model_name, n_samples,
                    temperature, top_p, max_new_tokens,
                )
                completions = self._lookup.get(key)
            else:
                # text-keyed: try canonical first, fall back to raw prompt
                completions = self._lookup.get(canonical)
                if completions is None:
                    completions = self._lookup.get(prompt)

            if completions is None:
                miss += 1
                results.append([""] * n_samples)
            else:
                if len(completions) >= n_samples:
                    results.append(completions[:n_samples])
                else:
                    short += 1
                    padded = completions + [""] * (n_samples - len(completions))
                    results.append(padded)

        if miss:
            logger.warning(
                "[DatasetTeacher] %d / %d prompts NOT found in dataset (mode=%s). "
                "These receive empty-string completions — teacher signal LOST. "
                "Check TEACHER_DATASET_PATH and that n_samples/temperature/top_p/"
                "max_new_tokens match the dataset export parameters.",
                miss, len(prompts),
                "hash" if self._hash_mode else "text",
            )
        if short:
            logger.warning(
                "[DatasetTeacher] %d / %d prompts have fewer than %d completions — "
                "padded with empty strings. Reduce CF_TEACHER_N_SAMPLES or re-export "
                "dataset with a larger n_samples value.",
                short, len(prompts), n_samples,
            )
        return results

# ---------------------------------------------------------------------------
# Remote provider
# ---------------------------------------------------------------------------

class RemoteTeacherProvider(BaseTeacherProvider):
    """HTTP client for an OpenAI / vLLM compatible teacher completion service.

    Supports two API styles:
      - ``completions``:      POST /v1/completions      (text completion)
      - ``chat_completions``: POST /v1/chat/completions  (chat completion)

    Error strategy (v1): **fail-fast**.  If all retries are exhausted the
    exception propagates and training stops.  This prevents silent target
    contamination.
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: str = "EMPTY",
        api_style: str = "completions",
        timeout: int = 120,
        max_retries: int = 3,
        batch_size: int = 8,
        cache: Optional[TeacherCache] = None,
    ):
        if api_style not in ("completions", "chat_completions"):
            raise ValueError(
                f"api_style must be 'completions' or 'chat_completions', got {api_style!r}"
            )

        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.api_style = api_style
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.cache = cache

        self._session = requests.Session()
        adapter = HTTPAdapter(pool_maxsize=max(batch_size, 10))
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
        )

        logger.info(
            "[RemoteTeacher] Init: api_base=%s, model=%s, style=%s, "
            "timeout=%ds, retries=%d, concurrency=%d, cache=%s",
            api_base, model_name, api_style,
            timeout, max_retries, batch_size,
            "ON" if cache else "OFF",
        )

    # ---- single-prompt request with retry --------------------------------

    def _build_request(self, prompt: str, n_samples: int,
                       temperature: float, top_p: float,
                       max_new_tokens: int):
        """Return (url, payload) for the configured API style."""
        if self.api_style == "chat_completions":
            url = f"{self.api_base}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "n": n_samples,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
            }
        else:
            url = f"{self.api_base}/completions"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "n": n_samples,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_new_tokens,
            }
        return url, payload

    def _parse_completions(self, data: dict) -> List[str]:
        """Extract completion texts from the API response."""
        choices = data.get("choices", [])
        if self.api_style == "chat_completions":
            return [
                c.get("message", {}).get("content", "")
                for c in choices
            ]
        else:
            return [c.get("text", "") for c in choices]

    def _request_single(
        self,
        prompt: str,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> List[str]:
        # Canonicalize before cache lookup so trailing whitespace/CR differences
        # do not produce spurious misses.
        prompt = TeacherCache._canonicalize(prompt)

        if self.cache:
            cached = self.cache.get(
                prompt, self.model_name, n_samples, temperature, top_p, max_new_tokens,
                api_style=self.api_style,
            )
            if cached is not None:
                logger.debug("[RemoteTeacher] cache HIT (prompt len=%d)", len(prompt))
                return cached

        url, payload = self._build_request(
            prompt, n_samples, temperature, top_p, max_new_tokens,
        )

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                completions = self._parse_completions(data)
                if len(completions) < n_samples:
                    raise ValueError(
                        f"Server returned {len(completions)} choices, expected {n_samples}"
                    )
                completions = completions[:n_samples]

                if self.cache:
                    self.cache.put(
                        prompt, self.model_name, n_samples,
                        temperature, top_p, max_new_tokens, completions,
                        api_style=self.api_style,
                    )

                return completions

            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "[RemoteTeacher] attempt %d/%d failed: %s  (retry in %ds)",
                        attempt, self.max_retries, exc, wait,
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"[RemoteTeacher] All {self.max_retries} attempts failed "
            f"(prompt len={len(prompt)}). Last error: {last_exc}"
        ) from last_exc

    # ---- batch interface -------------------------------------------------

    def sample_targets(
        self,
        prompts: List[str],
        n_samples: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> List[List[str]]:
        logger.info(
            "[RemoteTeacher] Requesting %d prompts x %d samples "
            "(temp=%.2f, top_p=%.2f, max_tokens=%d)",
            len(prompts), n_samples, temperature, top_p, max_new_tokens,
        )
        t0 = time.time()
        results: List[Optional[List[str]]] = [None] * len(prompts)
        cache_hits = 0

        with ThreadPoolExecutor(max_workers=self.batch_size) as pool:
            futures = {}
            for idx, prompt in enumerate(prompts):
                fut = pool.submit(
                    self._request_single,
                    prompt, n_samples, temperature, top_p, max_new_tokens,
                )
                futures[fut] = idx

            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()

        if self.cache:
            for prompt in prompts:
                if self.cache.get(
                    prompt, self.model_name, n_samples, temperature, top_p, max_new_tokens
                ) is not None:
                    cache_hits += 1

        elapsed = time.time() - t0
        logger.info(
            "[RemoteTeacher] Done: %d prompts in %.1fs, cache_hits=%d",
            len(prompts), elapsed, cache_hits,
        )
        return results


# ---------------------------------------------------------------------------
# Multi-worker provider (round-robin load balancing across N vLLM workers)
# ---------------------------------------------------------------------------

class MultiWorkerTeacherProvider(BaseTeacherProvider):
    """Round-robin load balancer across multiple RemoteTeacherProvider workers.

    Each worker corresponds to one independently running vLLM server instance.
    Requests are distributed across workers so that all servers stay busy.

    Cache is **per-worker** (each worker has its own SQLite file in a
    sub-directory of ``cache_dir``).  This avoids SQLite write-lock contention
    when multiple threads hit different workers simultaneously.

    Args:
        api_bases:    List of base URLs, one per vLLM worker.
                      e.g. ["http://172.17.0.26:8000/v1",
                             "http://172.17.0.26:8001/v1"]
        model_name:   Model name served at every endpoint (must be identical).
        api_key:      Shared bearer token.
        api_style:    ``"completions"`` or ``"chat_completions"``.
        timeout:      Per-request HTTP timeout in seconds.
        max_retries:  Retries per request (with exponential back-off).
        batch_size:   Total concurrent in-flight requests across ALL workers.
        cache_dir:    Root cache directory; each worker gets its own sub-dir.
    """

    def __init__(
        self,
        api_bases: List[str],
        model_name: str,
        api_key: str = "EMPTY",
        api_style: str = "completions",
        timeout: int = 120,
        max_retries: int = 3,
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
    ):
        if not api_bases:
            raise ValueError("api_bases must contain at least one URL")

        self._num_workers = len(api_bases)
        # per-worker concurrency: distribute total batch_size evenly
        per_worker_concurrency = max(1, batch_size // self._num_workers)

        self._workers: List[RemoteTeacherProvider] = []
        for i, base in enumerate(api_bases):
            worker_cache: Optional[TeacherCache] = None
            if cache_dir is not None:
                worker_cache_dir = os.path.join(cache_dir, f"worker_{i}")
                worker_cache = TeacherCache(worker_cache_dir)
            self._workers.append(
                RemoteTeacherProvider(
                    api_base=base,
                    model_name=model_name,
                    api_key=api_key,
                    api_style=api_style,
                    timeout=timeout,
                    max_retries=max_retries,
                    batch_size=per_worker_concurrency,
                    cache=worker_cache,
                )
            )

        # Shared atomic counter for round-robin assignment (thread-safe via GIL)
        self._counter = 0
        self._total_batch_size = batch_size

        logger.info(
            "[MultiWorkerTeacher] Init: %d workers, total_concurrency=%d, "
            "per_worker_concurrency=%d\n  workers: %s",
            self._num_workers, batch_size, per_worker_concurrency,
            ", ".join(api_bases),
        )

    @staticmethod
    def _hrw_score(prompt: str, worker_url: str) -> int:
        """Rendezvous / Highest Random Weight (HRW) score for one (prompt, worker) pair.

        HRW guarantees that when num_workers changes N → N+1, only ~1/(N+1) of
        prompts are remapped (to the new worker).  This is strictly better than
        modulo hashing, which remaps ~(N-1)/N * 2/3 of keys on a 2→3 change.

        The same prompt always picks the same worker (highest score wins),
        giving each worker's SQLite cache a stable, disjoint partition of the
        prompt space.
        """
        raw = f"{prompt}\x00{worker_url}"
        return int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16)

    def _pick_worker(self, prompt: str, exclude: set = None) -> tuple:
        """Select worker via Rendezvous (HRW) hashing with fallback support.

        Returns (worker_index, RemoteTeacherProvider).

        Args:
            prompt:  Canonicalized prompt text (call TeacherCache._canonicalize first).
            exclude: Set of worker indices to skip (used for fault-tolerance fallback).
        """
        exclude = exclude or set()
        best_idx, best_score = -1, -1
        for i, w in enumerate(self._workers):
            if i in exclude:
                continue
            score = self._hrw_score(prompt, w.api_base)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1:
            raise RuntimeError("All workers excluded — no worker available for this prompt")
        return best_idx, self._workers[best_idx]

    def _request_with_fallback(
        self,
        prompt: str,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> List[str]:
        """Send request to HRW-selected worker; fall back to next candidate on failure.

        On the first failure (all retries exhausted on primary worker), the
        primary worker index is added to ``exclude`` and the HRW algorithm
        selects the next best worker.  This avoids a cache-miss storm because
        the fallback worker may already have a cache entry (from warmup) for
        this prompt if it was the secondary HRW candidate.

        At most ``num_workers`` attempts are made in total.
        """
        prompt = TeacherCache._canonicalize(prompt)
        excluded: set = set()
        last_exc: Optional[Exception] = None

        for _ in range(self._num_workers):
            try:
                worker_idx, worker = self._pick_worker(prompt, exclude=excluded)
                return worker._request_single(
                    prompt, n_samples, temperature, top_p, max_new_tokens
                )
            except RuntimeError as exc:
                # All workers excluded
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[MultiWorkerTeacher] worker %d (%s) failed for prompt "
                    "(len=%d): %s — trying next HRW candidate",
                    worker_idx, self._workers[worker_idx].api_base, len(prompt), exc,
                )
                excluded.add(worker_idx)

        raise RuntimeError(
            f"[MultiWorkerTeacher] All {self._num_workers} workers failed. "
            f"Last error: {last_exc}"
        ) from last_exc

    def sample_targets(
        self,
        prompts: List[str],
        n_samples: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
    ) -> List[List[str]]:
        """Distribute prompts via HRW hashing with per-worker fault tolerance.

        Each prompt is canonicalized, then deterministically routed to its
        HRW-selected worker.  If that worker fails, the next HRW candidate
        is tried (up to num_workers attempts).  All requests run concurrently
        inside a shared ThreadPoolExecutor.
        """
        logger.info(
            "[MultiWorkerTeacher] Requesting %d prompts x %d samples "
            "across %d workers (total_concurrency=%d)",
            len(prompts), n_samples, self._num_workers, self._total_batch_size,
        )
        t0 = time.time()
        results: List[Optional[List[str]]] = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=self._total_batch_size) as pool:
            futures = {}
            for idx, prompt in enumerate(prompts):
                fut = pool.submit(
                    self._request_with_fallback,
                    prompt, n_samples, temperature, top_p, max_new_tokens,
                )
                futures[fut] = idx

            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()

        elapsed = time.time() - t0
        logger.info(
            "[MultiWorkerTeacher] Done: %d prompts in %.1fs (%.1f prompts/s)",
            len(prompts), elapsed, len(prompts) / max(elapsed, 1e-6),
        )
        return results
        """Check reachability of all workers.  Returns {url: ok/error}."""
        import requests as _req
        status = {}
        for w in self._workers:
            url = f"{w.api_base}/models"
            try:
                r = _req.get(url, headers={"Authorization": f"Bearer {w.api_key}"}, timeout=10)
                status[w.api_base] = "ok" if r.ok else f"http_{r.status_code}"
            except Exception as e:
                status[w.api_base] = f"error: {e}"
        return status


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_teacher_provider(args) -> Optional[BaseTeacherProvider]:
    """Build the appropriate teacher provider from CLI args.

    Returns ``None`` for the local backend (the existing SamplesGenerator
    path is used directly, no provider wrapper needed).
    """
    backend = getattr(args, "teacher_backend", "local")

    if backend == "local":
        return None

    if backend == "dataset":
        dataset_path = getattr(args, "teacher_dataset_path", None)
        if not dataset_path:
            raise ValueError("--teacher_dataset_path is required when --teacher_backend=dataset")
        return DatasetTeacherProvider(dataset_path)

    if backend != "remote":
        raise ValueError(f"Unknown teacher_backend: {backend!r}")

    api_base_raw = getattr(args, "teacher_api_base", None)
    if not api_base_raw:
        raise ValueError("--teacher_api_base is required when --teacher_backend=remote")

    model_name = getattr(args, "teacher_model_name", None)
    if not model_name:
        raise ValueError("--teacher_model_name is required when --teacher_backend=remote")

    # Support comma-separated list of worker URLs for multi-worker deployment.
    # Single URL → original RemoteTeacherProvider (no overhead).
    # Multiple URLs → MultiWorkerTeacherProvider with round-robin load balancing.
    api_bases = [u.strip() for u in api_base_raw.split(",") if u.strip()]

    cache_dir = None
    if getattr(args, "teacher_cache_enable", False):
        cache_dir = getattr(args, "teacher_cache_dir", None) or os.path.join(
            getattr(args, "save_path", "./ckpt"), "teacher_cache"
        )
        logger.info(
            "[TeacherProvider] Cache enabled, root dir: %s (%d worker sub-dirs)",
            cache_dir, len(api_bases),
        )

    batch_size = int(getattr(args, "teacher_remote_batch_size", 8))
    api_key = getattr(args, "teacher_api_key", "EMPTY")
    api_style = getattr(args, "teacher_api_style", "completions")
    timeout = int(getattr(args, "teacher_timeout", 120))
    max_retries = int(getattr(args, "teacher_max_retries", 3))

    if len(api_bases) == 1:
        # Single worker — use original provider (no extra abstraction layer)
        cache = None
        if cache_dir is not None:
            cache = TeacherCache(cache_dir)
        return RemoteTeacherProvider(
            api_base=api_bases[0],
            model_name=model_name,
            api_key=api_key,
            api_style=api_style,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size,
            cache=cache,
        )
    else:
        # Multiple workers — round-robin across all endpoints
        logger.info(
            "[TeacherProvider] Multi-worker mode: %d workers detected from "
            "--teacher_api_base (comma-separated)",
            len(api_bases),
        )
        return MultiWorkerTeacherProvider(
            api_bases=api_bases,
            model_name=model_name,
            api_key=api_key,
            api_style=api_style,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )
