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
    def _make_key(
        prompt: str,
        model_name: str,
        n_samples: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> str:
        raw = json.dumps(
            {
                "prompt": prompt,
                "model": model_name,
                "n": n_samples,
                "temp": round(temperature, 4),
                "top_p": round(top_p, 4),
                "max_tokens": max_new_tokens,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(
        self, prompt, model_name, n_samples, temperature, top_p, max_new_tokens
    ) -> Optional[List[str]]:
        key = self._make_key(prompt, model_name, n_samples, temperature, top_p, max_new_tokens)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT completions FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(
        self, prompt, model_name, n_samples, temperature, top_p, max_new_tokens, completions
    ):
        key = self._make_key(prompt, model_name, n_samples, temperature, top_p, max_new_tokens)
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
        if self.cache:
            cached = self.cache.get(
                prompt, self.model_name, n_samples, temperature, top_p, max_new_tokens
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

    if backend != "remote":
        raise ValueError(f"Unknown teacher_backend: {backend!r}")

    api_base = getattr(args, "teacher_api_base", None)
    if not api_base:
        raise ValueError("--teacher_api_base is required when --teacher_backend=remote")

    model_name = getattr(args, "teacher_model_name", None)
    if not model_name:
        raise ValueError("--teacher_model_name is required when --teacher_backend=remote")

    cache = None
    if getattr(args, "teacher_cache_enable", False):
        cache_dir = getattr(args, "teacher_cache_dir", None) or os.path.join(
            getattr(args, "save_path", "./ckpt"), "teacher_cache"
        )
        cache = TeacherCache(cache_dir)
        logger.info("[RemoteTeacher] Cache dir: %s", cache_dir)

    return RemoteTeacherProvider(
        api_base=api_base,
        model_name=model_name,
        api_key=getattr(args, "teacher_api_key", "EMPTY"),
        api_style=getattr(args, "teacher_api_style", "completions"),
        timeout=int(getattr(args, "teacher_timeout", 120)),
        max_retries=int(getattr(args, "teacher_max_retries", 3)),
        batch_size=int(getattr(args, "teacher_remote_batch_size", 8)),
        cache=cache,
    )
