#!/usr/bin/env python3
"""Standalone smoke test for teacher_provider.py (no GPU required).

Starts the mock teacher server, verifies the RemoteTeacherProvider can
fetch completions, and validates caching + retry behaviour.

Usage:
    python scripts/test_teacher_provider.py
"""

import os
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PORT = 18291


def start_mock_server():
    proc = subprocess.Popen(
        [sys.executable, os.path.join(REPO, "scripts", "mock_teacher_server.py"),
         "--port", str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    time.sleep(2)
    assert proc.poll() is None, "Mock server failed to start"
    return proc


def test_remote_provider():
    from openrlhf.utils.teacher_provider import RemoteTeacherProvider, TeacherCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TeacherCache(tmpdir)
        provider = RemoteTeacherProvider(
            api_base=f"http://localhost:{PORT}/v1",
            model_name="mock-teacher",
            api_key="EMPTY",
            timeout=10,
            max_retries=2,
            batch_size=2,
            cache=cache,
        )

        prompts = ["Hello world", "What is 2+2?", "Tell me a story"]
        n_samples = 3

        # First call: should hit the API
        results = provider.sample_targets(prompts, n_samples, temperature=0.7, top_p=0.9, max_new_tokens=32)
        assert len(results) == len(prompts), f"Expected {len(prompts)} results, got {len(results)}"
        for i, r in enumerate(results):
            assert len(r) == n_samples, f"Prompt {i}: expected {n_samples} completions, got {len(r)}"
            for j, text in enumerate(r):
                assert isinstance(text, str), f"Completion [{i}][{j}] is not str: {type(text)}"
                assert len(text) > 0, f"Completion [{i}][{j}] is empty"

        print(f"[PASS] Remote provider returned {len(prompts)} x {n_samples} completions")
        print(f"  Example: {results[0][0][:80]}...")

        # Second call: should hit cache
        results2 = provider.sample_targets(prompts, n_samples, temperature=0.7, top_p=0.9, max_new_tokens=32)
        assert results2 == results, "Cached results differ from original"
        print("[PASS] Cache returns identical results")


def test_cache_standalone():
    from openrlhf.utils.teacher_provider import TeacherCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TeacherCache(tmpdir)
        key_args = ("test prompt", "model-x", 2, 0.7, 0.9, 64)

        assert cache.get(*key_args) is None, "Cache should be empty initially"
        cache.put(*key_args, completions=["hello", "world"])
        result = cache.get(*key_args)
        assert result == ["hello", "world"], f"Cache get returned: {result}"
        print("[PASS] TeacherCache put/get works")


def test_build_factory():
    from openrlhf.utils.teacher_provider import build_teacher_provider
    from types import SimpleNamespace

    args_local = SimpleNamespace(teacher_backend="local")
    assert build_teacher_provider(args_local) is None, "local backend should return None"
    print("[PASS] build_teacher_provider(local) returns None")

    args_remote = SimpleNamespace(
        teacher_backend="remote",
        teacher_api_base=f"http://localhost:{PORT}/v1",
        teacher_model_name="mock",
        teacher_api_key="EMPTY",
        teacher_timeout=10,
        teacher_max_retries=1,
        teacher_remote_batch_size=2,
        teacher_cache_enable=False,
        teacher_cache_dir=None,
        save_path="/tmp",
    )
    provider = build_teacher_provider(args_remote)
    assert provider is not None, "remote backend should return a provider"
    print("[PASS] build_teacher_provider(remote) returns RemoteTeacherProvider")


def test_per_block_distinct_completions():
    """Verify that different block contexts produce different completions.

    The mock server returns deterministic pseudo-random text seeded by
    the prompt hash.  Different block contexts must yield different
    completions — if they were identical, it would mean per-block
    dispatch is broken.
    """
    from openrlhf.utils.teacher_provider import RemoteTeacherProvider

    provider = RemoteTeacherProvider(
        api_base=f"http://localhost:{PORT}/v1",
        model_name="mock-teacher",
        timeout=10,
        max_retries=1,
        batch_size=4,
    )

    block_contexts = [
        "Block context zero: the quick brown fox",
        "Block context one: jumped over the lazy dog",
        "Block context two: in the distant mountains",
        "Block context three: near the flowing river",
    ]
    M = 2
    results = provider.sample_targets(block_contexts, M, 0.7, 0.9, 32)

    assert len(results) == len(block_contexts)
    for i, r in enumerate(results):
        assert len(r) == M, f"Block {i}: expected {M} completions, got {len(r)}"

    all_first_samples = [results[i][0] for i in range(len(block_contexts))]
    unique_first = set(all_first_samples)
    assert len(unique_first) == len(block_contexts), (
        f"Per-block completions are NOT distinct — got {len(unique_first)} unique "
        f"out of {len(block_contexts)} blocks.  Completion replication detected!"
    )
    print(f"[PASS] Per-block completions are distinct ({len(unique_first)} unique out of {len(block_contexts)})")


def test_cache_isolation():
    """Verify cache keys are per-prompt (per-block context), not shared."""
    from openrlhf.utils.teacher_provider import TeacherCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = TeacherCache(tmpdir)
        ctx_a = ("Block 0 context text", "model-x", 2, 0.7, 0.9, 64)
        ctx_b = ("Block 1 context text", "model-x", 2, 0.7, 0.9, 64)
        cache.put(*ctx_a, completions=["comp_a1", "comp_a2"])
        cache.put(*ctx_b, completions=["comp_b1", "comp_b2"])

        assert cache.get(*ctx_a) == ["comp_a1", "comp_a2"]
        assert cache.get(*ctx_b) == ["comp_b1", "comp_b2"]

        # Different context → different cache entry (no cross-contamination)
        ctx_c = ("Block 2 context text", "model-x", 2, 0.7, 0.9, 64)
        assert cache.get(*ctx_c) is None
        print("[PASS] Cache entries are isolated per block context")


def main():
    print("=" * 60)
    print("Teacher Provider Smoke Tests")
    print("=" * 60)

    proc = start_mock_server()
    try:
        test_cache_standalone()
        test_cache_isolation()
        test_build_factory()
        test_remote_provider()
        test_per_block_distinct_completions()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
