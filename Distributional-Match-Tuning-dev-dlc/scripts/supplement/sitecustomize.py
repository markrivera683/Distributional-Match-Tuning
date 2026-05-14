from __future__ import annotations

import os
import sys

if os.environ.get("REPO_QWEN35_TEXT_ONLY_SHIM") == "1":
    try:
        from qwen35_text_only_shim import (
            ensure_qwen35_text_only_runtime_patch_installed,
            ensure_qwen35_text_only_shim_registered,
        )

        ensure_qwen35_text_only_shim_registered()
        ensure_qwen35_text_only_runtime_patch_installed()
    except Exception as exc:
        print(
            f"[compat] failed to register qwen3.5 text-only shim in sitecustomize: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise
