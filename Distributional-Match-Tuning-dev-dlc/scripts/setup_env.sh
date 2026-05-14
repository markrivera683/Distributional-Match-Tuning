#!/usr/bin/env bash
#
# Bootstrap the two virtualenvs this repo expects:
#   - ${REPO_ROOT}/.venv         student / training / OpenRLHF env
#   - ${REPO_ROOT}/.teacherVenv  teacher / vLLM env
#
# Both envs are created with `uv venv --seed --python 3.12.12` directly under
# the repo root, because every run_G{1,2,3}_*.sh, reproduction_*.sh,
# supplement*/*.sh, dlc*/*.sh and benchmark script in this repo references
# them as `${REPO_ROOT}/.venv/bin/python` and `${REPO_ROOT}/.teacherVenv/bin/vllm`.
#
# Locked package versions follow scripts/stash/recreate_current_env.sh
# (commit 0a9b59b9 snapshot). Use that script if you need the fully
# parameterised reference (apt deps, repo checkout, snapshot restore, etc.).
#
# Usage:
#   bash scripts/setup_env.sh                       # build both envs
#   SKIP_STUDENT=1 bash scripts/setup_env.sh        # only build .teacherVenv
#   SKIP_TEACHER=1 bash scripts/setup_env.sh        # only build .venv
#   PYTHON_VERSION=3.12.12 \
#     STUDENT_VENV=/path/to/.venv \
#     TEACHER_VENV=/path/to/.teacherVenv \
#     bash scripts/setup_env.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths and toggles
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.12}"

STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"
TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"

SKIP_STUDENT="${SKIP_STUDENT:-0}"
SKIP_TEACHER="${SKIP_TEACHER:-0}"

# ---------------------------------------------------------------------------
# Locked versions (keep aligned with scripts/stash/recreate_current_env.sh)
# ---------------------------------------------------------------------------
STUDENT_TORCH_INDEX_URL="${STUDENT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
STUDENT_TORCH_VERSION="${STUDENT_TORCH_VERSION:-2.5.1}"
STUDENT_TORCHVISION_VERSION="${STUDENT_TORCHVISION_VERSION:-0.20.1}"
STUDENT_TORCHAUDIO_VERSION="${STUDENT_TORCHAUDIO_VERSION:-2.5.1}"
STUDENT_FLASH_ATTN_VERSION="${STUDENT_FLASH_ATTN_VERSION:-2.8.3}"
STUDENT_TRANSFORMERS_REF="${STUDENT_TRANSFORMERS_REF:-db9f18c370c92e971172e69bc9a88854947d9fc5}"

TEACHER_VLLM_VERSION="${TEACHER_VLLM_VERSION:-0.19.0}"
TEACHER_TORCH_VERSION="${TEACHER_TORCH_VERSION:-2.10.0}"
TEACHER_TORCHVISION_VERSION="${TEACHER_TORCHVISION_VERSION:-0.25.0}"
TEACHER_TORCHAUDIO_VERSION="${TEACHER_TORCHAUDIO_VERSION:-2.10.0}"
TEACHER_TRANSFORMERS_VERSION="${TEACHER_TRANSFORMERS_VERSION:-5.5.0}"
TEACHER_HF_HUB_VERSION="${TEACHER_HF_HUB_VERSION:-1.9.0}"
TEACHER_FLASHINFER_VERSION="${TEACHER_FLASHINFER_VERSION:-0.6.6}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }
warn() { printf "warning: %s\n" "$*" >&2; }
die()  { printf "error: %s\n" "$*" >&2; exit 1; }

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return
  fi
  command -v curl >/dev/null 2>&1 || die "Need either 'uv' or 'curl' on PATH"
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="${HOME}/.local/bin/uv"
  [[ -x "${UV_BIN}" ]] || die "uv installation failed"
}

create_venv() {
  local venv_path="$1"
  log "Creating virtualenv at ${venv_path} (python ${PYTHON_VERSION})"
  rm -rf "${venv_path}"
  "${UV_BIN}" venv --seed --python "${PYTHON_VERSION}" "${venv_path}"
  [[ -x "${venv_path}/bin/python" ]] || die "venv creation failed for ${venv_path}"
  "${venv_path}/bin/python" -m pip install --upgrade \
    pip "setuptools==82.0.1" "wheel==0.46.3"
}

# ---------------------------------------------------------------------------
# Student env (.venv): training / OpenRLHF / Ray / DeepSpeed
# ---------------------------------------------------------------------------
install_student() {
  local py="${STUDENT_VENV}/bin/python"

  log "Installing student PyTorch CUDA 12.4 stack"
  "${py}" -m pip install \
    --extra-index-url "${STUDENT_TORCH_INDEX_URL}" \
    "torch==${STUDENT_TORCH_VERSION}" \
    "torchvision==${STUDENT_TORCHVISION_VERSION}" \
    "torchaudio==${STUDENT_TORCHAUDIO_VERSION}"

  log "Installing student training stack"
  "${py}" -m pip install \
    "accelerate==1.13.0" \
    "bitsandbytes==0.49.2" \
    "datasets==4.8.4" \
    "deepspeed==0.18.9" \
    "einops==0.8.2" \
    "grpcio==1.80.0" \
    "huggingface_hub==1.10.1" \
    "jsonlines==4.0.0" \
    "loralib==0.1.2" \
    "matplotlib==3.10.8" \
    "numpy==2.2.6" \
    "nvitop==1.6.2" \
    "optimum==2.1.0" \
    "optree==0.19.0" \
    "packaging==26.0" \
    "pandas==3.0.2" \
    "peft==0.18.1" \
    "psutil==7.2.2" \
    "pyarrow==23.0.1" \
    "pynvml==13.0.1" \
    "pyyaml==6.0.3" \
    "ray[default]==2.48.0" \
    "safetensors==0.7.0" \
    "sentencepiece==0.2.1" \
    "tensorboard==2.20.0" \
    "tokenizers==0.22.2" \
    "torchdata==0.11.0" \
    "torchmetrics==1.9.0" \
    "tqdm==4.67.3" \
    "transformers-stream-generator==0.0.5" \
    "wandb==0.25.1"

  log "Installing student transformers from git ref ${STUDENT_TRANSFORMERS_REF}"
  "${py}" -m pip install \
    "git+https://github.com/huggingface/transformers.git@${STUDENT_TRANSFORMERS_REF}"

  log "Installing student flash-attn==${STUDENT_FLASH_ATTN_VERSION}"
  if ! "${py}" -m pip install "flash-attn==${STUDENT_FLASH_ATTN_VERSION}"; then
    log "Falling back to source install for flash-attn"
    "${py}" -m pip install --no-build-isolation \
      "flash-attn==${STUDENT_FLASH_ATTN_VERSION}"
  fi

  log "Installing this repo (editable) into ${STUDENT_VENV}"
  "${py}" -m pip install -e "${REPO_ROOT}"
}

verify_student() {
  log "Verifying student env"
  "${STUDENT_VENV}/bin/python" - <<'PY'
import sys
import deepspeed
import ray
import torch
import transformers

print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
print("ray", ray.__version__)
print("deepspeed", deepspeed.__version__)
print("transformers", transformers.__version__)
try:
    import openrlhf
    print("openrlhf", getattr(openrlhf, "__version__", "editable"))
except Exception as exc:
    print("openrlhf_import_error", exc)
PY
}

# ---------------------------------------------------------------------------
# Teacher env (.teacherVenv): vLLM serving only
# ---------------------------------------------------------------------------
install_teacher() {
  local py="${TEACHER_VENV}/bin/python"

  log "Installing teacher PyTorch stack"
  "${py}" -m pip install \
    "torch==${TEACHER_TORCH_VERSION}" \
    "torchvision==${TEACHER_TORCHVISION_VERSION}" \
    "torchaudio==${TEACHER_TORCHAUDIO_VERSION}"

  log "Installing teacher vLLM stack (vllm ${TEACHER_VLLM_VERSION})"
  "${py}" -m pip install \
    "vllm==${TEACHER_VLLM_VERSION}" \
    "flashinfer-python==${TEACHER_FLASHINFER_VERSION}" \
    "tqdm==4.67.3"

  "${py}" -m pip install "huggingface_hub==${TEACHER_HF_HUB_VERSION}"

  log "Pinning teacher transformers to ${TEACHER_TRANSFORMERS_VERSION} (no-deps)"
  "${py}" -m pip install --no-deps \
    "transformers==${TEACHER_TRANSFORMERS_VERSION}"
}

verify_teacher() {
  log "Verifying teacher env"
  "${TEACHER_VENV}/bin/python" - <<'PY'
import shutil
import sys
from pathlib import Path

import torch
import transformers
import vllm

print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("vllm_cli", Path(sys.executable).with_name("vllm"))
print("vllm_cli_on_path", shutil.which("vllm"))
PY
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  [[ -d "${REPO_ROOT}" ]] || die "REPO_ROOT not found: ${REPO_ROOT}"
  [[ -f "${REPO_ROOT}/setup.py" ]] || die "REPO_ROOT does not look like the repo root: ${REPO_ROOT}"

  log "REPO_ROOT       = ${REPO_ROOT}"
  log "STUDENT_VENV    = ${STUDENT_VENV}"
  log "TEACHER_VENV    = ${TEACHER_VENV}"
  log "PYTHON_VERSION  = ${PYTHON_VERSION}"

  ensure_uv

  if [[ "${SKIP_STUDENT}" != "1" ]]; then
    create_venv "${STUDENT_VENV}"
    install_student
    verify_student
  else
    log "Skipping student env (SKIP_STUDENT=1)"
  fi

  if [[ "${SKIP_TEACHER}" != "1" ]]; then
    create_venv "${TEACHER_VENV}"
    install_teacher
    verify_teacher
  else
    log "Skipping teacher env (SKIP_TEACHER=1)"
  fi

  log "Done."
  printf "  student env:  source %s/bin/activate\n" "${STUDENT_VENV}"
  printf "  teacher env:  source %s/bin/activate\n" "${TEACHER_VENV}"
}

main "$@"
