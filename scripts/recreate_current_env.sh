#!/usr/bin/env bash
set -euo pipefail

# Recreate the current Distributional-Match-Tuning layout on another machine.
# This script creates two virtualenvs under the repo root:
# - .venv        : training / analysis / OpenRLHF environment
# - .teacherVenv : teacher-model / vLLM environment
#
# Snapshot source:
# - repo commit: 0a9b59b933d04441485a4aa8f74c9bd65af5ac23
# - student env: Python 3.12.12, torch 2.5.1+cu124, torchvision 0.20.1,
#   transformers@db9f18c370c92e971172e69bc9a88854947d9fc5, deepspeed 0.18.9
# - teacher env: Python 3.12.12, vllm 0.19.0, torch 2.10.0, torchvision 0.25.0,
#   torchaudio 2.10.0, transformers 5.5.0, huggingface_hub 1.9.0

REPO_URL_DEFAULT="https://github.com/markrivera683/Distributional-Match-Tuning.git"
REPO_COMMIT_DEFAULT="0a9b59b933d04441485a4aa8f74c9bd65af5ac23"
PYTHON_VERSION_DEFAULT="3.12.12"

STUDENT_TORCH_INDEX_URL_DEFAULT=""
STUDENT_TRANSFORMERS_REF_DEFAULT="db9f18c370c92e971172e69bc9a88854947d9fc5"
STUDENT_FLASH_ATTN_VERSION_DEFAULT="2.8.3"

TEACHER_VLLM_VERSION_DEFAULT="0.19.0"
TEACHER_TORCH_VERSION_DEFAULT="2.10.0"
TEACHER_TORCHVISION_VERSION_DEFAULT="0.25.0"
TEACHER_TORCHAUDIO_VERSION_DEFAULT="2.10.0"
TEACHER_TRANSFORMERS_VERSION_DEFAULT="5.5.0"
TEACHER_HF_HUB_VERSION_DEFAULT="1.9.0"
TEACHER_FLASHINFER_VERSION_DEFAULT="0.6.6"

log() {
  printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"
}

warn() {
  printf "warning: %s\n" "$*" >&2
}

die() {
  printf "error: %s\n" "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/recreate_current_env.sh

This creates two virtualenvs:
  $REPO_DIR/.venv
  $REPO_DIR/.teacherVenv

Optional environment variables:
  REPO_URL                        Repo to clone when REPO_DIR is missing
  REPO_DIR                        Target repo path. Defaults to the parent of this script
  REPO_COMMIT                     Repo commit to checkout when the repo is clean
  REPO_SYNC_MODE                  auto (default), skip, or git
  PYTHON_VERSION                  Defaults to 3.12.12
  INSTALL_APT_DEPS                1 (default) to install Ubuntu/Debian build deps
  RECREATE_STUDENT_ENV            1 (default) to rebuild .venv
  RECREATE_TEACHER_ENV            1 (default) to rebuild .teacherVenv
  STUDENT_VENV                    Defaults to $REPO_DIR/.venv
  TEACHER_VENV                    Defaults to $REPO_DIR/.teacherVenv
  STUDENT_TORCH_INDEX_URL         Defaults to https://download.pytorch.org/whl/cu124
  STUDENT_TRANSFORMERS_REF        Defaults to the current student transformers git ref
  STUDENT_FLASH_ATTN_VERSION      Defaults to 2.8.3
  STUDENT_FLASH_ATTN_WHEEL        Optional wheel path or URL for flash-attn
  STUDENT_FLASH_ATTN_STRATEGY     auto (default) or skip
  STUDENT_CAUSAL_CONV1D_SPEC      Optional pip spec/path/URL for causal-conv1d
  TEACHER_TORCH_INDEX_URL         Optional extra index for teacher torch packages
  TEACHER_VLLM_VERSION            Defaults to 0.19.0
  TEACHER_TORCH_VERSION           Defaults to 2.10.0
  TEACHER_TORCHVISION_VERSION     Defaults to 0.25.0
  TEACHER_TORCHAUDIO_VERSION      Defaults to 2.10.0
  TEACHER_TRANSFORMERS_VERSION    Defaults to 5.5.0
  TEACHER_HF_HUB_VERSION          Defaults to 1.9.0
  TEACHER_FLASHINFER_VERSION      Defaults to 0.6.6

Examples:
  bash scripts/recreate_current_env.sh
  RECREATE_TEACHER_ENV=0 bash scripts/recreate_current_env.sh
  STUDENT_FLASH_ATTN_STRATEGY=skip bash scripts/recreate_current_env.sh
  STUDENT_CAUSAL_CONV1D_SPEC=causal-conv1d==1.6.0 bash scripts/recreate_current_env.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REPO_URL="${REPO_URL:-$REPO_URL_DEFAULT}"
REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"
REPO_COMMIT="${REPO_COMMIT:-$REPO_COMMIT_DEFAULT}"
REPO_SYNC_MODE="${REPO_SYNC_MODE:-auto}"
PYTHON_VERSION="${PYTHON_VERSION:-$PYTHON_VERSION_DEFAULT}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"
RECREATE_STUDENT_ENV="${RECREATE_STUDENT_ENV:-1}"
RECREATE_TEACHER_ENV="${RECREATE_TEACHER_ENV:-1}"

STUDENT_VENV="${STUDENT_VENV:-$REPO_DIR/.venv}"
TEACHER_VENV="${TEACHER_VENV:-$REPO_DIR/.teacherVenv}"

STUDENT_TORCH_INDEX_URL="${STUDENT_TORCH_INDEX_URL:-$STUDENT_TORCH_INDEX_URL_DEFAULT}"
STUDENT_TRANSFORMERS_REF="${STUDENT_TRANSFORMERS_REF:-$STUDENT_TRANSFORMERS_REF_DEFAULT}"
STUDENT_FLASH_ATTN_VERSION="${STUDENT_FLASH_ATTN_VERSION:-$STUDENT_FLASH_ATTN_VERSION_DEFAULT}"
STUDENT_FLASH_ATTN_WHEEL="${STUDENT_FLASH_ATTN_WHEEL:-}"
STUDENT_FLASH_ATTN_STRATEGY="${STUDENT_FLASH_ATTN_STRATEGY:-auto}"
STUDENT_CAUSAL_CONV1D_SPEC="${STUDENT_CAUSAL_CONV1D_SPEC:-}"

TEACHER_TORCH_INDEX_URL="${TEACHER_TORCH_INDEX_URL:-}"
TEACHER_VLLM_VERSION="${TEACHER_VLLM_VERSION:-$TEACHER_VLLM_VERSION_DEFAULT}"
TEACHER_TORCH_VERSION="${TEACHER_TORCH_VERSION:-$TEACHER_TORCH_VERSION_DEFAULT}"
TEACHER_TORCHVISION_VERSION="${TEACHER_TORCHVISION_VERSION:-$TEACHER_TORCHVISION_VERSION_DEFAULT}"
TEACHER_TORCHAUDIO_VERSION="${TEACHER_TORCHAUDIO_VERSION:-$TEACHER_TORCHAUDIO_VERSION_DEFAULT}"
TEACHER_TRANSFORMERS_VERSION="${TEACHER_TRANSFORMERS_VERSION:-$TEACHER_TRANSFORMERS_VERSION_DEFAULT}"
TEACHER_HF_HUB_VERSION="${TEACHER_HF_HUB_VERSION:-$TEACHER_HF_HUB_VERSION_DEFAULT}"
TEACHER_FLASHINFER_VERSION="${TEACHER_FLASHINFER_VERSION:-$TEACHER_FLASHINFER_VERSION_DEFAULT}"

run_apt_install() {
  if [[ "$INSTALL_APT_DEPS" != "1" ]]; then
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    return
  fi

  local apt_runner=()
  if command -v sudo >/dev/null 2>&1; then
    apt_runner=(sudo)
  elif [[ "$(id -u)" -eq 0 ]]; then
    apt_runner=()
  else
    warn "Skipping apt dependencies because sudo is unavailable."
    return
  fi

  log "Installing Ubuntu/Debian build dependencies"
  "${apt_runner[@]}" apt-get update
  "${apt_runner[@]}" apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    ninja-build \
    pkg-config
}

ensure_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: $cmd"
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return
  fi

  ensure_cmd curl
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="${HOME}/.local/bin/uv"
  [[ -x "$UV_BIN" ]] || die "uv installation failed"
}

ensure_repo() {
  mkdir -p "$(dirname "$REPO_DIR")"

  case "$REPO_SYNC_MODE" in
    auto|skip|git)
      ;;
    *)
      die "Unsupported REPO_SYNC_MODE=$REPO_SYNC_MODE"
      ;;
  esac

  if [[ "$REPO_SYNC_MODE" == "skip" ]]; then
    [[ -d "$REPO_DIR" ]] || die "REPO_SYNC_MODE=skip but REPO_DIR does not exist: $REPO_DIR"
    log "Using existing repo snapshot at $REPO_DIR (REPO_SYNC_MODE=skip)"
    return
  fi

  if [[ -d "$REPO_DIR/.git" ]]; then
    log "Using existing repo at $REPO_DIR"
  elif [[ -d "$REPO_DIR" ]] && [[ -n "$(ls -A "$REPO_DIR" 2>/dev/null || true)" ]]; then
    if [[ "$REPO_SYNC_MODE" == "git" ]]; then
      die "REPO_SYNC_MODE=git requires a git repo at $REPO_DIR"
    fi
    log "Using existing non-git repo snapshot at $REPO_DIR"
    return
  else
    log "Cloning repo into $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
  fi

  if [[ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null || true)" ]]; then
    warn "Repo at $REPO_DIR has local changes; leaving the current checkout untouched."
    return
  fi

  log "Checking out repo commit $REPO_COMMIT"
  git -C "$REPO_DIR" fetch --all --tags --prune
  git -C "$REPO_DIR" checkout "$REPO_COMMIT"
  git -C "$REPO_DIR" submodule update --init --recursive
}

create_venv() {
  local venv_path="$1"
  local python_bin

  mkdir -p "$(dirname "$venv_path")"
  log "Creating virtualenv at $venv_path"
  rm -rf "$venv_path"
  "$UV_BIN" venv --seed --python "$PYTHON_VERSION" "$venv_path"
  python_bin="$venv_path/bin/python"
  [[ -x "$python_bin" ]] || die "Virtualenv creation failed for $venv_path"
  "$python_bin" -m pip install --upgrade pip setuptools==82.0.1 wheel==0.46.3
}

install_student_torch_stack() {
  log "Installing student PyTorch CUDA 12.4 stack into $STUDENT_VENV"
  local torch_args=()
  if [[ -n "$STUDENT_TORCH_INDEX_URL" ]]; then
    torch_args+=(--extra-index-url "$STUDENT_TORCH_INDEX_URL")
  fi
  "$STUDENT_PYTHON_BIN" -m pip install \
    "${torch_args[@]}" \
    "torch==2.5.1" \
    "torchvision==0.20.1" \
    "torchaudio==2.5.1"
}

install_student_core_python_packages() {
  log "Installing student training stack into $STUDENT_VENV"
  "$STUDENT_PYTHON_BIN" -m pip install \
    "accelerate==1.13.0" \
    "autopep8==2.3.2" \
    "bitsandbytes==0.49.2" \
    "black==26.3.1" \
    "datasets==4.8.4" \
    "deepspeed==0.18.9" \
    "einops==0.8.2" \
    "grpcio==1.80.0" \
    "huggingface_hub==1.10.1" \
    "isort==8.0.1" \
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
}

install_student_transformers() {
  log "Installing student transformers from git ref $STUDENT_TRANSFORMERS_REF"
  "$STUDENT_PYTHON_BIN" -m pip install \
    "git+https://github.com/huggingface/transformers.git@${STUDENT_TRANSFORMERS_REF}"
}

install_student_flash_attn() {
  if [[ "$STUDENT_FLASH_ATTN_STRATEGY" == "skip" ]]; then
    log "Skipping student flash-attn"
    return
  fi

  if [[ "$STUDENT_FLASH_ATTN_STRATEGY" != "auto" ]]; then
    die "Unsupported STUDENT_FLASH_ATTN_STRATEGY=$STUDENT_FLASH_ATTN_STRATEGY"
  fi

  if [[ -n "$STUDENT_FLASH_ATTN_WHEEL" ]]; then
    log "Installing student flash-attn from $STUDENT_FLASH_ATTN_WHEEL"
    "$STUDENT_PYTHON_BIN" -m pip install "$STUDENT_FLASH_ATTN_WHEEL"
    return
  fi

  log "Installing student flash-attn==$STUDENT_FLASH_ATTN_VERSION"
  if "$STUDENT_PYTHON_BIN" -m pip install "flash-attn==${STUDENT_FLASH_ATTN_VERSION}"; then
    return
  fi

  log "Falling back to source install for student flash-attn"
  "$STUDENT_PYTHON_BIN" -m pip install --no-build-isolation "flash-attn==${STUDENT_FLASH_ATTN_VERSION}"
}

install_student_optional_packages() {
  if [[ -n "$STUDENT_CAUSAL_CONV1D_SPEC" ]]; then
    log "Installing student causal-conv1d from $STUDENT_CAUSAL_CONV1D_SPEC"
    "$STUDENT_PYTHON_BIN" -m pip install "$STUDENT_CAUSAL_CONV1D_SPEC"
  else
    log "Skipping student causal-conv1d because the source machine used a non-portable local wheel"
  fi
}

install_student_repo() {
  log "Installing repo into student env in editable mode"
  "$STUDENT_PYTHON_BIN" -m pip install -e "$REPO_DIR"
}

verify_student_env() {
  log "Verifying student env"
  "$STUDENT_PYTHON_BIN" - <<'PY'
import sys
import deepspeed
import ray
import torch
import transformers

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
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

install_teacher_torch_stack() {
  log "Installing teacher torch stack into $TEACHER_VENV"
  local torch_args=()
  if [[ -n "$TEACHER_TORCH_INDEX_URL" ]]; then
    torch_args+=(--index-url "$TEACHER_TORCH_INDEX_URL")
  fi

  "$TEACHER_PYTHON_BIN" -m pip install \
    "${torch_args[@]}" \
    "torch==${TEACHER_TORCH_VERSION}" \
    "torchvision==${TEACHER_TORCHVISION_VERSION}" \
    "torchaudio==${TEACHER_TORCHAUDIO_VERSION}"
}

install_teacher_core_python_packages() {
  log "Installing teacher vLLM stack into $TEACHER_VENV"
  "$TEACHER_PYTHON_BIN" -m pip install \
    "huggingface_hub==${TEACHER_HF_HUB_VERSION}" \
    "transformers==${TEACHER_TRANSFORMERS_VERSION}" \
    "flashinfer-python==${TEACHER_FLASHINFER_VERSION}" \
    "tqdm==4.67.3" \
    "vllm==${TEACHER_VLLM_VERSION}"
}

verify_teacher_env() {
  log "Verifying teacher env"
  "$TEACHER_PYTHON_BIN" - <<'PY'
import importlib
import shutil
import sys
from pathlib import Path

import torch
import transformers
import vllm

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("vllm_cli", Path(sys.executable).with_name("vllm"))
print("vllm_cli_on_path", shutil.which("vllm"))
importlib.import_module("vllm.model_executor.models.qwen3_5")
importlib.import_module("vllm.transformers_utils.configs.qwen3_5")
print("qwen3_5_support", True)
PY
}

main() {
  local need_git="0"

  run_apt_install

  if [[ "$REPO_SYNC_MODE" == "git" ]]; then
    need_git="1"
  elif [[ "$REPO_SYNC_MODE" == "auto" ]]; then
    if [[ ! -d "$REPO_DIR" ]] || [[ -d "$REPO_DIR/.git" ]]; then
      need_git="1"
    fi
  fi

  if [[ "$need_git" == "1" ]]; then
    ensure_cmd git
  fi

  ensure_uv
  ensure_repo

  if [[ "$RECREATE_STUDENT_ENV" == "1" ]]; then
    create_venv "$STUDENT_VENV"
    STUDENT_PYTHON_BIN="$STUDENT_VENV/bin/python"
    install_student_torch_stack
    install_student_core_python_packages
    install_student_transformers
    install_student_flash_attn
    install_student_optional_packages
    install_student_repo
    verify_student_env
  else
    log "Skipping student env rebuild"
  fi

  if [[ "$RECREATE_TEACHER_ENV" == "1" ]]; then
    create_venv "$TEACHER_VENV"
    TEACHER_PYTHON_BIN="$TEACHER_VENV/bin/python"
    install_teacher_torch_stack
    install_teacher_core_python_packages
    verify_teacher_env
  else
    log "Skipping teacher env rebuild"
  fi

  log "Done"
  if [[ "$RECREATE_STUDENT_ENV" == "1" ]]; then
    printf "Student env:\n  source \"%s/bin/activate\"\n" "$STUDENT_VENV"
  fi
  if [[ "$RECREATE_TEACHER_ENV" == "1" ]]; then
    printf "Teacher env:\n  source \"%s/bin/activate\"\n" "$TEACHER_VENV"
  fi
}

main "$@"
