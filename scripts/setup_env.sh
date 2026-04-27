#!/usr/bin/env bash
#
# Bootstrap the two virtualenvs this repo expects:
#   - ${LOCAL_ROOT}/venvs/.venv         student / training / OpenRLHF env
#   - ${LOCAL_ROOT}/venvs/.teacherVenv  teacher / vLLM env
#
# DSW-specific storage layout:
#   /mnt/data           ossfs2 (OSS), persistent across reboots, but:
#                       - no symlink / no hardlink
#                       - no "seek + write into existing file" (EINVAL,
#                         which fuse mis-reports as ENOSPC; this kills
#                         zipfile writes AND g++/nvcc/triton .o emission)
#   /mnt/workspace      ext4 on /vda, ~30G, wiped on restart, full POSIX
#
# State buckets:
#
#   On local ext4 (lost on restart, but cheap to recreate):
#     - venvs                       (need bin/python symlink)
#     - uv-managed Python           (contains symlinks)
#     - PIP_CACHE_DIR               (zipfile needs seek+truncate)
#     - TORCH_EXTENSIONS_DIR        (g++/nvcc need seek+write)
#     - TRITON_CACHE_DIR            (triton JIT same)
#
#   On persistent OSS (survive restart):
#     - HF_HOME           huge model blobs, append-only writes
#     - WHEEL_HOUSE_DIR   precious pre-built / hand-staged wheels
#     - SNAPSHOTS_DIR     tar.zst of fully-installed venvs (~5GB total)
#
# After a successful install we tar+zstd each venv into
# ${CACHE_ROOT}/snapshots/{student,teacher}-venv.tar.zst. The next run on
# a fresh machine restores from those snapshots (~1-2 minutes) and skips
# uv venv + pip install entirely. Set EBFT_REBUILD_VENV=1 to force a
# clean rebuild, or EBFT_USE_VENV_SNAPSHOT=0 to disable the mechanism.
#
# The run_G{1,2,3}_*.sh, reproduction_*.sh, supplement*/*.sh, dlc*/*.sh and
# benchmark scripts honour STUDENT_VENV / TEACHER_VENV env vars. Export them
# (see the message printed at the end) or symlink the paths into REPO_ROOT
# from a non-OSS location if you need ${REPO_ROOT}/.venv to keep working.
#
# Locked package versions follow scripts/stash/recreate_current_env.sh
# (commit 0a9b59b9 snapshot). Use that script if you need the fully
# parameterised reference (apt deps, repo checkout, snapshot restore, etc.).
#
# Usage:
#   bash scripts/setup_env.sh                       # build both envs (or
#                                                   # restore from snapshot
#                                                   # if one exists)
#   SKIP_STUDENT=1 bash scripts/setup_env.sh        # only build .teacherVenv
#   SKIP_TEACHER=1 bash scripts/setup_env.sh        # only build .venv
#   SNAPSHOT_ONLY=1 bash scripts/setup_env.sh       # tar existing venvs to
#                                                   # OSS without reinstall
#   EBFT_REBUILD_VENV=1 bash scripts/setup_env.sh   # ignore snapshot, do
#                                                   # a clean rebuild and
#                                                   # overwrite the snapshot
#   EBFT_USE_VENV_SNAPSHOT=0 bash scripts/setup_env.sh
#                                                   # disable snapshot logic
#                                                   # entirely
#   LOCAL_ROOT=/some/ext4 CACHE_ROOT=/mnt/data/my-cache \
#     bash scripts/setup_env.sh                     # custom paths
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

# DSW storage layout:
#   /mnt/data           ossfs2 (OSS)         — survives container REBUILD,
#                                               huge, slow (network), NO
#                                               symlink and NO hardlink support
#   /mnt/workspace      ext4 on /vda (30G)   — survives container REBUILD,
#                                               fast, symlink + hardlink OK,
#                                               but 30G quota fills up once
#                                               venvs (~17G) and pip cache
#                                               (~11G) coexist
#   /              overlayfs over ext4       — wiped on container REBUILD,
#                                               survives plain reboots, fast,
#                                               4T+ free, symlink + hardlink OK
#
# Therefore venvs (need bin/python symlink) and the uv-managed Python
# (internally contains symlinks) MUST live on real ext4. We default venvs
# to /mnt/workspace because:
#   - it survives container REBUILD (rootfs gets wiped on rebuild, not just
#     on plain reboot);
#   - the 30G quota is enough in practice: student venv (~7G) + teacher
#     venv (~10G) ≈ 17G, leaving ~13G for compile caches and headroom;
#   - all run_*.sh scripts in this repo also default to /mnt/workspace so
#     keeping setup_env.sh's default in sync avoids the "venv path
#     mismatch" footgun where setup builds in one place and runners look
#     in another.
# Override with LOCAL_ROOT=/root if you specifically want rootfs (faster
# rebuild restore from snapshot, but loses persistence on container
# rebuild and forces every run_*.sh to also override its venv path).
# Caches that are pure files (pip wheels, HF blobs) still live on OSS so
# the venvs can be rebuilt from snapshot in minutes after a restart.
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/workspace}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/data/ebft-distribution-new/caches}"

STUDENT_VENV="${STUDENT_VENV:-${LOCAL_ROOT}/venvs/.venv}"
TEACHER_VENV="${TEACHER_VENV:-${LOCAL_ROOT}/venvs/.teacherVenv}"

# HF blobs go on persistent OSS (downloads are tmp-file+rename, OSS-safe;
# weights are huge so persistence is mandatory).
HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"

# Compile caches MUST be on local ext4: ossfs2 rejects "seek + write into
# an existing file" with EINVAL (fuse mis-translates this to ENOSPC, which
# you'll see as 'No space left on device' from g++/nvcc/triton). g++/nvcc/
# triton all need exactly that operation when emitting .o / .cubin / .so
# during JIT, so anything compiled at runtime (deepspeed FusedAdam, fused
# kernels, triton matmul, ...) must build on /mnt/workspace.
#
# Trade-off: these caches don't survive a container restart, so the first
# training step after a reboot pays a ~30-60s recompilation cost. Cheap.
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_ROOT}/.torch_extensions}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_ROOT}/.triton_cache}"

# Pre-built / hand-staged wheels on persistent OSS. Pip is told to look here
# (--find-links) so that heavy sdist builds (flash-attn, deepspeed,
# transformers-stream-generator, ...) only need to compile once per arch /
# CUDA combo. Drop a .whl into this dir and the next setup_env.sh run will
# pick it up instead of rebuilding.
WHEEL_HOUSE_DIR="${WHEEL_HOUSE_DIR:-${CACHE_ROOT}/wheels}"

# pip cache MUST be on local ext4: ossfs2 does not support the seek+truncate
# operations zipfile needs to write a .whl, so any wheel build that lands in
# PIP_CACHE_DIR on OSS dies with `[Errno 22] Invalid argument`. Keeping it
# local means downloaded wheels are re-fetched after every container restart,
# but Aliyun's mirror is fast enough (~24 MB/s) that this is acceptable.
#
# Default to /root/.cache/pip (container rootfs ext4, ~4T free). Historically
# this lived under /mnt/workspace but that volume's 30G quota filled up once
# venvs (~17G) coexisted with the wheel cache (~11G). /root/.cache/pip is also
# the standard XDG location pip would use without any override.
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/.cache/pip}"

# uv internals on local ext4 (uv hardlinks within its cache, which
# ossfs2 does not support; the uv-managed Python contains symlinks).
# Co-locating with the venvs under LOCAL_ROOT keeps hardlinks valid.
UV_INSTALL_DIR="${UV_INSTALL_DIR:-${LOCAL_ROOT}/.uv/bin}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${LOCAL_ROOT}/.uv/cache}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${LOCAL_ROOT}/.uv/python}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"

export PIP_CACHE_DIR WHEEL_HOUSE_DIR HF_HOME TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR \
       UV_INSTALL_DIR UV_CACHE_DIR UV_PYTHON_INSTALL_DIR UV_LINK_MODE

# Default to Aliyun's PyPI mirror because DSW lives in Aliyun's network and
# direct pypi.org transfers regularly suffer from mid-stream IncompleteRead
# errors here. Override with PIP_INDEX_URL=https://pypi.org/simple/ or
# PIP_INDEX_URL="" to keep pip's built-in default.
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
PIP_RETRIES="${PIP_RETRIES:-10}"
PIP_TIMEOUT="${PIP_TIMEOUT:-120}"
PIP_INSTALL_ATTEMPTS="${PIP_INSTALL_ATTEMPTS:-3}"
export PIP_INDEX_URL PIP_RETRIES PIP_TIMEOUT

SKIP_STUDENT="${SKIP_STUDENT:-0}"
SKIP_TEACHER="${SKIP_TEACHER:-0}"

# When set, do not (re)install anything; just tar already-installed venvs
# into SNAPSHOTS_DIR. Useful right after a manual install when you want to
# capture the current state without going through pip again.
SNAPSHOT_ONLY="${SNAPSHOT_ONLY:-0}"

# Venv snapshots on persistent OSS. After a successful install, each venv
# is tarred + zstd-compressed under SNAPSHOTS_DIR. On a fresh machine /
# container the next run untars those tarballs back to LOCAL_ROOT and
# skips the entire pip install pipeline (~1-2 minutes vs. ~20 minutes).
#
#   EBFT_USE_VENV_SNAPSHOT=0  disable both saving and restoring snapshots
#   EBFT_REBUILD_VENV=1       ignore any existing snapshot and rebuild
#                             from scratch (the new venv will overwrite
#                             the old snapshot on success)
#   SNAPSHOT_ZSTD_LEVEL=N     compression level (default 3, range 1..22)
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-${CACHE_ROOT}/snapshots}"
EBFT_USE_VENV_SNAPSHOT="${EBFT_USE_VENV_SNAPSHOT:-1}"
EBFT_REBUILD_VENV="${EBFT_REBUILD_VENV:-0}"
SNAPSHOT_ZSTD_LEVEL="${SNAPSHOT_ZSTD_LEVEL:-3}"

# ---------------------------------------------------------------------------
# Locked versions (keep aligned with scripts/stash/recreate_current_env.sh)
# ---------------------------------------------------------------------------
# PyTorch's cu124 wheels are not on PyPI (PyPI only ships CPU torch). The
# Aliyun mirror at mirrors.aliyun.com/pytorch-wheels/cu124/ is a flat
# directory listing (not a PEP 503 simple index) — it works as pip
# --find-links but NOT as --extra-index-url. Going through find-links keeps
# pip from sneaking small deps like sympy through download.pytorch.org's
# (much slower from CN) CDN, which is the dominant cause of stalls during
# the torch install on DSW.
STUDENT_TORCH_FIND_LINKS="${STUDENT_TORCH_FIND_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu124/}"
STUDENT_TORCH_VERSION="${STUDENT_TORCH_VERSION:-2.5.1+cu124}"
STUDENT_TORCHVISION_VERSION="${STUDENT_TORCHVISION_VERSION:-0.20.1+cu124}"
STUDENT_TORCHAUDIO_VERSION="${STUDENT_TORCHAUDIO_VERSION:-2.5.1+cu124}"
STUDENT_FLASH_ATTN_VERSION="${STUDENT_FLASH_ATTN_VERSION:-2.8.3}"
# Default to the prebuilt wheel that ships next to this repo (cu124 + torch
# 2.5.1 + cp312). Override STUDENT_FLASH_ATTN_WHEEL_FILE to point pip at a
# different .whl, or set both STUDENT_FLASH_ATTN_WHEEL_FILE="" and
# STUDENT_FLASH_ATTN_WHEEL_URL="" to skip the wheel entirely and fall back to
# PyPI / source build. If the file is missing and a URL is set, the wheel is
# downloaded into STUDENT_FLASH_ATTN_WHEEL_DIR and installed from there.
STUDENT_FLASH_ATTN_WHEEL_FILE="${STUDENT_FLASH_ATTN_WHEEL_FILE:-${REPO_ROOT}/flash_attn-${STUDENT_FLASH_ATTN_VERSION}+cu124torch2.5-cp312-cp312-linux_x86_64.whl}"
STUDENT_FLASH_ATTN_WHEEL_DIR="${STUDENT_FLASH_ATTN_WHEEL_DIR:-${CACHE_ROOT}/wheels}"
STUDENT_FLASH_ATTN_WHEEL_URL="${STUDENT_FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v${STUDENT_FLASH_ATTN_VERSION}/flash_attn-${STUDENT_FLASH_ATTN_VERSION}+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl}"
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
  # Reuse a uv binary that already lives on local ext4.
  if [[ -x "${UV_INSTALL_DIR}/uv" ]]; then
    UV_BIN="${UV_INSTALL_DIR}/uv"
    return
  fi
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return
  fi
  command -v curl >/dev/null 2>&1 || die "Need either 'uv' or 'curl' on PATH"
  mkdir -p "${UV_INSTALL_DIR}"
  log "Installing uv into ${UV_INSTALL_DIR}"
  # The official installer honors UV_INSTALL_DIR + UV_NO_MODIFY_PATH.
  UV_INSTALL_DIR="${UV_INSTALL_DIR}" UV_NO_MODIFY_PATH=1 \
    curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="${UV_INSTALL_DIR}/uv"
  [[ -x "${UV_BIN}" ]] || die "uv installation failed (looked for ${UV_BIN})"
}

# Run `${py} -m pip install <args>` with retries to survive flaky network
# (IncompleteRead, RemoteDisconnected, transient 5xx). The inner pip already
# retries each chunk via --retries; the outer loop covers the case where pip
# gives up entirely after a mid-stream protocol error.
#
# Every call also passes --find-links ${WHEEL_HOUSE_DIR}, so any pre-built
# wheel staged on OSS (flash-attn, deepspeed, ...) is reused without rebuild.
pip_install() {
  local py="$1"; shift
  local attempt
  local find_links_args=()
  if [[ -n "${WHEEL_HOUSE_DIR}" && -d "${WHEEL_HOUSE_DIR}" ]]; then
    find_links_args+=(--find-links "${WHEEL_HOUSE_DIR}")
  fi
  for attempt in $(seq 1 "${PIP_INSTALL_ATTEMPTS}"); do
    if "${py}" -m pip install \
        --retries "${PIP_RETRIES}" \
        --timeout "${PIP_TIMEOUT}" \
        "${find_links_args[@]}" \
        "$@"; then
      return 0
    fi
    if [[ "${attempt}" -lt "${PIP_INSTALL_ATTEMPTS}" ]]; then
      warn "pip install attempt ${attempt}/${PIP_INSTALL_ATTEMPTS} failed; retrying in 5s"
      sleep 5
    fi
  done
  return 1
}

# After a successful install, copy any wheels pip locally built (under
# PIP_CACHE_DIR/wheels/) into the OSS wheelhouse so the next machine /
# restart picks them up via --find-links instead of rebuilding from sdist.
# Safe no-op if there are no built wheels yet.
stage_built_wheels() {
  [[ -n "${WHEEL_HOUSE_DIR}" ]] || return 0
  local pip_wheels_dir="${PIP_CACHE_DIR}/wheels"
  [[ -d "${pip_wheels_dir}" ]] || return 0
  mkdir -p "${WHEEL_HOUSE_DIR}"
  local count=0
  while IFS= read -r whl; do
    local base
    base="$(basename "${whl}")"
    if [[ ! -f "${WHEEL_HOUSE_DIR}/${base}" ]]; then
      cp -f "${whl}" "${WHEEL_HOUSE_DIR}/${base}.part" \
        && mv "${WHEEL_HOUSE_DIR}/${base}.part" "${WHEEL_HOUSE_DIR}/${base}" \
        && count=$((count + 1))
    fi
  done < <(find "${pip_wheels_dir}" -type f -name '*.whl' 2>/dev/null)
  if [[ "${count}" -gt 0 ]]; then
    log "Staged ${count} new wheel(s) into ${WHEEL_HOUSE_DIR}"
  fi
}

# Quick smoke-import test for a restored venv. We only verify that the
# binary extensions actually load (the typical breakage when a snapshot
# is restored to an incompatible host: glibc / libstdc++ / CUDA mismatch
# manifest as ImportError on the .so files). Anything beyond that is
# left to verify_{student,teacher}.
_smoke_imports_for_tag() {
  case "$1" in
    student) echo "import torch; import deepspeed; import transformers" ;;
    teacher) echo "import torch; import vllm" ;;
    *)       echo "import sys" ;;
  esac
}

# tar + zstd the given venv into ${SNAPSHOTS_DIR}/${tag}-venv.tar.zst on
# OSS, with a sibling .meta file recording the source path / python
# version so try_restore_venv can refuse a restore into a different
# layout. Atomic via .part tmp files.
#
# Skipped when EBFT_USE_VENV_SNAPSHOT=0 or zstd is missing.
snapshot_venv() {
  local venv_path="$1"
  local tag="$2"

  [[ "${EBFT_USE_VENV_SNAPSHOT}" == "1" ]] || return 0
  if ! command -v zstd >/dev/null 2>&1; then
    warn "zstd not on PATH; skipping snapshot of ${tag} venv"
    return 0
  fi
  if [[ ! -x "${venv_path}/bin/python" ]]; then
    warn "${venv_path} has no bin/python; skipping snapshot of ${tag}"
    return 0
  fi

  local snap="${SNAPSHOTS_DIR}/${tag}-venv.tar.zst"
  local meta="${SNAPSHOTS_DIR}/${tag}-venv.meta"
  local tmp_snap="${snap}.part"
  local tmp_meta="${meta}.part"
  local parent base
  parent="$(dirname "${venv_path}")"
  base="$(basename "${venv_path}")"

  mkdir -p "${SNAPSHOTS_DIR}"
  rm -f "${tmp_snap}" "${tmp_meta}"

  log "Snapshotting ${tag} venv (${venv_path}) -> ${snap}"
  log "  zstd level=${SNAPSHOT_ZSTD_LEVEL}, threads=$(nproc), this takes ~30-60s"
  if ! tar --exclude='*/__pycache__' \
           --use-compress-program="zstd -${SNAPSHOT_ZSTD_LEVEL} -T0 --long" \
           -cf "${tmp_snap}" \
           -C "${parent}" "${base}"; then
    warn "Snapshot of ${tag} failed; leaving previous snapshot intact"
    rm -f "${tmp_snap}"
    return 0
  fi

  local py_ver="unknown"
  py_ver="$("${venv_path}/bin/python" -V 2>&1 | awk '{print $2}')"
  cat > "${tmp_meta}" <<EOF
venv_path=${venv_path}
python_version=${py_ver}
repo_root=${REPO_ROOT}
created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
host=$(hostname)
EOF

  mv "${tmp_snap}" "${snap}"
  mv "${tmp_meta}" "${meta}"

  local size
  size="$(du -h "${snap}" | awk '{print $1}')"
  log "Snapshot OK: ${snap} (${size})"
}

# Try to restore a venv from its OSS snapshot. Returns 0 on success
# (caller can skip the install pipeline entirely), non-zero otherwise.
#
# Refuses to restore when:
#   - EBFT_USE_VENV_SNAPSHOT=0
#   - EBFT_REBUILD_VENV=1
#   - the snapshot does not exist
#   - the snapshot was taken for a different venv path (uv venvs bake
#     absolute paths into pyvenv.cfg, so cross-path restore is unsafe)
#   - bin/python is missing after extraction
#   - the smoke import fails (host glibc / CUDA mismatch)
#
# On any failure the partially-restored venv is removed so the caller
# falls back to a clean uv venv + pip install.
try_restore_venv() {
  local venv_path="$1"
  local tag="$2"

  [[ "${EBFT_USE_VENV_SNAPSHOT}" == "1" ]] || return 1
  if [[ "${EBFT_REBUILD_VENV}" == "1" ]]; then
    log "EBFT_REBUILD_VENV=1, ignoring any existing ${tag} snapshot"
    return 1
  fi
  command -v zstd >/dev/null 2>&1 || return 1

  local snap="${SNAPSHOTS_DIR}/${tag}-venv.tar.zst"
  local meta="${SNAPSHOTS_DIR}/${tag}-venv.meta"
  if [[ ! -f "${snap}" ]]; then
    log "No ${tag} snapshot at ${snap}; will install from scratch"
    return 1
  fi

  if [[ -f "${meta}" ]]; then
    local saved_path
    saved_path="$(awk -F= '$1=="venv_path"{print $2}' "${meta}" 2>/dev/null)"
    if [[ -n "${saved_path}" && "${saved_path}" != "${venv_path}" ]]; then
      warn "${tag} snapshot was taken at ${saved_path}, refusing to restore into ${venv_path}"
      warn "(set EBFT_REBUILD_VENV=1 to overwrite, or align STUDENT_VENV/TEACHER_VENV)"
      return 1
    fi
  fi

  local parent base
  parent="$(dirname "${venv_path}")"
  base="$(basename "${venv_path}")"
  local size
  size="$(du -h "${snap}" 2>/dev/null | awk '{print $1}')"

  log "Restoring ${tag} venv from ${snap} (${size}) -> ${venv_path}"
  mkdir -p "${parent}"
  rm -rf "${venv_path}"

  if ! tar --use-compress-program="unzstd --long=27" \
           -xf "${snap}" -C "${parent}"; then
    warn "Snapshot extraction failed for ${tag}; falling back to clean install"
    rm -rf "${venv_path}"
    return 1
  fi

  if [[ ! -x "${venv_path}/bin/python" ]]; then
    warn "Restored ${tag} venv has no bin/python; falling back to clean install"
    rm -rf "${venv_path}"
    return 1
  fi

  local imports
  imports="$(_smoke_imports_for_tag "${tag}")"
  log "Smoke-testing restored ${tag} venv: ${imports}"
  if ! "${venv_path}/bin/python" -c "${imports}" 2>&1; then
    warn "Smoke import failed for restored ${tag} venv (host mismatch?); falling back to clean install"
    rm -rf "${venv_path}"
    return 1
  fi

  log "${tag} venv restored from snapshot OK"
  return 0
}

# Download "$2" -> "$1" if "$1" doesn't exist yet. Uses curl, then wget as
# fallback. Atomic via a .part tmp file so an aborted download won't be
# mistaken for a complete wheel on the next run.
download_file() {
  local dest="$1"
  local url="$2"
  if [[ -f "${dest}" ]]; then
    log "Reusing cached file ${dest}"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  local tmp="${dest}.part"
  rm -f "${tmp}"
  log "Downloading ${url} -> ${dest}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "${tmp}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${tmp}" "${url}"
  else
    die "Need either 'curl' or 'wget' to download ${url}"
  fi
  mv "${tmp}" "${dest}"
}

create_venv() {
  local venv_path="$1"
  log "Creating virtualenv at ${venv_path} (python ${PYTHON_VERSION})"
  rm -rf "${venv_path}"
  "${UV_BIN}" venv --seed --python "${PYTHON_VERSION}" "${venv_path}"
  [[ -x "${venv_path}/bin/python" ]] || die "venv creation failed for ${venv_path}"
  pip_install "${venv_path}/bin/python" --upgrade \
    pip "setuptools==82.0.1" "wheel==0.46.3"
}

# ---------------------------------------------------------------------------
# Student env (.venv): training / OpenRLHF / Ray / DeepSpeed
# ---------------------------------------------------------------------------
install_student() {
  local py="${STUDENT_VENV}/bin/python"

  log "Installing student PyTorch CUDA 12.4 stack (via ${STUDENT_TORCH_FIND_LINKS})"
  pip_install "${py}" \
    --find-links "${STUDENT_TORCH_FIND_LINKS}" \
    "torch==${STUDENT_TORCH_VERSION}" \
    "torchvision==${STUDENT_TORCHVISION_VERSION}" \
    "torchaudio==${STUDENT_TORCHAUDIO_VERSION}"

  log "Installing student training stack"
  pip_install "${py}" \
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
  pip_install "${py}" \
    "git+https://github.com/huggingface/transformers.git@${STUDENT_TRANSFORMERS_REF}"

  log "Installing student flash-attn==${STUDENT_FLASH_ATTN_VERSION}"
  local flash_installed=0
  local flash_wheel="${STUDENT_FLASH_ATTN_WHEEL_FILE}"
  if [[ -n "${flash_wheel}" && ! -f "${flash_wheel}" && -n "${STUDENT_FLASH_ATTN_WHEEL_URL}" ]]; then
    local fallback_wheel="${STUDENT_FLASH_ATTN_WHEEL_DIR}/$(basename "${STUDENT_FLASH_ATTN_WHEEL_URL}")"
    warn "Default wheel not found at ${flash_wheel}; downloading from ${STUDENT_FLASH_ATTN_WHEEL_URL}"
    if download_file "${fallback_wheel}" "${STUDENT_FLASH_ATTN_WHEEL_URL}"; then
      flash_wheel="${fallback_wheel}"
    else
      warn "flash-attn wheel download failed"
      flash_wheel=""
    fi
  fi
  if [[ -n "${flash_wheel}" && -f "${flash_wheel}" ]]; then
    log "Installing flash-attn from local wheel ${flash_wheel}"
    if pip_install "${py}" "${flash_wheel}"; then
      flash_installed=1
    else
      warn "Local flash-attn wheel install failed; falling back to PyPI"
    fi
  fi
  if [[ "${flash_installed}" -eq 0 ]]; then
    if ! pip_install "${py}" "flash-attn==${STUDENT_FLASH_ATTN_VERSION}"; then
      log "Falling back to source install for flash-attn"
      pip_install "${py}" --no-build-isolation \
        "flash-attn==${STUDENT_FLASH_ATTN_VERSION}"
    fi
  fi

  log "Installing this repo (editable) into ${STUDENT_VENV}"
  pip_install "${py}" -e "${REPO_ROOT}"

  stage_built_wheels
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
  pip_install "${py}" \
    "torch==${TEACHER_TORCH_VERSION}" \
    "torchvision==${TEACHER_TORCHVISION_VERSION}" \
    "torchaudio==${TEACHER_TORCHAUDIO_VERSION}"

  log "Installing teacher vLLM stack (vllm ${TEACHER_VLLM_VERSION})"
  pip_install "${py}" \
    "vllm==${TEACHER_VLLM_VERSION}" \
    "flashinfer-python==${TEACHER_FLASHINFER_VERSION}" \
    "tqdm==4.67.3"

  pip_install "${py}" "huggingface_hub==${TEACHER_HF_HUB_VERSION}"

  log "Pinning teacher transformers to ${TEACHER_TRANSFORMERS_VERSION} (no-deps)"
  pip_install "${py}" --no-deps \
    "transformers==${TEACHER_TRANSFORMERS_VERSION}"

  stage_built_wheels
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
  log "LOCAL_ROOT      = ${LOCAL_ROOT}        (local ext4; survives reboots, lost on container rebuild)"
  log "CACHE_ROOT      = ${CACHE_ROOT}        (persistent OSS)"
  log "STUDENT_VENV    = ${STUDENT_VENV}"
  log "TEACHER_VENV    = ${TEACHER_VENV}"
  log "PYTHON_VERSION  = ${PYTHON_VERSION}"
  log "PIP_CACHE_DIR        = ${PIP_CACHE_DIR}        (local ext4; OSS can't zipfile)"
  log "PIP_INDEX_URL        = ${PIP_INDEX_URL}"
  log "PIP_RETRIES          = ${PIP_RETRIES} (timeout=${PIP_TIMEOUT}s, attempts=${PIP_INSTALL_ATTEMPTS})"
  log "WHEEL_HOUSE_DIR      = ${WHEEL_HOUSE_DIR}        (OSS, --find-links)"
  log "SNAPSHOTS_DIR        = ${SNAPSHOTS_DIR}        (OSS, venv tarballs)"
  log "  EBFT_USE_VENV_SNAPSHOT=${EBFT_USE_VENV_SNAPSHOT}  EBFT_REBUILD_VENV=${EBFT_REBUILD_VENV}"
  log "HF_HOME              = ${HF_HOME}        (OSS, persistent model blobs)"
  log "TORCH_EXTENSIONS_DIR = ${TORCH_EXTENSIONS_DIR}        (local ext4; OSS can't host g++/nvcc builds)"
  log "TRITON_CACHE_DIR     = ${TRITON_CACHE_DIR}        (local ext4; same reason)"
  log "UV_CACHE_DIR         = ${UV_CACHE_DIR}"
  log "UV_PYTHON_INSTALL_DIR= ${UV_PYTHON_INSTALL_DIR}"

  mkdir -p "$(dirname "${STUDENT_VENV}")" "$(dirname "${TEACHER_VENV}")" \
           "${PIP_CACHE_DIR}" "${HF_HOME}" \
           "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" \
           "${UV_INSTALL_DIR}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" \
           "${WHEEL_HOUSE_DIR}" "${STUDENT_FLASH_ATTN_WHEEL_DIR}" \
           "${SNAPSHOTS_DIR}"

  if [[ "${SNAPSHOT_ONLY}" == "1" ]]; then
    log "SNAPSHOT_ONLY=1: tarring existing venvs into ${SNAPSHOTS_DIR} without reinstalling"
    if [[ "${SKIP_STUDENT}" != "1" ]]; then
      if [[ -x "${STUDENT_VENV}/bin/python" ]]; then
        snapshot_venv "${STUDENT_VENV}" "student"
      else
        warn "Cannot snapshot student: ${STUDENT_VENV} missing or broken (no bin/python)"
      fi
    fi
    if [[ "${SKIP_TEACHER}" != "1" ]]; then
      if [[ -x "${TEACHER_VENV}/bin/python" ]]; then
        snapshot_venv "${TEACHER_VENV}" "teacher"
      else
        warn "Cannot snapshot teacher: ${TEACHER_VENV} missing or broken (no bin/python)"
      fi
    fi
    log "Done (SNAPSHOT_ONLY mode)."
    return 0
  fi

  ensure_uv

  if [[ "${SKIP_STUDENT}" != "1" ]]; then
    if try_restore_venv "${STUDENT_VENV}" "student"; then
      verify_student
    else
      create_venv "${STUDENT_VENV}"
      install_student
      verify_student
      snapshot_venv "${STUDENT_VENV}" "student"
    fi
  else
    log "Skipping student env (SKIP_STUDENT=1)"
  fi

  if [[ "${SKIP_TEACHER}" != "1" ]]; then
    if try_restore_venv "${TEACHER_VENV}" "teacher"; then
      verify_teacher
    else
      create_venv "${TEACHER_VENV}"
      install_teacher
      verify_teacher
      snapshot_venv "${TEACHER_VENV}" "teacher"
    fi
  else
    log "Skipping teacher env (SKIP_TEACHER=1)"
  fi

  log "Done."
  printf "  student env:  source %s/bin/activate\n" "${STUDENT_VENV}"
  printf "  teacher env:  source %s/bin/activate\n" "${TEACHER_VENV}"
  printf "\n"
  printf "Note: venvs live on ephemeral ext4 (%s); snapshots in %s\n" \
         "${LOCAL_ROOT}" "${SNAPSHOTS_DIR}"
  printf "On a fresh machine / after a container restart, re-running this\n"
  printf "script will untar the snapshots back into %s in ~1-2 minutes\n" \
         "${LOCAL_ROOT}"
  printf "instead of re-installing from pip. To force a clean rebuild:\n"
  printf "  EBFT_REBUILD_VENV=1 bash scripts/setup_env.sh\n"
  printf "To disable snapshots entirely:\n"
  printf "  EBFT_USE_VENV_SNAPSHOT=0 bash scripts/setup_env.sh\n"
  printf "\n"
  printf "When invoking run_*.sh / supplement*/*.sh, export:\n"
  printf "  export STUDENT_VENV=%s\n" "${STUDENT_VENV}"
  printf "  export TEACHER_VENV=%s\n" "${TEACHER_VENV}"
  printf "  export HF_HOME=%s\n" "${HF_HOME}"
  printf "  export PIP_CACHE_DIR=%s\n" "${PIP_CACHE_DIR}"
  printf "  export TORCH_EXTENSIONS_DIR=%s\n" "${TORCH_EXTENSIONS_DIR}"
  printf "  export TRITON_CACHE_DIR=%s\n" "${TRITON_CACHE_DIR}"
}

main "$@"
