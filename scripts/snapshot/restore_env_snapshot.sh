#!/usr/bin/env bash
set -euo pipefail

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
  bash tools/restore_env_snapshot.sh
  bash scripts/snapshot/restore_env_snapshot.sh

Restore a snapshot created by create_env_snapshot.sh.

Defaults:
  TARGET_REPO_DIR=/root/code/Distributional-Match-Tuning
  TARGET_STUDENT_VENV=$TARGET_REPO_DIR/.venv
  TARGET_TEACHER_VENV=$TARGET_REPO_DIR/.teacherVenv
  TARGET_STUDENT_BASE_SITE_PACKAGES_DIR=$TARGET_REPO_DIR/.snapshot_runtime/student_system_site_packages
  TARGET_HF_CACHE_DIR=/root/.cache/huggingface
  TARGET_PIP_CACHE_DIR=/root/.cache/pip
  TARGET_TEACHER_CACHE_DIR=/root/outputs/teacher_cache_shared

Safety:
  - if any non-empty target already exists, set ALLOW_OVERWRITE=1 to replace it

Useful environment variables:
  SNAPSHOT_DIR
  TARGET_REPO_DIR
  TARGET_STUDENT_VENV
  TARGET_TEACHER_VENV
  TARGET_STUDENT_BASE_SITE_PACKAGES_DIR
  TARGET_HF_CACHE_DIR
  TARGET_PIP_CACHE_DIR
  TARGET_TEACHER_CACHE_DIR
  TARGET_PYTHON_BIN
  RESTORE_HF_CACHE=1|0
  RESTORE_PIP_CACHE=1|0
  RESTORE_TEACHER_CACHE=1|0
  CREATE_COMPAT_SYMLINKS=1|0
  VERIFY_IMPORTS=1|0
  ALLOW_OVERWRITE=1|0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Restore script is stored at <snapshot>/tools/restore_env_snapshot.sh,
# so the snapshot root is exactly one level above SCRIPT_DIR.
SNAPSHOT_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SNAPSHOT_DIR="${SNAPSHOT_DIR:-${SNAPSHOT_DIR_DEFAULT}}"
METADATA_DIR="${SNAPSHOT_DIR}/metadata"
ARCHIVE_DIR="${SNAPSHOT_DIR}/archives"
MANIFEST_PATH="${METADATA_DIR}/snapshot_manifest.env"

[[ -f "${MANIFEST_PATH}" ]] || die "Missing snapshot manifest: ${MANIFEST_PATH}"
# shellcheck disable=SC1090
source "${MANIFEST_PATH}"

TARGET_REPO_DIR="${TARGET_REPO_DIR:-/root/code/Distributional-Match-Tuning}"
TARGET_STUDENT_VENV="${TARGET_STUDENT_VENV:-${TARGET_REPO_DIR}/.venv}"
TARGET_TEACHER_VENV="${TARGET_TEACHER_VENV:-${TARGET_REPO_DIR}/.teacherVenv}"
TARGET_STUDENT_BASE_SITE_PACKAGES_DIR="${TARGET_STUDENT_BASE_SITE_PACKAGES_DIR:-${TARGET_REPO_DIR}/.snapshot_runtime/student_system_site_packages}"
TARGET_HF_CACHE_DIR="${TARGET_HF_CACHE_DIR:-/root/.cache/huggingface}"
TARGET_PIP_CACHE_DIR="${TARGET_PIP_CACHE_DIR:-/root/.cache/pip}"
TARGET_TEACHER_CACHE_DIR="${TARGET_TEACHER_CACHE_DIR:-/root/outputs/teacher_cache_shared}"

RESTORE_HF_CACHE="${RESTORE_HF_CACHE:-${INCLUDE_HF_CACHE:-1}}"
RESTORE_PIP_CACHE="${RESTORE_PIP_CACHE:-${INCLUDE_PIP_CACHE:-1}}"
RESTORE_TEACHER_CACHE="${RESTORE_TEACHER_CACHE:-${INCLUDE_TEACHER_CACHE:-1}}"
CREATE_COMPAT_SYMLINKS="${CREATE_COMPAT_SYMLINKS:-1}"
VERIFY_IMPORTS="${VERIFY_IMPORTS:-1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
TARGET_PYTHON_BIN="${TARGET_PYTHON_BIN:-}"

ensure_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: ${cmd}"
}

ensure_snapshot_layout() {
  [[ -d "${SNAPSHOT_DIR}" ]] || die "SNAPSHOT_DIR not found: ${SNAPSHOT_DIR}"
  [[ -d "${ARCHIVE_DIR}" ]] || die "ARCHIVE_DIR not found: ${ARCHIVE_DIR}"
  [[ -f "${ARCHIVE_DIR}/repo.tar" ]] || die "Missing repo archive"
  [[ -f "${ARCHIVE_DIR}/student_venv.tar" ]] || die "Missing student_venv archive"
  [[ -f "${ARCHIVE_DIR}/teacher_venv.tar" ]] || die "Missing teacher_venv archive"
  [[ -f "${ARCHIVE_DIR}/student_base_site_packages.tar" ]] || die "Missing student_base_site_packages archive"
}

resolve_target_python() {
  if [[ -n "${TARGET_PYTHON_BIN}" ]]; then
    [[ -x "${TARGET_PYTHON_BIN}" ]] || die "TARGET_PYTHON_BIN is not executable: ${TARGET_PYTHON_BIN}"
  else
    local candidate
    for candidate in python3.12 python3 /usr/local/bin/python3.12 /usr/local/bin/python3; do
      if command -v "${candidate}" >/dev/null 2>&1; then
        TARGET_PYTHON_BIN="$(command -v "${candidate}")"
        break
      fi
      if [[ -x "${candidate}" ]]; then
        TARGET_PYTHON_BIN="${candidate}"
        break
      fi
    done
  fi

  [[ -n "${TARGET_PYTHON_BIN}" ]] || die "Could not find a usable target Python interpreter."

  local target_mm
  target_mm="$("${TARGET_PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
)"

  if [[ -n "${SOURCE_STUDENT_PYTHON_MM:-}" ]] && [[ "${target_mm}" != "${SOURCE_STUDENT_PYTHON_MM}" ]]; then
    die "Target Python major.minor ${target_mm} does not match snapshot requirement ${SOURCE_STUDENT_PYTHON_MM}"
  fi
}

prepare_target_dir() {
  local dir_path="$1"
  local label="$2"

  if [[ -e "${dir_path}" ]]; then
    if [[ -n "$(ls -A "${dir_path}" 2>/dev/null || true)" ]] && [[ "${ALLOW_OVERWRITE}" != "1" ]]; then
      die "${label} already exists and is not empty: ${dir_path}. Re-run with ALLOW_OVERWRITE=1 to replace it."
    fi
    rm -rf "${dir_path}"
  fi
  mkdir -p "${dir_path}"
}

extract_archive() {
  local archive_path="$1"
  local target_dir="$2"
  local label="$3"
  [[ -f "${archive_path}" ]] || die "Archive not found: ${archive_path}"
  prepare_target_dir "${target_dir}" "${label}"
  log "Extracting ${archive_path} -> ${target_dir}"
  (
    cd "${target_dir}"
    tar -xpf "${archive_path}"
  )
}

restore_repo() {
  extract_archive "${ARCHIVE_DIR}/repo.tar" "${TARGET_REPO_DIR}" "TARGET_REPO_DIR"
}

restore_student_runtime() {
  extract_archive "${ARCHIVE_DIR}/student_venv.tar" "${TARGET_STUDENT_VENV}" "TARGET_STUDENT_VENV"
  extract_archive "${ARCHIVE_DIR}/student_base_site_packages.tar" "${TARGET_STUDENT_BASE_SITE_PACKAGES_DIR}" "TARGET_STUDENT_BASE_SITE_PACKAGES_DIR"
}

restore_teacher_runtime() {
  extract_archive "${ARCHIVE_DIR}/teacher_venv.tar" "${TARGET_TEACHER_VENV}" "TARGET_TEACHER_VENV"
}

restore_optional_cache() {
  local archive_path="$1"
  local target_dir="$2"
  local enabled="$3"
  local label="$4"
  if [[ "${enabled}" != "1" ]]; then
    log "Skipping ${label}"
    return
  fi
  [[ -f "${archive_path}" ]] || die "Requested ${label}, but archive is missing: ${archive_path}"
  extract_archive "${archive_path}" "${target_dir}" "${label}"
}

update_pyvenv_cfg() {
  local venv_path="$1"
  local include_system_site_packages="$2"
  python3 - "${venv_path}/pyvenv.cfg" "${TARGET_PYTHON_BIN}" "${venv_path}" "${include_system_site_packages}" <<'PY'
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
python_bin = Path(sys.argv[2])
venv_path = Path(sys.argv[3])
include_system_site_packages = sys.argv[4]

entries = {}
if cfg_path.exists():
    for raw_line in cfg_path.read_text().splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        entries[key.strip()] = value.strip()

entries["home"] = str(python_bin.parent)
entries["include-system-site-packages"] = "true" if include_system_site_packages == "1" else "false"
entries["version"] = ".".join(map(str, sys.version_info[:3]))
entries["executable"] = str(python_bin)
if include_system_site_packages == "1":
    command = f"{python_bin} -m venv --system-site-packages {venv_path}"
else:
    command = f"{python_bin} -m venv {venv_path}"
entries["command"] = command

ordered_keys = ["home", "include-system-site-packages", "version", "executable", "command"]
cfg_path.write_text("".join(f"{key} = {entries[key]}\n" for key in ordered_keys))
PY
}

relink_venv_python() {
  local venv_path="$1"
  local target_python_mm
  target_python_mm="$("${TARGET_PYTHON_BIN}" - <<'PY'
import sys
print(f"python{sys.version_info[0]}.{sys.version_info[1]}")
PY
)"

  rm -f "${venv_path}/bin/python" "${venv_path}/bin/python3" "${venv_path}/bin/${target_python_mm}"
  ln -sfn "${TARGET_PYTHON_BIN}" "${venv_path}/bin/python"
  ln -sfn python "${venv_path}/bin/python3"
  ln -sfn python "${venv_path}/bin/${target_python_mm}"
}

rewrite_venv_scripts() {
  local venv_path="$1"
  shift
  python3 - "${venv_path}/bin" "${venv_path}" "$@" <<'PY'
import sys
from pathlib import Path

bin_dir = Path(sys.argv[1])
venv_path = sys.argv[2]
old_paths = [item for item in sys.argv[3:] if item]

for path in bin_dir.iterdir():
    if not path.is_file():
        continue
    try:
        text = path.read_text()
    except Exception:
        continue

    original = text
    if text.startswith("#!"):
        first_line, sep, remainder = text.partition("\n")
        if "python" in first_line:
            text = f"#!{venv_path}/bin/python"
            if sep:
                text += "\n" + remainder

    for old_path in old_paths:
        text = text.replace(old_path, venv_path)

    if text != original:
        path.write_text(text)
PY
}

install_student_base_sitepackages_hook() {
  local student_site_packages
  student_site_packages="$("${TARGET_STUDENT_VENV}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
  mkdir -p "${student_site_packages}"
  cat > "${student_site_packages}/_snapshot_student_base.pth" <<EOF
import site; site.addsitedir(r'${TARGET_STUDENT_BASE_SITE_PACKAGES_DIR}')
EOF
  cat > "${student_site_packages}/_snapshot_repo_root.pth" <<EOF
${TARGET_REPO_DIR}
EOF
}

patch_student_base_sitepackages() {
  python3 - "${TARGET_STUDENT_BASE_SITE_PACKAGES_DIR}" "${SOURCE_REPO_DIR:-}" "${TARGET_REPO_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
source_repo_dir = sys.argv[2]
target_repo_dir = sys.argv[3]
if not source_repo_dir:
    raise SystemExit(0)

patterns = [
    "__editable___openrlhf_*.py",
    "__editable__.openrlhf-*.pth",
    "openrlhf-*.dist-info/direct_url.json",
    "flash_attn-*.dist-info/direct_url.json",
]

for pattern in patterns:
    for path in root.glob(pattern):
        try:
            text = path.read_text()
        except Exception:
            continue
        if source_repo_dir in text:
            path.write_text(text.replace(source_repo_dir, target_repo_dir))
PY
}

repair_student_env() {
  update_pyvenv_cfg "${TARGET_STUDENT_VENV}" "0"
  relink_venv_python "${TARGET_STUDENT_VENV}"
  rewrite_venv_scripts "${TARGET_STUDENT_VENV}" "${SOURCE_STUDENT_VENV:-}" "/root/code/.venv"
  install_student_base_sitepackages_hook
  patch_student_base_sitepackages
}

repair_teacher_env() {
  update_pyvenv_cfg "${TARGET_TEACHER_VENV}" "0"
  relink_venv_python "${TARGET_TEACHER_VENV}"
  rewrite_venv_scripts "${TARGET_TEACHER_VENV}" "${SOURCE_TEACHER_VENV:-}" "/root/code/.teacherVenv"
}

create_compat_symlinks() {
  if [[ "${CREATE_COMPAT_SYMLINKS}" != "1" ]]; then
    return
  fi

  log "Creating compatibility symlinks under /root/code and /root"
  mkdir -p /root/code /root/outputs
  ln -sfn "${TARGET_STUDENT_VENV}" /root/code/.venv
  ln -sfn "${TARGET_TEACHER_VENV}" /root/code/.teacherVenv
  ln -sfn "${TARGET_TEACHER_CACHE_DIR}" /root/teacher_cache_shared
}

verify_student_imports() {
  "${TARGET_STUDENT_VENV}/bin/python" - <<'PY'
import sys
from pathlib import Path

import deepspeed
import openrlhf
import ray
import torch
import transformers

print("student_python", sys.version.split()[0])
print("student_executable", sys.executable)
print("student_torch", torch.__version__)
print("student_torch_cuda", torch.version.cuda)
print("student_ray", ray.__version__)
print("student_deepspeed", deepspeed.__version__)
print("student_transformers", transformers.__version__)
print("student_openrlhf_path", Path(openrlhf.__file__).resolve())
PY
}

verify_teacher_imports() {
  "${TARGET_TEACHER_VENV}/bin/python" - <<'PY'
import shutil
import sys
from pathlib import Path

import torch
import transformers
import vllm

print("teacher_python", sys.version.split()[0])
print("teacher_executable", sys.executable)
print("teacher_torch", torch.__version__)
print("teacher_torch_cuda", torch.version.cuda)
print("teacher_transformers", transformers.__version__)
print("teacher_vllm", vllm.__version__)
print("teacher_vllm_cli", Path(sys.executable).with_name("vllm"))
print("teacher_vllm_on_path", shutil.which("vllm"))
PY
}

verify_runtime() {
  if [[ "${VERIFY_IMPORTS}" != "1" ]]; then
    log "Skipping import verification"
    return
  fi

  log "Verifying student environment imports"
  verify_student_imports
  log "Verifying teacher environment imports"
  verify_teacher_imports
}

main() {
  ensure_cmd tar
  ensure_cmd python3
  ensure_snapshot_layout
  resolve_target_python

  restore_repo
  restore_student_runtime
  restore_teacher_runtime
  restore_optional_cache "${ARCHIVE_DIR}/huggingface_cache.tar" "${TARGET_HF_CACHE_DIR}" "${RESTORE_HF_CACHE}" "TARGET_HF_CACHE_DIR"
  restore_optional_cache "${ARCHIVE_DIR}/pip_cache.tar" "${TARGET_PIP_CACHE_DIR}" "${RESTORE_PIP_CACHE}" "TARGET_PIP_CACHE_DIR"
  restore_optional_cache "${ARCHIVE_DIR}/teacher_cache_shared.tar" "${TARGET_TEACHER_CACHE_DIR}" "${RESTORE_TEACHER_CACHE}" "TARGET_TEACHER_CACHE_DIR"

  repair_student_env
  repair_teacher_env
  create_compat_symlinks
  verify_runtime

  log "Restore completed"
  printf "Repo root: %s\n" "${TARGET_REPO_DIR}"
  printf "Student env: %s\n" "${TARGET_STUDENT_VENV}"
  printf "Teacher env: %s\n" "${TARGET_TEACHER_VENV}"
}

main "$@"
