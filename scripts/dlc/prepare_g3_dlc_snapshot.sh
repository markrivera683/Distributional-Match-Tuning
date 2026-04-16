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
  bash scripts/dlc/prepare_g3_dlc_snapshot.sh

What this script does:
1. Copy the current repo working tree into:
     /mnt/data/ebft-teacher-distribution/code/Distributional-Matching-Tuning-g3-dlc
   while excluding .venv/.teacherVenv and other local caches.
2. Snapshot the exact current student/teacher Python environments into
   .dlc_snapshot/*.txt and .dlc_snapshot/*.env under that copied repo.
3. Copy the current teacher cache snapshot into:
     /mnt/data/ebft-teacher-distribution/teacher_cache_shared_g3_dlc

Environment variables:
  SRC_REPO_DIR           Defaults to the current repo root
  DST_REPO_DIR           Defaults to /mnt/data/ebft-teacher-distribution/code/Distributional-Matching-Tuning-g3-dlc
  SRC_STUDENT_VENV       Defaults to $SRC_REPO_DIR/.venv
  SRC_TEACHER_VENV       Defaults to $SRC_REPO_DIR/.teacherVenv
  SRC_TEACHER_CACHE_DIR  Defaults to /root/outputs/teacher_cache_shared
  DST_TEACHER_CACHE_DIR  Defaults to /mnt/data/ebft-teacher-distribution/teacher_cache_shared_g3_dlc
  OVERWRITE_TARGET_REPO  1 (default) replaces DST_REPO_DIR
  SYNC_TEACHER_CACHE     1 (default) copies the teacher cache snapshot
  COPY_GIT_DIR           0 (default) excludes .git from the copied repo
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SRC_REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SRC_REPO_DIR="${SRC_REPO_DIR:-$DEFAULT_SRC_REPO_DIR}"
DST_REPO_DIR="${DST_REPO_DIR:-/mnt/data/ebft-teacher-distribution/code/Distributional-Matching-Tuning-g3-dlc}"
SRC_STUDENT_VENV="${SRC_STUDENT_VENV:-$SRC_REPO_DIR/.venv}"
SRC_TEACHER_VENV="${SRC_TEACHER_VENV:-$SRC_REPO_DIR/.teacherVenv}"
SRC_TEACHER_CACHE_DIR="${SRC_TEACHER_CACHE_DIR:-/root/outputs/teacher_cache_shared}"
DST_TEACHER_CACHE_DIR="${DST_TEACHER_CACHE_DIR:-/mnt/data/ebft-teacher-distribution/teacher_cache_shared_g3_dlc}"
OVERWRITE_TARGET_REPO="${OVERWRITE_TARGET_REPO:-1}"
SYNC_TEACHER_CACHE="${SYNC_TEACHER_CACHE:-1}"
COPY_GIT_DIR="${COPY_GIT_DIR:-0}"

SNAPSHOT_DIR="${DST_REPO_DIR}/.dlc_snapshot"

ensure_paths() {
  [[ -d "${SRC_REPO_DIR}" ]] || die "SRC_REPO_DIR not found: ${SRC_REPO_DIR}"
  [[ -x "${SRC_STUDENT_VENV}/bin/python" ]] || die "Student env missing: ${SRC_STUDENT_VENV}"
  [[ -x "${SRC_TEACHER_VENV}/bin/python" ]] || die "Teacher env missing: ${SRC_TEACHER_VENV}"
  if [[ "${SYNC_TEACHER_CACHE}" == "1" ]]; then
    [[ -d "${SRC_TEACHER_CACHE_DIR}" ]] || die "Teacher cache dir not found: ${SRC_TEACHER_CACHE_DIR}"
  fi
}

copy_repo_snapshot() {
  local excludes=(
    ".venv"
    ".teacherVenv"
    ".git"
    ".cursor"
    "__pycache__"
    ".pytest_cache"
    ".mypy_cache"
    ".ruff_cache"
  )

  if [[ "${COPY_GIT_DIR}" == "1" ]]; then
    excludes=(
      ".venv"
      ".teacherVenv"
      ".cursor"
      "__pycache__"
      ".pytest_cache"
      ".mypy_cache"
      ".ruff_cache"
    )
  fi

  if [[ -e "${DST_REPO_DIR}" ]]; then
    if [[ "${OVERWRITE_TARGET_REPO}" != "1" ]]; then
      die "DST_REPO_DIR already exists: ${DST_REPO_DIR}. Set OVERWRITE_TARGET_REPO=1 to replace it."
    fi
    log "Removing existing snapshot repo at ${DST_REPO_DIR}"
    rm -rf "${DST_REPO_DIR}"
  fi

  mkdir -p "${DST_REPO_DIR}"

  if command -v rsync >/dev/null 2>&1; then
    local rsync_args=(-a)
    local pattern
    for pattern in "${excludes[@]}"; do
      rsync_args+=(--exclude="${pattern}")
    done
    log "Copying repo snapshot into ${DST_REPO_DIR}"
    rsync "${rsync_args[@]}" "${SRC_REPO_DIR}/" "${DST_REPO_DIR}/"
    return
  fi

  local tar_excludes=()
  local pattern
  for pattern in "${excludes[@]}"; do
    tar_excludes+=(--exclude="${pattern}")
  done
  log "Copying repo snapshot with tar fallback into ${DST_REPO_DIR}"
  (
    cd "${SRC_REPO_DIR}"
    tar "${tar_excludes[@]}" -cf - .
  ) | (
    cd "${DST_REPO_DIR}"
    tar -xf -
  )
}

sanitize_freeze() {
  local input_path="$1"
  local output_path="$2"
  python - "${input_path}" "${output_path}" <<'PY'
import pathlib
import sys

input_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
lines = []
for raw_line in input_path.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("-e ") or line.startswith("--editable"):
        continue
    lines.append(line)
output_path.write_text("\n".join(lines) + ("\n" if lines else ""))
PY
}

write_packaging_meta() {
  local python_bin="$1"
  local output_path="$2"
  "${python_bin}" - <<'PY' > "${output_path}"
import pip
import setuptools
import sys
import wheel

print(f"PYTHON_VERSION={sys.version.split()[0]}")
print(f"PIP_VERSION={pip.__version__}")
print(f"SETUPTOOLS_VERSION={setuptools.__version__}")
print(f"WHEEL_VERSION={wheel.__version__}")
PY
}

write_environment_snapshots() {
  mkdir -p "${SNAPSHOT_DIR}"

  log "Writing student/teacher freeze snapshots under ${SNAPSHOT_DIR}"
  "${SRC_STUDENT_VENV}/bin/python" -m pip freeze > "${SNAPSHOT_DIR}/student.requirements.raw.txt"
  "${SRC_TEACHER_VENV}/bin/python" -m pip freeze > "${SNAPSHOT_DIR}/teacher.requirements.raw.txt"

  sanitize_freeze "${SNAPSHOT_DIR}/student.requirements.raw.txt" "${SNAPSHOT_DIR}/student.requirements.install.txt"
  sanitize_freeze "${SNAPSHOT_DIR}/teacher.requirements.raw.txt" "${SNAPSHOT_DIR}/teacher.requirements.install.txt"

  write_packaging_meta "${SRC_STUDENT_VENV}/bin/python" "${SNAPSHOT_DIR}/student.packaging.env"
  write_packaging_meta "${SRC_TEACHER_VENV}/bin/python" "${SNAPSHOT_DIR}/teacher.packaging.env"
}

write_manifest() {
  mkdir -p "${SNAPSHOT_DIR}"

  local git_head="unknown"
  local git_dirty="unknown"
  if [[ -d "${SRC_REPO_DIR}/.git" ]]; then
    git_head="$(git -C "${SRC_REPO_DIR}" rev-parse HEAD)"
    if [[ -n "$(git -C "${SRC_REPO_DIR}" status --porcelain)" ]]; then
      git_dirty="1"
    else
      git_dirty="0"
    fi
  fi

  {
    echo "# Auto-generated DLC snapshot manifest"
    echo "SNAPSHOT_CREATED_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf "SOURCE_REPO_DIR=%q\n" "${SRC_REPO_DIR}"
    printf "SNAPSHOT_REPO_DIR=%q\n" "${DST_REPO_DIR}"
    printf "SNAPSHOT_CACHE_DIR=%q\n" "${DST_TEACHER_CACHE_DIR}"
    printf "SOURCE_TEACHER_CACHE_DIR=%q\n" "${SRC_TEACHER_CACHE_DIR}"
    printf "SOURCE_GIT_HEAD=%q\n" "${git_head}"
    printf "SOURCE_GIT_DIRTY=%q\n" "${git_dirty}"
  } > "${SNAPSHOT_DIR}/snapshot_manifest.env"
}

sync_teacher_cache() {
  if [[ "${SYNC_TEACHER_CACHE}" != "1" ]]; then
    log "Skipping teacher cache sync"
    return
  fi

  mkdir -p "${DST_TEACHER_CACHE_DIR}"
  if command -v rsync >/dev/null 2>&1; then
    log "Syncing teacher cache snapshot into ${DST_TEACHER_CACHE_DIR}"
    rsync -a --delete "${SRC_TEACHER_CACHE_DIR}/" "${DST_TEACHER_CACHE_DIR}/"
    return
  fi

  log "Syncing teacher cache snapshot with tar fallback into ${DST_TEACHER_CACHE_DIR}"
  rm -rf "${DST_TEACHER_CACHE_DIR}"
  mkdir -p "${DST_TEACHER_CACHE_DIR}"
  (
    cd "${SRC_TEACHER_CACHE_DIR}"
    tar -cf - .
  ) | (
    cd "${DST_TEACHER_CACHE_DIR}"
    tar -xf -
  )
}

main() {
  ensure_paths

  log "Source repo:           ${SRC_REPO_DIR}"
  log "Snapshot repo target:  ${DST_REPO_DIR}"
  log "Source teacher cache:  ${SRC_TEACHER_CACHE_DIR}"
  log "Snapshot teacher cache:${DST_TEACHER_CACHE_DIR}"

  mkdir -p "$(dirname "${DST_REPO_DIR}")" "$(dirname "${DST_TEACHER_CACHE_DIR}")"
  copy_repo_snapshot
  write_environment_snapshots
  write_manifest
  sync_teacher_cache

  log "Done"
  printf "Snapshot repo:\n  %s\n" "${DST_REPO_DIR}"
  printf "Snapshot cache:\n  %s\n" "${DST_TEACHER_CACHE_DIR}"
  printf "DLC startup helper:\n  %s\n" "${DST_REPO_DIR}/scripts/dlc/bootstrap_g3_dlc_runtime.sh"
}

main "$@"
