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
  bash scripts/dlc/bootstrap_g3_dlc_runtime.sh [command ...]

What this script does:
1. Sync the code snapshot from /mnt/data into /root/code/Distributional-Matching-Tuning
2. Copy the teacher cache snapshot from /mnt/data into /root/outputs/teacher_cache_shared
3. Recreate .venv and .teacherVenv from the exact freeze snapshots
4. Export REPO_ROOT / STUDENT_VENV / TEACHER_VENV / TEACHER_CACHE_DIR
5. Optionally execute the provided command inside the prepared repo

Environment variables:
  SNAPSHOT_REPO_DIR         Defaults to /mnt/data/ebft-teacher-distribution/code/Distributional-Matching-Tuning-g3-dlc
  TARGET_REPO_DIR           Defaults to /root/code/Distributional-Matching-Tuning
  SNAPSHOT_DIR              Defaults to $SNAPSHOT_REPO_DIR/.dlc_snapshot
  SOURCE_TEACHER_CACHE_DIR  Defaults to the snapshot manifest value or
                            /mnt/data/ebft-teacher-distribution/teacher_cache_shared_g3_dlc
  TARGET_TEACHER_CACHE_DIR  Defaults to /root/outputs/teacher_cache_shared
  SYNC_REPO                 1 (default) or 0
  SYNC_TEACHER_CACHE        1 (default) or 0
  REBUILD_ENVS              1 (default) or 0
  FORCE_REBUILD_ENVS        0 (default) or 1
  INSTALL_APT_DEPS          Passed to recreate_env_from_freeze.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SNAPSHOT_REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SNAPSHOT_REPO_DIR="${SNAPSHOT_REPO_DIR:-$DEFAULT_SNAPSHOT_REPO_DIR}"
TARGET_REPO_DIR="${TARGET_REPO_DIR:-/root/code/Distributional-Matching-Tuning}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$SNAPSHOT_REPO_DIR/.dlc_snapshot}"
SYNC_REPO="${SYNC_REPO:-1}"
SYNC_TEACHER_CACHE="${SYNC_TEACHER_CACHE:-1}"
REBUILD_ENVS="${REBUILD_ENVS:-1}"
FORCE_REBUILD_ENVS="${FORCE_REBUILD_ENVS:-0}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"
TARGET_TEACHER_CACHE_DIR="${TARGET_TEACHER_CACHE_DIR:-/root/outputs/teacher_cache_shared}"
SOURCE_TEACHER_CACHE_DIR="${SOURCE_TEACHER_CACHE_DIR:-}"

load_snapshot_manifest() {
  local manifest_path="${SNAPSHOT_DIR}/snapshot_manifest.env"
  local explicit_source_teacher_cache_dir="${SOURCE_TEACHER_CACHE_DIR}"
  if [[ -f "${manifest_path}" ]]; then
    # shellcheck disable=SC1090
    source "${manifest_path}"
  fi

  if [[ -n "${explicit_source_teacher_cache_dir}" ]]; then
    SOURCE_TEACHER_CACHE_DIR="${explicit_source_teacher_cache_dir}"
  elif [[ -n "${SNAPSHOT_CACHE_DIR:-}" ]]; then
    SOURCE_TEACHER_CACHE_DIR="${SNAPSHOT_CACHE_DIR}"
  elif [[ -d "${SOURCE_TEACHER_CACHE_DIR:-}" ]]; then
    :
  else
    SOURCE_TEACHER_CACHE_DIR="/mnt/data/ebft-teacher-distribution/teacher_cache_shared_g3_dlc"
  fi
}

ensure_snapshot_layout() {
  [[ -d "${SNAPSHOT_REPO_DIR}" ]] || die "SNAPSHOT_REPO_DIR not found: ${SNAPSHOT_REPO_DIR}"
  [[ -d "${SNAPSHOT_DIR}" ]] || die "SNAPSHOT_DIR not found: ${SNAPSHOT_DIR}"
}

sync_repo_snapshot() {
  if [[ "${SYNC_REPO}" != "1" ]]; then
    log "Skipping repo sync"
    return
  fi

  mkdir -p "${TARGET_REPO_DIR}"
  if command -v rsync >/dev/null 2>&1; then
    log "Syncing repo snapshot into ${TARGET_REPO_DIR}"
    rsync -a --delete \
      --exclude=".venv" \
      --exclude=".teacherVenv" \
      --exclude=".git" \
      "${SNAPSHOT_REPO_DIR}/" "${TARGET_REPO_DIR}/"
    return
  fi

  log "Syncing repo snapshot with tar fallback into ${TARGET_REPO_DIR}"
  rm -rf "${TARGET_REPO_DIR}"
  mkdir -p "${TARGET_REPO_DIR}"
  (
    cd "${SNAPSHOT_REPO_DIR}"
    tar --exclude=".venv" --exclude=".teacherVenv" --exclude=".git" -cf - .
  ) | (
    cd "${TARGET_REPO_DIR}"
    tar -xf -
  )
}

sync_teacher_cache_snapshot() {
  if [[ "${SYNC_TEACHER_CACHE}" != "1" ]]; then
    log "Skipping teacher cache sync"
    return
  fi

  [[ -d "${SOURCE_TEACHER_CACHE_DIR}" ]] || die "SOURCE_TEACHER_CACHE_DIR not found: ${SOURCE_TEACHER_CACHE_DIR}"
  mkdir -p "${TARGET_TEACHER_CACHE_DIR}"

  if command -v rsync >/dev/null 2>&1; then
    log "Copying teacher cache into local path ${TARGET_TEACHER_CACHE_DIR}"
    rsync -a --delete "${SOURCE_TEACHER_CACHE_DIR}/" "${TARGET_TEACHER_CACHE_DIR}/"
    return
  fi

  log "Copying teacher cache with tar fallback into ${TARGET_TEACHER_CACHE_DIR}"
  rm -rf "${TARGET_TEACHER_CACHE_DIR}"
  mkdir -p "${TARGET_TEACHER_CACHE_DIR}"
  (
    cd "${SOURCE_TEACHER_CACHE_DIR}"
    tar -cf - .
  ) | (
    cd "${TARGET_TEACHER_CACHE_DIR}"
    tar -xf -
  )
}

recreate_envs() {
  if [[ "${REBUILD_ENVS}" != "1" ]]; then
    log "Skipping environment rebuild"
    return
  fi

  local recreate_script="${TARGET_REPO_DIR}/scripts/dlc/recreate_env_from_freeze.sh"
  [[ -f "${recreate_script}" ]] || die "Missing recreate script: ${recreate_script}"

  log "Recreating runtime environments under ${TARGET_REPO_DIR}"
  env \
    REPO_DIR="${TARGET_REPO_DIR}" \
    SNAPSHOT_DIR="${TARGET_REPO_DIR}/.dlc_snapshot" \
    STUDENT_VENV="${TARGET_REPO_DIR}/.venv" \
    TEACHER_VENV="${TARGET_REPO_DIR}/.teacherVenv" \
    INSTALL_APT_DEPS="${INSTALL_APT_DEPS}" \
    FORCE_REBUILD="${FORCE_REBUILD_ENVS}" \
    bash "${recreate_script}"
}

ensure_compat_symlinks() {
  mkdir -p /root/code /root/outputs
  ln -sfn "${TARGET_REPO_DIR}/.venv" /root/code/.venv
  ln -sfn "${TARGET_REPO_DIR}/.teacherVenv" /root/code/.teacherVenv
  ln -sfn "${TARGET_TEACHER_CACHE_DIR}" /root/teacher_cache_shared
}

main() {
  load_snapshot_manifest
  ensure_snapshot_layout
  sync_repo_snapshot
  sync_teacher_cache_snapshot
  recreate_envs
  ensure_compat_symlinks

  export REPO_ROOT="${TARGET_REPO_DIR}"
  export STUDENT_VENV="${TARGET_REPO_DIR}/.venv"
  export TEACHER_VENV="${TARGET_REPO_DIR}/.teacherVenv"
  export TEACHER_CACHE_DIR="${TARGET_TEACHER_CACHE_DIR}"

  log "Runtime prepared"
  printf "REPO_ROOT=%s\n" "${REPO_ROOT}"
  printf "STUDENT_VENV=%s\n" "${STUDENT_VENV}"
  printf "TEACHER_VENV=%s\n" "${TEACHER_VENV}"
  printf "TEACHER_CACHE_DIR=%s\n" "${TEACHER_CACHE_DIR}"

  cd "${TARGET_REPO_DIR}"
  if (( $# > 0 )); then
    exec "$@"
  fi
}

main "$@"
