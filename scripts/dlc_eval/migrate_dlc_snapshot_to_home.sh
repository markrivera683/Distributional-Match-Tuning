#!/usr/bin/env bash
set -euo pipefail

# Migrate a code snapshot from /mnt/data into ~/data, copy outputs, then
# recreate the two local virtualenvs under the target repo.

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
  bash scripts/dlc_eval/migrate_dlc_snapshot_to_home.sh

What this script does:
1. Copy the repo snapshot from /mnt/data to ~/data/Distributional-Matching-Tuning
2. Copy /mnt/data/ebft-teacher-distribution/outputs2/ into ~/outputs/
3. Create:
   - ~/data/Distributional-Matching-Tuning/.venv
   - ~/data/Distributional-Matching-Tuning/.teacherVenv

Optional environment variables:
  SRC_REPO_DIR               Source repo snapshot. Auto-detected if unset.
  TARGET_REPO_DIR            Defaults to $HOME/data/Distributional-Matching-Tuning
  SRC_OUTPUTS_DIR            Defaults to /mnt/data/ebft-teacher-distribution/outputs2
  TARGET_OUTPUTS_DIR         Defaults to $HOME/outputs
  COPY_OUTPUTS               1 (default) to copy outputs
  RECREATE_ENVS              1 (default) to build .venv and .teacherVenv
  OVERWRITE_TARGET_REPO      0 (default). Set to 1 to replace an existing target repo
  COPY_GIT_DIR               0 (default). Set to 1 to copy .git
  PYTHON_VERSION             Passed through to recreate_current_env.sh if set
  INSTALL_APT_DEPS           Passed through to recreate_current_env.sh if set
  STUDENT_FLASH_ATTN_STRATEGY Passed through if set
  STUDENT_FLASH_ATTN_WHEEL   Passed through if set
  STUDENT_CAUSAL_CONV1D_SPEC Passed through if set
  TEACHER_TORCH_INDEX_URL    Passed through if set

Examples:
  bash scripts/dlc_eval/migrate_dlc_snapshot_to_home.sh
  OVERWRITE_TARGET_REPO=1 bash scripts/dlc_eval/migrate_dlc_snapshot_to_home.sh
  SRC_REPO_DIR=/mnt/data/Distributional-Match-Tuning-eval-dlc bash scripts/dlc_eval/migrate_dlc_snapshot_to_home.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECREATE_SCRIPT="$(cd "${SCRIPT_DIR}/.." && pwd)/recreate_current_env.sh"
[[ -f "$RECREATE_SCRIPT" ]] || die "Missing recreate script: $RECREATE_SCRIPT"

TARGET_REPO_DIR="${TARGET_REPO_DIR:-$HOME/data/Distributional-Matching-Tuning}"
SRC_OUTPUTS_DIR="${SRC_OUTPUTS_DIR:-/mnt/data/ebft-teacher-distribution/outputs2}"
TARGET_OUTPUTS_DIR="${TARGET_OUTPUTS_DIR:-$HOME/outputs}"
COPY_OUTPUTS="${COPY_OUTPUTS:-1}"
RECREATE_ENVS="${RECREATE_ENVS:-1}"
OVERWRITE_TARGET_REPO="${OVERWRITE_TARGET_REPO:-0}"
COPY_GIT_DIR="${COPY_GIT_DIR:-0}"

pick_source_repo() {
  if [[ -n "${SRC_REPO_DIR:-}" ]]; then
    [[ -d "$SRC_REPO_DIR" ]] || die "SRC_REPO_DIR does not exist: $SRC_REPO_DIR"
    return
  fi

  local candidates=(
    "/mnt/data/Distributional-Match-Tuning-eval-dlc"
    "/mnt/data/ebft-teacher-distribution/code/Distributional-Match-Tuning-eval-dlc"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      SRC_REPO_DIR="$candidate"
      return
    fi
  done

  die "Could not auto-detect the DLC repo snapshot. Set SRC_REPO_DIR explicitly."
}

copy_repo_snapshot() {
  local excludes=(
    ".venv"
    ".teacherVenv"
    "__pycache__"
    ".pytest_cache"
    ".mypy_cache"
    ".ruff_cache"
  )

  if [[ "$COPY_GIT_DIR" != "1" ]]; then
    excludes+=(".git")
  fi

  if [[ -e "$TARGET_REPO_DIR" ]]; then
    if [[ "$OVERWRITE_TARGET_REPO" != "1" ]]; then
      die "Target repo already exists: $TARGET_REPO_DIR. Set OVERWRITE_TARGET_REPO=1 to replace it."
    fi
    log "Removing existing target repo at $TARGET_REPO_DIR"
    rm -rf "$TARGET_REPO_DIR"
  fi

  mkdir -p "$TARGET_REPO_DIR"

  if command -v rsync >/dev/null 2>&1; then
    local rsync_args=(-a)
    local pattern
    for pattern in "${excludes[@]}"; do
      rsync_args+=(--exclude="$pattern")
    done
    log "Copying repo snapshot with rsync"
    rsync "${rsync_args[@]}" "$SRC_REPO_DIR"/ "$TARGET_REPO_DIR"/
    return
  fi

  local tar_excludes=()
  local pattern
  for pattern in "${excludes[@]}"; do
    tar_excludes+=(--exclude="$pattern")
  done
  log "Copying repo snapshot with tar fallback"
  (
    cd "$SRC_REPO_DIR"
    tar "${tar_excludes[@]}" -cf - .
  ) | (
    cd "$TARGET_REPO_DIR"
    tar -xf -
  )
}

copy_outputs_dir() {
  [[ -d "$SRC_OUTPUTS_DIR" ]] || die "SRC_OUTPUTS_DIR does not exist: $SRC_OUTPUTS_DIR"
  mkdir -p "$TARGET_OUTPUTS_DIR"

  if command -v rsync >/dev/null 2>&1; then
    log "Copying outputs contents into $TARGET_OUTPUTS_DIR"
    rsync -a "$SRC_OUTPUTS_DIR"/ "$TARGET_OUTPUTS_DIR"/
    return
  fi

  log "Copying outputs contents into $TARGET_OUTPUTS_DIR with tar fallback"
  (
    cd "$SRC_OUTPUTS_DIR"
    tar -cf - .
  ) | (
    cd "$TARGET_OUTPUTS_DIR"
    tar -xf -
  )
}

recreate_target_envs() {
  local -a env_cmd=(
    env
    "REPO_DIR=$TARGET_REPO_DIR"
    "REPO_SYNC_MODE=skip"
    "RECREATE_STUDENT_ENV=1"
    "RECREATE_TEACHER_ENV=1"
    "STUDENT_VENV=$TARGET_REPO_DIR/.venv"
    "TEACHER_VENV=$TARGET_REPO_DIR/.teacherVenv"
  )

  local passthrough_vars=(
    PYTHON_VERSION
    INSTALL_APT_DEPS
    STUDENT_FLASH_ATTN_STRATEGY
    STUDENT_FLASH_ATTN_WHEEL
    STUDENT_CAUSAL_CONV1D_SPEC
    TEACHER_TORCH_INDEX_URL
  )

  local name
  for name in "${passthrough_vars[@]}"; do
    if [[ -n "${!name:-}" ]]; then
      env_cmd+=("${name}=${!name}")
    fi
  done

  log "Recreating local .venv and .teacherVenv under $TARGET_REPO_DIR"
  "${env_cmd[@]}" bash "$RECREATE_SCRIPT"
}

main() {
  pick_source_repo

  log "Source repo snapshot: $SRC_REPO_DIR"
  log "Target repo path:     $TARGET_REPO_DIR"
  log "Source outputs path:  $SRC_OUTPUTS_DIR"
  log "Target outputs path:  $TARGET_OUTPUTS_DIR"

  mkdir -p "$(dirname "$TARGET_REPO_DIR")"
  copy_repo_snapshot

  if [[ "$COPY_OUTPUTS" == "1" ]]; then
    copy_outputs_dir
  else
    log "Skipping outputs copy"
  fi

  if [[ "$RECREATE_ENVS" == "1" ]]; then
    recreate_target_envs
  else
    log "Skipping environment recreation"
  fi

  log "Done"
  printf "Repo:\n  %s\n" "$TARGET_REPO_DIR"
  printf "Student env:\n  %s\n" "$TARGET_REPO_DIR/.venv"
  printf "Teacher env:\n  %s\n" "$TARGET_REPO_DIR/.teacherVenv"
  printf "Outputs:\n  %s\n" "$TARGET_OUTPUTS_DIR"
}

main "$@"
