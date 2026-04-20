#!/usr/bin/env bash
set -euo pipefail

log() {
  printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"
}

die() {
  printf "error: %s\n" "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dlc_eval/migrate_eval_dlc_to_home.sh

Default behavior:
  1. Copy repo snapshot from:
       /mnt/data/Distributional-Match-Tuning-eval-dlc
     or
       /mnt/data/ebft-teacher-distribution/code/Distributional-Match-Tuning-eval-dlc
     to:
       ~/data/Distributional-Match-Tuning

  2. Copy outputs from:
       /mnt/data/ebft-teacher-distribution/outputs2
     to:
       ~/outputs

  3. Recreate:
       ~/data/Distributional-Match-Tuning/.venv
       ~/data/Distributional-Match-Tuning/.teacherVenv

Optional environment variables:
  SRC_REPO                 Explicit source repo path
  DST_REPO                 Defaults to ~/data/Distributional-Match-Tuning
  SRC_OUTPUTS              Defaults to /mnt/data/ebft-teacher-distribution/outputs2
  DST_OUTPUTS              Defaults to ~/outputs
  COPY_REPO                1 (default) or 0
  COPY_OUTPUTS             1 (default) or 0
  RECREATE_ENVS            1 (default) or 0
  REPLACE_DST_REPO         1 (default) or 0
  INSTALL_APT_DEPS         Passed through to recreate_current_env.sh
  REPO_SYNC_MODE_FOR_ENV   Defaults to skip

Examples:
  bash scripts/dlc_eval/migrate_eval_dlc_to_home.sh
  COPY_OUTPUTS=0 bash scripts/dlc_eval/migrate_eval_dlc_to_home.sh
  SRC_REPO=/mnt/data/Distributional-Match-Tuning-eval-dlc bash scripts/dlc_eval/migrate_eval_dlc_to_home.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_REPO="${SRC_REPO:-}"
DST_REPO="${DST_REPO:-${HOME}/data/Distributional-Match-Tuning}"
SRC_OUTPUTS="${SRC_OUTPUTS:-/mnt/data/ebft-teacher-distribution/outputs2}"
DST_OUTPUTS="${DST_OUTPUTS:-${HOME}/outputs}"
COPY_REPO="${COPY_REPO:-1}"
COPY_OUTPUTS="${COPY_OUTPUTS:-1}"
RECREATE_ENVS="${RECREATE_ENVS:-1}"
REPLACE_DST_REPO="${REPLACE_DST_REPO:-1}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"
REPO_SYNC_MODE_FOR_ENV="${REPO_SYNC_MODE_FOR_ENV:-skip}"

pick_default_src_repo() {
  local candidate
  for candidate in \
    "/mnt/data/Distributional-Match-Tuning-eval-dlc" \
    "/mnt/data/ebft-teacher-distribution/code/Distributional-Match-Tuning-eval-dlc"
  do
    if [[ -d "$candidate" ]]; then
      SRC_REPO="$candidate"
      return
    fi
  done

  die "Could not auto-detect source repo. Set SRC_REPO=/path/to/Distributional-Match-Tuning-eval-dlc"
}

safe_remove_dir() {
  local path="$1"
  [[ -n "$path" ]] || die "Refusing to remove an empty path"
  [[ "$path" != "/" ]] || die "Refusing to remove /"
  [[ "$path" != "$HOME" ]] || die "Refusing to remove \$HOME"
  rm -rf "$path"
}

copy_repo_snapshot() {
  [[ "$COPY_REPO" == "1" ]] || return

  if [[ -z "$SRC_REPO" ]]; then
    pick_default_src_repo
  fi

  [[ -d "$SRC_REPO" ]] || die "Source repo directory not found: $SRC_REPO"
  mkdir -p "$(dirname "$DST_REPO")"

  if [[ "$REPLACE_DST_REPO" == "1" && -e "$DST_REPO" ]]; then
    log "Replacing destination repo at $DST_REPO"
    safe_remove_dir "$DST_REPO"
  fi

  mkdir -p "$DST_REPO"
  log "Copying repo snapshot from $SRC_REPO to $DST_REPO"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude '.git/' \
      --exclude '.venv/' \
      --exclude '.teacherVenv/' \
      --exclude '__pycache__/' \
      --exclude '.mypy_cache/' \
      --exclude '.pytest_cache/' \
      --exclude '.ruff_cache/' \
      --exclude '*.pyc' \
      "$SRC_REPO/" "$DST_REPO/"
  else
    (
      cd "$SRC_REPO"
      tar \
        --exclude='.git' \
        --exclude='.venv' \
        --exclude='.teacherVenv' \
        --exclude='__pycache__' \
        --exclude='.mypy_cache' \
        --exclude='.pytest_cache' \
        --exclude='.ruff_cache' \
        --exclude='*.pyc' \
        -cf - .
    ) | (
      cd "$DST_REPO"
      tar -xf -
    )
  fi

  [[ -f "$DST_REPO/scripts/recreate_current_env.sh" ]] || die "Copied repo is missing scripts/recreate_current_env.sh"
}

copy_outputs_tree() {
  [[ "$COPY_OUTPUTS" == "1" ]] || return

  [[ -d "$SRC_OUTPUTS" ]] || die "Source outputs directory not found: $SRC_OUTPUTS"
  mkdir -p "$DST_OUTPUTS"

  log "Copying outputs from $SRC_OUTPUTS to $DST_OUTPUTS"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$SRC_OUTPUTS/" "$DST_OUTPUTS/"
  else
    (
      cd "$SRC_OUTPUTS"
      tar -cf - .
    ) | (
      cd "$DST_OUTPUTS"
      tar -xf -
    )
  fi
}

recreate_envs() {
  [[ "$RECREATE_ENVS" == "1" ]] || return
  local recreate_script="$DST_REPO/scripts/recreate_current_env.sh"
  [[ -f "$recreate_script" ]] || die "Missing recreate script in destination repo: $recreate_script"
  [[ -d "$DST_REPO" ]] || die "Destination repo directory not found: $DST_REPO"

  log "Recreating .venv and .teacherVenv under $DST_REPO"
  REPO_DIR="$DST_REPO" \
  REPO_SYNC_MODE="$REPO_SYNC_MODE_FOR_ENV" \
  INSTALL_APT_DEPS="$INSTALL_APT_DEPS" \
  STUDENT_TORCH_INDEX_URL="${STUDENT_TORCH_INDEX_URL:-}" \
  STUDENT_FLASH_ATTN_WHEEL="${STUDENT_FLASH_ATTN_WHEEL:-/mnt/data/ebft-teacher-distribution/flash_attn-2.8.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl}" \
  bash "$recreate_script"
}

main() {
  copy_repo_snapshot
  copy_outputs_tree
  recreate_envs

  log "Migration complete"
  printf "Repo path:\n  %s\n" "$DST_REPO"
  printf "Outputs path:\n  %s\n" "$DST_OUTPUTS"
}

main "$@"
