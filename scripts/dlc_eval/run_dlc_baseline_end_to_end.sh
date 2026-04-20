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
  bash scripts/dlc_eval/run_dlc_baseline_end_to_end.sh

What this script does:
1. Migrate the DLC repo snapshot to ~/data/Distributional-Match-Tuning
2. Recreate .venv and .teacherVenv in the migrated repo
3. Run scripts/dlc_eval/dlc_baseline_eval.sh from the migrated repo
   - first pass: full eval at 16k
   - second pass: retry non-correct prompts at 32k
4. Copy the finished run directory back to /mnt/data/ebft-teacher-distribution/outputs3

Progress monitoring from another terminal:
  bash ~/data/Distributional-Match-Tuning/scripts/dlc_eval/dlc_baseline_eval_progress.sh

Optional environment variables:
  DST_REPO               Defaults to ~/data/Distributional-Match-Tuning
  RUN_DIR                Defaults to ~/outputs/dlc_baseline_eval_retry16k_to_32k_<timestamp>
  OUTPUTS3_ROOT          Defaults to /mnt/data/ebft-teacher-distribution/outputs3
  COPY_REPO              Passed to migrate_eval_dlc_to_home.sh, defaults to 1
  COPY_OUTPUTS           Passed to migrate_eval_dlc_to_home.sh, defaults to 1
  REPLACE_DST_REPO       Passed to migrate_eval_dlc_to_home.sh, defaults to 1
  INSTALL_APT_DEPS       Passed to migrate_eval_dlc_to_home.sh
  MODEL_PATH             Passed through to dlc_baseline_eval.sh if set
  EVAL_DATA              Passed through to dlc_baseline_eval.sh if set
  MODEL_CUDA_VISIBLE_DEVICES Passed through to dlc_baseline_eval.sh if set
  VLLM_TP_SIZE           Passed through to dlc_baseline_eval.sh if set
  VLLM_MAX_NUM_SEQS      Passed through to dlc_baseline_eval.sh if set
  FIRST_PASS_MAX_NEW_TOKENS  Passed through to dlc_baseline_eval.sh if set
  SECOND_PASS_MAX_NEW_TOKENS Passed through to dlc_baseline_eval.sh if set
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATE_SCRIPT="${SCRIPT_DIR}/migrate_eval_dlc_to_home.sh"

[[ -f "${MIGRATE_SCRIPT}" ]] || die "Missing migrate script: ${MIGRATE_SCRIPT}"

TS="${TS:-$(date +%m%d_%H%M)}"
DST_REPO="${DST_REPO:-${HOME}/data/Distributional-Match-Tuning}"
RUN_DIR="${RUN_DIR:-${HOME}/outputs/dlc_baseline_eval_retry16k_to_32k_${TS}}"
OUTPUTS3_ROOT="${OUTPUTS3_ROOT:-/mnt/data/ebft-teacher-distribution/outputs3}"
CURRENT_PROGRESS_POINTER="${CURRENT_PROGRESS_POINTER:-${DST_REPO}/.dlc_baseline_eval_current.env}"
COPY_REPO="${COPY_REPO:-1}"
COPY_OUTPUTS="${COPY_OUTPUTS:-1}"
REPLACE_DST_REPO="${REPLACE_DST_REPO:-1}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"

sync_run_dir_back() {
  local src_run_dir="$1"
  local dst_root="$2"
  local dst_run_dir="${dst_root}/$(basename "${src_run_dir}")"

  [[ -d "${src_run_dir}" ]] || die "Run directory not found for sync-back: ${src_run_dir}"
  mkdir -p "${dst_root}"

  if command -v rsync >/dev/null 2>&1; then
    log "Syncing run directory back to ${dst_run_dir}"
    rsync -a "${src_run_dir}/" "${dst_run_dir}/"
    return
  fi

  log "Syncing run directory back to ${dst_run_dir} with tar fallback"
  mkdir -p "${dst_run_dir}"
  (
    cd "${src_run_dir}"
    tar -cf - .
  ) | (
    cd "${dst_run_dir}"
    tar -xf -
  )
}

main() {
  log "Step 1/3: migrate repo snapshot and recreate environments"
  DST_REPO="${DST_REPO}" \
  COPY_REPO="${COPY_REPO}" \
  COPY_OUTPUTS="${COPY_OUTPUTS}" \
  RECREATE_ENVS=1 \
  REPLACE_DST_REPO="${REPLACE_DST_REPO}" \
  INSTALL_APT_DEPS="${INSTALL_APT_DEPS}" \
  STUDENT_TORCH_INDEX_URL="${STUDENT_TORCH_INDEX_URL:-}" \
  STUDENT_FLASH_ATTN_WHEEL="${STUDENT_FLASH_ATTN_WHEEL:-}" \
  bash "${MIGRATE_SCRIPT}"

  local eval_script="${DST_REPO}/scripts/dlc_eval/dlc_baseline_eval.sh"
  local progress_script="${DST_REPO}/scripts/dlc_eval/dlc_baseline_eval_progress.sh"

  [[ -f "${eval_script}" ]] || die "Missing eval script after migration: ${eval_script}"
  [[ -f "${progress_script}" ]] || die "Missing progress script after migration: ${progress_script}"

  log "Step 2/3: run baseline eval"
  printf "Monitor progress from another terminal with:\n  bash \"%s\"\n" "${progress_script}"

  local eval_exit_code=0
  REPO_ROOT="${DST_REPO}" \
  RUN_DIR="${RUN_DIR}" \
  CURRENT_PROGRESS_POINTER="${CURRENT_PROGRESS_POINTER}" \
  bash "${eval_script}" || eval_exit_code=$?

  log "Step 3/3: sync outputs back to ${OUTPUTS3_ROOT}"
  if [[ -d "${RUN_DIR}" ]]; then
    sync_run_dir_back "${RUN_DIR}" "${OUTPUTS3_ROOT}"
    printf "Synced run dir:\n  %s\n" "${OUTPUTS3_ROOT}/$(basename "${RUN_DIR}")"
  else
    log "WARNING: RUN_DIR does not exist, nothing to sync: ${RUN_DIR}"
  fi

  printf "Local run dir:\n  %s\n" "${RUN_DIR}"

  if (( eval_exit_code != 0 )); then
    log "ERROR: eval script exited with code ${eval_exit_code}. Partial results have been synced back."
    exit "${eval_exit_code}"
  fi

  log "Done"
}

main "$@"
