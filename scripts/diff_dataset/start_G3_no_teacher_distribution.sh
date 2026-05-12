#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export LOCAL_ROOT="${LOCAL_ROOT:-/mnt/workspace}"
export STUDENT_VENV="${STUDENT_VENV:-${LOCAL_ROOT}/venvs/.venv}"
export TEACHER_VENV="${TEACHER_VENV:-${LOCAL_ROOT}/venvs/.teacherVenv}"
export ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"

if [[ "${RUN_SETUP_ENV:-true}" == "true" ]]; then
  if [[ ! -x "${STUDENT_VENV}/bin/python" || ! -x "${TEACHER_VENV}/bin/python" ]]; then
    echo "[env] venv not found under ${LOCAL_ROOT}; running setup_env.sh"
    LOCAL_ROOT="${LOCAL_ROOT}" bash "${REPO_ROOT}/scripts/setup_env.sh"
  else
    echo "[env] using existing venvs under ${LOCAL_ROOT}"
  fi
fi

export CF_TARGET_MODE="${CF_TARGET_MODE:-vicinal}"
export RUN_NAME="${RUN_NAME:-diff_g3_no_teacher_distribution_vicinal_ema099_$(date +%m%d_%H%M)}"

exec bash "${SCRIPT_DIR}/run_G3_rebase_no_teacher_distribution_2node_once.sh" "$@"
