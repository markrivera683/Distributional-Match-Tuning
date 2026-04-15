#!/usr/bin/env bash
# Monitor a running DLC baseline eval launched by `dlc_baseline_eval.sh`.
# By default it auto-detects the latest active progress file.
set -euo pipefail

die() {
  printf "error: %s\n" "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dlc_eval/dlc_baseline_eval_progress.sh
  WATCH=0 bash scripts/dlc_eval/dlc_baseline_eval_progress.sh
  RUN_DIR=~/outputs/dlc_baseline_eval_retry16k_to_32k_xxx bash scripts/dlc_eval/dlc_baseline_eval_progress.sh
  PROGRESS_JSON_PATH=/path/to/progress.json bash scripts/dlc_eval/dlc_baseline_eval_progress.sh

Defaults:
  - REPO_ROOT defaults to ~/data/Distributional-Matching-Tuning
  - The script first tries the current pointer file written by dlc_baseline_eval.sh
  - If that is missing, it falls back to the newest progress_*.json under ~/outputs

Optional environment variables:
  REPO_ROOT                Defaults to ~/data/Distributional-Matching-Tuning
  CURRENT_PROGRESS_POINTER Defaults to $REPO_ROOT/.dlc_baseline_eval_current.env
  PIPELINE_STATE_PATH      Explicit pipeline state JSON to inspect
  PROGRESS_JSON_PATH       Explicit progress JSON to inspect
  RUN_DIR                  Eval run directory; script will auto-find progress_*.json inside it
  WATCH                    1 (default) to refresh repeatedly, 0 for a one-shot snapshot
  WATCH_INTERVAL           Seconds between refreshes when WATCH=1. Defaults to 5
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="${REPO_ROOT:-${HOME}/data/Distributional-Matching-Tuning}"
CURRENT_PROGRESS_POINTER="${CURRENT_PROGRESS_POINTER:-${REPO_ROOT}/.dlc_baseline_eval_current.env}"
PIPELINE_STATE_PATH="${PIPELINE_STATE_PATH:-}"
PROGRESS_JSON_PATH="${PROGRESS_JSON_PATH:-}"
RUN_DIR="${RUN_DIR:-${1:-}}"
WATCH="${WATCH:-1}"
WATCH_INTERVAL="${WATCH_INTERVAL:-5}"

pick_latest_progress_file() {
  local latest=""
  latest="$(ls -td "${HOME}"/outputs/*/supplement_logs/progress_*.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    printf "%s\n" "${latest}"
  fi
}

pick_latest_pipeline_state_file() {
  local latest=""
  latest="$(ls -td "${HOME}"/outputs/*/supplement_logs/pipeline_state_*.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest}" ]]; then
    printf "%s\n" "${latest}"
  fi
}

resolve_pipeline_state() {
  if [[ -n "${PIPELINE_STATE_PATH}" ]]; then
    return
  fi

  if [[ -n "${RUN_DIR}" ]]; then
    PIPELINE_STATE_PATH="$(ls -td "${RUN_DIR}"/supplement_logs/pipeline_state_*.json 2>/dev/null | head -n 1 || true)"
    [[ -n "${PIPELINE_STATE_PATH}" ]] && return
  fi

  if [[ -f "${CURRENT_PROGRESS_POINTER}" ]]; then
    # shellcheck disable=SC1090
    source "${CURRENT_PROGRESS_POINTER}"
    if [[ -n "${PIPELINE_STATE_PATH:-}" ]]; then
      return
    fi
  fi

  PIPELINE_STATE_PATH="$(pick_latest_pipeline_state_file || true)"
}

resolve_progress_json() {
  if [[ -n "${PROGRESS_JSON_PATH}" ]]; then
    return
  fi

  if [[ -n "${RUN_DIR}" ]]; then
    PROGRESS_JSON_PATH="$(ls -td "${RUN_DIR}"/supplement_logs/progress_*.json 2>/dev/null | head -n 1 || true)"
    [[ -n "${PROGRESS_JSON_PATH}" ]] && return
  fi

  if [[ -f "${CURRENT_PROGRESS_POINTER}" ]]; then
    # shellcheck disable=SC1090
    source "${CURRENT_PROGRESS_POINTER}"
    if [[ -n "${PROGRESS_JSON_PATH:-}" ]]; then
      return
    fi
  fi

  PROGRESS_JSON_PATH="$(pick_latest_progress_file || true)"
}

read_progress_field() {
  local field="$1"
  python3 - "$PROGRESS_JSON_PATH" "$field" <<'PY'
import json
import sys

path = sys.argv[1]
field = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
value = data.get(field, "")
if value is None:
    value = ""
print(value)
PY
}

read_pipeline_field() {
  local field="$1"
  python3 - "$PIPELINE_STATE_PATH" "$field" <<'PY'
import json
import sys

path = sys.argv[1]
field = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
value = data.get(field, "")
if value is None:
    value = ""
print(value)
PY
}

render_pipeline_state() {
  python3 - "$PIPELINE_STATE_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("==== DLC Baseline Eval Pipeline ====")
print(f"status:               {data.get('status', 'unknown')}")
print(f"phase:                {data.get('phase', 'unknown')}")
print(f"message:              {data.get('message', '')}")
print(f"run_dir:              {data.get('run_dir', 'n/a')}")
print(f"model_path:           {data.get('model_path', 'n/a')}")
print(f"eval_data:            {data.get('eval_data', 'n/a')}")
print(f"started_at:           {data.get('started_at', 'n/a')}")
print(f"updated_at:           {data.get('updated_at', 'n/a')}")
print(f"finished_at:          {data.get('finished_at', 'n/a')}")
if data.get("retry_count") is not None:
    print(f"retry_count:          {data.get('retry_count')}")
if data.get("active_max_new_tokens") is not None:
    print(f"active_max_new_tokens:{data.get('active_max_new_tokens')}")
if data.get("final_analysis_report_path"):
    print(f"final_report:         {data.get('final_analysis_report_path')}")
if data.get("error"):
    print(f"error:                {data['error']}")
PY
}

render_progress() {
  python3 - "$PROGRESS_JSON_PATH" <<'PY'
import json
import sys


def fmt_seconds(value):
    if value in (None, "", 0):
        return "n/a"
    value = float(value)
    if value < 0:
        return "n/a"
    mins, secs = divmod(int(value), 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m {secs}s"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

status = data.get("status", "unknown")
written = int(data.get("written", 0) or 0)
total_outputs = int(data.get("total_outputs", 0) or 0)
remaining_outputs = int(data.get("remaining_outputs", max(total_outputs - written, 0)) or 0)
percent = float(data.get("percent", (written / total_outputs * 100.0) if total_outputs else 0.0) or 0.0)
chunk_idx = int(data.get("chunk_idx", 0) or 0)
chunk_count = int(data.get("chunk_count", 0) or 0)
rate = float(data.get("rate_samples_per_sec", 0.0) or 0.0)
eta_seconds = remaining_outputs / rate if rate > 0 and status not in {"completed", "failed"} else None

print("==== DLC Baseline Eval Progress ====")
print(f"status:               {status}")
print(f"model_path:           {data.get('model_path', 'n/a')}")
print(f"dataset_path:         {data.get('dataset_path', 'n/a')}")
print(f"output_path:          {data.get('output_path', 'n/a')}")
print(f"written/total:        {written}/{total_outputs}")
print(f"percent:              {percent:.2f}%")
print(f"chunk:                {chunk_idx}/{chunk_count}")
print(f"rate:                 {rate:.2f} samples/s")
print(f"remaining_outputs:    {remaining_outputs}")
print(f"eta:                  {fmt_seconds(eta_seconds)}")
print(f"started_at:           {data.get('started_at', 'n/a')}")
print(f"updated_at:           {data.get('updated_at', 'n/a')}")
print(f"finished_at:          {data.get('finished_at', 'n/a')}")
if data.get("error"):
    print(f"error:                {data['error']}")
PY

  if [[ -n "${RUN_DIR:-}" ]]; then
    printf "run_dir:              %s\n" "${RUN_DIR}"
  fi
  if [[ -n "${POST_EVAL_LOG_PATH:-}" ]]; then
    printf "eval_log:             %s\n" "${POST_EVAL_LOG_PATH}"
  fi
  if [[ -n "${SCRIPT_LOG_PATH:-}" ]]; then
    printf "script_log:           %s\n" "${SCRIPT_LOG_PATH}"
  fi
  if [[ -n "${ANALYSIS_REPORT_PATH:-}" ]]; then
    printf "analysis_report:      %s\n" "${ANALYSIS_REPORT_PATH}"
  fi
  printf "progress_json:        %s\n" "${PROGRESS_JSON_PATH}"
}

while true; do
  resolve_pipeline_state
  resolve_progress_json

  if [[ -t 1 ]]; then
    printf '\033c'
  fi

  if [[ -f "${CURRENT_PROGRESS_POINTER}" ]]; then
    # shellcheck disable=SC1090
    source "${CURRENT_PROGRESS_POINTER}" || true
  fi

  if [[ -n "${PIPELINE_STATE_PATH}" && -f "${PIPELINE_STATE_PATH}" ]]; then
    render_pipeline_state
    printf "\n"
  fi

  if [[ -z "${PROGRESS_JSON_PATH}" ]]; then
    if [[ "${WATCH}" == "1" ]]; then
      printf "Waiting for progress JSON path...\n"
      sleep "${WATCH_INTERVAL}"
      continue
    fi
    die "Could not locate a progress JSON file"
  fi

  if [[ ! -f "${PROGRESS_JSON_PATH}" ]]; then
    if [[ "${WATCH}" == "1" ]]; then
      printf "Waiting for progress JSON to appear: %s\n" "${PROGRESS_JSON_PATH}"
      sleep "${WATCH_INTERVAL}"
      continue
    fi
    die "Progress JSON not found: ${PROGRESS_JSON_PATH}"
  fi

  render_progress

  if [[ -n "${PIPELINE_STATE_PATH}" && -f "${PIPELINE_STATE_PATH}" ]]; then
    status="$(read_pipeline_field status)"
  else
    status="$(read_progress_field status)"
    if [[ "${status}" == "completed" ]]; then
      resolve_pipeline_state
      if [[ -n "${PIPELINE_STATE_PATH}" && -f "${PIPELINE_STATE_PATH}" ]]; then
        status="$(read_pipeline_field status)"
      fi
    fi
  fi
  if [[ "${WATCH}" != "1" ]]; then
    break
  fi
  if [[ "${status}" == "completed" || "${status}" == "failed" ]]; then
    break
  fi

  sleep "${WATCH_INTERVAL}"
done
