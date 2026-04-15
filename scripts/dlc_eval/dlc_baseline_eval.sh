#!/usr/bin/env bash
# DLC Qwen3.5-0.8B post-eval script.
# Intended usage:
#   1. Run `bash scripts/dlc_eval/migrate_eval_dlc_to_home.sh`
#   2. Run this script inside DLC
#
# Defaults assume the migrated layout:
#   repo    -> ~/data/Distributional-Matching-Tuning
#   outputs -> ~/outputs
#
# Default model auto-detection is intentionally restricted to Qwen3.5-0.8B.
# If you want to evaluate a specific fine-tuned checkpoint, pass MODEL_PATH.
#
# This script targets an 8xA100 DLC setup and runs a two-stage vLLM eval:
# first pass on the full dataset with 16k generation, then a 32k retry pass
# only on prompts that were judged incorrect on the first pass.
# It also writes progress/pipeline pointer files so an external call to
# `scripts/dlc_eval/dlc_baseline_eval_progress.sh` can inspect live status.
set -euo pipefail

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "$csv" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"$csv"
}

pick_latest_model_dir() {
  local pattern="$1"
  local latest=""
  latest="$(ls -td ${pattern} 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest" ]]; then
    printf "%s\n" "$latest"
  fi
}

pick_default_model_path() {
  local candidate=""

  candidate="$(pick_latest_model_dir "${HOME}/outputs/*0.8B*/model")"
  [[ -n "$candidate" ]] && { printf "%s\n" "$candidate"; return; }

  candidate="$(pick_latest_model_dir "${HOME}/outputs/*0.8b*/model")"
  [[ -n "$candidate" ]] && { printf "%s\n" "$candidate"; return; }

  for candidate in \
    "/mnt/data/Qwen3.5-0.8B" \
    "/mnt/data/models/Qwen3.5-0.8B" \
    "/mnt/data/models/qwen3.5-0.8b" \
    "/mnt/data/teacher_model/models/Qwen3.5-0.8B" \
    "/mnt/data/teacher_model/models/qwen3.5-0.8b"
  do
    if [[ -e "$candidate" ]]; then
      printf "%s\n" "$candidate"
      return
    fi
  done
}

write_json_file() {
  local target_path="$1"
  local json_payload="$2"
  mkdir -p "$(dirname "$target_path")"
  python3 - "$target_path" "$json_payload" <<'PY'
import json
import os
import sys
import tempfile

target_path = sys.argv[1]
payload = json.loads(sys.argv[2])
target_dir = os.path.dirname(target_path) or "."
os.makedirs(target_dir, exist_ok=True)
fd, tmp_path = tempfile.mkstemp(prefix=".pipeline-", suffix=".json", dir=target_dir)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, target_path)
finally:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
PY
}

update_pipeline_state() {
  local status="$1"
  local phase="$2"
  local message="${3:-}"
  local extra_json="${4:-{}}"

  python3 - "${PIPELINE_STATE_PATH}" "${status}" "${phase}" "${message}" "${extra_json}" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

path, status, phase, message, extra_json = sys.argv[1:6]

def parse_extra_json(raw):
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        idx = 0
        parsed_parts = []
        while idx < len(raw):
            while idx < len(raw) and raw[idx].isspace():
                idx += 1
            if idx >= len(raw):
                break
            try:
                part, end = decoder.raw_decode(raw, idx)
            except json.JSONDecodeError:
                # Keep pipeline alive even if shell-side payload has trailing junk.
                return {}
            parsed_parts.append(part)
            idx = end

        if not parsed_parts:
            return {}
        if len(parsed_parts) == 1:
            parsed = parsed_parts[0]
        else:
            parsed = {}
            for part in parsed_parts:
                if isinstance(part, dict):
                    parsed.update(part)
    if isinstance(parsed, dict):
        return parsed
    return {}

extra = parse_extra_json(extra_json)

def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = {}

data.update(extra)
data["status"] = status
data["phase"] = phase
data["message"] = message
data["updated_at"] = now_iso()
data.setdefault("started_at", data["updated_at"])
if status in {"completed", "failed"}:
    data["finished_at"] = data["updated_at"]

target_dir = os.path.dirname(path) or "."
os.makedirs(target_dir, exist_ok=True)
fd, tmp_path = tempfile.mkstemp(prefix=".pipeline-state-", suffix=".json", dir=target_dir)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)
finally:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
PY
}

write_current_progress_pointer() {
  local pointer_path="$1"
  mkdir -p "$(dirname "$pointer_path")"
  {
    printf 'PROGRESS_JSON_PATH=%q\n' "${PROGRESS_JSON_PATH}"
    printf 'PIPELINE_STATE_PATH=%q\n' "${PIPELINE_STATE_PATH}"
    printf 'RUN_DIR=%q\n' "${RUN_DIR}"
    printf 'MODEL_PATH=%q\n' "${MODEL_PATH}"
    printf 'REPO_ROOT=%q\n' "${REPO_ROOT}"
    printf 'POST_EVAL_OUTPUT_PATH=%q\n' "${POST_EVAL_OUTPUT_PATH}"
    printf 'POST_EVAL_LOG_PATH=%q\n' "${POST_EVAL_LOG_PATH}"
    printf 'SCRIPT_LOG_PATH=%q\n' "${SCRIPT_LOG_PATH}"
    printf 'ANALYSIS_REPORT_PATH=%q\n' "${ANALYSIS_REPORT_PATH}"
    printf 'ANALYSIS_LOG_PATH=%q\n' "${ANALYSIS_LOG_PATH}"
    printf 'FIRST_PASS_OUTPUT_PATH=%q\n' "${FIRST_PASS_OUTPUT_PATH}"
    printf 'FIRST_PASS_ANALYSIS_REPORT_PATH=%q\n' "${FIRST_PASS_ANALYSIS_REPORT_PATH}"
    printf 'SECOND_PASS_DATASET_PATH=%q\n' "${SECOND_PASS_DATASET_PATH}"
    printf 'SECOND_PASS_OUTPUT_PATH=%q\n' "${SECOND_PASS_OUTPUT_PATH}"
    printf 'SECOND_PASS_ANALYSIS_REPORT_PATH=%q\n' "${SECOND_PASS_ANALYSIS_REPORT_PATH}"
    printf 'FINAL_ANALYSIS_REPORT_PATH=%q\n' "${FINAL_ANALYSIS_REPORT_PATH}"
  } > "${pointer_path}"
}

run_generation_stage() {
  local stage_name="$1"
  local dataset_path="$2"
  local output_path="$3"
  local log_path="$4"
  local progress_json_path="$5"
  local max_new_tokens="$6"
  local extra_json="$7"

  [[ -e "${dataset_path}" ]] || {
    echo "[ERROR] ${stage_name} dataset not found: ${dataset_path}"
    exit 1
  }

  PROGRESS_JSON_PATH="${progress_json_path}"
  POST_EVAL_OUTPUT_PATH="${output_path}"
  POST_EVAL_LOG_PATH="${log_path}"
  write_current_progress_pointer "${CURRENT_PROGRESS_POINTER}"
  update_pipeline_state "running" "${stage_name}" "Running ${stage_name}" "${extra_json}"

  local -a vllm_cmd=(
    "${TEACHER_PYTHON_BIN}" "${PROGRESS_HELPER}"
    --pretrain "${MODEL_PATH}"
    --dataset "${dataset_path}"
    --input_key question
    --output_path "${output_path}"
    --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}"
    --max_new_tokens "${max_new_tokens}"
    --temperature "${POST_EVAL_TEMPERATURE}"
    --top_p "${POST_EVAL_TOP_P}"
    --repetition_penalty "${POST_EVAL_REPETITION_PENALTY}"
    --max_samples "${POST_EVAL_MAX_SAMPLES}"
    --best_of_n "${POST_EVAL_BEST_OF_N}"
    --tp_size "${VLLM_TP_SIZE}"
    --max_num_seqs "${VLLM_MAX_NUM_SEQS}"
    --progress_json_path "${progress_json_path}"
    --seed "${VLLM_SEED}"
  )

  if [[ -n "${INPUT_TEMPLATE}" ]]; then
    vllm_cmd+=(--input_template "${INPUT_TEMPLATE}")
  fi

  if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "true" ]]; then
    vllm_cmd+=(--enable_prefix_caching)
  fi

  CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
  "${vllm_cmd[@]}" \
    2>&1 | tee "${log_path}"
}

run_analysis_stage() {
  local stage_name="$1"
  local eval_results_path="$2"
  local eval_dataset_path="$3"
  local report_path="$4"
  local log_path="$5"
  local extra_json="$6"

  ANALYSIS_REPORT_PATH="${report_path}"
  ANALYSIS_LOG_PATH="${log_path}"
  write_current_progress_pointer "${CURRENT_PROGRESS_POINTER}"
  update_pipeline_state "running" "${stage_name}" "Running ${stage_name}" "${extra_json}"

  "${ANALYSIS_PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_eval_results.py" \
    --eval_results "${eval_results_path}" \
    --eval_dataset "${eval_dataset_path}" \
    --input_key question --label_key answer \
    --report_path "${report_path}" \
    2>&1 | tee "${log_path}"
}

extract_retry_subset() {
  local source_dataset_path="$1"
  local analysis_report_path="$2"
  local subset_dataset_path="$3"
  local subset_metadata_path="$4"

  python3 - "${source_dataset_path}" "${analysis_report_path}" "${subset_dataset_path}" "${subset_metadata_path}" <<'PY'
import json
import os
import sys

source_dataset_path, analysis_report_path, subset_dataset_path, subset_metadata_path = sys.argv[1:5]


def load_rows(path):
    if os.path.isdir(path):
        for name in ("test.jsonl", "test_qa.jsonl", "eval.jsonl"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No supported eval file found under directory: {path}")

    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("test", "data", "rows"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"Unsupported JSON dataset structure in: {path}")

    raise ValueError(f"Unsupported dataset path: {path}")


rows = load_rows(source_dataset_path)
with open(analysis_report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

records = report.get("records", [])
retry_candidates = []
seen = set()
for record in records:
    is_correct = record.get("is_correct")
    if is_correct is True:
        continue
    source_idx = record.get("source_idx")
    if source_idx is None:
        source_idx = record.get("idx")
    if source_idx is None:
        continue
    try:
        source_idx = int(source_idx)
    except Exception:
        continue
    if source_idx < 0 or source_idx >= len(rows):
        continue
    if source_idx in seen:
        continue
    seen.add(source_idx)
    row = dict(rows[source_idx])
    row["source_idx"] = source_idx
    retry_candidates.append(row)

os.makedirs(os.path.dirname(os.path.abspath(subset_dataset_path)), exist_ok=True)
with open(subset_dataset_path, "w", encoding="utf-8") as f:
    for row in retry_candidates:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

metadata = {
    "source_dataset_path": source_dataset_path,
    "analysis_report_path": analysis_report_path,
    "retry_count": len(retry_candidates),
    "source_indices": [row["source_idx"] for row in retry_candidates],
}
with open(subset_metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
}

build_final_report() {
  local first_report_path="$1"
  local second_report_path="$2"
  local retry_metadata_path="$3"
  local final_report_path="$4"

  python3 - "${first_report_path}" "${second_report_path}" "${retry_metadata_path}" "${final_report_path}" <<'PY'
import json
import os
import sys
from collections import Counter

first_report_path, second_report_path, retry_metadata_path, final_report_path = sys.argv[1:5]

with open(first_report_path, "r", encoding="utf-8") as f:
    first_report = json.load(f)

with open(second_report_path, "r", encoding="utf-8") as f:
    second_report = json.load(f)

with open(retry_metadata_path, "r", encoding="utf-8") as f:
    retry_metadata = json.load(f)

first_records = first_report.get("records", [])
second_records = second_report.get("records", [])

final_records = list(first_records)
retry_indices = set()
for record in second_records:
    source_idx = record.get("source_idx")
    if source_idx is None:
        source_idx = record.get("idx")
    if source_idx is None:
        continue
    source_idx = int(source_idx)
    retry_indices.add(source_idx)
    if 0 <= source_idx < len(final_records):
        merged = dict(final_records[source_idx])
        merged["first_pass"] = dict(final_records[source_idx])
        merged["second_pass"] = dict(record)
        merged["prompt"] = record.get("prompt", merged.get("prompt", ""))
        merged["model_output"] = record.get("model_output", merged.get("model_output", ""))
        merged["gold_answer"] = record.get("gold_answer", merged.get("gold_answer"))
        merged["is_correct"] = record.get("is_correct")
        merged["category"] = record.get("category")
        merged["detail"] = record.get("detail")
        merged["retry_applied"] = True
        final_records[source_idx] = merged

for idx, record in enumerate(final_records):
    record.setdefault("retry_applied", idx in retry_indices)

matched = sum(1 for r in final_records if r.get("gold_answer") is not None)
unmatched = sum(1 for r in final_records if r.get("gold_answer") is None)
evaluated = [r for r in final_records if r.get("is_correct") is not None]
correct = sum(1 for r in evaluated if r.get("is_correct"))
accuracy_pct = round((correct / len(evaluated) * 100.0), 2) if evaluated else 0.0
categories = Counter(r.get("category", "unknown") for r in final_records)
avg_output_length_chars = round(
    sum(len(r.get("model_output", "")) for r in final_records) / max(1, len(final_records)),
    1,
)
empty_or_very_short = sum(1 for r in final_records if len(r.get("model_output", "")) < 5)
improved_after_retry = 0
still_incorrect_after_retry = 0
for idx in retry_indices:
    if not (0 <= idx < len(first_records) and 0 <= idx < len(final_records)):
        continue
    first_correct = first_records[idx].get("is_correct")
    final_correct = final_records[idx].get("is_correct")
    if first_correct is not True and final_correct is True:
        improved_after_retry += 1
    elif final_correct is not True:
        still_incorrect_after_retry += 1

summary = {
    "total_predictions": len(final_records),
    "matched": matched,
    "unmatched": unmatched,
    "evaluated": len(evaluated),
    "correct": correct,
    "accuracy_pct": accuracy_pct,
    "avg_output_length_chars": avg_output_length_chars,
    "empty_or_very_short": empty_or_very_short,
    "categories": dict(categories),
    "first_pass_correct": first_report.get("summary", {}).get("correct"),
    "first_pass_accuracy_pct": first_report.get("summary", {}).get("accuracy_pct"),
    "second_pass_retry_count": retry_metadata.get("retry_count", len(retry_indices)),
    "retry_improved_to_correct": improved_after_retry,
    "retry_still_incorrect": still_incorrect_after_retry,
}

final_report = {
    "summary": summary,
    "first_pass_report_path": first_report_path,
    "second_pass_report_path": second_report_path,
    "retry_metadata": retry_metadata,
    "records": final_records,
}

os.makedirs(os.path.dirname(os.path.abspath(final_report_path)), exist_ok=True)
with open(final_report_path, "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
}

get_retry_count() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(int(data.get("retry_count", 0)))
PY
}

handle_failure() {
  local failed_command="${1:-unknown command}"
  local error_json
  trap - ERR
  error_json="$(python3 - "${failed_command}" <<'PY'
import json
import sys

print(json.dumps({"error": sys.argv[1]}))
PY
)"
  update_pipeline_state "failed" "failed" "Two-stage eval pipeline failed" "${error_json}" || true
}

REPO_ROOT="${REPO_ROOT:-${HOME}/data/Distributional-Matching-Tuning}"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"
TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
ANALYSIS_VENV="${ANALYSIS_VENV:-${REPO_ROOT}/.venv}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"
PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"
CURRENT_PROGRESS_POINTER="${CURRENT_PROGRESS_POINTER:-${REPO_ROOT}/.dlc_baseline_eval_current.env}"

SCRIPT_NAME="$(basename "$0" .sh)"
TS="${TS:-$(date +%m%d_%H%M)}"
MODEL_PATH="${MODEL_PATH:-${1:-}}"
RUN_DIR="${RUN_DIR:-${2:-${HOME}/outputs/dlc_baseline_eval_retry16k_to_32k_${TS}}}"
EVAL_TAG="${EVAL_TAG:-baseline_retry16k_to_32k}"

MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPU_COUNT="$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-${GPU_COUNT}}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"

LOG_DIR="${LOG_DIR:-${RUN_DIR}/supplement_logs}"
SCRIPT_LOG_PATH="${SCRIPT_LOG_PATH:-${LOG_DIR}/${SCRIPT_NAME}_${EVAL_TAG}_${TS}.log}"
PIPELINE_STATE_PATH="${PIPELINE_STATE_PATH:-${LOG_DIR}/pipeline_state_${EVAL_TAG}_${TS}.json}"
PROGRESS_JSON_PATH="${PROGRESS_JSON_PATH:-${LOG_DIR}/progress_stage1_${EVAL_TAG}_${TS}.json}"
POST_EVAL_OUTPUT_PATH="${POST_EVAL_OUTPUT_PATH:-${LOG_DIR}/eval_results_stage1_${EVAL_TAG}_${TS}.jsonl}"
POST_EVAL_LOG_PATH="${POST_EVAL_LOG_PATH:-${LOG_DIR}/eval_stage1_${EVAL_TAG}_${TS}.log}"
ANALYSIS_REPORT_PATH="${ANALYSIS_REPORT_PATH:-${LOG_DIR}/eval_analysis_stage1_${EVAL_TAG}_${TS}.json}"
ANALYSIS_LOG_PATH="${ANALYSIS_LOG_PATH:-${LOG_DIR}/eval_analysis_stage1_${EVAL_TAG}_${TS}.log}"
FIRST_PASS_PROGRESS_JSON_PATH="${FIRST_PASS_PROGRESS_JSON_PATH:-${PROGRESS_JSON_PATH}}"
FIRST_PASS_OUTPUT_PATH="${FIRST_PASS_OUTPUT_PATH:-${POST_EVAL_OUTPUT_PATH}}"
FIRST_PASS_LOG_PATH="${FIRST_PASS_LOG_PATH:-${POST_EVAL_LOG_PATH}}"
FIRST_PASS_ANALYSIS_REPORT_PATH="${FIRST_PASS_ANALYSIS_REPORT_PATH:-${ANALYSIS_REPORT_PATH}}"
FIRST_PASS_ANALYSIS_LOG_PATH="${FIRST_PASS_ANALYSIS_LOG_PATH:-${ANALYSIS_LOG_PATH}}"
SECOND_PASS_DATASET_PATH="${SECOND_PASS_DATASET_PATH:-${LOG_DIR}/eval_retry_subset_${EVAL_TAG}_${TS}.jsonl}"
SECOND_PASS_METADATA_PATH="${SECOND_PASS_METADATA_PATH:-${LOG_DIR}/eval_retry_subset_meta_${EVAL_TAG}_${TS}.json}"
SECOND_PASS_PROGRESS_JSON_PATH="${SECOND_PASS_PROGRESS_JSON_PATH:-${LOG_DIR}/progress_stage2_${EVAL_TAG}_${TS}.json}"
SECOND_PASS_OUTPUT_PATH="${SECOND_PASS_OUTPUT_PATH:-${LOG_DIR}/eval_results_stage2_${EVAL_TAG}_${TS}.jsonl}"
SECOND_PASS_LOG_PATH="${SECOND_PASS_LOG_PATH:-${LOG_DIR}/eval_stage2_${EVAL_TAG}_${TS}.log}"
SECOND_PASS_ANALYSIS_REPORT_PATH="${SECOND_PASS_ANALYSIS_REPORT_PATH:-${LOG_DIR}/eval_analysis_stage2_${EVAL_TAG}_${TS}.json}"
SECOND_PASS_ANALYSIS_LOG_PATH="${SECOND_PASS_ANALYSIS_LOG_PATH:-${LOG_DIR}/eval_analysis_stage2_${EVAL_TAG}_${TS}.log}"
FINAL_ANALYSIS_REPORT_PATH="${FINAL_ANALYSIS_REPORT_PATH:-${LOG_DIR}/eval_analysis_final_${EVAL_TAG}_${TS}.json}"
FINAL_ANALYSIS_LOG_PATH="${FINAL_ANALYSIS_LOG_PATH:-${LOG_DIR}/eval_analysis_final_${EVAL_TAG}_${TS}.log}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="$(pick_default_model_path || true)"
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "$(dirname "${FIRST_PASS_OUTPUT_PATH}")" "$(dirname "${FIRST_PASS_ANALYSIS_REPORT_PATH}")"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[ERROR] REPO_ROOT not found: ${REPO_ROOT}"
  echo "        Run scripts/dlc_eval/migrate_eval_dlc_to_home.sh first or override REPO_ROOT."
  exit 1
fi

if [[ ! -x "${TEACHER_PYTHON_BIN}" ]]; then
  echo "[ERROR] TEACHER_PYTHON_BIN not executable: ${TEACHER_PYTHON_BIN}"
  echo "        Expected migrated teacher env under: ${TEACHER_VENV}"
  exit 1
fi

if [[ ! -x "${ANALYSIS_PYTHON_BIN}" ]]; then
  echo "[ERROR] ANALYSIS_PYTHON_BIN not executable: ${ANALYSIS_PYTHON_BIN}"
  echo "        Expected migrated analysis env under: ${ANALYSIS_VENV}"
  exit 1
fi

if [[ ! -f "${PROGRESS_HELPER}" ]]; then
  echo "[ERROR] PROGRESS_HELPER not found: ${PROGRESS_HELPER}"
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[ERROR] Could not auto-detect a Qwen3.5-0.8B model/checkpoint."
  echo "        Auto-detection only searches 0.8B-looking paths."
  echo "        For a specific checkpoint, pass MODEL_PATH=/path/to/checkpoint"
  exit 1
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi

if (( GPU_COUNT != 8 )); then
  echo "[WARN] This script is tuned for 8xA100, but MODEL_CUDA_VISIBLE_DEVICES resolves to ${GPU_COUNT} GPU(s)."
fi

if (( VLLM_TP_SIZE < 1 )); then
  echo "[ERROR] VLLM_TP_SIZE must be >= 1, got: ${VLLM_TP_SIZE}"
  exit 1
fi

if (( VLLM_TP_SIZE > GPU_COUNT )); then
  echo "[ERROR] VLLM_TP_SIZE=${VLLM_TP_SIZE} exceeds visible GPU count=${GPU_COUNT}"
  exit 1
fi

if (( POST_EVAL_BEST_OF_N != 1 )); then
  echo "[ERROR] Two-stage retry eval currently requires POST_EVAL_BEST_OF_N=1, got: ${POST_EVAL_BEST_OF_N}"
  exit 1
fi

cd "${REPO_ROOT}"

trap 'handle_failure "$BASH_COMMAND"' ERR

echo "========== DLC Qwen3.5-0.8B Two-Stage Eval via vLLM (8xA100) =========="
echo "RUN_DIR:                       ${RUN_DIR}"
echo "REPO_ROOT:                     ${REPO_ROOT}"
echo "MODEL_PATH:                    ${MODEL_PATH}"
echo "EVAL_DATA:                     ${EVAL_DATA}"
echo "SCRIPT_LOG_PATH:               ${SCRIPT_LOG_PATH}"
echo "PIPELINE_STATE_PATH:           ${PIPELINE_STATE_PATH}"
echo "FIRST_PASS_LOG_PATH:           ${FIRST_PASS_LOG_PATH}"
echo "FIRST_PASS_ANALYSIS_LOG_PATH:  ${FIRST_PASS_ANALYSIS_LOG_PATH}"
echo "SECOND_PASS_LOG_PATH:          ${SECOND_PASS_LOG_PATH}"
echo "SECOND_PASS_ANALYSIS_LOG_PATH: ${SECOND_PASS_ANALYSIS_LOG_PATH}"
echo "MODEL_CUDA_VISIBLE_DEVICES:    ${MODEL_CUDA_VISIBLE_DEVICES}"
echo "VLLM_TP_SIZE:                  ${VLLM_TP_SIZE}"
echo "VLLM_MAX_NUM_SEQS:             ${VLLM_MAX_NUM_SEQS}"
echo "VLLM_ENABLE_PREFIX_CACHING:    ${VLLM_ENABLE_PREFIX_CACHING}"
echo "POST_EVAL_PROMPT_MAX_LEN:      ${POST_EVAL_PROMPT_MAX_LEN}"
echo "FIRST_PASS_MAX_NEW_TOKENS:     ${FIRST_PASS_MAX_NEW_TOKENS}"
echo "SECOND_PASS_MAX_NEW_TOKENS:    ${SECOND_PASS_MAX_NEW_TOKENS}"
echo "POST_EVAL_BEST_OF_N:           ${POST_EVAL_BEST_OF_N}"
echo "POST_EVAL_MAX_SAMPLES:         ${POST_EVAL_MAX_SAMPLES}"
echo "FIRST_PASS_PROGRESS_JSON_PATH: ${FIRST_PASS_PROGRESS_JSON_PATH}"
echo "SECOND_PASS_PROGRESS_JSON_PATH:${SECOND_PASS_PROGRESS_JSON_PATH}"
echo "CURRENT_PROGRESS_POINTER:      ${CURRENT_PROGRESS_POINTER}"
echo "FIRST_PASS_OUTPUT_PATH:        ${FIRST_PASS_OUTPUT_PATH}"
echo "SECOND_PASS_OUTPUT_PATH:       ${SECOND_PASS_OUTPUT_PATH}"
echo "FINAL_ANALYSIS_REPORT_PATH:    ${FINAL_ANALYSIS_REPORT_PATH}"
echo "========================================================"

update_pipeline_state "initializing" "setup" "Initializing two-stage eval pipeline" "$(cat <<EOF
{"run_dir":"${RUN_DIR}","repo_root":"${REPO_ROOT}","model_path":"${MODEL_PATH}","eval_data":"${EVAL_DATA}","first_pass_progress_json_path":"${FIRST_PASS_PROGRESS_JSON_PATH}","second_pass_progress_json_path":"${SECOND_PASS_PROGRESS_JSON_PATH}","pipeline_state_path":"${PIPELINE_STATE_PATH}"}
EOF
)"
write_current_progress_pointer "${CURRENT_PROGRESS_POINTER}"

echo "[stage1] Running first-pass full eval at 16k generation length"
run_generation_stage \
  "stage1_generate_16k" \
  "${EVAL_DATA}" \
  "${FIRST_PASS_OUTPUT_PATH}" \
  "${FIRST_PASS_LOG_PATH}" \
  "${FIRST_PASS_PROGRESS_JSON_PATH}" \
  "${FIRST_PASS_MAX_NEW_TOKENS}" \
  "$(cat <<EOF
{"active_progress_json_path":"${FIRST_PASS_PROGRESS_JSON_PATH}","active_output_path":"${FIRST_PASS_OUTPUT_PATH}","active_log_path":"${FIRST_PASS_LOG_PATH}","active_max_new_tokens":${FIRST_PASS_MAX_NEW_TOKENS}}
EOF
)"

echo "[stage1] Saved: ${FIRST_PASS_OUTPUT_PATH}"
echo "[stage1] Log:   ${FIRST_PASS_LOG_PATH}"

echo "[stage1-analysis] Analyzing first-pass results"
run_analysis_stage \
  "stage1_analysis" \
  "${FIRST_PASS_OUTPUT_PATH}" \
  "${EVAL_DATA}" \
  "${FIRST_PASS_ANALYSIS_REPORT_PATH}" \
  "${FIRST_PASS_ANALYSIS_LOG_PATH}" \
  "$(cat <<EOF
{"active_analysis_report_path":"${FIRST_PASS_ANALYSIS_REPORT_PATH}","active_analysis_log_path":"${FIRST_PASS_ANALYSIS_LOG_PATH}","active_progress_json_path":"${FIRST_PASS_PROGRESS_JSON_PATH}"}
EOF
)"

echo "[stage1-analysis] Report: ${FIRST_PASS_ANALYSIS_REPORT_PATH}"
echo "[stage1-analysis] Log:    ${FIRST_PASS_ANALYSIS_LOG_PATH}"

echo "[stage2-prepare] Building retry subset from first-pass non-correct prompts"
update_pipeline_state "running" "stage2_prepare_retry_subset" "Extracting retry subset from first-pass analysis" "$(cat <<EOF
{"active_analysis_report_path":"${FIRST_PASS_ANALYSIS_REPORT_PATH}","retry_subset_path":"${SECOND_PASS_DATASET_PATH}","retry_metadata_path":"${SECOND_PASS_METADATA_PATH}"}
EOF
)"
extract_retry_subset \
  "${EVAL_DATA}" \
  "${FIRST_PASS_ANALYSIS_REPORT_PATH}" \
  "${SECOND_PASS_DATASET_PATH}" \
  "${SECOND_PASS_METADATA_PATH}"

RETRY_COUNT="$(get_retry_count "${SECOND_PASS_METADATA_PATH}")"
echo "[stage2-prepare] Retry subset size: ${RETRY_COUNT}"

LAST_GENERATION_PROGRESS_JSON_PATH="${FIRST_PASS_PROGRESS_JSON_PATH}"
LAST_GENERATION_OUTPUT_PATH="${FIRST_PASS_OUTPUT_PATH}"
LAST_GENERATION_LOG_PATH="${FIRST_PASS_LOG_PATH}"

if (( RETRY_COUNT > 0 )); then
  echo "[stage2] Re-running ${RETRY_COUNT} non-correct prompts at 32k generation length"
  run_generation_stage \
    "stage2_generate_32k_retry" \
    "${SECOND_PASS_DATASET_PATH}" \
    "${SECOND_PASS_OUTPUT_PATH}" \
    "${SECOND_PASS_LOG_PATH}" \
    "${SECOND_PASS_PROGRESS_JSON_PATH}" \
    "${SECOND_PASS_MAX_NEW_TOKENS}" \
    "$(cat <<EOF
{"active_progress_json_path":"${SECOND_PASS_PROGRESS_JSON_PATH}","active_output_path":"${SECOND_PASS_OUTPUT_PATH}","active_log_path":"${SECOND_PASS_LOG_PATH}","active_max_new_tokens":${SECOND_PASS_MAX_NEW_TOKENS},"retry_count":${RETRY_COUNT}}
EOF
)"

  echo "[stage2] Saved: ${SECOND_PASS_OUTPUT_PATH}"
  echo "[stage2] Log:   ${SECOND_PASS_LOG_PATH}"
  LAST_GENERATION_PROGRESS_JSON_PATH="${SECOND_PASS_PROGRESS_JSON_PATH}"
  LAST_GENERATION_OUTPUT_PATH="${SECOND_PASS_OUTPUT_PATH}"
  LAST_GENERATION_LOG_PATH="${SECOND_PASS_LOG_PATH}"

  echo "[stage2-analysis] Analyzing retry-pass results"
  run_analysis_stage \
    "stage2_analysis" \
    "${SECOND_PASS_OUTPUT_PATH}" \
    "${EVAL_DATA}" \
    "${SECOND_PASS_ANALYSIS_REPORT_PATH}" \
    "${SECOND_PASS_ANALYSIS_LOG_PATH}" \
    "$(cat <<EOF
{"active_analysis_report_path":"${SECOND_PASS_ANALYSIS_REPORT_PATH}","active_analysis_log_path":"${SECOND_PASS_ANALYSIS_LOG_PATH}","active_progress_json_path":"${SECOND_PASS_PROGRESS_JSON_PATH}","retry_count":${RETRY_COUNT}}
EOF
)"
else
  echo "[stage2] No non-correct prompts from first pass. Skipping retry generate."
  write_json_file "${SECOND_PASS_ANALYSIS_REPORT_PATH}" '{"summary":{"total_predictions":0,"matched":0,"unmatched":0,"evaluated":0,"correct":0,"accuracy_pct":0.0,"avg_output_length_chars":0.0,"empty_or_very_short":0,"categories":{},"math_verify_available":true},"records":[]}'
  : > "${SECOND_PASS_OUTPUT_PATH}"
  : > "${SECOND_PASS_LOG_PATH}"
  : > "${SECOND_PASS_ANALYSIS_LOG_PATH}"
fi

echo "[final-analysis] Building merged final report"
update_pipeline_state "running" "final_analysis" "Building merged final report" "$(cat <<EOF
{"active_analysis_report_path":"${FINAL_ANALYSIS_REPORT_PATH}","first_pass_analysis_report_path":"${FIRST_PASS_ANALYSIS_REPORT_PATH}","second_pass_analysis_report_path":"${SECOND_PASS_ANALYSIS_REPORT_PATH}","retry_metadata_path":"${SECOND_PASS_METADATA_PATH}"}
EOF
)"
{
  build_final_report \
    "${FIRST_PASS_ANALYSIS_REPORT_PATH}" \
    "${SECOND_PASS_ANALYSIS_REPORT_PATH}" \
    "${SECOND_PASS_METADATA_PATH}" \
    "${FINAL_ANALYSIS_REPORT_PATH}"
  echo "[final-analysis] Merged report written to ${FINAL_ANALYSIS_REPORT_PATH}"
} 2>&1 | tee "${FINAL_ANALYSIS_LOG_PATH}"

ANALYSIS_REPORT_PATH="${FINAL_ANALYSIS_REPORT_PATH}"
ANALYSIS_LOG_PATH="${FINAL_ANALYSIS_LOG_PATH}"
POST_EVAL_OUTPUT_PATH="${LAST_GENERATION_OUTPUT_PATH}"
POST_EVAL_LOG_PATH="${LAST_GENERATION_LOG_PATH}"
PROGRESS_JSON_PATH="${LAST_GENERATION_PROGRESS_JSON_PATH}"
write_current_progress_pointer "${CURRENT_PROGRESS_POINTER}"
update_pipeline_state "completed" "done" "Two-stage eval pipeline completed" "$(cat <<EOF
{"final_analysis_report_path":"${FINAL_ANALYSIS_REPORT_PATH}","first_pass_output_path":"${FIRST_PASS_OUTPUT_PATH}","second_pass_output_path":"${SECOND_PASS_OUTPUT_PATH}","retry_count":${RETRY_COUNT},"active_progress_json_path":"${LAST_GENERATION_PROGRESS_JSON_PATH}"}
EOF
)"

echo "[final-analysis] Final report: ${FINAL_ANALYSIS_REPORT_PATH}"
echo "[script]   Log:    ${SCRIPT_LOG_PATH}"
