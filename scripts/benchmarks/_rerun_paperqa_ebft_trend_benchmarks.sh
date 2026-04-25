#!/usr/bin/env bash
# One-off helper: re-run offline benchmarks for the paper QA EBFT trend run after
# fixing the right-padding bug in run_code_generation_benchmarks.py.
# Each checkpoint is evaluated in parallel on its own GPU.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/paperqa_ebft_trend_seed43}"
CKPT_PATH="${CKPT_PATH:-${RUN_ROOT}/checkpoints}"
BENCH_ROOT="${BENCH_ROOT:-${RUN_ROOT}/offline_benchmarks}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
CODE_BENCHMARK_SCRIPT="${REPO_ROOT}/scripts/benchmarks/run_code_generation_benchmarks.py"

GREEDY_BATCH_SIZE="${GREEDY_BATCH_SIZE:-16}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-10}"
SEED="${SEED:-43}"

DOWNSTREAM_HUMANEVAL_DATASET="${DOWNSTREAM_HUMANEVAL_DATASET:-openai/openai_humaneval}"
DOWNSTREAM_HUMANEVAL_SPLIT="${DOWNSTREAM_HUMANEVAL_SPLIT:-test}"
DOWNSTREAM_MBPP_DATASET="${DOWNSTREAM_MBPP_DATASET:-google-research-datasets/mbpp}"
DOWNSTREAM_MBPP_CONFIG="${DOWNSTREAM_MBPP_CONFIG:-sanitized}"
DOWNSTREAM_MBPP_SPLIT="${DOWNSTREAM_MBPP_SPLIT:-test}"

mkdir -p "${BENCH_ROOT}"

shopt -s nullglob
checkpoint_dirs=( "${CKPT_PATH}"/global_step*_hf )
if [[ ${#checkpoint_dirs[@]} -eq 0 ]]; then
  echo "[ERROR] No HF checkpoints under ${CKPT_PATH}"
  exit 1
fi

IFS=$'\n' read -r -d '' -a sorted_checkpoint_dirs < <(printf '%s\n' "${checkpoint_dirs[@]}" | sort -V && printf '\0')

run_one() {
  local checkpoint_dir="$1"
  local gpu_id="$2"
  local checkpoint_name
  checkpoint_name="$(basename "${checkpoint_dir}")"
  local benchmark_dir="${BENCH_ROOT}/${checkpoint_name}"
  mkdir -p "${benchmark_dir}"

  echo "[$(date -u '+%H:%M:%S')] [GPU ${gpu_id}] start ${checkpoint_name}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  "${STUDENT_PYTHON_BIN}" \
    "${CODE_BENCHMARK_SCRIPT}" \
    --model_path "${checkpoint_dir}" \
    --output_dir "${benchmark_dir}" \
    --backend hf \
    --benchmarks "humaneval,mbpp" \
    --prompt_max_len "${PROMPT_MAX_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --top_p 1.0 \
    --greedy_temperature 0.0 \
    --greedy_only \
    --n_samples 1 \
    --seed "${SEED}" \
    --greedy_batch_size "${GREEDY_BATCH_SIZE}" \
    --sample_batch_size 1 \
    --max_num_seqs 128 \
    --tp_size 1 \
    --timeout_seconds "${TIMEOUT_SECONDS}" \
    --max_samples_per_benchmark 0 \
    --skip_missing_toolchains \
    --humaneval_dataset "${DOWNSTREAM_HUMANEVAL_DATASET}" \
    --humaneval_split "${DOWNSTREAM_HUMANEVAL_SPLIT}" \
    --mbpp_dataset "${DOWNSTREAM_MBPP_DATASET}" \
    --mbpp_config "${DOWNSTREAM_MBPP_CONFIG}" \
    --mbpp_split "${DOWNSTREAM_MBPP_SPLIT}" \
    > "${benchmark_dir}/benchmark.log" 2>&1
  local rc=$?

  echo "[$(date -u '+%H:%M:%S')] [GPU ${gpu_id}] done ${checkpoint_name} (rc=${rc})"
  return ${rc}
}

pids=()
gpu_id=0
for checkpoint_dir in "${sorted_checkpoint_dirs[@]}"; do
  run_one "${checkpoint_dir}" "${gpu_id}" &
  pids+=("$!")
  gpu_id=$((gpu_id + 1))
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=$((failed + 1))
  fi
done

echo ""
echo "All checkpoints finished, failures=${failed}"

"${STUDENT_PYTHON_BIN}" - "${BENCH_ROOT}" <<'PY'
import json
import re
import sys
from pathlib import Path

bench_root = Path(sys.argv[1])
rows = []
pattern = re.compile(r"global_step(\d+)_hf")

for summary_path in sorted(bench_root.glob("global_step*_hf/benchmark_summary.json")):
    match = pattern.search(summary_path.parent.name)
    if not match:
        continue
    row = {
        "checkpoint": summary_path.parent.name,
        "global_step": int(match.group(1)),
    }
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for summary in payload.get("summaries", []):
        row[summary["benchmark"]] = summary.get("greedy_accuracy")
    rows.append(row)

rows.sort(key=lambda item: item["global_step"])

(bench_root / "trend_summary.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

with (bench_root / "trend_summary.tsv").open("w", encoding="utf-8") as handle:
    handle.write("checkpoint\tglobal_step\tHumanEval\tMBPP\n")
    for row in rows:
        handle.write(
            f"{row['checkpoint']}\t{row['global_step']}\t"
            f"{row.get('HumanEval', '')}\t{row.get('MBPP', '')}\n"
        )

print("trend_summary.json:")
print((bench_root / "trend_summary.json").read_text(encoding="utf-8"))
PY

exit ${failed}
