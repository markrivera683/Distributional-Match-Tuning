#!/usr/bin/env bash
# Batch code eval for baseline/G1/G2/G3 checkpoints.
# For each model and each benchmark (HumanEval, MBPP), run once:
#   sample pass@166 with temperature=0.6

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "${csv}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${csv}"
}

resolve_model_path() {
  local raw_path="$1"
  if [[ -d "${raw_path}/model" ]]; then
    printf '%s\n' "${raw_path}/model"
  elif [[ -f "${raw_path}/config.json" && -f "${raw_path}/tokenizer_config.json" ]]; then
    printf '%s\n' "${raw_path}"
  else
    printf '%s\n' "${raw_path}"
  fi
}

run_one_eval() {
  local model_label="$1"
  local model_path="$2"
  local benchmark="$3"
  local repeat_id="$4"
  local task_name="$5"
  local output_dir="$6"
  local log_path="$7"

  local benchmark_args=()
  local task_args=()

  case "${benchmark}" in
    humaneval)
      benchmark_args=(
        --benchmarks humaneval
        --humaneval_dataset "${HUMANEVAL_EVAL_DATA}"
        --humaneval_split "${HUMANEVAL_EVAL_SPLIT}"
      )
      ;;
    mbpp)
      benchmark_args=(
        --benchmarks mbpp
        --mbpp_dataset "${MBPP_EVAL_DATA}"
        --mbpp_config "${MBPP_EVAL_CONFIG}"
        --mbpp_split "${MBPP_EVAL_SPLIT}"
      )
      ;;
    *)
      echo "[ERROR] unknown benchmark: ${benchmark}"
      exit 1
      ;;
  esac

  case "${task_name}" in
    greedy)
      task_args=(
        --greedy_only
        --greedy_temperature "${GREEDY_TEMPERATURE}"
        --sample_temperature "${PASS16_TEMPERATURE}"
        --n_samples 1
        --passk_list 1
      )
      ;;
    pass16_temp06)
      task_args=(
        --greedy_temperature 0.0
        --sample_temperature "${PASS16_TEMPERATURE}"
        --n_samples 16
        --passk_list 16
      )
      ;;
    *)
      echo "[ERROR] unknown task: ${task_name}"
      exit 1
      ;;
  esac

  mkdir -p "${output_dir}" "$(dirname "${log_path}")"

  echo ""
  echo "================================================================"
  echo "[eval] model=${model_label}"
  echo "[eval] model_path=${model_path}"
  echo "[eval] benchmark=${benchmark}"
  echo "[eval] repeat=${repeat_id}"
  echo "[eval] task=${task_name}"
  echo "[eval] output_dir=${output_dir}"
  echo "================================================================"

  CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
  "${CODE_BENCHMARK_PYTHON_BIN}" "${CODE_BENCHMARK_SCRIPT}" \
    --model_path "${model_path}" \
    --output_dir "${output_dir}" \
    --backend "${CODE_BENCHMARK_BACKEND}" \
    "${benchmark_args[@]}" \
    --prompt_max_len "${PROMPT_MAX_LEN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --top_p "${TOP_P}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    "${task_args[@]}" \
    --tp_size "${VLLM_TP_SIZE}" \
    --max_num_seqs "${VLLM_MAX_NUM_SEQS}" \
    --seed "$((BASE_SEED + repeat_id))" \
    --greedy_batch_size "${GREEDY_BATCH_SIZE}" \
    --sample_batch_size "${SAMPLE_BATCH_SIZE}" \
    --max_samples_per_benchmark "${MAX_SAMPLES_PER_BENCHMARK}" \
    --timeout_seconds "${TIMEOUT_SECONDS}" \
    --detail_preview_chars "${DETAIL_PREVIEW_CHARS}" \
    2>&1 | tee "${log_path}"
}

# ---------------------------------------------------------------------------
# Explicit runtime paths
# ---------------------------------------------------------------------------
LOCAL_ROOT="${LOCAL_ROOT:-/mnt/workspace}"
TEACHER_VENV="${TEACHER_VENV:-${LOCAL_ROOT}/venvs/.teacherVenv}"
CODE_BENCHMARK_PYTHON_BIN="${CODE_BENCHMARK_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
CODE_BENCHMARK_SCRIPT="${CODE_BENCHMARK_SCRIPT:-${REPO_ROOT}/scripts/benchmarks/run_code_generation_benchmarks.py}"

PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-/mnt/data/ebft-distribution-new/outputs/diff_dataset_prepared}"
HUMANEVAL_EVAL_DATA="${HUMANEVAL_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa.jsonl}"
MBPP_EVAL_DATA="${MBPP_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"
HUMANEVAL_EVAL_SPLIT="${HUMANEVAL_EVAL_SPLIT:-test}"
MBPP_EVAL_CONFIG="${MBPP_EVAL_CONFIG:-}"
MBPP_EVAL_SPLIT="${MBPP_EVAL_SPLIT:-test}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs/diff_dataset/code_eval_repeats}"
RUN_NAME="${RUN_NAME:-baseline_g1_g2_g3_code_eval_pass16_once_$(date +%m%d_%H%M)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
SUMMARY_MANIFEST="${SUMMARY_MANIFEST:-${RUN_DIR}/manifest.tsv}"

# ---------------------------------------------------------------------------
# Explicit generation/eval hyperparameters
# ---------------------------------------------------------------------------
CODE_BENCHMARK_BACKEND="${CODE_BENCHMARK_BACKEND:-vllm}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
BASE_SEED="${BASE_SEED:-1234}"
NUM_REPEATS="${NUM_REPEATS:-1}"
SMOKE_TEST="${SMOKE_TEST:-false}"
SKIP_COMPLETED="${SKIP_COMPLETED:-false}"

PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
GREEDY_TEMPERATURE="${GREEDY_TEMPERATURE:-0.0}"
PASS16_TEMPERATURE="${PASS16_TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
GREEDY_BATCH_SIZE="${GREEDY_BATCH_SIZE:-16}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
MAX_SAMPLES_PER_BENCHMARK="${MAX_SAMPLES_PER_BENCHMARK:-0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-10}"
DETAIL_PREVIEW_CHARS="${DETAIL_PREVIEW_CHARS:-4096}"

export HF_HOME="${HF_HOME:-/mnt/data/ebft-distribution-new/caches/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
[[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

# ---------------------------------------------------------------------------
# Explicit model list
# Format: label|path. If path/model exists, it is used as the checkpoint path.
# ---------------------------------------------------------------------------
MODEL_SPECS=(
  "baseline_qwen35_4b|/mnt/data/models/Qwen3.5-4B"
  "g1_0507_0206_500step|/mnt/data/ebft-distribution-new/outputs/diff_dataset/diff_g1_qwen35_4b_0507_0206"
  "g1_0508_2209_500step|/mnt/data/ebft-distribution-new/outputs/diff_dataset/diff_g1_qwen35_4b_0508_2209"
  "g1_0511_0039_500step|/mnt/data/ebft-distribution-new/outputs/diff_dataset/diff_g1_qwen35_4b_0511_0039"
  "g2_no_teacher_vicinal|/mnt/data/ebft-distribution-new/outputs/diff_dataset/g2_diff_dataset_no_teacher_vicinal/diff_g2_no_teacher_vicinal_dlc1nta2oa0ac22v"
  "g2_normal|/mnt/data/ebft-distribution-new/outputs/diff_dataset/g2_dlc1ihgqh6motwz1"
  "g2_no_teacher_distribution_single|/mnt/data/ebft-distribution-new/outputs/diff_dataset/g2_outputs_diff_dataset_no_teacher_distribution/diff_g2_no_teacher_distribution_qwen35_4b_1node_0511_1240"
  "g3_dlc8pd4wa5cqtgur|/mnt/data/ebft-distribution-new/outputs/diff_dataset/g3_dlc8pd4wa5cqtgur"
)

BENCHMARKS=(humaneval)
TASKS=(pass16_temp06)

if [[ "${SMOKE_TEST}" == "true" ]]; then
  MODEL_SPECS=("baseline_qwen35_4b|/mnt/data/models/Qwen3.5-4B")
  BENCHMARKS=(humaneval)
  TASKS=(pass16_temp06)
  NUM_REPEATS=1
  MAX_SAMPLES_PER_BENCHMARK=1
  MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-64}"
  VLLM_MAX_NUM_SEQS="${SMOKE_VLLM_MAX_NUM_SEQS:-8}"
  echo "[smoke] enabled: baseline + HumanEval + pass@16 + 1 benchmark prompt"
fi

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

[[ -x "${CODE_BENCHMARK_PYTHON_BIN}" ]] || { echo "[ERROR] python not executable: ${CODE_BENCHMARK_PYTHON_BIN}"; exit 1; }
[[ -f "${CODE_BENCHMARK_SCRIPT}" ]] || { echo "[ERROR] benchmark script not found: ${CODE_BENCHMARK_SCRIPT}"; exit 1; }
[[ -e "${HUMANEVAL_EVAL_DATA}" ]] || { echo "[ERROR] HumanEval data not found: ${HUMANEVAL_EVAL_DATA}"; exit 1; }
[[ -e "${MBPP_EVAL_DATA}" ]] || { echo "[ERROR] MBPP data not found: ${MBPP_EVAL_DATA}"; exit 1; }

if ! "${CODE_BENCHMARK_PYTHON_BIN}" -c "import datasets, torch, vllm" >/dev/null 2>&1; then
  echo "[ERROR] benchmark python lacks datasets/torch/vllm: ${CODE_BENCHMARK_PYTHON_BIN}"
  echo "        Try: ${CODE_BENCHMARK_PYTHON_BIN} -m pip install datasets==4.8.4"
  exit 1
fi

if [[ "${SKIP_COMPLETED}" == "true" && -s "${SUMMARY_MANIFEST}" ]]; then
  echo "[resume] keeping existing manifest: ${SUMMARY_MANIFEST}"
else
  printf 'model_label\tmodel_path\tbenchmark\trepeat\ttask\toutput_dir\tlog_path\n' > "${SUMMARY_MANIFEST}"
fi

echo "========== Code Eval Repeats =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "CODE_BENCHMARK_PYTHON_BIN:  ${CODE_BENCHMARK_PYTHON_BIN}"
echo "CODE_BENCHMARK_SCRIPT:      ${CODE_BENCHMARK_SCRIPT}"
echo "MODEL_CUDA_VISIBLE_DEVICES: ${MODEL_CUDA_VISIBLE_DEVICES}"
echo "VLLM_TP_SIZE:               ${VLLM_TP_SIZE}"
echo "NUM_REPEATS:                ${NUM_REPEATS}"
echo "BENCHMARKS:                 ${BENCHMARKS[*]}"
echo "TASKS:                      ${TASKS[*]}"
echo "greedy temp:                ${GREEDY_TEMPERATURE}"
echo "sample pass@16 temp:         ${PASS16_TEMPERATURE}"
echo "======================================="

for spec in "${MODEL_SPECS[@]}"; do
  model_label="${spec%%|*}"
  raw_model_path="${spec#*|}"
  model_path="$(resolve_model_path "${raw_model_path}")"

  if [[ ! -e "${model_path}" ]]; then
    echo "[ERROR] model path not found for ${model_label}: ${model_path}"
    exit 1
  fi
  if [[ ! -f "${model_path}/config.json" || ! -f "${model_path}/tokenizer_config.json" ]]; then
    echo "[ERROR] resolved path is not a HF model directory for ${model_label}: ${model_path}"
    echo "        Expected config.json and tokenizer_config.json."
    exit 1
  fi

  for benchmark in "${BENCHMARKS[@]}"; do
    for repeat_id in $(seq 1 "${NUM_REPEATS}"); do
      for task_name in "${TASKS[@]}"; do
        output_dir="${RUN_DIR}/${model_label}/${benchmark}/repeat_${repeat_id}/${task_name}"
        log_path="${LOG_DIR}/${model_label}_${benchmark}_repeat${repeat_id}_${task_name}.log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${model_label}" "${model_path}" "${benchmark}" "${repeat_id}" "${task_name}" "${output_dir}" "${log_path}" >> "${SUMMARY_MANIFEST}"
        if [[ "${SKIP_COMPLETED}" == "true" && -s "${output_dir}/benchmark_summary.json" ]]; then
          echo "[skip] existing benchmark_summary.json: ${output_dir}"
          continue
        fi
        run_one_eval "${model_label}" "${model_path}" "${benchmark}" "${repeat_id}" "${task_name}" "${output_dir}" "${log_path}"
      done
    done
  done
done

echo "[done] manifest: ${SUMMARY_MANIFEST}"
