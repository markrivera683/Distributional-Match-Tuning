#!/usr/bin/env bash
# Common helpers for 8k benchmark post-eval scripts.

benchmark_dataset_path() {
  case "$1" in
    aime24) echo "${REPO_ROOT}/benchmarks/aime24" ;;
    aime25) echo "${REPO_ROOT}/benchmarks/aime25" ;;
    amc23) echo "${REPO_ROOT}/benchmarks/amc23" ;;
    math500) echo "${REPO_ROOT}/benchmarks/math500" ;;
    minervamath) echo "${REPO_ROOT}/benchmarks/minervamath" ;;
    olympiadbench) echo "${REPO_ROOT}/benchmarks/olympiadbench" ;;
    *)
      echo "[ERROR] Unknown benchmark: $1" >&2
      return 1
      ;;
  esac
}

parse_selected_benchmarks() {
  local csv="${BENCHMARKS:-aime24,aime25,amc23,math500,minervamath,olympiadbench}"
  IFS=',' read -r -a SELECTED_BENCHMARKS <<< "${csv}"
  if (( ${#SELECTED_BENCHMARKS[@]} == 0 )); then
    echo "[ERROR] BENCHMARKS is empty" >&2
    return 1
  fi

  local cleaned=()
  local bench=""
  for bench in "${SELECTED_BENCHMARKS[@]}"; do
    bench="${bench// /}"
    [[ -z "${bench}" ]] && continue
    benchmark_dataset_path "${bench}" >/dev/null
    cleaned+=("${bench}")
  done
  SELECTED_BENCHMARKS=("${cleaned[@]}")
}

init_benchmark_suite_env() {
  : "${REPO_ROOT:?REPO_ROOT must be set}"
  : "${MODEL_LABEL:?MODEL_LABEL must be set}"
  : "${LOAD_MODEL_PATH:?LOAD_MODEL_PATH must be set}"
  : "${RUN_DIR:?RUN_DIR must be set}"

  STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"
  STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
  TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
  TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
  ANALYSIS_VENV="${ANALYSIS_VENV:-${STUDENT_VENV}}"
  ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"
  VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-${TEACHER_PYTHON_BIN}}"
  PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"
  SUITE_TS="${SUITE_TS:-$(date +%m%d_%H%M)}"

  EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
  POST_EVAL_MAX_NEW_TOKENS="${POST_EVAL_MAX_NEW_TOKENS:-8192}"
  POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
  POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
  POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
  POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
  POST_EVAL_MASTER_PORT_BASE="${POST_EVAL_MASTER_PORT_BASE:-29530}"
  POST_EVAL_MAX_SAMPLES_OVERRIDE="${POST_EVAL_MAX_SAMPLES:-}"
  POST_EVAL_NNODES="${POST_EVAL_NNODES:-1}"

  IFS=',' read -r -a _VISIBLE_GPUS <<< "${EVAL_CUDA_VISIBLE_DEVICES}"
  VISIBLE_GPU_COUNT="${#_VISIBLE_GPUS[@]}"
  VLLM_TP_SIZE="${VLLM_TP_SIZE:-${VISIBLE_GPU_COUNT}}"
  VLLM_DP_SIZE="${VLLM_DP_SIZE:-1}"
  VLLM_PP_SIZE="${VLLM_PP_SIZE:-1}"
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
  VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-16}"
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
  VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
  VLLM_SEED="${VLLM_SEED:-1234}"

  BENCHMARK_LOG_ROOT="${BENCHMARK_LOG_ROOT:-${RUN_DIR}/benchmark_logs}"
  SUITE_LOG_PATH="${SUITE_LOG_PATH:-${BENCHMARK_LOG_ROOT}/${MODEL_LABEL}_benchmarks_${SUITE_TS}.log}"

  export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

  if [[ ! -x "${ANALYSIS_PYTHON_BIN}" ]]; then
    echo "[ERROR] ANALYSIS_PYTHON_BIN not executable: ${ANALYSIS_PYTHON_BIN}" >&2
    return 1
  fi
  if [[ ! -x "${VLLM_PYTHON_BIN}" ]]; then
    echo "[ERROR] VLLM_PYTHON_BIN not executable: ${VLLM_PYTHON_BIN}" >&2
    return 1
  fi
  if [[ ! -f "${PROGRESS_HELPER}" ]]; then
    echo "[ERROR] PROGRESS_HELPER not found: ${PROGRESS_HELPER}" >&2
    return 1
  fi
  if [[ ! -e "${LOAD_MODEL_PATH}" ]]; then
    echo "[ERROR] LOAD_MODEL_PATH not found: ${LOAD_MODEL_PATH}" >&2
    return 1
  fi
  if (( VISIBLE_GPU_COUNT < 1 )); then
    echo "[ERROR] EVAL_CUDA_VISIBLE_DEVICES is empty: ${EVAL_CUDA_VISIBLE_DEVICES}" >&2
    return 1
  fi
  if (( VLLM_TP_SIZE < 1 || VLLM_TP_SIZE > VISIBLE_GPU_COUNT )); then
    echo "[ERROR] VLLM_TP_SIZE=${VLLM_TP_SIZE} is invalid for visible GPU count=${VISIBLE_GPU_COUNT}" >&2
    echo "        EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES}" >&2
    return 1
  fi
  if (( POST_EVAL_NNODES != 1 || VLLM_DP_SIZE != 1 || VLLM_PP_SIZE != 1 )); then
    echo "[ERROR] benchmark vLLM suite currently supports single-node TP-only inference." >&2
    echo "        got POST_EVAL_NNODES=${POST_EVAL_NNODES}, VLLM_DP_SIZE=${VLLM_DP_SIZE}, VLLM_PP_SIZE=${VLLM_PP_SIZE}" >&2
    return 1
  fi

  mkdir -p "${RUN_DIR}" "${BENCHMARK_LOG_ROOT}"

  if [[ "${BENCHMARK_SUITE_LOGGING_INITIALIZED:-false}" != "true" ]]; then
    exec > >(tee -a "${SUITE_LOG_PATH}") 2>&1
    BENCHMARK_SUITE_LOGGING_INITIALIZED="true"
  fi

  parse_selected_benchmarks
}

run_vllm_generation() {
  local normalized_eval_path="$1"
  local output_path="$2"
  local log_path="$3"
  local max_samples="$4"
  local progress_json_path="$5"

  local -a vllm_cmd=(
    "${VLLM_PYTHON_BIN}" "${PROGRESS_HELPER}"
    --pretrain "${LOAD_MODEL_PATH}"
    --dataset "${normalized_eval_path}"
    --input_key question
    --output_path "${output_path}"
    --prompt_max_len "${POST_EVAL_PROMPT_MAX_LEN}"
    --max_new_tokens "${POST_EVAL_MAX_NEW_TOKENS}"
    --temperature "${POST_EVAL_TEMPERATURE}"
    --top_p "${POST_EVAL_TOP_P}"
    --repetition_penalty "${POST_EVAL_REPETITION_PENALTY}"
    --max_samples "${max_samples}"
    --best_of_n "${POST_EVAL_BEST_OF_N}"
    --tp_size "${VLLM_TP_SIZE}"
    --dp_size "${VLLM_DP_SIZE}"
    --pp_size "${VLLM_PP_SIZE}"
    --max_num_seqs "${VLLM_MAX_NUM_SEQS}"
    --progress_batch_size "${VLLM_PROGRESS_BATCH_SIZE}"
    --seed "${VLLM_SEED}"
    --progress_json_path "${progress_json_path}"
  )

  if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "true" ]]; then
    vllm_cmd+=(--enable_prefix_caching)
  fi
  if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION}" ]]; then
    vllm_cmd+=(--gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
  fi

  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
  "${vllm_cmd[@]}" 2>&1 | tee "${log_path}"
}

run_one_benchmark_eval() {
  local benchmark="$1"
  local index="$2"
  local dataset_path
  dataset_path="$(benchmark_dataset_path "${benchmark}")"

  local benchmark_dir="${BENCHMARK_LOG_ROOT}/${benchmark}"
  local normalized_eval_path="${benchmark_dir}/normalized_eval_dataset.jsonl"
  local output_path="${benchmark_dir}/eval_results_${MODEL_LABEL}_${SUITE_TS}.jsonl"
  local eval_log_path="${benchmark_dir}/eval_${MODEL_LABEL}_${SUITE_TS}.log"
  local analysis_report_path="${benchmark_dir}/eval_analysis_${MODEL_LABEL}_${SUITE_TS}.json"
  local analysis_log_path="${benchmark_dir}/eval_analysis_${MODEL_LABEL}_${SUITE_TS}.log"
  local progress_json_path="${benchmark_dir}/eval_progress_${MODEL_LABEL}_${SUITE_TS}.json"
  local master_port=$((POST_EVAL_MASTER_PORT_BASE + index))

  mkdir -p "${benchmark_dir}"

  echo ""
  echo "================================================================"
  echo "  Benchmark: ${benchmark}"
  echo "================================================================"
  echo "  dataset_path:          ${dataset_path}"
  echo "  normalized_eval_path:  ${normalized_eval_path}"
  echo "  output_path:           ${output_path}"
  echo "  eval_log_path:         ${eval_log_path}"
  echo "  analysis_report_path:  ${analysis_report_path}"
  echo "  progress_json_path:    ${progress_json_path}"
  echo "  port_slot:             ${master_port}"
  echo ""

  "${ANALYSIS_PYTHON_BIN}" "${REPO_ROOT}/scripts/benchmarks/normalize_benchmark_eval_dataset.py" \
    --dataset_path "${dataset_path}" \
    --benchmark_name "${benchmark}" \
    --output_path "${normalized_eval_path}"

  local dataset_size
  dataset_size="$(wc -l < "${normalized_eval_path}")"
  dataset_size="${dataset_size// /}"

  local benchmark_max_samples="${dataset_size}"
  if [[ -n "${POST_EVAL_MAX_SAMPLES_OVERRIDE}" ]]; then
    benchmark_max_samples="${POST_EVAL_MAX_SAMPLES_OVERRIDE}"
  fi

  echo "  dataset_size:          ${dataset_size}"
  echo "  max_samples_used:      ${benchmark_max_samples}"

  run_vllm_generation \
    "${normalized_eval_path}" \
    "${output_path}" \
    "${eval_log_path}" \
    "${benchmark_max_samples}" \
    "${progress_json_path}"

  "${ANALYSIS_PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_eval_results.py" \
    --eval_results "${output_path}" \
    --eval_dataset "${normalized_eval_path}" \
    --input_key question \
    --label_key answer \
    --report_path "${analysis_report_path}" \
    2>&1 | tee "${analysis_log_path}"

  echo "[benchmark-done] ${benchmark}"
  echo "  results:   ${output_path}"
  echo "  analysis:  ${analysis_report_path}"
  echo "  eval log:  ${eval_log_path}"
}

run_benchmark_eval_suite() {
  init_benchmark_suite_env

  echo "================================================================"
  echo "  8K Benchmark Completion Suite"
  echo "================================================================"
  echo "  model_label:                ${MODEL_LABEL}"
  echo "  load_model_path:            ${LOAD_MODEL_PATH}"
  echo "  run_dir:                    ${RUN_DIR}"
  echo "  benchmark_log_root:         ${BENCHMARK_LOG_ROOT}"
  echo "  suite_log_path:             ${SUITE_LOG_PATH}"
  echo "  benchmarks:                 ${SELECTED_BENCHMARKS[*]}"
  echo "  vllm_python_bin:            ${VLLM_PYTHON_BIN}"
  echo "  analysis_python_bin:        ${ANALYSIS_PYTHON_BIN}"
  echo "  vllm_tp_size:               ${VLLM_TP_SIZE}"
  echo "  vllm_max_num_seqs:          ${VLLM_MAX_NUM_SEQS}"
  echo "  vllm_progress_batch_size:   ${VLLM_PROGRESS_BATCH_SIZE}"
  echo "  vllm_prefix_caching:        ${VLLM_ENABLE_PREFIX_CACHING}"
  echo "  vllm_gpu_mem_util:          ${VLLM_GPU_MEMORY_UTILIZATION:-<default>}"
  echo "  prompt_max_len:             ${POST_EVAL_PROMPT_MAX_LEN}"
  echo "  max_new_tokens:             ${POST_EVAL_MAX_NEW_TOKENS}"
  echo "  eval cuda visible devices:  ${EVAL_CUDA_VISIBLE_DEVICES}"
  echo "================================================================"

  local idx=0
  local benchmark=""
  for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
    run_one_benchmark_eval "${benchmark}" "${idx}"
    idx=$((idx + 1))
  done

  echo ""
  echo "[suite-done] model=${MODEL_LABEL}"
  echo "[suite-done] logs=${BENCHMARK_LOG_ROOT}"
}
