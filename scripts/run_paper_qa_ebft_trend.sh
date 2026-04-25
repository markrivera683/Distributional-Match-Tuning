#!/usr/bin/env bash
set -euo pipefail

has_model_weights() {
  local model_dir="$1"
  local candidate

  [[ -d "${model_dir}" ]] || return 1

  for candidate in \
    "${model_dir}"/*.safetensors "${model_dir}"/*.bin "${model_dir}"/*.pt \
    "${model_dir}"/*/*.safetensors "${model_dir}"/*/*.bin "${model_dir}"/*/*.pt; do
    if [[ -e "${candidate}" ]]; then
      return 0
    fi
  done

  return 1
}

resolve_dataset_spec() {
  local value="$1"
  local parquet_glob

  if [[ -d "${value}" ]]; then
    parquet_glob="${value}/data/*.parquet"
    if compgen -G "${parquet_glob}" >/dev/null; then
      echo "${parquet_glob}"
      return
    fi
  fi

  echo "${value}"
}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
CRITIC_GPUS="${CRITIC_GPUS:-4}"
REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Match-Tuning}"
MODEL_PATH="${MODEL_PATH:-/root/model}"
TRAIN_DATA="${TRAIN_DATA:-sjelassi/opencode-instruct_100k_200tok}"
PROMPT_SPLIT="${PROMPT_SPLIT:-train}"
INPUT_KEY="${INPUT_KEY:-question}"
OUTPUT_KEY="${OUTPUT_KEY:-answer}"
LABEL_KEY="${LABEL_KEY:-answer}"

GLOBAL_SEED="${GLOBAL_SEED:-43}"
RUN_TAG="${RUN_TAG:-paperqa_ebft_trend_seed${GLOBAL_SEED}}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}"
SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/final_model}"
CKPT_PATH="${CKPT_PATH:-${RUN_ROOT}/checkpoints}"
TB_ROOT="${TB_ROOT:-${RUN_ROOT}/tensorboard}"
BENCH_ROOT="${BENCH_ROOT:-${RUN_ROOT}/offline_benchmarks}"

TRAIN_DATA="$(resolve_dataset_spec "${TRAIN_DATA}")"

FEATURE_MAP_TYPE="${FEATURE_MAP_TYPE:-identity}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
SAVE_EPOCH_FRACTIONS="${SAVE_EPOCH_FRACTIONS:-0.02,0.05,0.1,0.2,0.5}"
STOP_AFTER_EPOCH_FRACTION="${STOP_AFTER_EPOCH_FRACTION:-0.5}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-8}"
MICRO_REWARD_BATCH_SIZE="${MICRO_REWARD_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"

PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-1024}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-8}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-8}"
STRIDE="${STRIDE:-8}"

TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-1e-6}"
INIT_KL_COEF="${INIT_KL_COEF:-0.0}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.03}"
EMA_BETA="${EMA_BETA:-0.9}"
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-0.5}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"

CODE_BENCHMARK_SCRIPT="${CODE_BENCHMARK_SCRIPT:-${REPO_ROOT}/scripts/benchmarks/run_code_generation_benchmarks.py}"
CODE_BENCHMARK_BACKEND="${CODE_BENCHMARK_BACKEND:-auto}"
CODE_BENCHMARK_PROMPT_MAX_LEN="${CODE_BENCHMARK_PROMPT_MAX_LEN:-1024}"
CODE_BENCHMARK_MAX_NEW_TOKENS="${CODE_BENCHMARK_MAX_NEW_TOKENS:-512}"
CODE_BENCHMARK_GREEDY_BATCH_SIZE="${CODE_BENCHMARK_GREEDY_BATCH_SIZE:-16}"
CODE_BENCHMARK_MAX_NUM_SEQS="${CODE_BENCHMARK_MAX_NUM_SEQS:-128}"
CODE_BENCHMARK_TP_SIZE="${CODE_BENCHMARK_TP_SIZE:-1}"
CODE_BENCHMARK_TIMEOUT_SECONDS="${CODE_BENCHMARK_TIMEOUT_SECONDS:-10}"
CODE_BENCHMARK_MAX_SAMPLES_PER_BENCHMARK="${CODE_BENCHMARK_MAX_SAMPLES_PER_BENCHMARK:-0}"
CODE_BENCHMARK_ENABLE_PREFIX_CACHING="${CODE_BENCHMARK_ENABLE_PREFIX_CACHING:-false}"

DOWNSTREAM_HUMANEVAL_DATASET="${DOWNSTREAM_HUMANEVAL_DATASET:-openai/openai_humaneval}"
DOWNSTREAM_HUMANEVAL_SPLIT="${DOWNSTREAM_HUMANEVAL_SPLIT:-test}"
DOWNSTREAM_MBPP_DATASET="${DOWNSTREAM_MBPP_DATASET:-google-research-datasets/mbpp}"
DOWNSTREAM_MBPP_CONFIG="${DOWNSTREAM_MBPP_CONFIG:-sanitized}"
DOWNSTREAM_MBPP_SPLIT="${DOWNSTREAM_MBPP_SPLIT:-test}"

STUDENT_VENV="${STUDENT_VENV:-${REPO_ROOT}/.venv}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"

mkdir -p "${RUN_ROOT}" "${SAVE_PATH}" "${CKPT_PATH}" "${TB_ROOT}" "${BENCH_ROOT}"
exec > >(tee -a "${RUN_ROOT}/run.log") 2>&1

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] Missing MODEL_PATH: ${MODEL_PATH}"
  exit 2
fi

if ! has_model_weights "${MODEL_PATH}"; then
  echo "[ERROR] MODEL_PATH exists but no model weights were found: ${MODEL_PATH}"
  exit 3
fi

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting paper QA EBFT trend run"
echo "REPO_ROOT=${REPO_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_DATA=${TRAIN_DATA}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SAVE_EPOCH_FRACTIONS=${SAVE_EPOCH_FRACTIONS}"
echo "STOP_AFTER_EPOCH_FRACTION=${STOP_AFTER_EPOCH_FRACTION}"
echo "FEATURE_MAP_TYPE=${FEATURE_MAP_TYPE}"
echo "Paper-shaped training values:"
echo "  distribution_reward_type=pointwise"
echo "  cf_target_mode=single"
echo "  n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}"
echo "  rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
echo "  train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  prompt_max_len=${PROMPT_MAX_LEN}"
echo "  context_max_len=${CONTEXT_MAX_LEN}"
echo "  generate_max_len=${GENERATE_MAX_LEN}"
echo "  stride=${STRIDE}"
echo "  temperature=${TEMPERATURE}"
echo "  actor_learning_rate=${ACTOR_LEARNING_RATE}"
echo "  advantage_estimator=rloo"
echo "  init_kl_coef=${INIT_KL_COEF}"
echo "  use_whitening=true"
echo "  hidden_state_method=concat"
echo "  embed_method=last_token"
echo "  critic_sequence_level=last_token"
echo "  critic_learning_rate=0"
echo "  critic_lr_head=0"
echo "  critic_classifier_loss_coef=0.0"
echo "  lr_warmup_ratio=${LR_WARMUP_RATIO}"
echo "  adam_betas=(0.9, 0.95)"
echo "  max_epochs=${MAX_EPOCHS}"
echo "Offline benchmark values:"
echo "  benchmarks=HumanEval,MBPP"
echo "  greedy_temperature=0"
echo "  num_generations_per_prompt=1"
echo "  pass@k=disabled"

ray stop --force 2>/dev/null || true
sleep 2

cd "${REPO_ROOT}"

train_cmd=(
  "${STUDENT_PYTHON_BIN}" -m openrlhf.cli.train_ebft_ray
  --bf16
  --flash_attn
  --gradient_checkpointing
  --pretrain_mode
  --no_chat_template
  --disable_ds_ckpt
  --colocate_actor_ref
  --colocate_critic_reward
  --use_kl_loss
  --use_whitening
  --enable_ema
  --distribution_reward_type pointwise
  --cf_target_mode single
  --feature_map_type "${FEATURE_MAP_TYPE}"
  --pretrain "${MODEL_PATH}"
  --critic_pretrain "${MODEL_PATH}"
  --prompt_data "${TRAIN_DATA}"
  --input_key "${INPUT_KEY}"
  --label_key "${LABEL_KEY}"
  --output_key "${OUTPUT_KEY}"
  --prompt_split "${PROMPT_SPLIT}"
  --prompt_max_len "${PROMPT_MAX_LEN}"
  --context_max_len "${CONTEXT_MAX_LEN}"
  --generate_max_len "${GENERATE_MAX_LEN}"
  --stride "${STRIDE}"
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}"
  --micro_rollout_batch_size "${MICRO_ROLLOUT_BATCH_SIZE}"
  --micro_reward_batch_size "${MICRO_REWARD_BATCH_SIZE}"
  --max_samples "${MAX_SAMPLES}"
  --num_episodes "${NUM_EPISODES}"
  --max_epochs "${MAX_EPOCHS}"
  --actor_num_nodes 1
  --actor_num_gpus_per_node "${ACTOR_GPUS}"
  --critic_num_nodes 1
  --critic_num_gpus_per_node "${CRITIC_GPUS}"
  --ref_num_nodes 1
  --ref_num_gpus_per_node "${REF_GPUS}"
  --reward_num_nodes 1
  --reward_num_gpus_per_node "${REWARD_GPUS}"
  --advantage_estimator rloo
  --init_kl_coef "${INIT_KL_COEF}"
  --kl_estimator k2
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --actor_learning_rate "${ACTOR_LEARNING_RATE}"
  --critic_learning_rate 0.0
  --critic_lr_head 0.0
  --critic_classifier_loss_coef 0.0
  --lr_warmup_ratio "${LR_WARMUP_RATIO}"
  --lr_scheduler cosine_with_min_lr
  --adam_betas 0.9 0.95
  --zero_stage 2
  --seed "${GLOBAL_SEED}"
  --ema_beta "${EMA_BETA}"
  --hidden_state_method concat
  --embed_method last_token
  --critic_sequence_level last_token
  --classifier_sequence_selection closest
  --ce_loss_coef "${CE_LOSS_COEF}"
  --rl_loss_coef 1.0
  --diversity_rew_coef "${DIVERSITY_REW_COEF}"
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}"
  --eval_steps -1
  --eval_down_steps -1
  --logging_steps 1
  --save_steps -1
  --save_hf_ckpt
  --save_epoch_fractions "${SAVE_EPOCH_FRACTIONS}"
  --stop_after_epoch_fraction "${STOP_AFTER_EPOCH_FRACTION}"
  --save_path "${SAVE_PATH}"
  --ckpt_path "${CKPT_PATH}"
  --use_tensorboard "${TB_ROOT}"
  --wandb_run_name "${RUN_TAG}_ebft_trend"
)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${train_cmd[@]}" 2>&1 | tee "${RUN_ROOT}/train.log"

ray stop --force 2>/dev/null || true
sleep 2

shopt -s nullglob
checkpoint_dirs=( "${CKPT_PATH}"/global_step*_hf )
if [[ ${#checkpoint_dirs[@]} -eq 0 ]]; then
  echo "[ERROR] No HF checkpoints were saved under ${CKPT_PATH}"
  exit 4
fi

IFS=$'\n' read -r -d '' -a sorted_checkpoint_dirs < <(printf '%s\n' "${checkpoint_dirs[@]}" | sort -V && printf '\0')

for checkpoint_dir in "${sorted_checkpoint_dirs[@]}"; do
  checkpoint_name="$(basename "${checkpoint_dir}")"
  benchmark_dir="${BENCH_ROOT}/${checkpoint_name}"
  mkdir -p "${benchmark_dir}"

  echo ""
  echo "===== Offline greedy benchmarks: ${checkpoint_name} ====="

  benchmark_cmd=(
    "${STUDENT_PYTHON_BIN}"
    "${CODE_BENCHMARK_SCRIPT}"
    --model_path "${checkpoint_dir}"
    --output_dir "${benchmark_dir}"
    --backend "${CODE_BENCHMARK_BACKEND}"
    --benchmarks "humaneval,mbpp"
    --prompt_max_len "${CODE_BENCHMARK_PROMPT_MAX_LEN}"
    --max_new_tokens "${CODE_BENCHMARK_MAX_NEW_TOKENS}"
    --top_p 1.0
    --greedy_temperature 0.0
    --greedy_only
    --n_samples 1
    --seed "${GLOBAL_SEED}"
    --greedy_batch_size "${CODE_BENCHMARK_GREEDY_BATCH_SIZE}"
    --sample_batch_size 1
    --max_num_seqs "${CODE_BENCHMARK_MAX_NUM_SEQS}"
    --tp_size "${CODE_BENCHMARK_TP_SIZE}"
    --timeout_seconds "${CODE_BENCHMARK_TIMEOUT_SECONDS}"
    --max_samples_per_benchmark "${CODE_BENCHMARK_MAX_SAMPLES_PER_BENCHMARK}"
    --skip_missing_toolchains
    --humaneval_dataset "${DOWNSTREAM_HUMANEVAL_DATASET}"
    --humaneval_split "${DOWNSTREAM_HUMANEVAL_SPLIT}"
    --mbpp_dataset "${DOWNSTREAM_MBPP_DATASET}"
    --mbpp_config "${DOWNSTREAM_MBPP_CONFIG}"
    --mbpp_split "${DOWNSTREAM_MBPP_SPLIT}"
  )

  if [[ "${CODE_BENCHMARK_ENABLE_PREFIX_CACHING}" == "true" ]]; then
    benchmark_cmd+=(--enable_prefix_caching)
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${benchmark_cmd[@]}" 2>&1 | tee "${benchmark_dir}/benchmark.log"
done

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
PY

echo ""
echo "===== Finished ====="
echo "Run root:        ${RUN_ROOT}"
echo "Final model:     ${SAVE_PATH}"
echo "Checkpoint root: ${CKPT_PATH}"
echo "Benchmark root:  ${BENCH_ROOT}"
echo "Trend summary:   ${BENCH_ROOT}/trend_summary.json"
