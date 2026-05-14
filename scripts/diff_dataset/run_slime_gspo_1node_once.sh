#!/usr/bin/env bash
# Standalone slime GRPO/GSPO smoke launcher for the diff-dataset pipeline.
# This script intentionally does not modify or depend on the OpenRLHF G1/G2/G3 launchers.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SLIME_ENV_FILE="${SLIME_ENV_FILE:-/root/slime_runtime/slime_env.sh}"

if [[ -f "${SLIME_ENV_FILE}" && "${SOURCE_SLIME_ENV:-true}" == "true" ]]; then
  # shellcheck disable=SC1090
  source "${SLIME_ENV_FILE}"
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if [[ -z "${SLIME_ROOT:-}" ]]; then
  for candidate in "/mnt/data/distribution-matching-slime/code/slime-0.2.4" "/root/slime_runtime/slime" "/root/slime"; do
    if [[ -f "${candidate}/train.py" ]]; then
      SLIME_ROOT="${candidate}"
      break
    fi
  done
fi
SLIME_ROOT="${SLIME_ROOT:-/mnt/data/distribution-matching-slime/code/slime-0.2.4}"
TRAIN_DRIVER="${TRAIN_DRIVER:-train.py}"

if [[ -z "${MEGATRON_PATH:-}" ]]; then
  for candidate in "/root/slime_runtime/Megatron-LM" "/mnt/data/utils/slime_runtime/Megatron-LM" "/root/Megatron-LM"; do
    if [[ -d "${candidate}/megatron" ]]; then
      MEGATRON_PATH="${candidate}"
      break
    fi
  done
fi
MEGATRON_PATH="${MEGATRON_PATH:-/root/slime_runtime/Megatron-LM}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/root/venvs/slime/bin/python" ]]; then
    PYTHON_BIN="/root/venvs/slime/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/${MODEL_NAME}}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
if [[ -z "${MODEL_ARGS_SCRIPT:-}" ]]; then
  for candidate in \
    "${SLIME_ROOT}/slime/scripts/models/${MODEL_NAME}.sh" \
    "${SLIME_ROOT}/slime/scripts/models/${MODEL_NAME,,}.sh" \
    "${SLIME_ROOT}/scripts/models/${MODEL_NAME}.sh" \
    "${SLIME_ROOT}/scripts/models/${MODEL_NAME,,}.sh"; do
    if [[ -f "${candidate}" ]]; then
      MODEL_ARGS_SCRIPT="${candidate}"
      break
    fi
  done
fi
ALLOW_INFER_MODEL_ARGS="${ALLOW_INFER_MODEL_ARGS:-false}"
if [[ -z "${REF_LOAD:-}" ]]; then
  for candidate in \
    "/root/slime_runtime/checkpoints/${MODEL_NAME}_torch_dist" \
    "/mnt/data/models/Megatron_convert_models/${MODEL_NAME}_torch_dist"; do
    if [[ -d "${candidate}" ]]; then
      REF_LOAD="${candidate}"
      break
    fi
  done
fi
REF_LOAD="${REF_LOAD:-/root/slime_runtime/checkpoints/${MODEL_NAME}_torch_dist}"
LOAD_PATH="${LOAD_PATH:-/mnt/data/ebft-distribution-new/outputs/diff_dataset/slime_${MODEL_NAME}_run}"
SAVE_PATH="${SAVE_PATH:-${LOAD_PATH}}"

PREPARED_DATA_DIR="${PREPARED_DATA_DIR:-/mnt/data/ebft-distribution-new/outputs/diff_dataset_prepared}"
TRAIN_DATA="${TRAIN_DATA:-${PREPARED_DATA_DIR}/opencodeinstruct_qa_100k.jsonl}"
SLIME_TRAIN_DATA="${SLIME_TRAIN_DATA:-${PREPARED_DATA_DIR}/opencodeinstruct_slime_qa_100k.jsonl}"
MBPP_EVAL_DATA="${MBPP_EVAL_DATA:-${PREPARED_DATA_DIR}/mbpp_eval_qa.jsonl}"
HUMANEVAL_EVAL_DATA="${HUMANEVAL_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa.jsonl}"
SLIME_EVAL_DATA="${SLIME_EVAL_DATA:-${PREPARED_DATA_DIR}/humaneval_eval_qa_slime.jsonl}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs/diff_dataset}"
RUN_NAME="${RUN_NAME:-slime_gspo_${MODEL_NAME}_$(date +%m%d_%H%M)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/ray/$(date +%m%d%H%M%S)}"

# ---------------------------------------------------------------------------
# Training and rollout knobs
# ---------------------------------------------------------------------------
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-gspo}"
RM_TYPE="${RM_TYPE:-deepscaler}"
USE_EBFT_CUSTOM_RM="${USE_EBFT_CUSTOM_RM:-true}"
GROUP_RM="${GROUP_RM:-false}"
CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-}"
CUSTOM_REWARD_POST_PROCESS_PATH="${CUSTOM_REWARD_POST_PROCESS_PATH:-}"
RM_URL="${RM_URL:-}"
EBFT_RM_MODE="${EBFT_RM_MODE:-nonempty}"
EBFT_FEATURE_MODEL_PATH="${EBFT_FEATURE_MODEL_PATH:-}"
EBFT_CF_TARGET_MODE="${EBFT_CF_TARGET_MODE:-single}"
if [[ "${USE_EBFT_CUSTOM_RM}" == "true" && -z "${CUSTOM_RM_PATH}" ]]; then
  if [[ "${GROUP_RM}" == "true" ]]; then
    CUSTOM_RM_PATH="scripts.diff_dataset.slime_ebft_custom_rm.batched_custom_rm"
  else
    CUSTOM_RM_PATH="scripts.diff_dataset.slime_ebft_custom_rm.custom_rm"
  fi
fi
NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.6}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
BALANCE_DATA="${BALANCE_DATA:-true}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
ENABLE_SLIME_EVAL="${ENABLE_SLIME_EVAL:-true}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-4}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1024}"

LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
ENTROPY_COEF="${ENTROPY_COEF:-0.0}"
USE_OPD="${USE_OPD:-false}"
OPD_TYPE="${OPD_TYPE:-}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"

# ---------------------------------------------------------------------------
# Resource allocation. Defaults are colocated single-node 8-GPU smoke settings.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-$(python - <<'PY2'
import os
csv=os.environ.get('CUDA_VISIBLE_DEVICES','')
print(len([x for x in csv.split(',') if x.strip()]) or 8)
PY2
)}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${NUM_GPUS}}"
COLOCATE="${COLOCATE:-true}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-0}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.7}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"

TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
PIPELINE_MODEL_PARALLEL_SIZE="${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
USE_DYNAMIC_BATCH_SIZE="${USE_DYNAMIC_BATCH_SIZE:-true}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"

export HF_HOME="${HF_HOME:-/mnt/data/ebft-distribution-new/caches/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
export PYTHONUNBUFFERED=1
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${MEGATRON_PATH}:${SLIME_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"
export VIRTUAL_ENV="${VIRTUAL_ENV:-$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${SAVE_PATH}" "${RAY_TMPDIR}"

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || { echo "[ERROR] required file not found: ${path}" >&2; exit 1; }
}
require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || { echo "[ERROR] required directory not found: ${path}" >&2; exit 1; }
}

require_dir "${SLIME_ROOT}"
require_dir "${MEGATRON_PATH}"
require_file "${SLIME_ROOT}/${TRAIN_DRIVER}"
require_dir "${MODEL_PATH}"
require_dir "${REF_LOAD}"
require_file "${RAY_BIN}"

if [[ ! -s "${TRAIN_DATA}" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_code_datasets.py" --output-dir "${PREPARED_DATA_DIR}"
fi

if [[ ! -s "${SLIME_TRAIN_DATA}" || "${PREPARE_SLIME_DATA_FORCE:-false}" == "true" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_slime_jsonl.py" \
    --input "${TRAIN_DATA}" \
    --output "${SLIME_TRAIN_DATA}" \
    --input-key question \
    --label-key answer
fi
if [[ -s "${HUMANEVAL_EVAL_DATA}" && ( ! -s "${SLIME_EVAL_DATA}" || "${PREPARE_SLIME_DATA_FORCE:-false}" == "true" ) ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_slime_jsonl.py" \
    --input "${HUMANEVAL_EVAL_DATA}" \
    --output "${SLIME_EVAL_DATA}" \
    --input-key question \
    --label-key answer
fi

if [[ -n "${MODEL_ARGS_SCRIPT:-}" ]]; then
  require_file "${MODEL_ARGS_SCRIPT}"
  # shellcheck disable=SC1090
  source "${MODEL_ARGS_SCRIPT}"
  echo "[model-args] using MODEL_ARGS_SCRIPT=${MODEL_ARGS_SCRIPT}"
elif [[ -f "${SLIME_ROOT}/scripts/models/${MODEL_NAME}.sh" ]]; then
  # shellcheck disable=SC1090
  source "${SLIME_ROOT}/scripts/models/${MODEL_NAME}.sh"
  echo "[model-args] using ${SLIME_ROOT}/scripts/models/${MODEL_NAME}.sh"
else
  if [[ "${ALLOW_INFER_MODEL_ARGS}" != "true" ]]; then
    echo "[ERROR] MODEL_ARGS_SCRIPT not found for ${MODEL_NAME}; set MODEL_ARGS_SCRIPT explicitly or ALLOW_INFER_MODEL_ARGS=true" >&2
    exit 1
  fi
  echo "[model-args] inferring Megatron args from ${MODEL_PATH}/config.json"
  mapfile -t MODEL_ARGS < <("${PYTHON_BIN}" - "${MODEL_PATH}" <<'PYARGS'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
cfg = data.get("text_config") if isinstance(data.get("text_config"), dict) else data


def pick(*names, default=None):
    for name in names:
        if cfg.get(name) is not None:
            return cfg[name]
        if data.get(name) is not None:
            return data[name]
    return default


root_model_type = str(data.get("model_type", "")).lower()
model_type = str(pick("model_type", default="")).lower()
hidden = int(pick("hidden_size"))
heads = int(pick("num_attention_heads"))
kv_heads = int(pick("num_key_value_heads", default=heads))
head_dim = pick("head_dim", default=None)
kv_channels = int(head_dim) if head_dim else hidden // heads
rope_parameters = pick("rope_parameters", default={}) or {}
rope_theta = pick("rope_theta", default=None) or rope_parameters.get("rope_theta") or 1000000
rotary_percent = rope_parameters.get("partial_rotary_factor", pick("rotary_percent", default=None))
norm_eps = pick("rms_norm_eps", "layer_norm_epsilon", default=1e-6)
attention_bias = bool(pick("attention_bias", default=False))
tie_word_embeddings = bool(pick("tie_word_embeddings", default=True))
qk_layernorm = bool(pick("qk_layernorm", "qk_layer_norm", default=False))
if "qwen" in model_type:
    qk_layernorm = True

args = []
if root_model_type in {"qwen3_5", "qwen3_5_moe"}:
    args += ["--spec", "slime_plugins.models.qwen3_5", "get_qwen3_5_spec"]
args += ["--swiglu"]
args += ["--num-layers", str(int(pick("num_hidden_layers")))]
args += ["--hidden-size", str(hidden)]
args += ["--ffn-hidden-size", str(int(pick("intermediate_size")))]
args += ["--num-attention-heads", str(heads)]
if kv_heads != heads:
    args += ["--group-query-attention", "--num-query-groups", str(kv_heads)]
if root_model_type in {"qwen3_5", "qwen3_5_moe"}:
    args += ["--use-gated-attention", "--position-embedding-type", "rope", "--apply-layernorm-1p"]
    if rotary_percent is not None:
        args += ["--rotary-percent", str(rotary_percent)]
else:
    args += ["--use-rotary-position-embeddings"]
if not attention_bias:
    args += ["--disable-bias-linear"]
args += ["--normalization", "RMSNorm"]
args += ["--norm-epsilon", str(norm_eps)]
args += ["--rotary-base", str(int(rope_theta))]
args += ["--vocab-size", str(int(pick("vocab_size")))]
args += ["--kv-channels", str(kv_channels)]
if qk_layernorm:
    args += ["--qk-layernorm"]
if not tie_word_embeddings:
    args += ["--untie-embeddings-and-output-weights"]
if root_model_type in {"qwen3_5", "qwen3_5_moe"}:
    args += ["--attention-output-gate"]

for item in args:
    print(item)
PYARGS
  )
fi

# Some Qwen3.5 configs carry legacy MoE placeholder fields even when the model
# is dense. Keep this opt-in only because Megatron rejects MoE args without
# `num_experts`.
if [[ "${ADD_QWEN35_MOE_VALIDATION_PLACEHOLDERS:-false}" == "true" ]]; then
  MODEL_ARGS+=(
    --moe-ffn-hidden-size "${SLIME_MOE_FFN_HIDDEN_SIZE:-512}"
    --moe-shared-expert-intermediate-size "${SLIME_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-512}"
  )
fi

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_PATH}"
  --ref-load "${REF_LOAD}"
  --load "${LOAD_PATH}"
  --save "${SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
  --prompt-data "${SLIME_TRAIN_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type "${RM_TYPE}"
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --rollout-top-p "${ROLLOUT_TOP_P}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
)
if [[ "${BALANCE_DATA}" == "true" ]]; then
  ROLLOUT_ARGS+=(--balance-data)
fi
if [[ -n "${CUSTOM_RM_PATH}" ]]; then
  ROLLOUT_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")
fi
if [[ -n "${CUSTOM_REWARD_POST_PROCESS_PATH}" ]]; then
  ROLLOUT_ARGS+=(--custom-reward-post-process-path "${CUSTOM_REWARD_POST_PROCESS_PATH}")
fi
if [[ -n "${RM_URL}" ]]; then
  ROLLOUT_ARGS+=(--rm-url "${RM_URL}")
fi
if [[ "${GROUP_RM}" == "true" ]]; then
  ROLLOUT_ARGS+=(--group-rm)
fi

EVAL_ARGS=()
if [[ "${ENABLE_SLIME_EVAL}" == "true" ]]; then
  EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data humaneval "${SLIME_EVAL_DATA}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-top-p 1
  )
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
)
if [[ "${USE_DYNAMIC_BATCH_SIZE}" == "true" ]]; then
  PERF_ARGS+=(--use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")
fi

RL_ARGS=(
  --advantage-estimator "${ADVANTAGE_ESTIMATOR}"
  --use-kl-loss
  --kl-loss-coef "${KL_LOSS_COEF}"
  --kl-loss-type low_var_kl
  --entropy-coef "${ENTROPY_COEF}"
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
)
if [[ "${USE_OPD}" == "true" ]]; then
  RL_ARGS+=(--use-opd --opd-type "${OPD_TYPE:-sglang}" --opd-kl-coef "${OPD_KL_COEF}")
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style constant
  --weight-decay "${WEIGHT_DECAY}"
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

RESOURCE_ARGS=(
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
)
if [[ "${COLOCATE}" == "true" ]]; then
  RESOURCE_ARGS+=(--colocate)
else
  RESOURCE_ARGS+=(--rollout-num-gpus "${ROLLOUT_NUM_GPUS}")
fi

{
  printf 'RUN_NAME=%s\n' "${RUN_NAME}"
  printf 'RUN_DIR=%s\n' "${RUN_DIR}"
  printf 'SLIME_ROOT=%s\n' "${SLIME_ROOT}"
  printf 'TRAIN_DRIVER=%s\n' "${TRAIN_DRIVER}"
  printf 'MEGATRON_PATH=%s\n' "${MEGATRON_PATH}"
  printf 'PYTHON_BIN=%s\n' "${PYTHON_BIN}"
  printf 'RAY_BIN=%s\n' "${RAY_BIN}"
  printf 'MODEL_PATH=%s\n' "${MODEL_PATH}"
  printf 'MODEL_ARGS_SCRIPT=%s\n' "${MODEL_ARGS_SCRIPT:-}"
  printf 'ALLOW_INFER_MODEL_ARGS=%s\n' "${ALLOW_INFER_MODEL_ARGS}"
  printf 'REF_LOAD=%s\n' "${REF_LOAD}"
  printf 'LOAD_PATH=%s\n' "${LOAD_PATH}"
  printf 'SAVE_PATH=%s\n' "${SAVE_PATH}"
  printf 'SLIME_TRAIN_DATA=%s\n' "${SLIME_TRAIN_DATA}"
  printf 'ADVANTAGE_ESTIMATOR=%s\n' "${ADVANTAGE_ESTIMATOR}"
  printf 'RM_TYPE=%s\n' "${RM_TYPE}"
  printf 'CUSTOM_RM_PATH=%s\n' "${CUSTOM_RM_PATH}"
  printf 'CUSTOM_REWARD_POST_PROCESS_PATH=%s\n' "${CUSTOM_REWARD_POST_PROCESS_PATH}"
  printf 'RM_URL=%s\n' "${RM_URL}"
  printf 'USE_EBFT_CUSTOM_RM=%s\n' "${USE_EBFT_CUSTOM_RM}"
  printf 'GROUP_RM=%s\n' "${GROUP_RM}"
  printf 'EBFT_RM_MODE=%s\n' "${EBFT_RM_MODE}"
  printf 'EBFT_FEATURE_MODEL_PATH=%s\n' "${EBFT_FEATURE_MODEL_PATH}"
  printf 'EBFT_CF_TARGET_MODE=%s\n' "${EBFT_CF_TARGET_MODE}"
  printf 'NUM_ROLLOUT=%s\n' "${NUM_ROLLOUT}"
  printf 'ROLLOUT_BATCH_SIZE=%s\n' "${ROLLOUT_BATCH_SIZE}"
  printf 'N_SAMPLES_PER_PROMPT=%s\n' "${N_SAMPLES_PER_PROMPT}"
  printf 'GLOBAL_BATCH_SIZE=%s\n' "${GLOBAL_BATCH_SIZE}"
  printf 'ENABLE_SLIME_EVAL=%s\n' "${ENABLE_SLIME_EVAL}"
  printf 'BALANCE_DATA=%s\n' "${BALANCE_DATA}"
  printf 'USE_DYNAMIC_BATCH_SIZE=%s\n' "${USE_DYNAMIC_BATCH_SIZE}"
  printf 'USE_OPD=%s\n' "${USE_OPD}"
  printf 'OPD_TYPE=%s\n' "${OPD_TYPE}"
  printf 'OPD_KL_COEF=%s\n' "${OPD_KL_COEF}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'NUM_GPUS=%s\n' "${NUM_GPUS}"
  printf 'COLOCATE=%s\n' "${COLOCATE}"
  printf 'RAY_TMPDIR=%s\n' "${RAY_TMPDIR}"
  printf 'SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=%s\n' "${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK}"
} > "${RUN_DIR}/run_context.env"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  echo "[dry-run] slime command would be submitted from ${SLIME_ROOT}"
  printf '%q ' "${PYTHON_BIN}" "${SLIME_ROOT}/${TRAIN_DRIVER}" "${RESOURCE_ARGS[@]}" "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${RL_ARGS[@]}" "${PERF_ARGS[@]}" "${EVAL_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}"
  echo
  exit 0
fi

"${RAY_BIN}" stop --force 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
sleep 3

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${RAY_BIN}" start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" --temp-dir "${RAY_TMPDIR}"

RUNTIME_ENV_JSON="$("${PYTHON_BIN}" - <<'PYJSON'
import json
import os

keys = [
    "PYTHONPATH",
    "PATH",
    "VIRTUAL_ENV",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_HUB_DISABLE_XET",
    "TOKENIZERS_PARALLELISM",
    "EBFT_RM_MODE",
    "EBFT_FEATURE_MODEL_PATH",
    "EBFT_CF_TARGET_MODE",
    "EBFT_N_SAMPLES_PER_PROMPT",
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK",
    "RAY_TMPDIR",
]
env_vars = {key: os.environ.get(key, "") for key in keys if os.environ.get(key, "")}
print(json.dumps({"env_vars": env_vars}, ensure_ascii=False))
PYJSON
)"

cd "${SLIME_ROOT}"
set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${RAY_BIN}" job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" "${SLIME_ROOT}/${TRAIN_DRIVER}" \
  "${RESOURCE_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${RL_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e

echo "TRAIN_RC=${TRAIN_RC}" > "${RUN_DIR}/final_status.env"
echo "SAVE_PATH=${SAVE_PATH}" >> "${RUN_DIR}/final_status.env"
echo "[done] slime training rc=${TRAIN_RC}; logs=${RUN_DIR}"
exit "${TRAIN_RC}"
