#!/usr/bin/env bash
# AOPS: single-node 8 GPU, online teacher + student training (G3 recipe).
#
# G2 vs G3 (this script = single-node G3):
#   G2: cf_l1oo + frozen critic head (critic_lr_head=0). NO feature adapter,
#       NO EMA, NO classifier loss, NO direct discrepancy, NO diversity/
#       alignment reward coefs, NO ce_loss_coef. Pure cf_l1oo with online
#       teacher. See scripts/run_G2_rebase.sh.
#   G3: G2 + enable_ema + feature_adapter (embedding-space training) +
#       trainable critic head (critic_lr_head=5e-5) + critic direct
#       discrepancy (coef=0.1, target=ema_gt) + explicit diversity/alignment
#       reward coefs + ce_loss (coef=0.03). Same teacher serving config
#       and same 128-completion/step teacher load profile as G2 single-node
#       because N_SAMPLES_PER_PROMPT=4 and ROLLOUT_BATCH_SIZE=32 are
#       unchanged; the G3 extras only touch the student side.
#
# Manual resource split:
#   - teacher GPU ids: TEACHER_CUDA_VISIBLE_DEVICES (default 0-5, 6 vLLM workers TP=1)
#   - student GPU ids: STUDENT_CUDA_VISIBLE_DEVICES (default 6-7)
#   - student actor/critic GPU counts: ACTOR_GPUS / CRITIC_GPUS (default 1/1)
#   - ref/reward follow actor/critic automatically (colocate)
set -euo pipefail

# Override examples:
#   TARGET_STEPS=500 bash scripts/run_G3_rebase.sh
#   MAX_SAMPLES=10000 bash scripts/run_G3_rebase.sh
#   ACTOR_LR=5e-7 CRITIC_LR_HEAD=1e-4 bash scripts/run_G3_rebase.sh

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "$csv" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"$csv"
}

# --------------------------------------------------------------------
# 0) MANUAL GPU ASSIGNMENT (edit these first)
# --------------------------------------------------------------------
TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
STUDENT_CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES:-6,7}"

ACTOR_GPUS="${ACTOR_GPUS:-1}"
CRITIC_GPUS="${CRITIC_GPUS:-1}"
# Keep ref/reward colocated and follow actor/critic world-size.
REF_GPUS="${ACTOR_GPUS}"
REWARD_GPUS="${CRITIC_GPUS}"

# --------------------------------------------------------------------
# 1) TEACHER SERVING CONFIG (local vLLM)
# --------------------------------------------------------------------
# Defaults match teacher_model/code/models/qwen_122b.env — the only
# teacher whose weights are fully downloaded on this machine.
# Override via env to use a different teacher (e.g. a smaller 7B/14B).
LAUNCH_TEACHER="${LAUNCH_TEACHER:-true}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/data/models/qwen3.5-27b}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-qwen3.5-27b}"
TEACHER_BASE_PORT="${TEACHER_BASE_PORT:-8004}"
TEACHER_API_KEY="${TEACHER_API_KEY:-teacher-local}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-2048}"
TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-32}"
TEACHER_MAX_BATCHED_TOKENS="${TEACHER_MAX_BATCHED_TOKENS:-16384}"
TEACHER_GPU_MEMORY_UTIL="${TEACHER_GPU_MEMORY_UTIL:-0.96}"
TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS:-1800}"

# Multi-worker: one vLLM worker per TP_SIZE GPUs, each on its own port.
IFS=',' read -r -a _TEACHER_GPU_IDS <<< "${TEACHER_CUDA_VISIBLE_DEVICES}"
TEACHER_WORKER_COUNT=$(( ${#_TEACHER_GPU_IDS[@]} / TEACHER_TP_SIZE ))
_DEFAULT_API_URLS=""
for (( _i=0; _i<TEACHER_WORKER_COUNT; _i++ )); do
  _port=$(( TEACHER_BASE_PORT + _i ))
  [[ -n "${_DEFAULT_API_URLS}" ]] && _DEFAULT_API_URLS="${_DEFAULT_API_URLS},"
  _DEFAULT_API_URLS="${_DEFAULT_API_URLS}http://127.0.0.1:${_port}/v1"
done
TEACHER_API_BASE="${TEACHER_API_BASE:-${_DEFAULT_API_URLS}}"

# --------------------------------------------------------------------
# 2) TRAINING DATA / MODEL PATHS
# --------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/mnt/data/ebft-distribution-new/code}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/gemma-4-E4B}"
DEFAULT_TRAIN_DATA="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
TRAIN_DATA="${TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"

# Venvs live on local ext4 (ossfs2 can't host venv symlinks). See
# scripts/setup_env.sh for the bootstrap that creates and snapshots them.
TEACHER_VENV="${TEACHER_VENV:-/mnt/workspace/venvs/.teacherVenv}"
STUDENT_VENV="${STUDENT_VENV:-/mnt/workspace/venvs/.venv}"

# HF blobs go on persistent OSS (model weights survive container restart;
# downloads are tmp+rename, OSS-safe).
export HF_HOME="${HF_HOME:-/mnt/data/ebft-distribution-new/caches/hf}"
# Compile caches MUST be on local ext4: ossfs2 rejects "seek + write into
# existing file" with EINVAL, which fuse mis-reports as 'No space left on
# device'. That kills g++/nvcc when emitting .o (FusedAdam, fused_adan,
# ...) and triton when emitting .cubin/.so. Cost of being on local ext4:
# ~30-60s recompile after a container restart.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/workspace/.torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/workspace/.triton_cache}"

# Reduce CUDA OOM under tight memory budgets. RLHF batches reshape every
# PPO step (rollout vs train, variable seq lens), so PyTorch's default
# fixed-size segments fragment fast. expandable_segments lets the
# allocator grow segments on demand and typically frees 1-2 GiB of
# headroom on an 80GB A100. PyTorch suggests this in the OOM message.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --------------------------------------------------------------------
# 3) TRAINING KNOBS
# --------------------------------------------------------------------
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE))}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-4}"
MICRO_REWARD_BATCH_SIZE="${MICRO_REWARD_BATCH_SIZE:-4}"

PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-384}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-8}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-8}"
STRIDE="${STRIDE:-8}"

NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
TARGET_STEPS="${TARGET_STEPS:-500}"
# Trainer computes:
#   max_steps = len(prompts_dataset) * n_samples_per_prompt / train_batch_size * num_episodes * max_epochs
# So we back-solve max_samples to hit TARGET_STEPS by default.
DEFAULT_MAX_SAMPLES="$((TARGET_STEPS * TRAIN_BATCH_SIZE / N_SAMPLES_PER_PROMPT / NUM_EPISODES / MAX_EPOCHS))"
MAX_SAMPLES="${MAX_SAMPLES:-${DEFAULT_MAX_SAMPLES}}"

CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.6}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-4}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-1024}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-200}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-48}"
TEACHER_SYSTEM_PROMPT_TEXT="${TEACHER_SYSTEM_PROMPT_TEXT:-You are a precise assistant. produce a correct and well-reasoned answer. Step by step when necessary. Keep reasoning sufficient. Final answer is clearly stated.}"
TEACHER_SYSTEM_PROMPT_ID="${TEACHER_SYSTEM_PROMPT_ID:-v1-balanced}"
# TEACHER_CACHE_DIR must be on local ext4 (NOT on ossfs2 / /mnt/data).
# The provider opens a SQLite DB at "${TEACHER_CACHE_DIR}/worker_<i>/teacher_cache.db",
# and SQLite on ossfs2 dies almost instantly with `sqlite3.OperationalError:
# disk I/O error` because:
#   - ossfs2 doesn't honor POSIX advisory locks (fcntl F_SETLK), which
#     SQLite's rollback-journal mode requires for the reserved/pending lock
#     transitions; the FUSE shim returns EINVAL/ENOTSUP and SQLite aborts.
#   - WAL mode would also fail because it depends on shared-memory mmap of
#     the .db-shm file, and ossfs2 only supports read-only mmap.
# Symptom in practice: training process crashes during EBFTTrainer.__init__()
# the moment build_teacher_provider() instantiates TeacherCache(...).
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-/mnt/workspace/teacher_cache_shared}"

ENABLE_TEACHER_PREFETCH="${ENABLE_TEACHER_PREFETCH:-false}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-2}"
PREFETCH_MAX_WORKERS="${PREFETCH_MAX_WORKERS:-6}"

# ── G3-only: feature adapter / EMA / trainable critic head / discrepancy ──
#
# These are the knobs that distinguish G3 from G2. The default values mirror
# the 2-node G3 launcher (scripts/run_G3_rebase_2node_once.sh) so single-node
# and 2-node G3 runs are directly comparable in TensorBoard.
#
# feature_adapter (residual_bottleneck):
#   Adds a small trainable adapter on top of the frozen embedding stream so
#   the critic can learn a task-specific feature space without unfreezing
#   the backbone. rank=64 / dropout=0 / unfreeze_layers=0 is the paper's
#   default; bump rank to 128 if the critic underfits.
FEATURE_ADAPTER_RANK="${FEATURE_ADAPTER_RANK:-64}"
FEATURE_ADAPTER_DROPOUT="${FEATURE_ADAPTER_DROPOUT:-0.0}"
UNFREEZE_LAYERS="${UNFREEZE_LAYERS:-0}"

# EMA over critic weights, consumed as ema_gt target in direct-discrepancy.
EMA_BETA="${EMA_BETA:-0.99}"

# Classifier-style CE loss coefficient on the critic output.
#   Paper gamma (CE-loss weight) = CE_LOSS_COEF, default 0.03.
CE_LOSS_COEF="${CE_LOSS_COEF:-0.03}"

# Learning rates. Key G3 vs G2 difference:
#   G2: critic_learning_rate=0, critic_lr_head=0  -> critic fully frozen
#   G3: critic_learning_rate=0, critic_lr_head=5e-5 -> embedding layers
#                                                       stay frozen, head
#                                                       trains from scratch
ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-5e-5}"

# Critic auxiliary losses.
#   - classifier loss: disabled by default in G3 (paper didn't show net win).
#   - direct discrepancy: regularize critic output toward EMA-smoothed GT
#     embedding; coef=0.1 is the paper's ablation sweet spot.
CRITIC_CLASSIFIER_LOSS_COEF="${CRITIC_CLASSIFIER_LOSS_COEF:-0.0}"
CRITIC_DIRECT_DISCREPANCY_COEF="${CRITIC_DIRECT_DISCREPANCY_COEF:-0.1}"
CRITIC_DIRECT_DISCREPANCY_TARGET="${CRITIC_DIRECT_DISCREPANCY_TARGET:-ema_gt}"

# Reward shape: r = ALIGNMENT_REW_COEF * gt_alignment - DIVERSITY_REW_COEF * diversity
#
# Mapping to the paper (Section 2.2 / Appendix C):
#   Paper alpha (alignment-bias) ≈ DIVERSITY_REW_COEF / ALIGNMENT_REW_COEF
#     alpha = 1.0  -> standard RLOO, full diversity penalty (DIVERSITY_REW_COEF=1.0)
#     alpha = 0.5  -> diversity penalty halved
#     alpha = 0    -> alignment-only, mode-seeking / unstable
# Default matches paper's standard configuration: alpha=1.
DIVERSITY_REW_COEF="${DIVERSITY_REW_COEF:-1.0}"
ALIGNMENT_REW_COEF="${ALIGNMENT_REW_COEF:-1.0}"

# ── Eval / Checkpoint ──
# IN-TRAINING EVAL DISABLED BY DEFAULT.
#   GENERATE_MAX_LEN=8 (the EBFT token-level rollout) makes the in-training
#   eval generate only 8 tokens, which is meaningless on AOPS — every prompt
#   ends up "no answer" / 0% acc and clutters TensorBoard with a flat-zero
#   curve that misleads early-stopping decisions. The post-training 2-round
#   vLLM eval (via supplement_2rounds/G3.sh) is what we actually trust for
#   accuracy reporting; that runs at FIRST_PASS_MAX_NEW_TOKENS=16384 / 32768.
#
#   To re-enable in-training eval explicitly, override EVAL_STEPS to a value
#   <= TARGET_STEPS, e.g.:
#       EVAL_STEPS=100 EVAL_GENERATE_MAX_LEN=2048 bash scripts/run_G3_rebase.sh
#   NOTE: only eval_steps == -1 is treated as "disabled" by ebft_trainer
#   (eval_steps == -1 -> float('inf')). A finite-but-large value still
#   triggers the trainer's step-0 initial eval, which under ZeRO-3
#   colocate_actor_ref deadlocks NCCL all-gather when batches don't
#   divide evenly across actor ranks. Use -1, not a big number.
EVAL_STEPS="${EVAL_STEPS:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"

# Post-training offline two-round eval (16k → 32k via vLLM in
# scripts/supplement_2rounds/G3.sh). Independent from in-training EVAL_STEPS.
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
RUN_TWO_ROUND_EVAL="${RUN_TWO_ROUND_EVAL:-${EVAL_AFTER_TRAIN}}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
# vLLM concurrency knobs. See scripts/supplement_2rounds/G3.sh for the full
# HOL-blocking rationale that motivated raising these from the legacy
# {32, hardcoded-16} defaults to {256, 256}.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-256}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
# vLLM eval grabs all 8 GPUs after teacher + RL training are torn down.
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-${TEACHER_CUDA_VISIBLE_DEVICES},${STUDENT_CUDA_VISIBLE_DEVICES}}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")}"

# Analysis venv (same as student by default; the analyzer is a pure-python
# script that just needs HF tokenizers, so no need for a separate venv).
ANALYSIS_VENV="${ANALYSIS_VENV:-${STUDENT_VENV}}"

# --------------------------------------------------------------------
# 4) ENV / RUN DIR
# --------------------------------------------------------------------
# HF_HOME and PYTORCH_CUDA_ALLOC_CONF are exported above (section 1) with
# DSW-specific defaults; do not redeclare here or the upper values would be
# silently shadowed if a user pre-exported only one of the two.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

# NCCL safety nets — applied to BOTH the training stage (DeepSpeed/Ray
# collectives between actor/critic/ref/reward on the student GPUs) AND
# the post-training vLLM stage (which also sources _vllm_runtime.sh and
# would re-apply these defaults; redundant export here makes the
# training stage benefit from the same protections):
#
#   NCCL_P2P_LEVEL=NVL   - allow P2P only over NVLink (HGX A100 NVSwitch
#                          full mesh on this box). Banning the previous
#                          NCCL_P2P_DISABLE=1 default that disabled NVLink
#                          P2P entirely and forced cross-NUMA traffic onto
#                          RoCE GDRDMA (where it tripped mlx5:1 async fatal
#                          QP / local access violation).
#   NCCL_NET_GDR_DISABLE=1 - even if NCCL still routes some path via NET,
#                          stage transfers through host RAM instead of GDR.
#                          Slower but eliminates the GDR-page-unmap window
#                          that fires QP-fatal asynchronously.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
[[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

RUN_NAME="${RUN_NAME:-g3_online_teacher_8gpu_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
TEACHER_LOG_DIR="${RUN_DIR}/teacher_logs"
mkdir -p "$RUN_DIR" "$SAVE_PATH" "$TB_DIR" "$TEACHER_CACHE_DIR" "$TEACHER_LOG_DIR"
# Note: post-eval output paths (eval_results_*.jsonl, eval_*.log,
# eval_analysis_*.json) are managed by scripts/supplement_2rounds/G3.sh
# under ${RUN_DIR}/supplement_logs/; no need to declare them here.

# --------------------------------------------------------------------
# 5) SANITY CHECK
# --------------------------------------------------------------------
teacher_gpu_count="$(count_csv_items "$TEACHER_CUDA_VISIBLE_DEVICES")"
student_gpu_count="$(count_csv_items "$STUDENT_CUDA_VISIBLE_DEVICES")"

TEACHER_VLLM_BIN="${TEACHER_VLLM_BIN:-${TEACHER_VENV}/bin/vllm}"
if [[ ! -x "${TEACHER_VLLM_BIN}" ]]; then
  echo "[ERROR] TEACHER_VLLM_BIN not executable: ${TEACHER_VLLM_BIN}"
  echo "        expected teacher env: ${TEACHER_VENV}"
  exit 1
fi

STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
if [[ ! -x "${STUDENT_PYTHON_BIN}" ]]; then
  echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"
  echo "        expected student env: ${STUDENT_VENV}"
  exit 1
fi

# Required only when the post-training two-round vLLM eval will run.
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"
if [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
  if [[ ! -x "${TEACHER_PYTHON_BIN}" ]]; then
    echo "[ERROR] TEACHER_PYTHON_BIN not executable: ${TEACHER_PYTHON_BIN}"
    echo "        (needed for vLLM 2-round eval; set RUN_TWO_ROUND_EVAL=false to skip)"
    exit 1
  fi
  if [[ ! -x "${ANALYSIS_PYTHON_BIN}" ]]; then
    echo "[ERROR] ANALYSIS_PYTHON_BIN not executable: ${ANALYSIS_PYTHON_BIN}"
    exit 1
  fi
fi

if (( teacher_gpu_count < TEACHER_TP_SIZE )); then
  echo "[ERROR] teacher GPU count (${teacher_gpu_count}) < TEACHER_TP_SIZE (${TEACHER_TP_SIZE})"
  exit 1
fi
if (( teacher_gpu_count % TEACHER_TP_SIZE != 0 )); then
  echo "[ERROR] teacher GPU count (${teacher_gpu_count}) not divisible by TEACHER_TP_SIZE (${TEACHER_TP_SIZE})"
  exit 1
fi

if (( ACTOR_GPUS + CRITIC_GPUS > student_gpu_count )); then
  echo "[ERROR] ACTOR_GPUS(${ACTOR_GPUS}) + CRITIC_GPUS(${CRITIC_GPUS}) > student GPU count(${student_gpu_count})"
  exit 1
fi

if (( TRAIN_BATCH_SIZE != N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE )); then
  echo "[ERROR] TRAIN_BATCH_SIZE must equal N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE"
  echo "        got ${TRAIN_BATCH_SIZE} vs ${N_SAMPLES_PER_PROMPT} * ${ROLLOUT_BATCH_SIZE}"
  exit 1
fi

if (( TRAIN_BATCH_SIZE % (MICRO_TRAIN_BATCH_SIZE * ACTOR_GPUS) != 0 )); then
  echo "[ERROR] train_batch_size % (micro_train_batch_size * actor_gpus) != 0"
  echo "        ${TRAIN_BATCH_SIZE} % (${MICRO_TRAIN_BATCH_SIZE} * ${ACTOR_GPUS}) != 0"
  exit 1
fi

if (( MICRO_TRAIN_BATCH_SIZE < N_SAMPLES_PER_PROMPT || MICRO_TRAIN_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] MICRO_TRAIN_BATCH_SIZE must be >= N_SAMPLES_PER_PROMPT and divisible by it"
  echo "        got MICRO_TRAIN_BATCH_SIZE=${MICRO_TRAIN_BATCH_SIZE}, N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}"
  exit 1
fi

if (( MICRO_ROLLOUT_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] MICRO_ROLLOUT_BATCH_SIZE must be divisible by N_SAMPLES_PER_PROMPT"
  exit 1
fi

if [[ "${TRAIN_DATA}" == "${DEFAULT_TRAIN_DATA}" && ! -e "${TRAIN_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
  echo "[WARN] TRAIN_DATA default not found, fallback to ${FALLBACK_LOCAL_DATA}"
  TRAIN_DATA="${FALLBACK_LOCAL_DATA}"
fi
if [[ "${EVAL_DATA}" == "${DEFAULT_EVAL_DATA}" && ! -e "${EVAL_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
  echo "[WARN] EVAL_DATA default not found, fallback to ${FALLBACK_LOCAL_DATA}"
  EVAL_DATA="${FALLBACK_LOCAL_DATA}"
fi
if [[ "${TRAIN_DATA}" == /* && ! -e "${TRAIN_DATA}" ]]; then
  echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"
  exit 1
fi
if [[ "${EVAL_DATA}" == /* && ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi

echo "========== AOPS online-teacher run =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "Teacher GPUs:               ${TEACHER_CUDA_VISIBLE_DEVICES} (count=${teacher_gpu_count})"
echo "Teacher workers:            ${TEACHER_WORKER_COUNT} x TP${TEACHER_TP_SIZE}"
echo "Teacher max_model_len:      ${TEACHER_MAX_MODEL_LEN}"
echo "Teacher max_num_seqs:       ${TEACHER_MAX_NUM_SEQS}"
echo "Teacher max_batched_tokens: ${TEACHER_MAX_BATCHED_TOKENS}"
echo "Teacher remote_batch_size:  ${TEACHER_REMOTE_BATCH_SIZE}"
echo "Student GPUs:               ${STUDENT_CUDA_VISIBLE_DEVICES} (count=${student_gpu_count})"
echo "Actor/Critic GPUs:          ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "Ref/Reward GPUs (colocate): ${REF_GPUS}/${REWARD_GPUS}"
echo "Teacher API:                ${TEACHER_API_BASE}"
echo "Teacher model:              ${TEACHER_MODEL_NAME}"
echo "Train data:                 ${TRAIN_DATA}"
echo "Eval data:                  ${EVAL_DATA}"
echo "target_steps:               ${TARGET_STEPS}"
echo "max_samples:                ${MAX_SAMPLES}"
echo "Teacher vLLM bin:           ${TEACHER_VLLM_BIN}"
echo "Student python:             ${STUDENT_PYTHON_BIN}"
echo "----- G3 recipe knobs -----"
echo "Actor/Critic/Head LR:       ${ACTOR_LR} / ${CRITIC_LR} / ${CRITIC_LR_HEAD}"
echo "Feature adapter:            rank=${FEATURE_ADAPTER_RANK}, dropout=${FEATURE_ADAPTER_DROPOUT}, unfreeze=${UNFREEZE_LAYERS}"
echo "EMA beta:                   ${EMA_BETA}"
echo "CE loss coef:               ${CE_LOSS_COEF}"
echo "Critic classifier coef:     ${CRITIC_CLASSIFIER_LOSS_COEF}"
echo "Critic direct discrepancy:  coef=${CRITIC_DIRECT_DISCREPANCY_COEF}, target=${CRITIC_DIRECT_DISCREPANCY_TARGET}"
echo "Reward coefs:               diversity=${DIVERSITY_REW_COEF}, alignment=${ALIGNMENT_REW_COEF}"
echo "============================================="

declare -a TEACHER_PIDS=()
declare -a TEACHER_PORTS=()
declare -a TEACHER_WORKER_LOGS=()

cleanup() {
  # 1) Stop the teacher vLLM worker processes we tracked.
  for _pid in "${TEACHER_PIDS[@]:-}"; do
    if [[ -n "${_pid}" ]] && kill -0 "${_pid}" 2>/dev/null; then
      echo "[cleanup] stopping teacher pid=${_pid}"
      kill "${_pid}" || true
    fi
  done
  for _pid in "${TEACHER_PIDS[@]:-}"; do
    [[ -n "${_pid}" ]] && wait "${_pid}" 2>/dev/null || true
  done

  # 2) Tear down the Ray cluster (actor/critic/ref/reward workers each hold
  #    a GPU; if training crashes mid-run, ray actors leak and OOM the next
  #    attempt).
  echo "[cleanup] stopping ray cluster..."
  ray stop --force >/dev/null 2>&1 || true

  # 3) Kill any orphan vLLM stage workers from the post-training
  #    supplement_2rounds/G3.sh (stage1 / stage2). When EngineCore raises
  #    EngineDeadError or the user Ctrl-C's, the parent Python exits but
  #    the multiproc TP workers keep holding all 8 GPUs on a dead NCCL
  #    collective. Same fault we hit on G1 stage2 (2026-04-28).
  echo "[cleanup] killing orphan vLLM stage workers (if any)..."
  pkill -9 -f 'multiproc_executor' 2>/dev/null || true
  pkill -9 -f 'vllm.v1.engine.core' 2>/dev/null || true
  pkill -9 -f 'EngineCore' 2>/dev/null || true
  pkill -9 -f 'vllm_generate_progress' 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_teacher_worker() {
  local idx="$1" pid="$2" port="$3" log="$4" waited=0
  until curl -sf "http://127.0.0.1:${port}/health" >/dev/null; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] Teacher worker ${idx} exited before health check. Log: ${log}"
      return 1
    fi
    sleep 3
    waited=$((waited + 3))
    if (( waited >= TEACHER_WAIT_SECONDS )); then
      echo "[ERROR] Teacher worker ${idx} health timeout (${TEACHER_WAIT_SECONDS}s). Log: ${log}"
      return 1
    fi
  done
  return 0
}

# Staggered launch: vLLM v1's EngineCore calls torch.distributed
# init_process_group() with TCPStore on a RANDOMLY CHOSEN high port.
# The port picker uses the classic `bind(port=0) -> getsockname() -> close()
# -> later bind(same port)` pattern, which has a TOCTOU race: between
# close() and the real bind(), the kernel may hand the same port to
# another process's ephemeral socket, producing
#   DistNetworkError: port NNN EADDRINUSE
# on the second bind(). Probability is tiny for 1-2 concurrent workers
# but shoots up to ~20-30% at 6 workers launched in a ~1s window, which
# is exactly what this script does. Sleeping between launches spreads
# the EngineCore startup sequence so no two are inside each other's
# port-picker TOCTOU window.
TEACHER_LAUNCH_STAGGER_SECONDS="${TEACHER_LAUNCH_STAGGER_SECONDS:-2}"

if [[ "${LAUNCH_TEACHER}" == "true" ]]; then
  echo "[teacher] launching ${TEACHER_WORKER_COUNT} workers (TP=${TEACHER_TP_SIZE}) ..."
  for (( _w=0; _w<TEACHER_WORKER_COUNT; _w++ )); do
    if (( _w > 0 && TEACHER_LAUNCH_STAGGER_SECONDS > 0 )); then
      sleep "${TEACHER_LAUNCH_STAGGER_SECONDS}"
    fi
    _port=$(( TEACHER_BASE_PORT + _w ))
    _gpu_start=$(( _w * TEACHER_TP_SIZE ))
    _worker_gpus=""
    for (( _g=_gpu_start; _g<_gpu_start+TEACHER_TP_SIZE; _g++ )); do
      [[ -n "${_worker_gpus}" ]] && _worker_gpus="${_worker_gpus},"
      _worker_gpus="${_worker_gpus}${_TEACHER_GPU_IDS[$_g]}"
    done
    _log="${TEACHER_LOG_DIR}/worker_${_w}.log"
    # TEACHER_LOG_DIR sits under RUN_DIR on ossfs2. `bash > existing_file`
    # opens with O_WRONLY|O_CREAT|O_TRUNC, and ossfs2 rejects truncate of
    # an existing object with EINVAL (fuse mis-reports as ENOSPC). Bash
    # then exits the redirect subshell BEFORE exec'ing vllm, so $! points
    # at a dead helper, no vllm actually runs -> GPU stays idle / curl
    # hangs for 1h. Only bites when the same RUN_NAME is reused within
    # the same minute (timestamp granularity), e.g. rapid Ctrl-C+restart
    # loops during debugging. Pre-delete forces O_CREAT-new-object path
    # which ossfs2 supports.
    rm -f "${_log}" 2>/dev/null || true

    CUDA_VISIBLE_DEVICES="${_worker_gpus}" \
    "${TEACHER_VLLM_BIN}" serve "${TEACHER_MODEL_PATH}" \
      --served-model-name "${TEACHER_MODEL_NAME}" \
      --host 0.0.0.0 \
      --port "${_port}" \
      --tensor-parallel-size "${TEACHER_TP_SIZE}" \
      --dtype "${TEACHER_DTYPE}" \
      --api-key "${TEACHER_API_KEY}" \
      --generation-config vllm \
      --max-model-len "${TEACHER_MAX_MODEL_LEN}" \
      --max-num-seqs "${TEACHER_MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${TEACHER_MAX_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${TEACHER_GPU_MEMORY_UTIL}" \
      --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
      --enable-chunked-prefill \
      > "${_log}" 2>&1 &

    TEACHER_PIDS+=("$!")
    TEACHER_PORTS+=("${_port}")
    TEACHER_WORKER_LOGS+=("${_log}")
    echo "[teacher] worker ${_w}: GPU ${_worker_gpus}, port ${_port}, log ${_log}"
  done

  for (( _w=0; _w<TEACHER_WORKER_COUNT; _w++ )); do
    wait_for_teacher_worker "${_w}" "${TEACHER_PIDS[$_w]}" "${TEACHER_PORTS[$_w]}" "${TEACHER_WORKER_LOGS[$_w]}"
    echo "[teacher] worker ${_w} healthy."
  done
  echo "[teacher] all ${TEACHER_WORKER_COUNT} workers ready."
fi

PREFETCH_FLAGS=()
if [[ "${ENABLE_TEACHER_PREFETCH}" == "true" ]]; then
  PREFETCH_FLAGS=(
    --enable_teacher_prefetch
    --prefetch_depth "${PREFETCH_DEPTH}"
    --prefetch_max_workers "${PREFETCH_MAX_WORKERS}"
  )
fi

ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" \
"${STUDENT_PYTHON_BIN}" -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \
  --gradient_checkpointing --gradient_checkpointing_use_reentrant --use_kl_loss --use_whitening --enable_ema \
  --feature_adapter_enable \
  --feature_adapter_type residual_bottleneck \
  --feature_adapter_rank "${FEATURE_ADAPTER_RANK}" \
  --feature_adapter_dropout "${FEATURE_ADAPTER_DROPOUT}" \
  --feature_adapter_unfreeze_layers "${UNFREEZE_LAYERS}" \
  --distribution_reward_type cf_l1oo \
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \
  --cf_num_freqs 128 --cf_sigma 1.0 --cf_seed 43 --cf_alpha 0.5 --cf_beta 0.5 --cf_reward_scale 1.0 \
  --cf_target_mode teacher --cf_teacher_lambda "${CF_TEACHER_LAMBDA}" --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}" \
  \
  --teacher_backend remote \
  --teacher_api_base "${TEACHER_API_BASE}" \
  --teacher_api_key "${TEACHER_API_KEY}" \
  --teacher_api_style completions \
  --teacher_model_name "${TEACHER_MODEL_NAME}" \
  --teacher_timeout "${TEACHER_TIMEOUT}" \
  --teacher_max_retries "${TEACHER_MAX_RETRIES}" \
  --teacher_remote_batch_size "${TEACHER_REMOTE_BATCH_SIZE}" \
  --teacher_temperature "${TEACHER_TEMPERATURE}" \
  --teacher_top_p "${TEACHER_TOP_P}" \
  --teacher_max_new_tokens "${TEACHER_MAX_NEW_TOKENS}" \
  --teacher_system_prompt_text "${TEACHER_SYSTEM_PROMPT_TEXT}" \
  --teacher_system_prompt_id "${TEACHER_SYSTEM_PROMPT_ID}" \
  --teacher_cache_enable --teacher_cache_dir "${TEACHER_CACHE_DIR}" \
  "${PREFETCH_FLAGS[@]}" \
  \
  --embed_method last_token --critic_sequence_level last_token \
  --critic_learning_rate "${CRITIC_LR}" \
  --critic_lr_head "${CRITIC_LR_HEAD}" \
  --critic_classifier_loss_coef "${CRITIC_CLASSIFIER_LOSS_COEF}" \
  --critic_direct_discrepancy_coef "${CRITIC_DIRECT_DISCREPANCY_COEF}" \
  --critic_direct_discrepancy_target "${CRITIC_DIRECT_DISCREPANCY_TARGET}" \
  --ema_beta "${EMA_BETA}" \
  --ce_loss_coef "${CE_LOSS_COEF}" \
  --diversity_rew_coef "${DIVERSITY_REW_COEF}" \
  --alignment_rew_coef "${ALIGNMENT_REW_COEF}" \
  --actor_learning_rate "${ACTOR_LR}" \
  --pretrain "${MODEL_PATH}" --critic_pretrain "${MODEL_PATH}" \
  --prompt_data "${TRAIN_DATA}" --eval_dataset "${EVAL_DATA}" \
  --input_key question --label_key answer --output_key answer \
  --prompt_split train --eval_split test \
  --prompt_max_len "${PROMPT_MAX_LEN}" \
  --context_max_len "${CONTEXT_MAX_LEN}" \
  --generate_max_len "${GENERATE_MAX_LEN}" \
  --stride "${STRIDE}" \
  --n_samples_per_prompt "${N_SAMPLES_PER_PROMPT}" \
  --rollout_batch_size "${ROLLOUT_BATCH_SIZE}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --micro_train_batch_size "${MICRO_TRAIN_BATCH_SIZE}" \
  --micro_rollout_batch_size "${MICRO_ROLLOUT_BATCH_SIZE}" \
  --micro_reward_batch_size "${MICRO_REWARD_BATCH_SIZE}" \
  --max_samples "${MAX_SAMPLES}" \
  --num_episodes "${NUM_EPISODES}" \
  --max_epochs "${MAX_EPOCHS}" \
  \
  --actor_num_nodes 1 --actor_num_gpus_per_node "${ACTOR_GPUS}" \
  --critic_num_nodes 1 --critic_num_gpus_per_node "${CRITIC_GPUS}" \
  --ref_num_nodes 1 --ref_num_gpus_per_node "${REF_GPUS}" \
  --reward_num_nodes 1 --reward_num_gpus_per_node "${REWARD_GPUS}" \
  \
  --advantage_estimator rloo --init_kl_coef 0.0 --kl_estimator k2 \
  --temperature 0.6 --top_p 1.0 \
  --zero_stage 3 --lr_warmup_ratio 0.03 --critic_lr_warmup_ratio 0.0 \
  --seed 43 \
  --eval_steps "${EVAL_STEPS}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}" \
  --logging_steps 10 \
  --save_steps "${SAVE_STEPS}" --save_even_count "${SAVE_EVEN_COUNT}" --save_hf_ckpt \
  --use_tensorboard "${TB_DIR}" \
  --save_path "${SAVE_PATH}" --ckpt_path "${SAVE_PATH}/ckpt" \
  --wandb_run_name "${RUN_NAME}" \
  2>&1 | tee "${RUN_DIR}/train.log"

# ====================================================================
# POST-TRAINING TWO-ROUND VLLM EVAL  (16k first pass -> 32k retry)
# Runs the same eval pipeline as G1/G2/baseline so checkpoint accuracy
# is directly comparable across methods.
# ====================================================================
ray stop --force 2>/dev/null || true

echo ""
echo "──────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  TRAINING FINISHED"
echo "  Logs:        ${RUN_DIR}/train.log"
echo "  TensorBoard: ${TB_DIR}"
echo "  Checkpoints: ${SAVE_PATH}"

if [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
  # Free teacher GPUs (vLLM eval will grab all 8 cards via TP=VLLM_TP_SIZE).
  if (( ${#TEACHER_PIDS[@]} > 0 )); then
    echo "[post-eval] stopping ${#TEACHER_PIDS[@]} teacher workers to free GPU memory..."
    for _pid in "${TEACHER_PIDS[@]}"; do
      kill "${_pid}" 2>/dev/null || true
    done
    for _pid in "${TEACHER_PIDS[@]}"; do
      wait "${_pid}" 2>/dev/null || true
    done
    TEACHER_PIDS=()
  fi

  echo ""
  echo "===== Running two-round 16k/32k completion eval ====="
  # Use export instead of inline VAR=val so child shell's `set -u` can't
  # surprise-fail on lookup; also mirrors how run_G1_rebase.sh does it.
  export RUN_DIR MODEL_PATH="${SAVE_PATH}"
  export TEACHER_VENV ANALYSIS_VENV
  export TEACHER_PYTHON_BIN ANALYSIS_PYTHON_BIN
  export MODEL_CUDA_VISIBLE_DEVICES VLLM_TP_SIZE
  export POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
  export FIRST_PASS_MAX_NEW_TOKENS SECOND_PASS_MAX_NEW_TOKENS
  export POST_EVAL_TEMPERATURE POST_EVAL_TOP_P
  export POST_EVAL_REPETITION_PENALTY POST_EVAL_BEST_OF_N
  export VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
  export EVAL_DATA
  bash "${REPO_ROOT}/scripts/supplement_2rounds/G3.sh"
fi

echo "──────────────────────────────────────────────────"
echo "G3 online teacher run completed at $(date)" > "${RUN_DIR}/status.txt"
echo "[done] logs: ${RUN_DIR}"


