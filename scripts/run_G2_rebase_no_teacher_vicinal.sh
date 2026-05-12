#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  G2 Rebase — no-teacher VICINAL distribution (single-node 8 GPU)║
# ║  cf_l1oo reward · vicinal target · 16k/32k two-round post-eval ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Aligned with scripts/run_G2_rebase_2node_once.sh on every knob that
# isn't teacher-specific. Key differences from 2node_once:
#
#   - SINGLE node, 8 GPUs total, no DLC autodetection / no SSH 2-node path.
#   - NO online teacher: no vLLM teacher fleet, no remote distribution
#     target, no shared teacher cache. cf_target_mode is `vicinal`
#     (estimate target distribution from CF_TARGET_NUM_REFS=8 nearest
#     reference samples) instead of `teacher`. cf_teacher_lambda is 0.
#   - actor + critic + ref + reward all colocate on the same 8 GPUs
#     (default ACTOR_GPUS=4 / CRITIC_GPUS=4, ZeRO-3, ref/reward share
#     actor/critic GPUs via colocate_actor_ref / colocate_critic_reward).
#
# Aligned-with-2node_once items (do not regress these without also
# updating the 2-node launcher):
#
#   - all training hyperparameters: N_SAMPLES_PER_PROMPT, ROLLOUT_BATCH_SIZE,
#     MICRO_*, PROMPT_MAX_LEN, CONTEXT_MAX_LEN, GENERATE_MAX_LEN, STRIDE,
#     TARGET_STEPS, ACTOR_LR, init_kl_coef, kl_estimator, advantage_estimator,
#     temperature, top_p, seed, lr_warmup_ratio, zero_stage, save_steps,
#     save_even_count, logging_steps, gradient_checkpointing, use_kl_loss,
#     use_whitening, use_tensorboard, save_hf_ckpt, distribution_reward_type,
#     feature_map_type, rff_*, cf_num_freqs, cf_sigma, cf_seed, cf_alpha,
#     cf_beta, cf_reward_scale, embed_method, critic_sequence_level.
#   - in-training eval is DISABLED by default (EVAL_STEPS=-1) for the
#     same reason as 2node: GENERATE_MAX_LEN=8 means in-training eval
#     generates 8 tokens, which is meaningless on AOPS and clutters TB
#     with a flat-zero curve. Trust the post-train two-round eval.
#     NOTE: only eval_steps == -1 is treated as "disabled" by the
#     trainer (ebft_trainer.py: eval_steps == -1 -> float('inf')). A
#     finite-but-large value like 999999 would still trigger the
#     trainer's "initial evaluation at step 0" path, which is gated on
#     `not math.isinf(eval_steps)` and runs the full eval dataloader
#     through samples_generator. With ZeRO-3 colocate_actor_ref, an
#     unbalanced round-robin batch distribution across actor ranks
#     deadlocks NCCL all-gather. Use -1, not a big number.
#   - post-train eval is ENABLED by default (EVAL_AFTER_TRAIN=true) and
#     goes through scripts/supplement_2rounds/G2.sh (the 1-node sibling
#     of G2_2node.sh) at 16k first pass + 32k retry pass.
#   - archive RUN_DIR to ARCHIVE_OUTPUT_ROOT after a successful run, so
#     OSS keeps a long-term record without us having to remember to mv.
#     Shared teacher cache archive is skipped here (we have no teacher).
#   - NCCL_P2P_LEVEL=NVL / NCCL_NET_GDR_DISABLE=1 (single-node makes the
#     GDR knob redundant, but keeping the same set guarantees that any
#     run-to-run delta is in training code, not NCCL config).
#   - whole-script log redirect via `exec > >(tee -a ...)` so any error
#     (including the post-eval phase) lands in one file under RUN_DIR.
#
# Usage:
#   bash scripts/run_G2_rebase_no_teacher_vicinal.sh
#   TARGET_STEPS=500 bash scripts/run_G2_rebase_no_teacher_vicinal.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ACTOR_GPUS=2 CRITIC_GPUS=2 \
#     bash scripts/run_G2_rebase_no_teacher_vicinal.sh
#   EVAL_AFTER_TRAIN=false bash scripts/run_G2_rebase_no_teacher_vicinal.sh
set -euo pipefail

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "${csv}" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${csv}"
}

# --------------------------------------------------------------------
# 0) GPU ASSIGNMENT — single node, 8 GPUs, no teacher
# --------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
CRITIC_GPUS="${CRITIC_GPUS:-4}"
REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
REF_NUM_NODES="${REF_NUM_NODES:-1}"
REWARD_NUM_NODES="${REWARD_NUM_NODES:-1}"

# --------------------------------------------------------------------
# 1) PATHS — model / data / venvs
# --------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/mnt/data/ebft-distribution-new/code}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/gemma-4-E4B}"
DEFAULT_TRAIN_DATA="/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
FALLBACK_LOCAL_DATA="${FALLBACK_LOCAL_DATA:-}"
TRAIN_DATA="${TRAIN_DATA:-${DEFAULT_TRAIN_DATA}}"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"

# Venvs live on local ext4 (ossfs2 can't host venv symlinks). See
# scripts/setup_env.sh for the bootstrap that creates and snapshots them.
#   STUDENT_VENV  ─ training (Ray/DeepSpeed/OpenRLHF/transformers)
#   TEACHER_VENV  ─ vLLM serving stack; here used ONLY by the post-train
#                   2-round eval (supplement_2rounds/G2.sh shells out to
#                   ${TEACHER_PYTHON_BIN} for vLLM generation). Keep this
#                   defined even though we don't run a teacher fleet, so
#                   the post-eval phase can find vllm_generate_progress.
STUDENT_VENV="${STUDENT_VENV:-/mnt/workspace/venvs/.venv}"
TEACHER_VENV="${TEACHER_VENV:-/mnt/workspace/venvs/.teacherVenv}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
ANALYSIS_VENV="${ANALYSIS_VENV:-${STUDENT_VENV}}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"

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
# 2) TRAINING KNOBS — aligned with run_G2_rebase_2node_once.sh
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
DEFAULT_MAX_SAMPLES="$((TARGET_STEPS * TRAIN_BATCH_SIZE / N_SAMPLES_PER_PROMPT / NUM_EPISODES / MAX_EPOCHS))"
MAX_SAMPLES="${MAX_SAMPLES:-${DEFAULT_MAX_SAMPLES}}"

# CF / RFF reward knobs (identical to 2node_once).
FEATURE_MAP_TYPE="${FEATURE_MAP_TYPE:-identity}"
RFF_NUM_FEATURES="${RFF_NUM_FEATURES:-128}"
RFF_SIGMA="${RFF_SIGMA:-1.0}"
RFF_SEED="${RFF_SEED:-43}"

CF_NUM_FREQS="${CF_NUM_FREQS:-128}"
CF_SIGMA="${CF_SIGMA:-1.0}"
CF_SEED="${CF_SEED:-43}"
CF_ALPHA="${CF_ALPHA:-0.5}"
CF_BETA="${CF_BETA:-0.5}"
CF_REWARD_SCALE="${CF_REWARD_SCALE:-1.0}"

# Vicinal target — the only non-teacher knobs that differ from 2node_once.
# CF_TARGET_NUM_REFS controls how many nearest reference samples are used
# to estimate the local target distribution; std + seed perturb the kernel.
CF_TARGET_NUM_REFS="${CF_TARGET_NUM_REFS:-8}"
CF_TARGET_STD="${CF_TARGET_STD:-0.05}"
CF_TARGET_SEED="${CF_TARGET_SEED:-43}"
# No teacher distribution mixed in. Keep cf_teacher_n_samples identical
# to 2node so the trainer's argparse contract stays uniform; the value
# is unused when cf_teacher_lambda == 0.
CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.0}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-${N_SAMPLES_PER_PROMPT}}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-0}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
GLOBAL_SEED="${GLOBAL_SEED:-43}"

# --------------------------------------------------------------------
# 3) IN-TRAINING EVAL — disabled by default (mirrors 2node_once)
# --------------------------------------------------------------------
# GENERATE_MAX_LEN=8 makes in-training eval generate 8 tokens, which is
# meaningless on AOPS — every prompt ends up "no answer" / 0% acc and
# clutters TensorBoard with a flat-zero curve. The post-training 2-round
# vLLM eval (16k first pass + 32k retry) is what we trust for accuracy.
# Disabled by passing eval_steps=-1 (the trainer's contract for "off");
# a finite-but-large value still fires the step-0 initial eval and
# deadlocks NCCL all-gather under ZeRO-3 colocate (see header comment).
# To re-enable in-training eval, ALSO bump generate_max_len so the eval
# is meaningful:
#     EVAL_STEPS=200 EVAL_GENERATE_MAX_LEN=2048 bash $0
EVAL_STEPS="${EVAL_STEPS:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"

# --------------------------------------------------------------------
# 4) POST-TRAIN TWO-ROUND EVAL — aligned with 2node_once + supplement_2rounds/G2.sh
# --------------------------------------------------------------------
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
RUN_TWO_ROUND_EVAL="${RUN_TWO_ROUND_EVAL:-${EVAL_AFTER_TRAIN}}"
# 1-node version of supplement_2rounds. The 2-node version (G2_2node.sh)
# expects an external Ray cluster; we run on a single host.
POST_EVAL_SCRIPT="${POST_EVAL_SCRIPT:-${REPO_ROOT}/scripts/supplement_2rounds/G2.sh}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
# vLLM concurrency knobs — see G2_2node.sh for the HOL-blocking rationale
# behind {256, 256}. Same defaults here so accuracy / wall-time match.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-256}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
# Single-host: vLLM eval just reuses all visible training GPUs at TP=N.
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")}"
POST_EVAL_TAG="${POST_EVAL_TAG:-post_train}"

# --------------------------------------------------------------------
# 5) ARCHIVE — mirror 2node_once. No shared teacher cache (no teacher).
# --------------------------------------------------------------------
ARCHIVE_OUTPUTS_AFTER_RUN="${ARCHIVE_OUTPUTS_AFTER_RUN:-true}"
ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g2_0.99}"

# --------------------------------------------------------------------
# 6) ENV / RUN DIR
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
# collectives across actor + critic on the same 8 GPUs via NVLink) AND
# the post-training vLLM stage. Single-node so NCCL_NET_GDR_DISABLE is
# nominally redundant, but we keep the same set as 2node_once so any
# run-to-run delta is purely training code, not NCCL env config.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
[[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE
export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

# Make sure the student venv's binaries (notably `ray`, plus python/pip)
# are first on PATH. Without this the launcher fails on a fresh DLC pod
# (ray is only inside the venv we just installed via setup_env.sh, but
# that doesn't activate the venv -- it just creates it). On DSW where
# the user typically already has the venv on PATH this is a no-op.
if [[ -d "${STUDENT_VENV}/bin" ]]; then
  export PATH="${STUDENT_VENV}/bin:${PATH}"
fi

RUN_NAME="${RUN_NAME:-g2_no_teacher_vicinal_8gpu_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
POST_EVAL_LOG_DIR="${POST_EVAL_LOG_DIR:-${RUN_DIR}/supplement_logs}"
SCRIPT_NAME="$(basename "$0" .sh)"
SCRIPT_LOG_PATH="${SCRIPT_LOG_PATH:-${RUN_DIR}/${SCRIPT_NAME}.log}"
SCRIPT_SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
LAUNCHER_SNAPSHOT_PATH="${RUN_DIR}/launcher_snapshot.sh"
TRAIN_CONFIG_SNAPSHOT_PATH="${RUN_DIR}/run_context.env"
TRAIN_CONFIG_SUMMARY_PATH="${RUN_DIR}/run_summary.txt"
mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${TB_DIR}" "${POST_EVAL_LOG_DIR}"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1

# --------------------------------------------------------------------
# 7) SANITY CHECK
# --------------------------------------------------------------------
gpu_count="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"

if [[ ! -x "${STUDENT_PYTHON_BIN}" ]]; then
  echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"
  echo "        expected student env: ${STUDENT_VENV}"
  exit 1
fi

if [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
  if [[ ! -x "${TEACHER_PYTHON_BIN}" ]]; then
    echo "[ERROR] TEACHER_PYTHON_BIN not executable: ${TEACHER_PYTHON_BIN}"
    echo "        post-eval (supplement_2rounds/G2.sh) shells out to it for"
    echo "        vLLM generation. Either bootstrap teacher venv via"
    echo "        scripts/setup_env.sh, or set RUN_TWO_ROUND_EVAL=false."
    exit 1
  fi
  if [[ ! -f "${POST_EVAL_SCRIPT}" ]]; then
    echo "[ERROR] POST_EVAL_SCRIPT not found: ${POST_EVAL_SCRIPT}"
    exit 1
  fi
fi

if (( ACTOR_GPUS + CRITIC_GPUS > gpu_count )); then
  echo "[ERROR] ACTOR_GPUS(${ACTOR_GPUS}) + CRITIC_GPUS(${CRITIC_GPUS}) > GPU count(${gpu_count})"
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

if [[ "${TRAIN_DATA}" == "${DEFAULT_TRAIN_DATA}" && -n "${FALLBACK_LOCAL_DATA}" && ! -e "${TRAIN_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
  echo "[WARN] TRAIN_DATA default not found, fallback to ${FALLBACK_LOCAL_DATA}"
  TRAIN_DATA="${FALLBACK_LOCAL_DATA}"
fi
if [[ "${EVAL_DATA}" == "${DEFAULT_EVAL_DATA}" && -n "${FALLBACK_LOCAL_DATA}" && ! -e "${EVAL_DATA}" && -f "${FALLBACK_LOCAL_DATA}" ]]; then
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
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

# --------------------------------------------------------------------
# 8) RUN METADATA — snapshot launcher + key vars under RUN_DIR
# --------------------------------------------------------------------
write_run_metadata() {
  local vars=(
    RUN_NAME OUTPUT_ROOT RUN_DIR SAVE_PATH TB_DIR POST_EVAL_LOG_DIR
    CUDA_VISIBLE_DEVICES MODEL_CUDA_VISIBLE_DEVICES
    ACTOR_GPUS CRITIC_GPUS REF_GPUS REWARD_GPUS
    ACTOR_NUM_NODES CRITIC_NUM_NODES REF_NUM_NODES REWARD_NUM_NODES
    REPO_ROOT MODEL_PATH TRAIN_DATA EVAL_DATA STUDENT_VENV TEACHER_VENV
    N_SAMPLES_PER_PROMPT ROLLOUT_BATCH_SIZE TRAIN_BATCH_SIZE
    MICRO_TRAIN_BATCH_SIZE MICRO_ROLLOUT_BATCH_SIZE MICRO_REWARD_BATCH_SIZE
    PROMPT_MAX_LEN CONTEXT_MAX_LEN GENERATE_MAX_LEN STRIDE
    NUM_EPISODES MAX_EPOCHS TARGET_STEPS MAX_SAMPLES
    FEATURE_MAP_TYPE RFF_NUM_FEATURES RFF_SIGMA RFF_SEED
    CF_NUM_FREQS CF_SIGMA CF_SEED CF_ALPHA CF_BETA CF_REWARD_SCALE
    CF_TARGET_NUM_REFS CF_TARGET_STD CF_TARGET_SEED
    CF_TEACHER_LAMBDA CF_TEACHER_N_SAMPLES
    ACTOR_LR CRITIC_LR CRITIC_LR_HEAD TEMPERATURE TOP_P GLOBAL_SEED
    EVAL_STEPS EVAL_MAX_SAMPLES EVAL_GENERATE_MAX_LEN SAVE_STEPS SAVE_EVEN_COUNT
    EVAL_AFTER_TRAIN RUN_TWO_ROUND_EVAL POST_EVAL_SCRIPT
    POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
    FIRST_PASS_MAX_NEW_TOKENS SECOND_PASS_MAX_NEW_TOKENS
    POST_EVAL_TEMPERATURE POST_EVAL_TOP_P POST_EVAL_REPETITION_PENALTY POST_EVAL_BEST_OF_N
    VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
    VLLM_TP_SIZE POST_EVAL_TAG POST_EVAL_LOG_DIR
    ARCHIVE_OUTPUTS_AFTER_RUN ARCHIVE_OUTPUT_ROOT
  )

  cp -f "${SCRIPT_SOURCE_PATH}" "${LAUNCHER_SNAPSHOT_PATH}" 2>/dev/null || true

  {
    echo "# Auto-generated run context snapshot"
    echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    for name in "${vars[@]}"; do
      printf "%s=%q\n" "${name}" "${!name-}"
    done
  } > "${TRAIN_CONFIG_SNAPSHOT_PATH}"

  {
    echo "run_name: ${RUN_NAME}"
    echo "run_dir: ${RUN_DIR}"
    echo "save_path: ${SAVE_PATH}"
    echo "model_path: ${MODEL_PATH}"
    echo "train_data: ${TRAIN_DATA}"
    echo "eval_data: ${EVAL_DATA}"
    echo "cf_target_mode: vicinal"
    echo "cf_target_num_refs: ${CF_TARGET_NUM_REFS}"
    echo "teacher_in_reward: false"
    echo "train_batch_size: ${TRAIN_BATCH_SIZE}"
    echo "target_steps: ${TARGET_STEPS}"
    echo "max_samples: ${MAX_SAMPLES}"
    echo "post_eval_script: ${POST_EVAL_SCRIPT}"
    echo "post_eval_max_samples: ${POST_EVAL_MAX_SAMPLES}"
    echo "archive_output_root: ${ARCHIVE_OUTPUT_ROOT}"
    echo "launcher_snapshot: ${LAUNCHER_SNAPSHOT_PATH}"
  } > "${TRAIN_CONFIG_SUMMARY_PATH}"
}

write_run_metadata

# --------------------------------------------------------------------
# 9) ARCHIVE HELPERS
# --------------------------------------------------------------------
archive_run_outputs() {
  local target_root="$1"
  local target_dir
  local old_run_dir="${RUN_DIR}"

  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[archive] skip: RUN_DIR not found: ${RUN_DIR}"
    return 0
  fi

  mkdir -p "${target_root}"
  target_dir="${target_root}/$(basename "${RUN_DIR}")"
  if [[ -e "${target_dir}" ]]; then
    target_dir="${target_dir}_$(date +%m%d_%H%M%S)"
  fi

  echo "[archive] moving run outputs to: ${target_dir}"
  mv "${RUN_DIR}" "${target_dir}"
  RUN_DIR="${target_dir}"
  SAVE_PATH="${RUN_DIR}/model"
  TB_DIR="${RUN_DIR}/tensorboard"
  if [[ -n "${POST_EVAL_LOG_DIR}" && "${POST_EVAL_LOG_DIR}" == "${old_run_dir}"* ]]; then
    POST_EVAL_LOG_DIR="${RUN_DIR}${POST_EVAL_LOG_DIR#${old_run_dir}}"
  fi
  TRAIN_CONFIG_SNAPSHOT_PATH="${RUN_DIR}/run_context.env"
  TRAIN_CONFIG_SUMMARY_PATH="${RUN_DIR}/run_summary.txt"
  LAUNCHER_SNAPSHOT_PATH="${RUN_DIR}/launcher_snapshot.sh"
  write_run_metadata
}

write_final_status() {
  {
    echo "# Auto-generated final status"
    echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf "TRAIN_RC=%q\n" "${TRAIN_RC:-0}"
    printf "EVAL_RC=%q\n" "${EVAL_RC:-0}"
    printf "ARCHIVE_RC=%q\n" "${ARCHIVE_RC:-0}"
    printf "FINAL_RC=%q\n" "${FINAL_RC:-0}"
    printf "RUN_DIR=%q\n" "${RUN_DIR:-}"
    printf "SAVE_PATH=%q\n" "${SAVE_PATH:-}"
    printf "POST_EVAL_LOG_DIR=%q\n" "${POST_EVAL_LOG_DIR:-}"
  } > "${RUN_DIR}/final_status.env"
}

# --------------------------------------------------------------------
# 10) BANNER
# --------------------------------------------------------------------
echo "========== AOPS G2 no-teacher VICINAL run =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "GPUs:                       ${CUDA_VISIBLE_DEVICES} (count=${gpu_count})"
echo "Actor/Critic GPUs:          ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "Ref/Reward GPUs (colocate): ${REF_GPUS}/${REWARD_GPUS}"
echo "Model:                      ${MODEL_PATH}"
echo "Train data:                 ${TRAIN_DATA}"
echo "Eval data:                  ${EVAL_DATA}"
echo "Student python:             ${STUDENT_PYTHON_BIN}"
echo "Teacher python (vLLM only): ${TEACHER_PYTHON_BIN}"
echo "distribution_reward:        cf_l1oo"
echo "cf_target_mode:             vicinal"
echo "cf_target_num_refs:         ${CF_TARGET_NUM_REFS}"
echo "cf_target_std:              ${CF_TARGET_STD}"
echo "cf_target_seed:             ${CF_TARGET_SEED}"
echo "cf_teacher_lambda:          ${CF_TEACHER_LAMBDA} (no teacher mix-in)"
echo "teacher_in_reward:          false"
echo "target_steps:               ${TARGET_STEPS}"
echo "max_samples:                ${MAX_SAMPLES}"
echo "eval_steps (in-train):      ${EVAL_STEPS}"
echo "save_steps:                 ${SAVE_STEPS}"
echo "run_two_round_eval:         ${RUN_TWO_ROUND_EVAL}"
echo "post_eval_script:           ${POST_EVAL_SCRIPT}"
echo "first/second pass tokens:   ${FIRST_PASS_MAX_NEW_TOKENS}/${SECOND_PASS_MAX_NEW_TOKENS}"
echo "vllm tp size:               ${VLLM_TP_SIZE}"
echo "archive_outputs:            ${ARCHIVE_OUTPUTS_AFTER_RUN} -> ${ARCHIVE_OUTPUT_ROOT}"
echo "===================================================="

# --------------------------------------------------------------------
# 11) TRAIN
# --------------------------------------------------------------------
ray stop --force 2>/dev/null || true
sleep 2
cd "${REPO_ROOT}"

TRAIN_RC=0
EVAL_RC=0
ARCHIVE_RC=0

train_cmd=(
  "${STUDENT_PYTHON_BIN}" -m openrlhf.cli.train_ebft_ray
  --bf16
  --flash_attn
  --pretrain_mode
  --no_chat_template
  --disable_ds_ckpt
  --colocate_actor_ref
  --colocate_critic_reward
  --gradient_checkpointing
  --use_kl_loss
  --use_whitening
  --distribution_reward_type cf_l1oo
  --feature_map_type "${FEATURE_MAP_TYPE}"
  --rff_num_features "${RFF_NUM_FEATURES}"
  --rff_sigma "${RFF_SIGMA}"
  --rff_seed "${RFF_SEED}"
  --cf_num_freqs "${CF_NUM_FREQS}"
  --cf_sigma "${CF_SIGMA}"
  --cf_seed "${CF_SEED}"
  --cf_alpha "${CF_ALPHA}"
  --cf_beta "${CF_BETA}"
  --cf_reward_scale "${CF_REWARD_SCALE}"
  --cf_target_mode vicinal
  --cf_target_num_refs "${CF_TARGET_NUM_REFS}"
  --cf_target_std "${CF_TARGET_STD}"
  --cf_target_seed "${CF_TARGET_SEED}"
  --cf_teacher_lambda "${CF_TEACHER_LAMBDA}"
  --cf_teacher_n_samples "${CF_TEACHER_N_SAMPLES}"
  --embed_method last_token
  --critic_sequence_level last_token
  --critic_learning_rate "${CRITIC_LR}"
  --critic_lr_head "${CRITIC_LR_HEAD}"
  --pretrain "${MODEL_PATH}"
  --critic_pretrain "${MODEL_PATH}"
  --prompt_data "${TRAIN_DATA}"
  --eval_dataset "${EVAL_DATA}"
  --input_key question
  --label_key answer
  --output_key answer
  --prompt_split train
  --eval_split test
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
  --actor_num_nodes "${ACTOR_NUM_NODES}"
  --actor_num_gpus_per_node "${ACTOR_GPUS}"
  --critic_num_nodes "${CRITIC_NUM_NODES}"
  --critic_num_gpus_per_node "${CRITIC_GPUS}"
  --ref_num_nodes "${REF_NUM_NODES}"
  --ref_num_gpus_per_node "${REF_GPUS}"
  --reward_num_nodes "${REWARD_NUM_NODES}"
  --reward_num_gpus_per_node "${REWARD_GPUS}"
  --advantage_estimator rloo
  --init_kl_coef 0.0
  --kl_estimator k2
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --actor_learning_rate "${ACTOR_LR}"
  --zero_stage 3
  --lr_warmup_ratio 0.03
  --critic_lr_warmup_ratio 0.0
  --seed "${GLOBAL_SEED}"
  --eval_steps "${EVAL_STEPS}"
  --eval_max_samples "${EVAL_MAX_SAMPLES}"
  --eval_generate_max_len "${EVAL_GENERATE_MAX_LEN}"
  --logging_steps 10
  --save_steps "${SAVE_STEPS}"
  --save_even_count "${SAVE_EVEN_COUNT}"
  --save_hf_ckpt
  --use_tensorboard "${TB_DIR}"
  --save_path "${SAVE_PATH}"
  --ckpt_path "${SAVE_PATH}/ckpt"
  --wandb_run_name "${RUN_NAME}"
)

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${train_cmd[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e

if (( TRAIN_RC != 0 )); then
  echo "[ERROR] training failed with exit code ${TRAIN_RC}"
fi

# --------------------------------------------------------------------
# 12) POST-TRAINING TWO-ROUND EVAL (only if training succeeded)
# --------------------------------------------------------------------
ray stop --force 2>/dev/null || true

echo ""
echo "──────────────────────────────────────────────────"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')  TRAINING FINISHED (rc=${TRAIN_RC})"
echo "  Logs:        ${RUN_DIR}/train.log"
echo "  TensorBoard: ${TB_DIR}"
echo "  Checkpoints: ${SAVE_PATH}"
echo "──────────────────────────────────────────────────"

if [[ "${RUN_TWO_ROUND_EVAL}" == "true" && "${TRAIN_RC}" -eq 0 ]]; then
  echo ""
  echo "===== Running two-round 16k/32k completion eval (single-node vLLM) ====="
  set +e
  RUN_DIR="${RUN_DIR}" \
  MODEL_PATH="${SAVE_PATH}" \
  EVAL_DATA="${EVAL_DATA}" \
  TEACHER_VENV="${TEACHER_VENV}" \
  TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN}" \
  ANALYSIS_VENV="${ANALYSIS_VENV}" \
  ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN}" \
  MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
  VLLM_TP_SIZE="${VLLM_TP_SIZE}" \
  POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES}" \
  POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN}" \
  FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS}" \
  SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS}" \
  POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE}" \
  POST_EVAL_TOP_P="${POST_EVAL_TOP_P}" \
  POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY}" \
  POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
  VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE}" \
  VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING}" \
  VLLM_SEED="${VLLM_SEED}" \
  EVAL_TAG="${POST_EVAL_TAG}" \
  LOG_DIR="${POST_EVAL_LOG_DIR}" \
  bash "${POST_EVAL_SCRIPT}"
  EVAL_RC=$?
  set -e
  if (( EVAL_RC != 0 )); then
    echo "[ERROR] post-eval (two-round vLLM) failed with exit code ${EVAL_RC}"
  fi
elif [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
  echo "[post-eval] skipped because training failed (TRAIN_RC=${TRAIN_RC})"
  EVAL_RC=0
else
  echo "[post-eval] disabled (RUN_TWO_ROUND_EVAL=${RUN_TWO_ROUND_EVAL})"
  EVAL_RC=0
fi

# --------------------------------------------------------------------
# 13) ARCHIVE
# --------------------------------------------------------------------
if [[ "${ARCHIVE_OUTPUTS_AFTER_RUN}" == "true" && "${TRAIN_RC}" -eq 0 ]]; then
  set +e
  archive_run_outputs "${ARCHIVE_OUTPUT_ROOT}"
  ARCHIVE_RC=$?
  set -e
  if (( ARCHIVE_RC != 0 )); then
    echo "[ERROR] archive_run_outputs failed with exit code ${ARCHIVE_RC}"
  fi
elif [[ "${ARCHIVE_OUTPUTS_AFTER_RUN}" == "true" ]]; then
  echo "[archive] skipped because training failed (TRAIN_RC=${TRAIN_RC})"
fi

# --------------------------------------------------------------------
# 14) FINAL STATUS
# --------------------------------------------------------------------
if (( TRAIN_RC != 0 )); then
  FINAL_RC="${TRAIN_RC}"
elif (( EVAL_RC != 0 )); then
  FINAL_RC="${EVAL_RC}"
elif (( ARCHIVE_RC != 0 )); then
  FINAL_RC="${ARCHIVE_RC}"
else
  FINAL_RC=0
fi

write_final_status

echo ""
echo "──────────────────────────────────────────────────"
echo "G2 no-teacher VICINAL run finished at $(date)"
echo "  TRAIN_RC=${TRAIN_RC}  EVAL_RC=${EVAL_RC}  ARCHIVE_RC=${ARCHIVE_RC}  FINAL_RC=${FINAL_RC}"
echo "  RUN_DIR: ${RUN_DIR}"
echo "──────────────────────────────────────────────────"
echo "G2 no-teacher VICINAL run completed at $(date) (rc=${FINAL_RC})" > "${RUN_DIR}/status.txt"
exit "${FINAL_RC}"
