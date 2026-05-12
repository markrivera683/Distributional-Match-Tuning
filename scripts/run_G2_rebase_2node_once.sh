#!/usr/bin/env bash
# Two-node launcher for the G2 rebase training recipe.
#
# G2 vs G3:
#   G2: cf_l1oo + frozen critic head (critic_lr_head=0). NO feature adapter,
#       NO EMA, NO classifier loss, NO direct discrepancy, NO diversity/
#       alignment reward, NO ce_loss_coef. Pure cf_l1oo with online teacher.
#   G3: G2 + enable_ema + feature_adapter + trainable critic head + classifier
#       loss + direct discrepancy + diversity/alignment reward + ce_loss.
#
# Same DLC / DSW-SSH / single-node mode autodetection as run_G3_rebase_2node_once.sh.
# Same 6-teacher / 2-student GPU split per node (head 6t+2s, worker 6t+2s).
# Post-train eval uses head node only at TP=8 via supplement_2rounds/G2.sh.
#
# Assumptions:
# 1) Run this script on the head node (DSW SSH) or it auto-detects DLC mode.
# 2) Passwordless SSH from head -> worker is available (DSW only).
# 3) Each node has 8 GPUs.
# 4) Teacher uses GPUs 0-5 on both nodes (12 GPUs total, 12 vLLM workers @ TP=1).
# 5) Student uses GPUs 6-7 on both nodes (4 GPUs total):
#      - actor/ref world:   1 node x 2 GPUs (head)
#      - critic/reward:     1 node x 2 GPUs (worker)
set -euo pipefail

count_csv_items() {
  local csv="${1// /}"
  if [[ -z "$csv" ]]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<<"${csv}"
}

resolve_host_ip() {
  local host="$1"
  local ip=""
  local waited=0
  local resolve_wait_seconds="${HOST_RESOLVE_WAIT_SECONDS:-60}"
  local resolve_retry_seconds="${HOST_RESOLVE_RETRY_SECONDS:-2}"

  if [[ "${host}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    echo "${host}"
    return 0
  fi

  while true; do
    ip="$(getent ahostsv4 "${host}" | awk 'NR==1 {print $1}')"
    if [[ -n "${ip}" ]]; then
      echo "${ip}"
      return 0
    fi
    if (( waited >= resolve_wait_seconds )); then
      echo "[ERROR] failed to resolve IPv4 for host: ${host}" >&2
      exit 1
    fi
    sleep "${resolve_retry_seconds}"
    waited=$((waited + resolve_retry_seconds))
  done
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ERROR] required command not found: ${cmd}"
    exit 1
  fi
}

build_teacher_urls() {
  local host_ip="$1"
  local base_port="$2"
  local worker_count="$3"
  local urls=""
  local port
  local i
  for (( i=0; i<worker_count; i++ )); do
    port=$((base_port + i))
    [[ -n "${urls}" ]] && urls+=","
    urls+="http://${host_ip}:${port}/v1"
  done
  echo "${urls}"
}

HEAD_NODE="${HEAD_NODE:-}"
WORKER_NODE="${WORKER_NODE:-}"
HEAD_NODE_IP="${HEAD_NODE_IP:-}"
WORKER_NODE_IP="${WORKER_NODE_IP:-}"
WORKER_SSH_HOST="${WORKER_SSH_HOST:-}"
SSH_USER="${SSH_USER:-}"
SSH_OPTS="${SSH_OPTS:-}"

# Single-node fallback: if neither HEAD_NODE nor WORKER_NODE is set,
# treat this as a single-node run on the current host. This makes the
# script work uniformly across:
#   - DSW 2-node SSH setup (set HEAD_NODE / WORKER_NODE explicitly)
#   - DLC 1-pod single-node deployment (no env vars needed)
# In single-node mode actor + critic + teacher all share the same 8
# GPUs (default split: teacher 0-5, actor 6, critic 7), no SSH or
# cross-node ray cluster is involved.
# =====================================================================
# DEPLOYMENT MODE DETECTION
# =====================================================================
# Three deployment modes:
#   - DSW 2-node SSH:  user explicitly sets HEAD_NODE / WORKER_NODE,
#                       head pod uses ssh to bring up worker pod.
#   - DLC multi-pod:   PAI DLC starts master + worker pod with the same
#                       startup command and injects RANK / WORLD_SIZE /
#                       MASTER_ADDR per pod. Both pods run THIS launcher;
#                       RANK=0 is master (acts like head node, runs
#                       train job), RANK>0 is worker pod (joins ray
#                       cluster + starts local teacher workers + waits).
#                       SSH is NOT used.
#   - Single-node:     no HEAD_NODE / WORKER_NODE, no DLC env vars.
#                       Everything runs on the current host, actor +
#                       critic + teacher share 8 GPUs (G2-style).
#
# DLC env conventions: PAI's PyTorch Job operator sets RANK,
# WORLD_SIZE, MASTER_ADDR, MASTER_PORT (via the kubedl pytorchReplicaSpec).
# Some PAI flavors also set PET_NODE_RANK / PET_MASTER_ADDR (torchrun
# elastic). We accept whichever pair is non-empty.
# ---------------------------------------------------------------------
SINGLE_NODE_MODE="false"
DLC_MODE="false"
DLC_NODE_RANK=""
DLC_MASTER_ADDR=""
DLC_WORLD_SIZE="${WORLD_SIZE:-${PET_WORLD_SIZE:-1}}"

if [[ -n "${PET_NODE_RANK:-}" ]]; then
  DLC_MODE="true"
  DLC_NODE_RANK="${PET_NODE_RANK}"
  DLC_MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
elif [[ -n "${RANK:-}" && -n "${MASTER_ADDR:-}" && "${DLC_WORLD_SIZE:-1}" -gt 1 ]]; then
  DLC_MODE="true"
  DLC_NODE_RANK="${RANK}"
  DLC_MASTER_ADDR="${MASTER_ADDR}"
fi

if [[ "${DLC_MODE}" == "true" ]]; then
  if [[ -z "${DLC_MASTER_ADDR}" ]]; then
    echo "[ERROR] DLC mode detected (RANK=${DLC_NODE_RANK} WORLD_SIZE=${DLC_WORLD_SIZE})"
    echo "        but MASTER_ADDR is empty. Cannot route ray join target."
    exit 1
  fi
  if [[ -z "${HEAD_NODE}" && -z "${WORKER_NODE}" ]]; then
    HEAD_NODE="${DLC_MASTER_ADDR}"
    WORKER_NODE="dlc-rank-${DLC_NODE_RANK}-pod"   # symbolic; SSH path is never used
  fi
  echo "[INFO] DLC multi-pod mode: rank=${DLC_NODE_RANK} world_size=${DLC_WORLD_SIZE} master=${DLC_MASTER_ADDR}"
elif [[ -z "${HEAD_NODE}" && -z "${WORKER_NODE}" ]]; then
  SINGLE_NODE_MODE="true"
  HEAD_NODE="$(hostname)"
  WORKER_NODE="${HEAD_NODE}"
  echo "[INFO] single-node mode: HEAD_NODE=WORKER_NODE=${HEAD_NODE}"
elif [[ -z "${HEAD_NODE}" || -z "${WORKER_NODE}" ]]; then
  echo "[ERROR] HEAD_NODE / WORKER_NODE must both be set (DSW 2-node ssh)"
  echo "        or both be unset (single-node / DLC autodetect)."
  echo "Examples:"
  echo "  HEAD_NODE=node0 WORKER_NODE=node1 bash scripts/run_G2_rebase_2node_once.sh   # DSW ssh"
  echo "  bash scripts/run_G2_rebase_2node_once.sh                                     # single-node / DLC"
  exit 1
fi

# Convenience: any non-DSW path (DLC or single-node) means SSH bootstrap is
# disabled and worker_node command execution is replaced by either the
# worker pod itself (DLC) or by being colocated on head (single-node).
SKIP_SSH_BOOTSTRAP="false"
if [[ "${SINGLE_NODE_MODE}" == "true" || "${DLC_MODE}" == "true" ]]; then
  SKIP_SSH_BOOTSTRAP="true"
fi

# In DLC mode the master pod does NOT know the worker pod's IP up-front
# (the worker pod will reach IN to the master via $MASTER_ADDR). So we
# leave WORKER_NODE_IP empty in that case; nothing on the master code
# path actually consumes it once SSH is disabled.
HEAD_NODE_IP="${HEAD_NODE_IP:-$(resolve_host_ip "${HEAD_NODE}")}"
if [[ "${DLC_MODE}" == "true" ]]; then
  WORKER_NODE_IP="${WORKER_NODE_IP:-${HEAD_NODE_IP}}"   # placeholder
else
  WORKER_NODE_IP="${WORKER_NODE_IP:-$(resolve_host_ip "${WORKER_NODE}")}"
fi
WORKER_SSH_HOST="${WORKER_SSH_HOST:-${WORKER_NODE_IP}}"

if [[ -n "${SSH_USER}" ]]; then
  WORKER_SSH_TARGET="${SSH_USER}@${WORKER_SSH_HOST}"
else
  WORKER_SSH_TARGET="${WORKER_SSH_HOST}"
fi

CURRENT_HOSTNAME="$(hostname)"
CURRENT_HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
CURRENT_HOST_IPS="$(hostname -I 2>/dev/null || true)"

# Head-only check: only enforce in DSW ssh mode. In DLC mode RANK>0 pods
# legitimately run this script (they take the worker bootstrap branch).
if [[ "${DLC_MODE}" != "true" && "${SINGLE_NODE_MODE}" != "true" ]]; then
  if [[ "${CURRENT_HOSTNAME}" != "${HEAD_NODE}" && "${CURRENT_HOSTNAME_SHORT}" != "${HEAD_NODE}" ]]; then
    case " ${CURRENT_HOST_IPS} " in
      *" ${HEAD_NODE_IP} "*) ;;
      *)
        echo "[ERROR] this launcher must be executed only on the head node."
        echo "        current host: ${CURRENT_HOSTNAME}"
        echo "        expected head: ${HEAD_NODE} (${HEAD_NODE_IP})"
        echo "        worker node will be started remotely through ssh and must not run this script directly."
        exit 1
        ;;
    esac
  fi
fi

HEAD_TEACHER_CUDA_VISIBLE_DEVICES="${HEAD_TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
HEAD_STUDENT_CUDA_VISIBLE_DEVICES="${HEAD_STUDENT_CUDA_VISIBLE_DEVICES:-6,7}"
if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  # Single-node mode: no worker pod, no remote teacher/student.
  WORKER_TEACHER_CUDA_VISIBLE_DEVICES="${WORKER_TEACHER_CUDA_VISIBLE_DEVICES:-}"
  WORKER_STUDENT_CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-}"
else
  WORKER_TEACHER_CUDA_VISIBLE_DEVICES="${WORKER_TEACHER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
  WORKER_STUDENT_CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES:-6,7}"
fi

# GPU split between actor and critic on the student side. Defaults differ
# between modes:
#   - 2-node mode: ACTOR=2, CRITIC=2 (head node hosts actor 2 GPUs,
#     worker node hosts critic 2 GPUs; total student world = 4 GPUs)
#   - single-node mode: ACTOR=1, CRITIC=1 (everything on head's 2 student
#     GPUs, mirrors run_G2_rebase.sh layout)
if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  ACTOR_GPUS="${ACTOR_GPUS:-1}"
  CRITIC_GPUS="${CRITIC_GPUS:-1}"
else
  ACTOR_GPUS="${ACTOR_GPUS:-2}"
  CRITIC_GPUS="${CRITIC_GPUS:-2}"
fi
REF_GPUS="${REF_GPUS:-${ACTOR_GPUS}}"
REWARD_GPUS="${REWARD_GPUS:-${CRITIC_GPUS}}"

ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
CRITIC_NUM_NODES="${CRITIC_NUM_NODES:-1}"
REF_NUM_NODES="${REF_NUM_NODES:-1}"
REWARD_NUM_NODES="${REWARD_NUM_NODES:-1}"

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/data/models/qwen3.5-27b}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-qwen3.5-27b}"
TEACHER_BASE_PORT="${TEACHER_BASE_PORT:-8004}"
TEACHER_API_KEY="${TEACHER_API_KEY:-teacher-local}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
TEACHER_DTYPE="${TEACHER_DTYPE:-bfloat16}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-2048}"
TEACHER_MAX_NUM_SEQS="${TEACHER_MAX_NUM_SEQS:-64}"
TEACHER_MAX_BATCHED_TOKENS="${TEACHER_MAX_BATCHED_TOKENS:-32768}"
TEACHER_GPU_MEMORY_UTIL="${TEACHER_GPU_MEMORY_UTIL:-0.96}"
TEACHER_WAIT_SECONDS="${TEACHER_WAIT_SECONDS:-3600}"

# vLLM prefix caching for the teacher serving fleet.
#   Why default true:
#     CF_TEACHER_N_SAMPLES (default 8) means each unique prompt is asked
#     to the teacher 8 times in quick succession. With prefix caching ON,
#     only the FIRST of those 8 calls pays the prefill cost; the other 7
#     hit the prefix KV cache and skip prefill entirely. On a 27B teacher
#     this typically wins 3-5x on the per-prompt teacher latency for the
#     repeated-sample workload, since prefill dominates at our prompt
#     lengths (PROMPT_MAX_LEN=384 + system prompt, vs max_new=768 decode).
#
#     Same wins apply across training steps when the SAME prompt recurs
#     (same RLOO doc reused across episodes). The per-RUN_DIR persistent
#     teacher_cache_shared SQLite is the long-term cache; vLLM prefix
#     caching is the short-term warm cache inside one teacher serving
#     instance.
#
#   When to disable:
#     - You are debugging a vLLM correctness issue and want to rule out
#       prefix-cache bugs (vLLM 0.x has had a few in the past).
#     - You see KV-cache-related crashes with high TEACHER_MAX_NUM_SEQS.
#     Set TEACHER_ENABLE_PREFIX_CACHING=false to opt out.
TEACHER_ENABLE_PREFIX_CACHING="${TEACHER_ENABLE_PREFIX_CACHING:-true}"
if [[ "${TEACHER_ENABLE_PREFIX_CACHING}" == "true" ]]; then
  TEACHER_PREFIX_CACHING_FLAG="--enable-prefix-caching"
else
  TEACHER_PREFIX_CACHING_FLAG=""
fi

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
TEACHER_VLLM_BIN="${TEACHER_VLLM_BIN:-${TEACHER_VENV}/bin/vllm}"
STUDENT_PYTHON_BIN="${STUDENT_PYTHON_BIN:-${STUDENT_VENV}/bin/python}"

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

CF_TEACHER_LAMBDA="${CF_TEACHER_LAMBDA:-0.6}"
CF_TEACHER_N_SAMPLES="${CF_TEACHER_N_SAMPLES:-4}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.7}"
TEACHER_TOP_P="${TEACHER_TOP_P:-0.95}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-1024}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-200}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
TEACHER_REMOTE_BATCH_SIZE="${TEACHER_REMOTE_BATCH_SIZE:-64}"
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
# Persistence across pod restarts is preserved by archive_shared_teacher_cache
# (RUN_DIR archive at end of run also copies the cache into the run's OSS dir).
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-/mnt/workspace/teacher_cache_shared}"

ENABLE_TEACHER_PREFETCH="${ENABLE_TEACHER_PREFETCH:-true}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-2}"
PREFETCH_MAX_WORKERS="${PREFETCH_MAX_WORKERS:-6}"

# IN-TRAINING EVAL DISABLED BY DEFAULT (mirrors run_G2_rebase.sh).
#   GENERATE_MAX_LEN=8 (the EBFT token-level rollout) makes in-training
#   eval generate only 8 tokens, which is meaningless on AOPS — every
#   prompt ends up "no answer" / 0% acc and clutters TensorBoard with
#   a flat-zero curve. The post-training 2-round vLLM eval (via
#   supplement_2rounds/G2.sh, FIRST_PASS_MAX_NEW_TOKENS=16384 / 32768)
#   is what we trust for accuracy reporting.
#
#   To re-enable in-training eval:
#       EVAL_STEPS=200 EVAL_GENERATE_MAX_LEN=2048 bash scripts/run_G2_rebase_2node_once.sh
#   NOTE: only eval_steps == -1 is treated as "disabled" by ebft_trainer
#   (eval_steps == -1 -> float('inf')). A finite-but-large value still
#   triggers the trainer's step-0 initial eval, which under ZeRO-3
#   colocate_actor_ref deadlocks NCCL all-gather when batches don't
#   divide evenly across actor ranks. Use -1, not a big number.
EVAL_STEPS="${EVAL_STEPS:--1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-50}"
EVAL_GENERATE_MAX_LEN="${EVAL_GENERATE_MAX_LEN:-${GENERATE_MAX_LEN}}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_EVEN_COUNT="${SAVE_EVEN_COUNT:-10}"
# Post-training offline two-round eval (16k -> 32k via vLLM in
# scripts/supplement_2rounds/G2.sh). Aligned with G1/G3/baseline so all
# four runs report directly comparable accuracy numbers.
#
# 2-node trade-off: training fully uses both nodes, but vLLM eval runs on
# the head node only (single-node TP=8). The worker node sits idle during
# post-eval. We picked this over "split dataset across nodes" because:
#   - cross-node vLLM TP=16 needs RDMA / Ray cluster and saves only
#     ~30-50% wall time on Gemma-4 8B (the model is small enough that
#     TP=8 already saturates a single A100 box);
#   - dataset-split parallelism adds ssh launch + merge logic that's
#     easy to get wrong, and stage 2 (32k retry) anyway depends on stage
#     1 results being merged before retry-subset selection.
# If you want 16-GPU eval later, override POST_EVAL_RUN_ON_BOTH_NODES=true
# and supply a custom PARALLEL_EVAL_SCRIPT.
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
RUN_TWO_ROUND_EVAL="${RUN_TWO_ROUND_EVAL:-${EVAL_AFTER_TRAIN}}"
POST_EVAL_SCRIPT="${POST_EVAL_SCRIPT:-${REPO_ROOT}/scripts/supplement_2rounds/G2_2node.sh}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
# vLLM concurrency knobs. See scripts/supplement_2rounds/G2_2node.sh for the
# full HOL-blocking rationale that motivated raising these from the legacy
# {32, hardcoded-16} defaults to {256, 256}.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-256}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
# vLLM eval lives on head node only; grab its 8 cards.
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(count_csv_items "${MODEL_CUDA_VISIBLE_DEVICES}")}"
ANALYSIS_VENV="${ANALYSIS_VENV:-${STUDENT_VENV}}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"
POST_EVAL_TAG="${POST_EVAL_TAG:-post_train}"
POST_EVAL_LOG_DIR="${POST_EVAL_LOG_DIR:-}"
ARCHIVE_OUTPUTS_AFTER_RUN="${ARCHIVE_OUTPUTS_AFTER_RUN:-true}"
ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g2_0.99}"
ARCHIVE_SHARED_TEACHER_CACHE_MODE="${ARCHIVE_SHARED_TEACHER_CACHE_MODE:-copy}"
ARCHIVE_SHARED_TEACHER_CACHE_DIR="${ARCHIVE_SHARED_TEACHER_CACHE_DIR:-${TEACHER_CACHE_DIR}}"

# G2 RL training knobs (no feature adapter / no EMA / frozen critic).
# G3-specific knobs (FEATURE_ADAPTER_*, UNFREEZE_LAYERS, CE_LOSS_COEF,
# EMA_BETA, CRITIC_CLASSIFIER_LOSS_COEF, CRITIC_DIRECT_DISCREPANCY_*,
# DIVERSITY_REW_COEF, ALIGNMENT_REW_COEF) are intentionally absent here
# so they don't leak into the G2 args block below. Don't add them back
# without also adding the matching --flag to the trainer cli.
ACTOR_LR="${ACTOR_LR:-1e-6}"
CRITIC_LR="${CRITIC_LR:-0}"
CRITIC_LR_HEAD="${CRITIC_LR_HEAD:-0}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_WAIT_SECONDS="${RAY_WAIT_SECONDS:-120}"

# In DLC mode use a stable RUN_NAME derived from the pod's hostname
# (e.g. dlcXXXXXX-master-0 -> g2_dlcXXXXXX) so master and worker pods
# converge on the same RUN_DIR even if they start a few seconds apart
# across a minute boundary. The IP rendezvous file lives under RUN_DIR,
# so master and worker MUST agree on it.
if [[ "${DLC_MODE}" == "true" && -z "${RUN_NAME:-}" ]]; then
  _dlc_job_id="$(hostname | sed -E 's/^(dlc[a-z0-9]+)-(master|worker)-[0-9]+$/\1/' || true)"
  if [[ -n "${_dlc_job_id}" && "${_dlc_job_id}" != "$(hostname)" ]]; then
    RUN_NAME="g2_${_dlc_job_id}"
  fi
fi
RUN_NAME="${RUN_NAME:-g2_rebase_2node_once_$(date +%m%d_%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/ebft-distribution-new/outputs}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
SAVE_PATH="${RUN_DIR}/model"
TB_DIR="${RUN_DIR}/tensorboard"
TEACHER_LOG_DIR="${RUN_DIR}/teacher_logs"
RAY_LOG_DIR="${RUN_DIR}/ray_logs"
PID_DIR="${RUN_DIR}/pids"
JOB_SCRIPT="${RUN_DIR}/run_train_once.sh"
JOB_LOG="${RUN_DIR}/ray_job.log"
POST_EVAL_LOG_DIR="${POST_EVAL_LOG_DIR:-${RUN_DIR}/supplement_logs}"
SCRIPT_SOURCE_PATH="${BASH_SOURCE[0]:-$0}"
TRAIN_CONFIG_SNAPSHOT_PATH="${RUN_DIR}/run_context.env"
TRAIN_CONFIG_SUMMARY_PATH="${RUN_DIR}/run_summary.txt"
LAUNCHER_SNAPSHOT_PATH="${RUN_DIR}/launcher_snapshot.sh"
mkdir -p "${RUN_DIR}" "${SAVE_PATH}" "${TB_DIR}" "${TEACHER_LOG_DIR}" "${RAY_LOG_DIR}" "${PID_DIR}" "${TEACHER_CACHE_DIR}"

# HF_HOME and PYTORCH_CUDA_ALLOC_CONF are exported above (section 1) with
# DSW-specific defaults; do not redeclare here or the upper values would be
# silently shadowed if a user pre-exported only one of the two. (The heredoc
# below for the worker-node job script still propagates them via '${HF_HOME}'
# / '${PYTORCH_CUDA_ALLOC_CONF}'.)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTHONUNBUFFERED=1

# NCCL safety nets — applied to BOTH the training stage (DeepSpeed/Ray
# collectives across actor 2-GPU + critic 2-GPU spread over 2 nodes via
# RoCE) AND the post-training vLLM stage (which also sources
# _vllm_runtime.sh and re-applies these defaults; redundant export here
# makes the training stage benefit from the same protections):
#
#   NCCL_P2P_LEVEL=NVL   - intra-node use NVLink/NVSwitch only; banning
#                          the previous NCCL_P2P_DISABLE=1 default that
#                          disabled NVLink P2P entirely and forced traffic
#                          onto RoCE GDRDMA (where it tripped mlx5:1
#                          async fatal QP / local access violation).
#   NCCL_NET_GDR_DISABLE=1 - inter-node traffic must go via host-staged
#                          RoCE instead of GPUDirect RDMA. Slower per-step
#                          but eliminates the GDR-page-unmap window that
#                          fires QP-fatal asynchronously on this fabric.
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

require_cmd curl
require_cmd ray
# ssh is only needed in DSW 2-node mode where the head pod uses ssh to
# bring up the worker pod. Both DLC mode and single-node mode bring up
# the worker side without ssh.
if [[ "${SKIP_SSH_BOOTSTRAP}" != "true" ]]; then
  require_cmd ssh
fi

if [[ ! -x "${TEACHER_VLLM_BIN}" ]]; then
  echo "[ERROR] TEACHER_VLLM_BIN not executable: ${TEACHER_VLLM_BIN}"
  exit 1
fi
if [[ ! -x "${STUDENT_PYTHON_BIN}" ]]; then
  echo "[ERROR] STUDENT_PYTHON_BIN not executable: ${STUDENT_PYTHON_BIN}"
  exit 1
fi
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[ERROR] REPO_ROOT not found: ${REPO_ROOT}"
  exit 1
fi
if [[ ! -e "${TRAIN_DATA}" ]]; then
  echo "[ERROR] TRAIN_DATA not found: ${TRAIN_DATA}"
  exit 1
fi
if [[ ! -e "${EVAL_DATA}" ]]; then
  echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"
  exit 1
fi

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
  TEACHER_LOG_DIR="${RUN_DIR}/teacher_logs"
  RAY_LOG_DIR="${RUN_DIR}/ray_logs"
  PID_DIR="${RUN_DIR}/pids"
  JOB_SCRIPT="${RUN_DIR}/run_train_once.sh"
  JOB_LOG="${RUN_DIR}/ray_job.log"
  if [[ -n "${POST_EVAL_LOG_DIR}" && "${POST_EVAL_LOG_DIR}" == "${old_run_dir}"* ]]; then
    POST_EVAL_LOG_DIR="${RUN_DIR}${POST_EVAL_LOG_DIR#${old_run_dir}}"
  fi
  TRAIN_CONFIG_SNAPSHOT_PATH="${RUN_DIR}/run_context.env"
  TRAIN_CONFIG_SUMMARY_PATH="${RUN_DIR}/run_summary.txt"
  LAUNCHER_SNAPSHOT_PATH="${RUN_DIR}/launcher_snapshot.sh"
  write_run_metadata
}

archive_shared_teacher_cache() {
  local mode="$1"
  local source_dir="$2"
  local dest_dir

  case "${mode}" in
    skip|"")
      echo "[archive] shared teacher cache archive skipped."
      return 0
      ;;
    copy|move)
      ;;
    *)
      echo "[ERROR] invalid ARCHIVE_SHARED_TEACHER_CACHE_MODE=${mode}. Use skip, copy, or move."
      return 1
      ;;
  esac

  if [[ ! -d "${source_dir}" ]]; then
    echo "[archive] shared teacher cache not found, skip: ${source_dir}"
    return 0
  fi

  dest_dir="${RUN_DIR}/$(basename "${source_dir}")"
  if [[ -e "${dest_dir}" ]]; then
    dest_dir="${dest_dir}_$(date +%m%d_%H%M%S)"
  fi

  echo "[archive] ${mode} shared teacher cache to: ${dest_dir}"
  if [[ "${mode}" == "copy" ]]; then
    cp -a "${source_dir}" "${dest_dir}"
  else
    mv "${source_dir}" "${dest_dir}"
  fi
}

write_run_metadata() {
  local vars=(
    RUN_NAME OUTPUT_ROOT RUN_DIR SAVE_PATH TB_DIR TEACHER_LOG_DIR RAY_LOG_DIR PID_DIR JOB_SCRIPT JOB_LOG
    HEAD_NODE HEAD_NODE_IP WORKER_NODE WORKER_NODE_IP WORKER_SSH_HOST SSH_USER SSH_OPTS
    HEAD_TEACHER_CUDA_VISIBLE_DEVICES WORKER_TEACHER_CUDA_VISIBLE_DEVICES
    HEAD_STUDENT_CUDA_VISIBLE_DEVICES WORKER_STUDENT_CUDA_VISIBLE_DEVICES
    ACTOR_GPUS CRITIC_GPUS REF_GPUS REWARD_GPUS
    ACTOR_NUM_NODES CRITIC_NUM_NODES REF_NUM_NODES REWARD_NUM_NODES
    TEACHER_MODEL_PATH TEACHER_MODEL_NAME TEACHER_BASE_PORT TEACHER_API_KEY TEACHER_TP_SIZE TEACHER_DTYPE
    TEACHER_MAX_MODEL_LEN TEACHER_MAX_NUM_SEQS TEACHER_MAX_BATCHED_TOKENS TEACHER_GPU_MEMORY_UTIL
    TEACHER_WAIT_SECONDS TEACHER_API_BASE TEACHER_CACHE_DIR ENABLE_TEACHER_PREFETCH PREFETCH_DEPTH PREFETCH_MAX_WORKERS
    REPO_ROOT MODEL_PATH TRAIN_DATA EVAL_DATA
    N_SAMPLES_PER_PROMPT ROLLOUT_BATCH_SIZE TRAIN_BATCH_SIZE MICRO_TRAIN_BATCH_SIZE MICRO_ROLLOUT_BATCH_SIZE MICRO_REWARD_BATCH_SIZE
    PROMPT_MAX_LEN CONTEXT_MAX_LEN GENERATE_MAX_LEN STRIDE NUM_EPISODES MAX_EPOCHS TARGET_STEPS MAX_SAMPLES
    CF_TEACHER_LAMBDA CF_TEACHER_N_SAMPLES TEACHER_TEMPERATURE TEACHER_TOP_P TEACHER_MAX_NEW_TOKENS
    TEACHER_TIMEOUT TEACHER_MAX_RETRIES TEACHER_REMOTE_BATCH_SIZE TEACHER_SYSTEM_PROMPT_ID
    ACTOR_LR CRITIC_LR CRITIC_LR_HEAD
    EVAL_STEPS EVAL_MAX_SAMPLES EVAL_GENERATE_MAX_LEN SAVE_STEPS SAVE_EVEN_COUNT
    EVAL_AFTER_TRAIN RUN_TWO_ROUND_EVAL POST_EVAL_SCRIPT
    MODEL_CUDA_VISIBLE_DEVICES VLLM_TP_SIZE
    POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
    FIRST_PASS_MAX_NEW_TOKENS SECOND_PASS_MAX_NEW_TOKENS
    POST_EVAL_TEMPERATURE POST_EVAL_TOP_P POST_EVAL_REPETITION_PENALTY POST_EVAL_BEST_OF_N
    VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
    POST_EVAL_TAG POST_EVAL_LOG_DIR
    ARCHIVE_OUTPUTS_AFTER_RUN ARCHIVE_OUTPUT_ROOT ARCHIVE_SHARED_TEACHER_CACHE_MODE ARCHIVE_SHARED_TEACHER_CACHE_DIR
  )

  mkdir -p "${RUN_DIR}"
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
    echo "teacher_api_base: ${TEACHER_API_BASE}"
    echo "teacher_cache_dir: ${TEACHER_CACHE_DIR}"
    echo "train_data: ${TRAIN_DATA}"
    echo "eval_data: ${EVAL_DATA}"
    echo "train_batch_size: ${TRAIN_BATCH_SIZE}"
    echo "target_steps: ${TARGET_STEPS}"
    echo "max_samples: ${MAX_SAMPLES}"
    echo "post_eval_script: ${POST_EVAL_SCRIPT}"
    echo "post_eval_max_samples: ${POST_EVAL_MAX_SAMPLES}"
    echo "archive_output_root: ${ARCHIVE_OUTPUT_ROOT}"
    echo "archive_shared_teacher_cache_mode: ${ARCHIVE_SHARED_TEACHER_CACHE_MODE}"
    echo "launcher_snapshot: ${LAUNCHER_SNAPSHOT_PATH}"
    echo "job_script: ${JOB_SCRIPT}"
  } > "${TRAIN_CONFIG_SUMMARY_PATH}"
}

write_final_status() {
  {
    echo "# Auto-generated final status"
    echo "# UTC: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf "TRAIN_RC=%q\n" "${TRAIN_RC:-0}"
    printf "EVAL_RC=%q\n" "${EVAL_RC:-0}"
    printf "ARCHIVE_RC=%q\n" "${ARCHIVE_RC:-0}"
    printf "SHARED_CACHE_ARCHIVE_RC=%q\n" "${SHARED_CACHE_ARCHIVE_RC:-0}"
    printf "FINAL_RC=%q\n" "${FINAL_RC:-0}"
    printf "RUN_DIR=%q\n" "${RUN_DIR:-}"
    printf "SAVE_PATH=%q\n" "${SAVE_PATH:-}"
    printf "POST_EVAL_LOG_DIR=%q\n" "${POST_EVAL_LOG_DIR:-}"
    printf "TRAIN_CONFIG_SNAPSHOT_PATH=%q\n" "${TRAIN_CONFIG_SNAPSHOT_PATH:-}"
    printf "TRAIN_CONFIG_SUMMARY_PATH=%q\n" "${TRAIN_CONFIG_SUMMARY_PATH:-}"
  } > "${RUN_DIR}/final_status.env"
}

head_teacher_gpu_count="$(count_csv_items "${HEAD_TEACHER_CUDA_VISIBLE_DEVICES}")"
worker_teacher_gpu_count="$(count_csv_items "${WORKER_TEACHER_CUDA_VISIBLE_DEVICES}")"
head_student_gpu_count="$(count_csv_items "${HEAD_STUDENT_CUDA_VISIBLE_DEVICES}")"
worker_student_gpu_count="$(count_csv_items "${WORKER_STUDENT_CUDA_VISIBLE_DEVICES}")"

if (( head_teacher_gpu_count % TEACHER_TP_SIZE != 0 )); then
  echo "[ERROR] head teacher gpu count must be divisible by TEACHER_TP_SIZE"
  exit 1
fi
if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  # Single-node: worker pod doesn't exist; head node hosts everything.
  # Force the cross-node "world size" knobs to 1 so the trainer launcher
  # doesn't try to reach a non-existent second node.
  ACTOR_NUM_NODES=1
  CRITIC_NUM_NODES=1
  REF_NUM_NODES=1
  REWARD_NUM_NODES=1
  if (( head_student_gpu_count != ACTOR_GPUS + CRITIC_GPUS )); then
    echo "[ERROR] (single-node) head student gpu count (${head_student_gpu_count})"
    echo "        must equal ACTOR_GPUS(${ACTOR_GPUS}) + CRITIC_GPUS(${CRITIC_GPUS})"
    echo "        Default is HEAD_STUDENT_CUDA_VISIBLE_DEVICES=6,7 with"
    echo "        ACTOR_GPUS=1, CRITIC_GPUS=1. Override one of these to fix."
    exit 1
  fi
else
  if (( worker_teacher_gpu_count % TEACHER_TP_SIZE != 0 )); then
    echo "[ERROR] worker teacher gpu count must be divisible by TEACHER_TP_SIZE"
    exit 1
  fi
  if (( head_student_gpu_count != ACTOR_GPUS )); then
    echo "[ERROR] head student gpu count must equal ACTOR_GPUS for this 2-node layout"
    exit 1
  fi
  if (( worker_student_gpu_count != CRITIC_GPUS )); then
    echo "[ERROR] worker student gpu count must equal CRITIC_GPUS for this 2-node layout"
    exit 1
  fi
fi
if (( TRAIN_BATCH_SIZE != N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE )); then
  echo "[ERROR] TRAIN_BATCH_SIZE must equal N_SAMPLES_PER_PROMPT * ROLLOUT_BATCH_SIZE"
  exit 1
fi
if (( TRAIN_BATCH_SIZE % (MICRO_TRAIN_BATCH_SIZE * ACTOR_GPUS) != 0 )); then
  echo "[ERROR] train_batch_size % (micro_train_batch_size * actor_gpus) != 0"
  exit 1
fi
if (( MICRO_TRAIN_BATCH_SIZE < N_SAMPLES_PER_PROMPT || MICRO_TRAIN_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] MICRO_TRAIN_BATCH_SIZE must be >= N_SAMPLES_PER_PROMPT and divisible by it"
  exit 1
fi
if (( MICRO_ROLLOUT_BATCH_SIZE % N_SAMPLES_PER_PROMPT != 0 )); then
  echo "[ERROR] MICRO_ROLLOUT_BATCH_SIZE must be divisible by N_SAMPLES_PER_PROMPT"
  exit 1
fi

# Required only when post-training two-round vLLM eval will run.
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
  if [[ ! -f "${POST_EVAL_SCRIPT}" ]]; then
    echo "[ERROR] POST_EVAL_SCRIPT not found: ${POST_EVAL_SCRIPT}"
    exit 1
  fi
fi

HEAD_TEACHER_WORKER_COUNT="$((head_teacher_gpu_count / TEACHER_TP_SIZE))"
WORKER_TEACHER_WORKER_COUNT="$((worker_teacher_gpu_count / TEACHER_TP_SIZE))"
HEAD_TEACHER_API_BASE="$(build_teacher_urls "${HEAD_NODE_IP}" "${TEACHER_BASE_PORT}" "${HEAD_TEACHER_WORKER_COUNT}")"
if [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
  # No worker pod -> only head-side teachers exist.
  WORKER_TEACHER_API_BASE=""
  TEACHER_API_BASE="${HEAD_TEACHER_API_BASE}"
else
  WORKER_TEACHER_API_BASE="$(build_teacher_urls "${WORKER_NODE_IP}" "${TEACHER_BASE_PORT}" "${WORKER_TEACHER_WORKER_COUNT}")"
  TEACHER_API_BASE="${HEAD_TEACHER_API_BASE},${WORKER_TEACHER_API_BASE}"
fi

echo "========== G2 2-node once launcher =========="
echo "RUN_DIR:                    ${RUN_DIR}"
echo "HEAD_NODE / IP:             ${HEAD_NODE} / ${HEAD_NODE_IP}"
echo "WORKER_NODE / IP:           ${WORKER_NODE} / ${WORKER_NODE_IP}"
echo "Head teacher GPUs:          ${HEAD_TEACHER_CUDA_VISIBLE_DEVICES}"
echo "Worker teacher GPUs:        ${WORKER_TEACHER_CUDA_VISIBLE_DEVICES}"
echo "Head student GPUs:          ${HEAD_STUDENT_CUDA_VISIBLE_DEVICES}"
echo "Worker student GPUs:        ${WORKER_STUDENT_CUDA_VISIBLE_DEVICES}"
echo "Teacher worker count:       ${HEAD_TEACHER_WORKER_COUNT} + ${WORKER_TEACHER_WORKER_COUNT}"
echo "Teacher API:                ${TEACHER_API_BASE}"
echo "Actor/Critic GPUs:          ${ACTOR_GPUS}/${CRITIC_GPUS}"
echo "Actor/Critic nodes:         ${ACTOR_NUM_NODES}/${CRITIC_NUM_NODES}"
echo "Target steps:               ${TARGET_STEPS}"
echo "Max samples:                ${MAX_SAMPLES}"
echo "Post-train eval:            ${EVAL_AFTER_TRAIN}"
echo "Post-eval script:           ${POST_EVAL_SCRIPT}"
echo "Post-eval VLLM_TP_SIZE:     ${VLLM_TP_SIZE} (head node only; worker idle during eval)"
echo "Post-eval first/second pass: ${FIRST_PASS_MAX_NEW_TOKENS}/${SECOND_PASS_MAX_NEW_TOKENS} tokens"
echo "Single training submitter:  ${HEAD_NODE}"
echo "============================================="

write_run_metadata

LOCAL_TEACHER_PIDS=()
RAY_HEAD_PID=""
RUNTIME_STOPPED=0

stop_runtime_processes() {
  local pid
  if [[ "${RUNTIME_STOPPED}" == "1" ]]; then
    return 0
  fi

  echo "[cleanup] stopping local teacher workers..."
  for pid in "${LOCAL_TEACHER_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${LOCAL_TEACHER_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
  done

  echo "[cleanup] stopping local ray..."
  if [[ -n "${RAY_HEAD_PID}" ]] && kill -0 "${RAY_HEAD_PID}" 2>/dev/null; then
    kill "${RAY_HEAD_PID}" 2>/dev/null || true
    wait "${RAY_HEAD_PID}" 2>/dev/null || true
  fi
  ray stop --force >/dev/null 2>&1 || true

  if [[ "${SKIP_SSH_BOOTSTRAP}" == "true" ]]; then
    # Single-node mode: no worker pod.
    # DLC mode: worker pod is its own process, will clean up via its own
    # trap on EXIT/SIGTERM (set in dlc_worker_bootstrap below).
    :
  else
    echo "[cleanup] stopping worker-side teacher/ray..."
    ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -s -- '${PID_DIR}'" <<'EOF' >/dev/null 2>&1 || true
set +e
PID_DIR="$1"
shopt -s nullglob
for pid_file in "${PID_DIR}"/teacher_worker_*.pid "${PID_DIR}"/ray_worker.pid; do
  if [[ -f "${pid_file}" ]]; then
    pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  fi
done
ray stop --force >/dev/null 2>&1 || true
EOF
  fi
  RUNTIME_STOPPED=1
}

cleanup() {
  stop_runtime_processes || true
}
trap cleanup EXIT INT TERM

wait_for_http_health() {
  local url="$1"
  local waited=0
  until curl -sf "${url%/v1}/health" >/dev/null; do
    sleep 3
    waited=$((waited + 3))
    if (( waited >= TEACHER_WAIT_SECONDS )); then
      echo "[ERROR] health timeout: ${url}"
      return 1
    fi
  done
}

# =====================================================================
# DLC WORKER POD ENTRY POINT
# =====================================================================
# In DLC multi-pod mode the worker pod (DLC_NODE_RANK > 0) takes a
# completely different code path: instead of waiting to be ssh'd into
# from the head, it actively (a) writes its IP to a shared rendezvous
# file so the master can build TEACHER_API_BASE, (b) launches its
# local teacher vLLM workers, (c) joins the master's ray cluster via
# MASTER_ADDR, and (d) blocks until ray is shut down. The master pod
# (DLC_NODE_RANK == 0) continues into the head launcher path below.
# =====================================================================
dlc_worker_bootstrap() {
  echo "================================================================"
  echo "[DLC worker rank=${DLC_NODE_RANK}] starting on $(hostname)"
  echo "[DLC worker rank=${DLC_NODE_RANK}] master address: ${DLC_MASTER_ADDR}:${RAY_PORT}"

  # Pick our routable IP (skip loopback / IPv6).
  local my_ip
  my_ip="$(hostname -I 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i !~ /^127\./ && $i !~ /:/) {print $i; exit}}')"
  if [[ -z "${my_ip}" ]]; then
    echo "[DLC worker] ERROR: could not determine own IP via 'hostname -I'"
    exit 1
  fi
  echo "[DLC worker] my IP: ${my_ip}"

  # Write IP to the rendezvous file so the master can read it and build
  # TEACHER_API_BASE that includes our teacher URLs.
  local rdv_dir="${RUN_DIR}/dlc_rendezvous"
  mkdir -p "${rdv_dir}"
  local ip_file="${rdv_dir}/worker_${DLC_NODE_RANK}_ip.txt"
  echo "${my_ip}" > "${ip_file}"
  echo "[DLC worker] wrote IP to ${ip_file}"

  # Start a keepalive that re-writes the IP file every few seconds.
  #
  # Why we need this: the master clears stale rendezvous files at the
  # very start of its DLC bootstrap (`rm -f .../worker_*_ip.txt`) to
  # protect against a previous AIMaster-restarted attempt unblocking
  # the new attempt with an old IP. But if the worker pod happens to
  # boot earlier than the master (which is the common case here, since
  # we ssh-execute setup_env.sh + run_*.sh in parallel and the master
  # has more setup work), the worker writes the file FIRST, then the
  # master deletes it, and the master then waits forever for an IP
  # nobody is going to write -> "[DLC master] ERROR: worker pod IP not
  # seen in 480s" deadlock. Re-writing the file from the worker side
  # makes the rendezvous robust to whichever pod starts first, while
  # still letting the master kill any genuinely stale file from a dead
  # previous run (since that previous run's keepalive is gone).
  (
    while :; do
      printf '%s\n' "${my_ip}" > "${ip_file}" 2>/dev/null || true
      sleep 5
    done
  ) &
  local _ip_keepalive_pid=$!
  echo "[DLC worker] IP keepalive started (pid=${_ip_keepalive_pid}, interval=5s)"

  # 1) Launch local teacher vLLM workers FIRST.
  #
  # We MUST launch teachers before waiting for the master's ray head.
  # The master's startup order is:
  #   [3/5] launch own teachers
  #   [4/5] wait for ALL teacher URLs (master + worker) to become healthy
  #   [5/5] start ray head
  # If we waited on ray head first, master would block in [4/5] forever
  # because our teachers would never come up -> master never reaches
  # [5/5] -> we never see ray head -> deadlock. Launching here breaks
  # the cycle: master's [4/5] eventually polls our /health, by which
  # point vLLM is up, and master proceeds to [5/5].
  echo "[DLC worker] launching ${WORKER_TEACHER_WORKER_COUNT} teacher worker(s)..."
  local _gpu_csv="${WORKER_TEACHER_CUDA_VISIBLE_DEVICES}"
  IFS=',' read -r -a _gpu_ids <<< "${_gpu_csv}"
  mkdir -p "${TEACHER_LOG_DIR}" "${PID_DIR}"
  local w port gpu_start g worker_gpus log_file
  for (( w=0; w<WORKER_TEACHER_WORKER_COUNT; w++ )); do
    port=$(( TEACHER_BASE_PORT + w ))
    gpu_start=$(( w * TEACHER_TP_SIZE ))
    worker_gpus=""
    for (( g=gpu_start; g<gpu_start+TEACHER_TP_SIZE; g++ )); do
      [[ -n "${worker_gpus}" ]] && worker_gpus+=","
      worker_gpus+="${_gpu_ids[$g]}"
    done
    log_file="${TEACHER_LOG_DIR}/worker_dlc_rank${DLC_NODE_RANK}_${w}.log"
    # AIMaster restarts re-enter this code with the same fixed log path.
    # On ossfs2 (where TEACHER_LOG_DIR lives) `bash > existing_file` does
    # an open(O_WRONLY|O_CREAT|O_TRUNC), and ossfs2 rejects truncate of
    # an existing object with EINVAL (fuse mis-reports as ENOSPC). bash
    # then exits the redirect subshell BEFORE exec'ing vllm, so $! ends
    # up pointing at a dead helper, the .pid file looks valid, but no
    # vllm process actually runs -> nvidia-smi 0 MiB / GPUs idle / the
    # master's [4/5] curls a port that nobody listens on for 1h.
    # Pre-deleting the file forces the open() to take the O_CREAT-new-
    # object path, which ossfs2 supports.
    rm -f "${log_file}" 2>/dev/null || true
    # PYTHONUNBUFFERED=1: When vLLM's stdout/stderr is redirected to a file
    # (this >${log_file}), Python defaults to FULL block-buffering (~8KB chunks)
    # instead of line-buffering. Combined with ossfs2 not auto-flushing dirty
    # pages, the log appears frozen at exactly ~6866 bytes for tens of minutes
    # while vLLM is in fact making progress (loading 27B weights, compiling
    # CUDA graphs, etc.). Forcing unbuffered stdout makes [4/5] hangs
    # diagnosable in real time instead of after a 1h timeout.
    nohup bash -lc "
      PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES='${worker_gpus}' \
      '${TEACHER_VLLM_BIN}' serve '${TEACHER_MODEL_PATH}' \
        --served-model-name '${TEACHER_MODEL_NAME}' \
        --host 0.0.0.0 \
        --port '${port}' \
        --tensor-parallel-size '${TEACHER_TP_SIZE}' \
        --dtype '${TEACHER_DTYPE}' \
        --api-key '${TEACHER_API_KEY}' \
        --generation-config vllm \
        --max-model-len '${TEACHER_MAX_MODEL_LEN}' \
        --max-num-seqs '${TEACHER_MAX_NUM_SEQS}' \
        --max-num-batched-tokens '${TEACHER_MAX_BATCHED_TOKENS}' \
        --gpu-memory-utilization '${TEACHER_GPU_MEMORY_UTIL}' \
        --limit-mm-per-prompt '{\"image\":0,\"video\":0,\"audio\":0}' \
        --enable-chunked-prefill \
        ${TEACHER_PREFIX_CACHING_FLAG}
    " > "${log_file}" 2>&1 &
    echo $! > "${PID_DIR}/dlc_worker_teacher_${w}.pid"
    echo "[DLC worker] teacher #${w} launched on GPUs ${worker_gpus} -> port ${port}, log ${log_file}"
  done

  # Install the cleanup trap NOW (before the ray-head wait can time
  # out) so any failure from this point on tears down the vLLM teacher
  # processes we just spawned. Otherwise an `exit 1` from the wait
  # below would leak 6 GPU-holding processes and make the next
  # AIMaster restart fail with CUDA OOM.
  _dlc_worker_cleanup() {
    echo "[DLC worker] cleaning up local teacher / ray..."
    if [[ -n "${_ip_keepalive_pid:-}" ]]; then
      kill "${_ip_keepalive_pid}" 2>/dev/null || true
    fi
    shopt -s nullglob
    for pid_file in "${PID_DIR}"/dlc_worker_teacher_*.pid; do
      [[ -f "${pid_file}" ]] || continue
      local p; p="$(cat "${pid_file}" 2>/dev/null || true)"
      [[ -n "${p}" ]] && kill "${p}" 2>/dev/null || true
    done
    ray stop --force >/dev/null 2>&1 || true
  }
  trap _dlc_worker_cleanup EXIT INT TERM

  # 2) Wait for master's ray head to come up.
  #
  # Master only reaches the `ray start --head` step AFTER its own
  # teacher health check loop has finished, which on a 27B teacher
  # cold-start can easily take 20-30 minutes. So this wait must
  # accommodate the full teacher startup budget, not just ray's own
  # post-startup cluster registration time.
  local _ray_head_wait="${DLC_WORKER_RAY_HEAD_WAIT_SECONDS:-${TEACHER_WAIT_SECONDS}}"
  if (( _ray_head_wait < RAY_WAIT_SECONDS * 4 )); then
    _ray_head_wait=$(( RAY_WAIT_SECONDS * 4 ))
  fi
  echo "[DLC worker] waiting for master ray head at ${DLC_MASTER_ADDR}:${RAY_PORT} (timeout ${_ray_head_wait}s)..."
  local waited=0
  until ray status --address "${DLC_MASTER_ADDR}:${RAY_PORT}" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= _ray_head_wait )); then
      echo "[DLC worker] ERROR: master ray head didn't come up in ${_ray_head_wait}s"
      exit 1
    fi
  done
  echo "[DLC worker] master ray head reachable after ${waited}s (loop iterations; wall time may be longer)"

  # 3) Join the ray cluster. --block keeps ray's worker process alive
  #    until ray cluster is torn down or we get killed.
  echo "[DLC worker] joining ray cluster via 'ray start --address=${DLC_MASTER_ADDR}:${RAY_PORT}'..."
  CUDA_VISIBLE_DEVICES="${WORKER_STUDENT_CUDA_VISIBLE_DEVICES}" \
    ray start --address "${DLC_MASTER_ADDR}:${RAY_PORT}" \
              --num-gpus "${worker_student_gpu_count}" \
              --block &
  local ray_pid=$!
  echo "[DLC worker] ray join started (pid=${ray_pid}); waiting until cluster shuts down..."
  echo "================================================================"

  # _dlc_worker_cleanup trap was already installed earlier (right after
  # teachers were launched). Just block on the ray join until shutdown.
  wait "${ray_pid}"
}

if [[ "${DLC_MODE}" == "true" && "${DLC_NODE_RANK}" -gt 0 ]]; then
  dlc_worker_bootstrap
  echo "[DLC worker rank=${DLC_NODE_RANK}] ray cluster shut down."

  # When post-train eval is enabled (RUN_TWO_ROUND_EVAL=true), stay alive
  # and participate as the "worker shard" of the 2-node post-eval via the
  # file-based rendezvous under ${RUN_DIR}/dlc_rendezvous/post_eval/.
  # Master writes a request, we fulfill it (run vLLM on our 8 GPUs),
  # write a done sentinel. When master marks the whole thing complete we
  # exit 0. See scripts/supplement_2rounds/_rendezvous_dlc.sh.
  #
  # If post-eval is disabled, exit immediately.
  if [[ "${RUN_TWO_ROUND_EVAL:-true}" == "true" ]]; then
    # Resolve paths we need on the worker side (master and worker use
    # identical code so these are already the right defaults).
    _rdv_helper_path="${REPO_ROOT}/scripts/supplement_2rounds/_rendezvous_dlc.sh"
    _vllm_runtime_path="${REPO_ROOT}/scripts/supplement_2rounds/_vllm_runtime.sh"
    if [[ ! -f "${_rdv_helper_path}" ]]; then
      echo "[DLC worker rank=${DLC_NODE_RANK}] post-eval rendezvous helper missing: ${_rdv_helper_path}"
      echo "                                    falling back to exit 0 (single-node eval only)."
      exit 0
    fi
    # Force all 8 of this pod's GPUs into vLLM's CUDA_VISIBLE_DEVICES;
    # the training-time visibility (2 student GPUs) is leftover from ray
    # join and would otherwise restrict the eval shard to 2 GPUs.
    export CUDA_VISIBLE_DEVICES="${POSTEVAL_WORKER_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    # These envs are consumed by _vllm_runtime.sh's pre-flight dump and by
    # the vLLM retry helper that fulfill() invokes.
    export MODEL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    _VISIBLE_COUNT="$(count_csv_items "${CUDA_VISIBLE_DEVICES}")"
    export VLLM_TP_SIZE="${POSTEVAL_WORKER_VLLM_TP_SIZE:-${_VISIBLE_COUNT}}"
    export REPO_ROOT TEACHER_VENV
    export TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
    export PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"
    # Source the vLLM retry helper first so posteval_worker_watch can use
    # run_vllm_generation_with_retry(). _vllm_runtime.sh also emits a
    # topology pre-flight dump which is handy for diagnosis.
    echo "[DLC worker rank=${DLC_NODE_RANK}] sourcing vLLM runtime helpers for post-eval"
    # shellcheck disable=SC1090
    source "${_vllm_runtime_path}"
    # shellcheck disable=SC1090
    source "${_rdv_helper_path}"
    rdv_init_root "${RUN_DIR}"
    echo "[DLC worker rank=${DLC_NODE_RANK}] entering post-eval rendezvous watch loop"
    set +e
    posteval_worker_watch
    _rdv_rc=$?
    set -e
    echo "[DLC worker rank=${DLC_NODE_RANK}] post-eval watcher exited rc=${_rdv_rc}; shutting down"
    exit "${_rdv_rc}"
  fi

  echo "[DLC worker rank=${DLC_NODE_RANK}] exiting"
  exit 0
fi

# In DLC master mode, wait for the worker pod to publish its IP, then
# rebuild TEACHER_API_BASE to include the worker's teacher URLs (the
# initial computation in section 6 used WORKER_NODE_IP=HEAD_NODE_IP as
# a placeholder because we didn't know the worker IP at that time).
if [[ "${DLC_MODE}" == "true" ]]; then
  rdv_dir="${RUN_DIR}/dlc_rendezvous"
  mkdir -p "${rdv_dir}"
  # Clear any stale rendezvous file from a previous (AIMaster-restarted)
  # attempt. RUN_DIR lives on persistent OSS so the file from attempt N
  # would otherwise unblock attempt N+1 immediately with the stale IP,
  # and the new worker pod's actual IP (likely different) would never
  # propagate to TEACHER_API_BASE -> [4/5] health check hangs forever.
  #
  # NOTE: this rm used to deadlock the launcher when the worker pod
  # happened to boot BEFORE the master pod: worker would write its IP,
  # master would then start and rm it, master would wait forever, and
  # the worker would already be past the IP-write step (stuck in the
  # ray-head-wait loop) so the file would never reappear. To keep the
  # rendezvous symmetric, the worker side now runs an IP-keepalive
  # background loop that re-writes the file every few seconds, so this
  # rm only kills genuinely stale files from a dead previous attempt
  # (whose keepalive is gone) and the live worker will repopulate the
  # file within 5s, bounded by the wait loop below.
  rm -f "${rdv_dir}"/worker_*_ip.txt 2>/dev/null || true
  echo "[DLC master] waiting for worker pod IP at ${rdv_dir}/worker_1_ip.txt..."
  waited=0
  while [[ ! -s "${rdv_dir}/worker_1_ip.txt" ]]; do
    sleep 3
    waited=$((waited + 3))
    if (( waited >= RAY_WAIT_SECONDS * 4 )); then
      echo "[DLC master] ERROR: worker pod IP not seen in $((RAY_WAIT_SECONDS * 4))s"
      exit 1
    fi
  done
  WORKER_NODE_IP="$(tr -d '[:space:]' < "${rdv_dir}/worker_1_ip.txt")"
  echo "[DLC master] worker IP: ${WORKER_NODE_IP} (after ${waited}s)"
  WORKER_TEACHER_API_BASE="$(build_teacher_urls "${WORKER_NODE_IP}" "${TEACHER_BASE_PORT}" "${WORKER_TEACHER_WORKER_COUNT}")"
  TEACHER_API_BASE="${HEAD_TEACHER_API_BASE},${WORKER_TEACHER_API_BASE}"
  echo "[DLC master] updated TEACHER_API_BASE: ${TEACHER_API_BASE}"
fi

if [[ "${SKIP_SSH_BOOTSTRAP}" == "true" ]]; then
  if [[ "${DLC_MODE}" == "true" ]]; then
    echo "[1/5] DLC mode: worker pod connects in via ray start; skipping ssh check"
    echo "[2/5] DLC mode: worker pod launches teacher locally; skipping ssh teacher launch"
  else
    echo "[1/5] single-node mode: skipping connectivity check (no worker pod)"
    echo "[2/5] single-node mode: skipping worker-side teacher launch"
  fi
else
echo "[1/5] connectivity check to worker..."
ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -lc 'hostname'" >/dev/null

echo "[2/5] launching worker-side teacher services..."
ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -s -- \
  '${RUN_DIR}' \
  '${TEACHER_VLLM_BIN}' \
  '${TEACHER_MODEL_PATH}' \
  '${TEACHER_MODEL_NAME}' \
  '${WORKER_TEACHER_CUDA_VISIBLE_DEVICES}' \
  '${TEACHER_BASE_PORT}' \
  '${TEACHER_TP_SIZE}' \
  '${TEACHER_DTYPE}' \
  '${TEACHER_API_KEY}' \
  '${TEACHER_MAX_MODEL_LEN}' \
  '${TEACHER_MAX_NUM_SEQS}' \
  '${TEACHER_MAX_BATCHED_TOKENS}' \
  '${TEACHER_GPU_MEMORY_UTIL}' \
  '${TEACHER_PREFIX_CACHING_FLAG}'" <<'EOF'
set -euo pipefail
RUN_DIR="$1"
TEACHER_VLLM_BIN="$2"
TEACHER_MODEL_PATH="$3"
TEACHER_MODEL_NAME="$4"
GPU_CSV="$5"
BASE_PORT="$6"
TP_SIZE="$7"
DTYPE="$8"
API_KEY="$9"
MAX_MODEL_LEN="${10}"
MAX_NUM_SEQS="${11}"
MAX_BATCHED_TOKENS="${12}"
GPU_MEMORY_UTIL="${13}"
PREFIX_CACHING_FLAG="${14}"

mkdir -p "${RUN_DIR}/teacher_logs"
mkdir -p "${RUN_DIR}/pids"
IFS=',' read -r -a GPU_IDS <<< "${GPU_CSV}"
WORKER_COUNT=$(( ${#GPU_IDS[@]} / TP_SIZE ))
for (( w=0; w<WORKER_COUNT; w++ )); do
  port=$(( BASE_PORT + w ))
  gpu_start=$(( w * TP_SIZE ))
  worker_gpus=""
  for (( g=gpu_start; g<gpu_start+TP_SIZE; g++ )); do
    [[ -n "${worker_gpus}" ]] && worker_gpus+=","
    worker_gpus+="${GPU_IDS[$g]}"
  done
  log_file="${RUN_DIR}/teacher_logs/worker_${HOSTNAME}_${w}.log"
  rm -f "${log_file}" 2>/dev/null || true   # ossfs2 O_TRUNC-on-existing safety
  nohup bash -lc "
    CUDA_VISIBLE_DEVICES='${worker_gpus}' \
    '${TEACHER_VLLM_BIN}' serve '${TEACHER_MODEL_PATH}' \
      --served-model-name '${TEACHER_MODEL_NAME}' \
      --host 0.0.0.0 \
      --port '${port}' \
      --tensor-parallel-size '${TP_SIZE}' \
      --dtype '${DTYPE}' \
      --api-key '${API_KEY}' \
      --generation-config vllm \
      --max-model-len '${MAX_MODEL_LEN}' \
      --max-num-seqs '${MAX_NUM_SEQS}' \
      --max-num-batched-tokens '${MAX_BATCHED_TOKENS}' \
      --gpu-memory-utilization '${GPU_MEMORY_UTIL}' \
      --limit-mm-per-prompt '{\"image\":0,\"video\":0,\"audio\":0}' \
      --enable-chunked-prefill \
      ${PREFIX_CACHING_FLAG}
  " > "${log_file}" 2>&1 &
  echo $! > "${RUN_DIR}/pids/teacher_worker_${w}.pid"
done
EOF
fi  # end of multi-node SSH worker-teacher launch

echo "[3/5] launching head-side teacher services..."
IFS=',' read -r -a _HEAD_GPU_IDS <<< "${HEAD_TEACHER_CUDA_VISIBLE_DEVICES}"
for (( _w=0; _w<HEAD_TEACHER_WORKER_COUNT; _w++ )); do
  _port=$(( TEACHER_BASE_PORT + _w ))
  _gpu_start=$(( _w * TEACHER_TP_SIZE ))
  _worker_gpus=""
  for (( _g=_gpu_start; _g<_gpu_start+TEACHER_TP_SIZE; _g++ )); do
    [[ -n "${_worker_gpus}" ]] && _worker_gpus+=","
    _worker_gpus+="${_HEAD_GPU_IDS[$_g]}"
  done
  _log="${TEACHER_LOG_DIR}/worker_${HEAD_NODE}_${_w}.log"
  # See worker-side comment: ossfs2 rejects O_TRUNC of an existing file
  # (EINVAL, fuse mis-reports as ENOSPC), which causes bash `>` to abort
  # the redirect subshell BEFORE exec'ing vllm on AIMaster restarts.
  rm -f "${_log}" 2>/dev/null || true
  # See worker-side launch: force unbuffered stdout so OSS-redirected logs
  # update in real time during 27B cold-start instead of looking frozen.
  PYTHONUNBUFFERED=1 \
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
    ${TEACHER_PREFIX_CACHING_FLAG} \
    > "${_log}" 2>&1 &
  LOCAL_TEACHER_PIDS+=("$!")
done

echo "[4/5] waiting for teacher health checks..."
IFS=',' read -r -a _ALL_TEACHER_URLS <<< "${TEACHER_API_BASE}"
for _url in "${_ALL_TEACHER_URLS[@]}"; do
  wait_for_http_health "${_url}"
  echo "  [teacher] healthy: ${_url}"
done

echo "[5/5] starting ray cluster..."
ray stop --force >/dev/null 2>&1 || true
# Same OSS-truncate-of-existing-file hazard as the teacher launches above.
rm -f "${RAY_LOG_DIR}/head.log" 2>/dev/null || true
CUDA_VISIBLE_DEVICES="${HEAD_STUDENT_CUDA_VISIBLE_DEVICES}" \
ray start --head \
  --node-ip-address "${HEAD_NODE_IP}" \
  --port "${RAY_PORT}" \
  --dashboard-host 0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}" \
  --num-gpus "${head_student_gpu_count}" \
  --block \
  > "${RAY_LOG_DIR}/head.log" 2>&1 &
RAY_HEAD_PID=$!

sleep 5

if [[ "${SKIP_SSH_BOOTSTRAP}" == "true" ]]; then
  if [[ "${DLC_MODE}" == "true" ]]; then
    echo "[ray] DLC mode: worker pod will join via 'ray start --address=${HEAD_NODE_IP}:${RAY_PORT}' on its own"
  else
    echo "[ray] single-node mode: no worker ray join"
  fi
else
  ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -lc '
    set -euo pipefail
    ray stop --force >/dev/null 2>&1 || true
    mkdir -p ${PID_DIR}
    nohup bash -lc \"CUDA_VISIBLE_DEVICES=${WORKER_STUDENT_CUDA_VISIBLE_DEVICES} ray start --address ${HEAD_NODE_IP}:${RAY_PORT} --num-gpus ${worker_student_gpu_count} --block\" > ${RAY_LOG_DIR}/worker.log 2>&1 &
    echo \$! > ${PID_DIR}/ray_worker.pid
  '"
fi

waited=0
until ray status --address "${HEAD_NODE_IP}:${RAY_PORT}" >/dev/null 2>&1; do
  sleep 2
  waited=$((waited + 2))
  if (( waited >= RAY_WAIT_SECONDS )); then
    echo "[ERROR] ray cluster did not become ready in time."
    exit 1
  fi
done

cat > "${JOB_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export HF_HOME='${HF_HOME}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}'
export HF_DATASETS_OFFLINE='${HF_DATASETS_OFFLINE}'
export HF_HUB_DISABLE_XET='${HF_HUB_DISABLE_XET}'
export TOKENIZERS_PARALLELISM=false
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD='${VLLM_WORKER_MULTIPROC_METHOD}'
export PYTORCH_CUDA_ALLOC_CONF='${PYTORCH_CUDA_ALLOC_CONF}'
export PYTHONUNBUFFERED=1
export RAY_ADDRESS=auto
# NCCL safety (mirror parent process; ensures Ray actors inherit even
# if some scheduler quirk drops the parent env)
export NCCL_P2P_LEVEL='${NCCL_P2P_LEVEL}'
export NCCL_NET_GDR_DISABLE='${NCCL_NET_GDR_DISABLE}'

cd '${REPO_ROOT}'

PREFETCH_FLAGS=()
if [[ '${ENABLE_TEACHER_PREFETCH}' == 'true' ]]; then
  PREFETCH_FLAGS=(
    --enable_teacher_prefetch
    --prefetch_depth '${PREFETCH_DEPTH}'
    --prefetch_max_workers '${PREFETCH_MAX_WORKERS}'
  )
fi

'${STUDENT_PYTHON_BIN}' -m openrlhf.cli.train_ebft_ray \
  --bf16 --flash_attn --pretrain_mode --no_chat_template \
  --disable_ds_ckpt --colocate_actor_ref --colocate_critic_reward \
  --gradient_checkpointing --gradient_checkpointing_use_reentrant --use_kl_loss --use_whitening \
  --distribution_reward_type cf_l1oo \
  --feature_map_type identity --rff_num_features 128 --rff_sigma 1.0 --rff_seed 43 \
  --cf_num_freqs 128 --cf_sigma 1.0 --cf_seed 43 --cf_alpha 0.5 --cf_beta 0.5 --cf_reward_scale 1.0 \
  --cf_target_mode teacher --cf_teacher_lambda '${CF_TEACHER_LAMBDA}' --cf_teacher_n_samples '${CF_TEACHER_N_SAMPLES}' \
  --teacher_backend remote \
  --teacher_api_base '${TEACHER_API_BASE}' \
  --teacher_api_key '${TEACHER_API_KEY}' \
  --teacher_api_style completions \
  --teacher_model_name '${TEACHER_MODEL_NAME}' \
  --teacher_timeout '${TEACHER_TIMEOUT}' \
  --teacher_max_retries '${TEACHER_MAX_RETRIES}' \
  --teacher_remote_batch_size '${TEACHER_REMOTE_BATCH_SIZE}' \
  --teacher_temperature '${TEACHER_TEMPERATURE}' \
  --teacher_top_p '${TEACHER_TOP_P}' \
  --teacher_max_new_tokens '${TEACHER_MAX_NEW_TOKENS}' \
  --teacher_system_prompt_text '${TEACHER_SYSTEM_PROMPT_TEXT}' \
  --teacher_system_prompt_id '${TEACHER_SYSTEM_PROMPT_ID}' \
  --teacher_cache_enable --teacher_cache_dir '${TEACHER_CACHE_DIR}' \
  "\${PREFETCH_FLAGS[@]}" \
  --embed_method last_token --critic_sequence_level last_token \
  --critic_learning_rate '${CRITIC_LR}' \
  --critic_lr_head '${CRITIC_LR_HEAD}' \
  --actor_learning_rate '${ACTOR_LR}' \
  --pretrain '${MODEL_PATH}' --critic_pretrain '${MODEL_PATH}' \
  --prompt_data '${TRAIN_DATA}' --eval_dataset '${EVAL_DATA}' \
  --input_key question --label_key answer --output_key answer \
  --prompt_split train --eval_split test \
  --prompt_max_len '${PROMPT_MAX_LEN}' \
  --context_max_len '${CONTEXT_MAX_LEN}' \
  --generate_max_len '${GENERATE_MAX_LEN}' \
  --stride '${STRIDE}' \
  --n_samples_per_prompt '${N_SAMPLES_PER_PROMPT}' \
  --rollout_batch_size '${ROLLOUT_BATCH_SIZE}' \
  --train_batch_size '${TRAIN_BATCH_SIZE}' \
  --micro_train_batch_size '${MICRO_TRAIN_BATCH_SIZE}' \
  --micro_rollout_batch_size '${MICRO_ROLLOUT_BATCH_SIZE}' \
  --micro_reward_batch_size '${MICRO_REWARD_BATCH_SIZE}' \
  --max_samples '${MAX_SAMPLES}' \
  --num_episodes '${NUM_EPISODES}' \
  --max_epochs '${MAX_EPOCHS}' \
  --actor_num_nodes '${ACTOR_NUM_NODES}' --actor_num_gpus_per_node '${ACTOR_GPUS}' \
  --critic_num_nodes '${CRITIC_NUM_NODES}' --critic_num_gpus_per_node '${CRITIC_GPUS}' \
  --ref_num_nodes '${REF_NUM_NODES}' --ref_num_gpus_per_node '${REF_GPUS}' \
  --reward_num_nodes '${REWARD_NUM_NODES}' --reward_num_gpus_per_node '${REWARD_GPUS}' \
  --advantage_estimator rloo --init_kl_coef 0.0 --kl_estimator k2 \
  --temperature 0.6 --top_p 1.0 \
  --zero_stage 3 --lr_warmup_ratio 0.03 --critic_lr_warmup_ratio 0.0 \
  --seed 43 \
  --eval_steps '${EVAL_STEPS}' \
  --eval_max_samples '${EVAL_MAX_SAMPLES}' \
  --eval_generate_max_len '${EVAL_GENERATE_MAX_LEN}' \
  --logging_steps 10 \
  --save_steps '${SAVE_STEPS}' --save_even_count '${SAVE_EVEN_COUNT}' --save_hf_ckpt \
  --use_tensorboard '${TB_DIR}' \
  --save_path '${SAVE_PATH}' --ckpt_path '${SAVE_PATH}/ckpt' \
  --wandb_run_name '${RUN_NAME}' \
  2>&1 | tee '${RUN_DIR}/train.log'
EOF

chmod +x "${JOB_SCRIPT}"

echo "[train] submitting one Ray job..."
echo "[train] the worker node does not run train_ebft_ray directly; it only hosts teacher/ray worker processes."
TRAIN_RC=0
EVAL_RC=0
ARCHIVE_RC=0
SHARED_CACHE_ARCHIVE_RC=0

set +e
ray job submit \
  --address "http://${HEAD_NODE_IP}:${RAY_DASHBOARD_PORT}" \
  -- bash "${JOB_SCRIPT}" | tee "${JOB_LOG}"
TRAIN_RC=$?
set -e

if (( TRAIN_RC != 0 )); then
  echo "[ERROR] training failed with exit code ${TRAIN_RC}"
fi

echo "[post-run] stopping teacher/ray processes before eval/archive ..."
stop_runtime_processes

if [[ "${RUN_TWO_ROUND_EVAL}" == "true" ]]; then
  if (( TRAIN_RC == 0 )); then
    echo ""
    echo "===== Running two-round 16k/32k completion eval (head node, 8 GPUs) ====="
    echo "[post-eval] running eval from checkpoint: ${SAVE_PATH}"
    # Use export so the worker shell sees these even under `set -u`. Mirror
    # how run_G1_rebase.sh / run_G2_rebase.sh invoke their G1.sh / G2.sh
    # workers, so all three two-round eval call sites stay consistent.
    set +e
    export RUN_DIR MODEL_PATH="${SAVE_PATH}"
    export REPO_ROOT
    export TEACHER_VENV ANALYSIS_VENV
    export TEACHER_PYTHON_BIN ANALYSIS_PYTHON_BIN
    export MODEL_CUDA_VISIBLE_DEVICES VLLM_TP_SIZE
    export POST_EVAL_MAX_SAMPLES POST_EVAL_PROMPT_MAX_LEN
    export FIRST_PASS_MAX_NEW_TOKENS SECOND_PASS_MAX_NEW_TOKENS
    export POST_EVAL_TEMPERATURE POST_EVAL_TOP_P
    export POST_EVAL_REPETITION_PENALTY POST_EVAL_BEST_OF_N
    export VLLM_MAX_NUM_SEQS VLLM_PROGRESS_BATCH_SIZE VLLM_ENABLE_PREFIX_CACHING VLLM_SEED
    export EVAL_DATA
    export LOG_DIR="${POST_EVAL_LOG_DIR}"
    export EVAL_TAG="${POST_EVAL_TAG}"
    # 2-node eval needs ssh plumbing + NCCL safety envs propagated to the
    # G{2,3}_2node.sh dispatcher (and from there into the worker-side ssh
    # invocation). Single-node G{2,3}.sh also tolerates these being set;
    # they're harmless when POST_EVAL_RUN_ON_BOTH_NODES=false or when
    # WORKER_SSH_TARGET is empty.
    #
    # Worker-dispatch backend depends on the deployment mode. G{2,3}_2node.sh
    # reads POSTEVAL_WORKER_DISPATCH (ssh | rendezvous) to decide how to
    # reach the worker pod:
    #   * DSW 2-node (passwordless ssh configured):
    #       dispatch=ssh. WORKER_SSH_TARGET is a real user@host.
    #   * DLC multi-pod (no sshd in worker pod):
    #       dispatch=rendezvous. Master writes request files to the
    #       OSS-shared RUN_DIR and the worker's post-eval watcher (parked
    #       after training finishes, see dlc_worker_bootstrap above) picks
    #       them up. WORKER_SSH_TARGET is cleared to make the misuse path
    #       impossible.
    #   * single-node (no worker):
    #       dispatch=ssh (default) but WORKER_SSH_TARGET is cleared; that
    #       triggers G{2,3}_2node.sh's single-node fallback into G{2,3}.sh.
    POSTEVAL_WORKER_DISPATCH="${POSTEVAL_WORKER_DISPATCH:-ssh}"
    if [[ "${DLC_MODE}" == "true" ]]; then
        POSTEVAL_WORKER_DISPATCH="rendezvous"
        WORKER_SSH_TARGET=""
    elif [[ "${SINGLE_NODE_MODE}" == "true" ]]; then
        WORKER_SSH_TARGET=""
    fi
    export POSTEVAL_WORKER_DISPATCH
    export WORKER_SSH_TARGET SSH_OPTS
    export NCCL_P2P_LEVEL NCCL_NET_GDR_DISABLE
    bash "${POST_EVAL_SCRIPT}" "${RUN_DIR}"
    EVAL_RC=$?
    set -e
    if (( EVAL_RC != 0 )); then
      echo "[ERROR] post-eval failed with exit code ${EVAL_RC}; run outputs will still be archived."
    fi
  else
    echo "[post-eval] skipped because training did not finish successfully."
  fi
fi

if [[ "${ARCHIVE_OUTPUTS_AFTER_RUN}" == "true" ]]; then
  set +e
  archive_run_outputs "${ARCHIVE_OUTPUT_ROOT}"
  ARCHIVE_RC=$?
  if (( ARCHIVE_RC == 0 )); then
    archive_shared_teacher_cache "${ARCHIVE_SHARED_TEACHER_CACHE_MODE}" "${ARCHIVE_SHARED_TEACHER_CACHE_DIR}"
    SHARED_CACHE_ARCHIVE_RC=$?
  else
    SHARED_CACHE_ARCHIVE_RC=0
  fi
  set -e
  if (( ARCHIVE_RC != 0 )); then
    echo "[ERROR] archiving run outputs failed with exit code ${ARCHIVE_RC}"
  fi
  if (( SHARED_CACHE_ARCHIVE_RC != 0 )); then
    echo "[ERROR] archiving shared teacher cache failed with exit code ${SHARED_CACHE_ARCHIVE_RC}"
  fi
fi

FINAL_RC=0
if (( TRAIN_RC != 0 )); then
  FINAL_RC=${TRAIN_RC}
elif (( EVAL_RC != 0 )); then
  FINAL_RC=${EVAL_RC}
elif (( ARCHIVE_RC != 0 )); then
  FINAL_RC=${ARCHIVE_RC}
elif (( SHARED_CACHE_ARCHIVE_RC != 0 )); then
  FINAL_RC=${SHARED_CACHE_ARCHIVE_RC}
fi

write_final_status

echo "[done] logs: ${RUN_DIR}"
exit "${FINAL_RC}"
