#!/usr/bin/env bash
# G2 two-round post-eval, 2-node parallel version.
#
# vs scripts/supplement_2rounds/G2.sh:
#   * G2.sh:     single-node TP=8 on the head node only (worker idle).
#   * G2_2node.sh: head and worker each run vLLM TP=8 in parallel on a
#                  disjoint half of the eval dataset (sharded by source_idx
#                  % 2). After both halves finish, results are concatenated
#                  before analysis / retry-subset extraction. Stage 2 retry
#                  is also sharded across both nodes (only the prompts that
#                  failed stage 1 are retried at 32k).
#
# Why dataset-split parallel vs cross-node TP=16:
#   * Gemma-class 8B models already saturate a single A100 box at TP=8;
#     cross-node TP=16 needs RDMA + ray cluster and saves only ~30-50%
#     wall time while doubling the failure surface (NCCL / fabric timeouts
#     during a 5328-prompt 16k-token batch).
#   * Dataset shard parallelism is embarrassingly parallel: each half is
#     a fully independent vLLM run; either node can fail/restart without
#     blocking the other; merge is a trivial concat by source_idx.
#
# Requirements:
#   * Both nodes must mount the same shared filesystem at /mnt/data/...
#     (this is true for the EBFT 2-node DLC + DSW setup; RUN_DIR /
#     SAVE_PATH / EVAL_DATA / LOG_DIR are all on the shared NAS).
#   * Passwordless SSH from head -> worker (DSW), or DLC mode where the
#     worker pod is reachable via WORKER_SSH_TARGET. Same SSH plumbing
#     as run_G2_rebase_2node_once.sh.
#   * Each node has 8 GPUs available for the post-eval stage (training
#     teachers / Ray have already been torn down by the launcher's
#     cleanup trap before this script runs).
#
# Usage:
#   bash scripts/supplement_2rounds/G2_2node.sh /mnt/data/.../outputs/g3_RUN
#   RUN_DIR=...   bash scripts/supplement_2rounds/G2_2node.sh
#
# Required env (typically set by run_G2_rebase_2node_once.sh):
#   WORKER_SSH_TARGET   user@host (or host) for ssh to worker
#   SSH_OPTS            optional ssh options string
#   REPO_ROOT           absolute path of the repo (same on both nodes)
#
# Optional env (with sensible defaults):
#   POST_EVAL_RUN_ON_BOTH_NODES   set false to fall back to single-node
#                                  G2.sh behavior (default: true)
#   N_SHARDS                       2 (head=shard 0, worker=shard 1)
#   HEAD_CUDA_VISIBLE_DEVICES      0,1,2,3,4,5,6,7
#   WORKER_CUDA_VISIBLE_DEVICES    0,1,2,3,4,5,6,7

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"

POST_EVAL_RUN_ON_BOTH_NODES="${POST_EVAL_RUN_ON_BOTH_NODES:-true}"
# Worker dispatch backend. The runner sets this explicitly:
#   ssh          DSW 2-node with passwordless ssh into worker pod (legacy)
#   rendezvous   DLC multi-pod: master writes request files under RUN_DIR
#                and worker's posteval watcher picks them up
# If unset here, we infer "ssh" and require WORKER_SSH_TARGET, matching
# the historical behavior.
POSTEVAL_WORKER_DISPATCH="${POSTEVAL_WORKER_DISPATCH:-ssh}"

# Fall back to single-node G2.sh if:
#   * the user explicitly opts out, OR
#   * dispatch=ssh but WORKER_SSH_TARGET is missing (DLC placeholder was
#     stripped, or script invoked standalone).
# In dispatch=rendezvous mode WORKER_SSH_TARGET is irrelevant.
_fallback_to_single_node=""
if [[ "${POST_EVAL_RUN_ON_BOTH_NODES}" != "true" ]]; then
  _fallback_to_single_node="opt-out (POST_EVAL_RUN_ON_BOTH_NODES=${POST_EVAL_RUN_ON_BOTH_NODES})"
elif [[ "${POSTEVAL_WORKER_DISPATCH}" == "ssh" && -z "${WORKER_SSH_TARGET:-}" ]]; then
  _fallback_to_single_node="ssh dispatch but WORKER_SSH_TARGET unset"
fi
if [[ -n "${_fallback_to_single_node}" ]]; then
  echo "[G2_2node] ${_fallback_to_single_node}, falling back to single-node G2.sh"
  exec bash "${SCRIPT_DIR}/G2.sh" "$@"
fi

_DEFAULT_TEACHER_VENV="/mnt/workspace/venvs/.teacherVenv"
[[ -d "${_DEFAULT_TEACHER_VENV}" ]] || _DEFAULT_TEACHER_VENV="${REPO_ROOT}/.teacherVenv"
TEACHER_VENV="${TEACHER_VENV:-${_DEFAULT_TEACHER_VENV}}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"

_DEFAULT_ANALYSIS_VENV="/mnt/workspace/venvs/.venv"
[[ -d "${_DEFAULT_ANALYSIS_VENV}" ]] || _DEFAULT_ANALYSIS_VENV="${REPO_ROOT}/.venv"
ANALYSIS_VENV="${ANALYSIS_VENV:-${_DEFAULT_ANALYSIS_VENV}}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"
PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"

RUN_DIR="${RUN_DIR:-${1:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: RUN_DIR=/path/to/run bash scripts/supplement_2rounds/G2_2node.sh"
  echo "   or: bash scripts/supplement_2rounds/G2_2node.sh /path/to/run"
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model}"
EVAL_TAG="${EVAL_TAG:-2rounds_vllm}"
SCRIPT_NAME="$(basename "$0" .sh)"
TS="${TS:-$(date +%m%d_%H%M)}"

N_SHARDS="${N_SHARDS:-2}"
HEAD_CUDA_VISIBLE_DEVICES="${HEAD_CUDA_VISIBLE_DEVICES:-${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
WORKER_CUDA_VISIBLE_DEVICES="${WORKER_CUDA_VISIBLE_DEVICES:-${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
IFS=',' read -r -a _HEAD_GPUS <<< "${HEAD_CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a _WORKER_GPUS <<< "${WORKER_CUDA_VISIBLE_DEVICES}"
HEAD_VLLM_TP_SIZE="${HEAD_VLLM_TP_SIZE:-${VLLM_TP_SIZE:-${#_HEAD_GPUS[@]}}}"
WORKER_VLLM_TP_SIZE="${WORKER_VLLM_TP_SIZE:-${VLLM_TP_SIZE:-${#_WORKER_GPUS[@]}}}"

POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
# vLLM concurrency knobs. Defaults raised after observing severe tail-blocking
# on the AOPS 32k retry pass: chunked submission of 16 prompts x max_num_seqs=32
# meant the slowest prompt in each chunk (typically the only one not yet at
# EOS) blocked the next chunk's launch for minutes while the GPUs ran 1-2 seqs
# -- wall-time throughput collapsed to a few hundred decode tok/s. Raising
# max_num_seqs to 256 just lifts the scheduler ceiling (KV cache pool sized via
# gpu_memory_utilization is the real cap, so this can't OOM at init), and
# VLLM_PROGRESS_BATCH_SIZE=256 means each tail-blocked chunk amortizes over 16x
# more prompts before the next chunk's HOL barrier. Both settings are pushed
# through to the worker-node ssh invocation below so the remote shard sees the
# same config.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_PROGRESS_BATCH_SIZE="${VLLM_PROGRESS_BATCH_SIZE:-256}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"

LOG_DIR="${LOG_DIR:-${RUN_DIR}/supplement_logs}"
SCRIPT_LOG_PATH="${SCRIPT_LOG_PATH:-${LOG_DIR}/${SCRIPT_NAME}_${EVAL_TAG}_${TS}.log}"
SHARD_DIR="${LOG_DIR}/shards_${EVAL_TAG}_${TS}"

# Per-shard / merged outputs.
SHARD_DATASET_PATTERN="${SHARD_DIR}/eval_data_shard_%d_of_${N_SHARDS}.jsonl"

FIRST_PASS_OUTPUT_PATH="${LOG_DIR}/eval_results_${EVAL_TAG}_stage1_${TS}.jsonl"
FIRST_PASS_OUTPUT_PATTERN="${SHARD_DIR}/eval_results_${EVAL_TAG}_stage1_shard_%d.jsonl"
FIRST_PASS_LOG_PATTERN="${SHARD_DIR}/eval_${EVAL_TAG}_stage1_shard_%d.log"
FIRST_PASS_ANALYSIS_REPORT_PATH="${LOG_DIR}/eval_analysis_${EVAL_TAG}_stage1_${TS}.json"
FIRST_PASS_ANALYSIS_LOG_PATH="${LOG_DIR}/eval_analysis_${EVAL_TAG}_stage1_${TS}.log"

SECOND_PASS_DATASET_PATH="${LOG_DIR}/eval_retry_subset_${EVAL_TAG}_${TS}.jsonl"
SECOND_PASS_DATASET_PATTERN="${SHARD_DIR}/eval_retry_subset_shard_%d.jsonl"
SECOND_PASS_METADATA_PATH="${LOG_DIR}/eval_retry_subset_meta_${EVAL_TAG}_${TS}.json"
SECOND_PASS_OUTPUT_PATH="${LOG_DIR}/eval_results_${EVAL_TAG}_stage2_${TS}.jsonl"
SECOND_PASS_OUTPUT_PATTERN="${SHARD_DIR}/eval_results_${EVAL_TAG}_stage2_shard_%d.jsonl"
SECOND_PASS_LOG_PATTERN="${SHARD_DIR}/eval_${EVAL_TAG}_stage2_shard_%d.log"
SECOND_PASS_ANALYSIS_REPORT_PATH="${LOG_DIR}/eval_analysis_${EVAL_TAG}_stage2_${TS}.json"
SECOND_PASS_ANALYSIS_LOG_PATH="${LOG_DIR}/eval_analysis_${EVAL_TAG}_stage2_${TS}.log"

FINAL_ANALYSIS_REPORT_PATH="${LOG_DIR}/eval_analysis_${EVAL_TAG}_final_${TS}.json"
FINAL_ANALYSIS_LOG_PATH="${LOG_DIR}/eval_analysis_${EVAL_TAG}_final_${TS}.log"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

source "${SCRIPT_DIR}/_vllm_runtime.sh"
# Rendezvous helpers only matter when POSTEVAL_WORKER_DISPATCH=rendezvous
# (DLC mode). In ssh mode this source is harmless -- it just defines a
# few helpers that are never called.
source "${SCRIPT_DIR}/_rendezvous_dlc.sh"

for _bin in "${TEACHER_PYTHON_BIN}" "${ANALYSIS_PYTHON_BIN}"; do
  [[ -x "${_bin}" ]] || { echo "[ERROR] Not executable: ${_bin}"; exit 1; }
done
[[ -d "${RUN_DIR}" ]] || { echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"; exit 1; }
[[ -e "${MODEL_PATH}" ]] || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -e "${EVAL_DATA}" ]] || { echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"; exit 1; }
[[ -f "${PROGRESS_HELPER}" ]] || { echo "[ERROR] PROGRESS_HELPER not found: ${PROGRESS_HELPER}"; exit 1; }

mkdir -p "${LOG_DIR}" "${SHARD_DIR}"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1
cd "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Split EVAL_DATA into N_SHARDS jsonl files using `idx % N_SHARDS == shard_id`.
# We add a `source_idx` field so that the merge step can stably reconstruct
# the original ordering by source_idx (matches G2.sh's
# extract_retry_subset() expectation).
shard_dataset() {
  local source_path="$1"
  local total="$2"
  local shard_dir="$3"
  local pattern="$4"

  python3 - "$source_path" "$total" "$shard_dir" "$pattern" <<'PY'
import json, os, sys
src, total, shard_dir, pattern = sys.argv[1:5]
total = int(total)

def load_rows(p):
    if os.path.isdir(p):
        for n in ("test.jsonl","test_qa.jsonl","eval.jsonl"):
            c = os.path.join(p,n)
            if os.path.isfile(c): p=c; break
        else: raise FileNotFoundError(f"No eval file under: {p}")
    if p.endswith(".jsonl"):
        return [json.loads(l) for l in open(p) if l.strip()]
    if p.endswith(".json"):
        d = json.load(open(p))
        if isinstance(d,list): return d
        if isinstance(d,dict):
            for k in ("test","data","rows"):
                if isinstance(d.get(k),list): return d[k]
        raise ValueError(f"Bad JSON: {p}")
    raise ValueError(f"Bad path: {p}")

rows = load_rows(src)
os.makedirs(shard_dir, exist_ok=True)
buckets = {i: [] for i in range(total)}
for i, r in enumerate(rows):
    row = dict(r)
    row.setdefault("source_idx", i)
    buckets[i % total].append(row)

for sid, rs in buckets.items():
    out_path = pattern % sid
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[shard] shard_{sid}: {len(rs)} rows -> {out_path}")
PY
}

# Concat shard outputs into a single jsonl. Order does not have to match the
# original dataset because analyze_eval_results.py keys by source_idx (the
# field we added in shard_dataset), but we still sort by source_idx for
# deterministic downstream consumption / git-friendly diffs.
merge_shards() {
  local pattern="$1"
  local total="$2"
  local out_path="$3"

  python3 - "$pattern" "$total" "$out_path" <<'PY'
import json, os, sys
pattern, total, out_path = sys.argv[1:4]
total = int(total)

rows = []
for sid in range(total):
    p = pattern % sid
    if not os.path.isfile(p):
        raise FileNotFoundError(f"missing shard output: {p}")
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

def key(r):
    si = r.get("source_idx")
    if si is None:
        si = r.get("idx")
    return (0, int(si)) if si is not None else (1, 0)
rows.sort(key=key)

os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[merge] {len(rows)} rows -> {out_path}")
PY
}

# Build the vLLM offline-batch command as a single shell-quoted string. The
# helper preserves spaces in INPUT_TEMPLATE / model paths and is used for
# both the local-shell-eval (head) and the ssh-eval (worker) call sites.
build_vllm_remote_cmd() {
  local cuda_devices="$1"
  local tp_size="$2"
  local dataset_path="$3"
  local output_path="$4"
  local max_new_tokens="$5"
  local log_path="$6"

  # Build the cmd with quoting that survives ssh.
  local quoted=""
  _q() { printf "%q" "$1"; }

  quoted+="cd $(_q "${REPO_ROOT}") && "
  quoted+="export HF_HOME=$(_q "${HF_HOME}") "
  quoted+="HF_HUB_OFFLINE=$(_q "${HF_HUB_OFFLINE}") "
  quoted+="HF_DATASETS_OFFLINE=$(_q "${HF_DATASETS_OFFLINE}") "
  quoted+="HF_HUB_DISABLE_XET=$(_q "${HF_HUB_DISABLE_XET}") "
  quoted+="TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
  quoted+="VLLM_WORKER_MULTIPROC_METHOD=$(_q "${VLLM_WORKER_MULTIPROC_METHOD}") "
  quoted+="NCCL_P2P_LEVEL=$(_q "${NCCL_P2P_LEVEL:-NVL}") "
  quoted+="NCCL_NET_GDR_DISABLE=$(_q "${NCCL_NET_GDR_DISABLE:-1}") "
  quoted+="CUDA_VISIBLE_DEVICES=$(_q "${cuda_devices}") && "
  quoted+="$(_q "${TEACHER_PYTHON_BIN}") $(_q "${PROGRESS_HELPER}") "
  quoted+="--pretrain $(_q "${MODEL_PATH}") "
  quoted+="--dataset $(_q "${dataset_path}") "
  quoted+="--input_key question "
  quoted+="--output_path $(_q "${output_path}") "
  quoted+="--prompt_max_len $(_q "${POST_EVAL_PROMPT_MAX_LEN}") "
  quoted+="--max_new_tokens $(_q "${max_new_tokens}") "
  quoted+="--temperature $(_q "${POST_EVAL_TEMPERATURE}") "
  quoted+="--top_p $(_q "${POST_EVAL_TOP_P}") "
  quoted+="--repetition_penalty $(_q "${POST_EVAL_REPETITION_PENALTY}") "
  quoted+="--max_samples $(_q "${POST_EVAL_MAX_SAMPLES}") "
  quoted+="--best_of_n $(_q "${POST_EVAL_BEST_OF_N}") "
  quoted+="--tp_size $(_q "${tp_size}") "
  quoted+="--max_num_seqs $(_q "${VLLM_MAX_NUM_SEQS}") "
  quoted+="--progress_batch_size $(_q "${VLLM_PROGRESS_BATCH_SIZE}") "
  quoted+="--seed $(_q "${VLLM_SEED}") "
  if [[ -n "${INPUT_TEMPLATE}" ]]; then
    quoted+="--input_template $(_q "${INPUT_TEMPLATE}") "
  fi
  if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "true" ]]; then
    quoted+="--enable_prefix_caching "
  fi
  quoted+="2>&1 | tee $(_q "${log_path}")"
  printf '%s' "${quoted}"
}

# Run a single shard locally (head node). Resume-skips if output is already
# complete (matches G2.sh's _stage_output_already_complete behavior).
run_shard_local() {
  local stage_name="$1"
  local dataset_path="$2"
  local output_path="$3"
  local log_path="$4"
  local max_new_tokens="$5"
  local cuda_devices="$6"
  local tp_size="$7"

  if _stage_output_already_complete "${stage_name}" "${output_path}" "${dataset_path}"; then
    return 0
  fi

  echo "[${stage_name}] (local) max_new=${max_new_tokens} tp=${tp_size} cuda=${cuda_devices}"
  local -a vllm_cmd=(
    env "CUDA_VISIBLE_DEVICES=${cuda_devices}"
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
    --tp_size "${tp_size}"
    --max_num_seqs "${VLLM_MAX_NUM_SEQS}"
    --progress_batch_size "${VLLM_PROGRESS_BATCH_SIZE}"
    --seed "${VLLM_SEED}"
  )
  [[ -n "${INPUT_TEMPLATE}" ]] && vllm_cmd+=(--input_template "${INPUT_TEMPLATE}")
  [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "true" ]] && vllm_cmd+=(--enable_prefix_caching)

  run_vllm_generation_with_retry "${stage_name}" "${log_path}" "${output_path}" "${dataset_path}" "${vllm_cmd[@]}"
}

# Run a single shard on the worker node via ssh. The worker side has the
# same shared NAS mount, so dataset / output / log paths are passed through
# directly.
run_shard_ssh() {
  local stage_name="$1"
  local dataset_path="$2"
  local output_path="$3"
  local log_path="$4"
  local max_new_tokens="$5"
  local cuda_devices="$6"
  local tp_size="$7"

  if _stage_output_already_complete "${stage_name}" "${output_path}" "${dataset_path}"; then
    return 0
  fi

  echo "[${stage_name}] (ssh ${WORKER_SSH_TARGET}) max_new=${max_new_tokens} tp=${tp_size} cuda=${cuda_devices}"
  local remote_cmd
  remote_cmd="$(build_vllm_remote_cmd "${cuda_devices}" "${tp_size}" "${dataset_path}" "${output_path}" "${max_new_tokens}" "${log_path}")"

  # We do NOT wrap this side in run_vllm_generation_with_retry because the
  # retry helper's orphan-killer / GPU-drain only addresses the local box.
  # Instead we rely on _stage_output_already_complete to short-circuit on
  # rerun, and the worker's vllm_generate_progress.py has its own internal
  # batch resume logic on output_path.
  # shellcheck disable=SC2029
  ssh ${SSH_OPTS} "${WORKER_SSH_TARGET}" "bash -lc $(printf '%q' "${remote_cmd}")"
}

# Dispatch one stage across both nodes in parallel. Returns nonzero if any
# shard fails.
run_stage_parallel() {
  local stage_name="$1"
  local dataset_pattern="$2"
  local output_pattern="$3"
  local log_pattern="$4"
  local max_new_tokens="$5"

  local head_dataset; head_dataset="$(printf "${dataset_pattern}" 0)"
  local head_output;  head_output="$(printf "${output_pattern}" 0)"
  local head_log;     head_log="$(printf "${log_pattern}" 0)"

  local worker_dataset; worker_dataset="$(printf "${dataset_pattern}" 1)"
  local worker_output;  worker_output="$(printf "${output_pattern}" 1)"
  local worker_log;     worker_log="$(printf "${log_pattern}" 1)"

  # If a shard input file is empty we synthesize an empty output and skip
  # vLLM (e.g. all stage1 prompts went to the other shard, or retry subset
  # was empty for this shard).
  local head_pid worker_pid head_rc=0 worker_rc=0

  if [[ -s "${head_dataset}" ]]; then
    ( run_shard_local "${stage_name}-head"   "${head_dataset}"   "${head_output}"   "${head_log}"   "${max_new_tokens}" "${HEAD_CUDA_VISIBLE_DEVICES}"   "${HEAD_VLLM_TP_SIZE}"   ) &
    head_pid=$!
  else
    echo "[${stage_name}-head] empty shard, writing empty output"
    : > "${head_output}"
    head_pid=""
  fi

  if [[ -s "${worker_dataset}" ]]; then
    # Choose worker dispatch backend:
    #   * ssh (default) - DSW 2-node mode: master has passwordless ssh
    #     into worker pod.
    #   * rendezvous    - DLC multi-pod mode: no sshd in worker pod, so
    #     master writes a JSON request file in OSS-shared RUN_DIR and a
    #     worker-side watcher loop fulfills it (see _rendezvous_dlc.sh).
    if [[ "${POSTEVAL_WORKER_DISPATCH:-ssh}" == "rendezvous" ]]; then
      ( run_shard_rendezvous "${stage_name}-worker" "${worker_dataset}" "${worker_output}" "${worker_log}" "${max_new_tokens}" "${WORKER_CUDA_VISIBLE_DEVICES}" "${WORKER_VLLM_TP_SIZE}" ) &
    else
      ( run_shard_ssh "${stage_name}-worker" "${worker_dataset}" "${worker_output}" "${worker_log}" "${max_new_tokens}" "${WORKER_CUDA_VISIBLE_DEVICES}" "${WORKER_VLLM_TP_SIZE}" ) &
    fi
    worker_pid=$!
  else
    echo "[${stage_name}-worker] empty shard, writing empty output"
    : > "${worker_output}"
    worker_pid=""
  fi

  if [[ -n "${head_pid}" ]]; then
    wait "${head_pid}";   head_rc=$?
  fi
  if [[ -n "${worker_pid}" ]]; then
    wait "${worker_pid}"; worker_rc=$?
  fi

  if (( head_rc != 0 || worker_rc != 0 )); then
    echo "[${stage_name}] shard failure: head_rc=${head_rc} worker_rc=${worker_rc}"
    return 1
  fi
  return 0
}

run_analysis() {
  local stage_name="$1"
  local eval_results_path="$2"
  local report_path="$3"
  local log_path="$4"
  # Optional 5th arg: the generation cap for this stage. When provided,
  # analyze_eval_results.py will also report token-length stats and the
  # fraction of outputs that hit max_new_tokens (along with cleaner units).
  local max_new_tokens="${5:-}"

  echo "[${stage_name}] Analyzing results"
  local -a analysis_cmd=(
    "${ANALYSIS_PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_eval_results.py"
    --eval_results "${eval_results_path}"
    --eval_dataset "${EVAL_DATA}"
    --input_key question --label_key answer
    --report_path "${report_path}"
    --tokenizer_path "${MODEL_PATH}"
  )
  [[ -n "${max_new_tokens}" ]] && analysis_cmd+=(--max_new_tokens "${max_new_tokens}")
  "${analysis_cmd[@]}" 2>&1 | tee "${log_path}"
}

extract_retry_subset() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, os, sys
source_path, report_path, subset_path, meta_path = sys.argv[1:5]
def load_rows(p):
    if os.path.isdir(p):
        for n in ("test.jsonl","test_qa.jsonl","eval.jsonl"):
            c = os.path.join(p,n)
            if os.path.isfile(c): p=c; break
        else: raise FileNotFoundError(f"No eval file under: {p}")
    if p.endswith(".jsonl"):
        return [json.loads(l) for l in open(p) if l.strip()]
    if p.endswith(".json"):
        d = json.load(open(p))
        if isinstance(d,list): return d
        if isinstance(d,dict):
            for k in ("test","data","rows"):
                if isinstance(d.get(k),list): return d[k]
        raise ValueError(f"Bad JSON: {p}")
    raise ValueError(f"Bad path: {p}")
rows = load_rows(source_path)
report = json.load(open(report_path))
cands, seen = [], set()
for r in report.get("records",[]):
    if r.get("is_correct") is True: continue
    si = r.get("source_idx") or r.get("idx")
    if si is None: continue
    si = int(si)
    if si<0 or si>=len(rows) or si in seen: continue
    seen.add(si); row=dict(rows[si]); row["source_idx"]=si; cands.append(row)
os.makedirs(os.path.dirname(os.path.abspath(subset_path)),exist_ok=True)
with open(subset_path,"w") as f:
    for row in cands: f.write(json.dumps(row,ensure_ascii=False)+"\n")
meta={"source_dataset_path":source_path,"analysis_report_path":report_path,
      "retry_count":len(cands),"source_indices":[r["source_idx"] for r in cands]}
with open(meta_path,"w") as f: json.dump(meta,f,indent=2,ensure_ascii=False); f.write("\n")
PY
}

# Re-shard the retry subset by source_idx % N_SHARDS so stage 2 also runs
# in parallel. We keep the same head=shard_0 / worker=shard_1 mapping as
# stage 1.
shard_retry_subset() {
  local subset_path="$1"
  local total="$2"
  local pattern="$3"

  python3 - "$subset_path" "$total" "$pattern" <<'PY'
import json, os, sys
src, total, pattern = sys.argv[1:4]
total = int(total)
buckets = {i: [] for i in range(total)}
if os.path.isfile(src):
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            si = r.get("source_idx")
            if si is None:
                si = r.get("idx", 0)
            buckets[int(si) % total].append(r)
for sid, rs in buckets.items():
    out = pattern % sid
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[shard-retry] shard_{sid}: {len(rs)} rows -> {out}")
PY
}

build_final_report() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, os, sys

fp, sp, mp, op = sys.argv[1:5]
with open(fp, "r", encoding="utf-8") as f:
    fr = json.load(f)
with open(sp, "r", encoding="utf-8") as f:
    sr = json.load(f)
with open(mp, "r", encoding="utf-8") as f:
    rm = json.load(f)

frc = fr.get("records", [])
src = sr.get("records", [])
final = list(frc)
ri = set()
second_by_idx = {}

def get_source_idx(record):
    value = record.get("source_idx")
    if value is None:
        value = record.get("idx")
    return None if value is None else int(value)

for r in src:
    si = get_source_idx(r)
    if si is None:
        continue
    ri.add(si)
    second_by_idx[si] = r
    if 0 <= si < len(final):
        m = dict(final[si])
        m["first_pass"] = dict(final[si])
        m["second_pass"] = dict(r)
        m["prompt"] = r.get("prompt", m.get("prompt", ""))
        m["model_output"] = r.get("model_output", m.get("model_output", ""))
        m["gold_answer"] = r.get("gold_answer", m.get("gold_answer"))
        m["is_correct"] = r.get("is_correct")
        m["category"] = r.get("category")
        m["detail"] = r.get("detail")
        m["retry_applied"] = True
        final[si] = m

for i, r in enumerate(final):
    r.setdefault("retry_applied", i in ri)

ev = [r for r in final if r.get("is_correct") is not None]
cor = sum(1 for r in ev if r.get("is_correct"))
acc = round(cor / len(ev) * 100, 2) if ev else 0.0
imp = sum(
    1
    for i in ri
    if 0 <= i < len(frc)
    and 0 <= i < len(final)
    and frc[i].get("is_correct") is not True
    and final[i].get("is_correct") is True
)
stw = sum(1 for i in ri if 0 <= i < len(final) and final[i].get("is_correct") is not True)

oracle_union_evaluated = 0
oracle_both_correct = 0
oracle_stage1_only_correct = 0
oracle_stage2_only_correct = 0
for i in range(len(final)):
    first_correct = frc[i].get("is_correct") if i < len(frc) else None
    second_record = second_by_idx.get(i)
    second_correct = None if second_record is None else second_record.get("is_correct")
    if first_correct is not None or second_correct is not None:
        oracle_union_evaluated += 1
    if first_correct is True and second_correct is True:
        oracle_both_correct += 1
    elif first_correct is True and second_correct is not True:
        oracle_stage1_only_correct += 1
    elif second_correct is True and first_correct is not True:
        oracle_stage2_only_correct += 1

oracle_union_correct = (
    oracle_both_correct
    + oracle_stage1_only_correct
    + oracle_stage2_only_correct
)
oracle_union_accuracy_pct = (
    round(oracle_union_correct / oracle_union_evaluated * 100, 2)
    if oracle_union_evaluated
    else 0.0
)

summary = {
    "total_predictions": len(final),
    "evaluated": len(ev),
    "correct": cor,
    "accuracy_pct": acc,
    "first_pass_correct": fr.get("summary", {}).get("correct"),
    "first_pass_accuracy_pct": fr.get("summary", {}).get("accuracy_pct"),
    "second_pass_retry_count": rm.get("retry_count", len(ri)),
    "retry_improved_to_correct": imp,
    "retry_still_incorrect": stw,
    "oracle_union_evaluated": oracle_union_evaluated,
    "oracle_union_correct": oracle_union_correct,
    "oracle_union_accuracy_pct": oracle_union_accuracy_pct,
    "oracle_both_correct": oracle_both_correct,
    "oracle_stage1_only_correct": oracle_stage1_only_correct,
    "oracle_stage2_only_correct": oracle_stage2_only_correct,
}
out = {
    "summary": summary,
    "first_pass_report_path": fp,
    "second_pass_report_path": sp,
    "retry_metadata": rm,
    "records": final,
}
os.makedirs(os.path.dirname(os.path.abspath(op)), exist_ok=True)
with open(op, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("======================================================================")
print("  Final Merged Report (2-node parallel)")
print("======================================================================")
print(f"  Total predictions:      {summary['total_predictions']}")
print(f"  Evaluated:              {summary['evaluated']}")
print(f"  Correct:                {summary['correct']}")
print(f"  Accuracy:               {summary['accuracy_pct']}%")
print(f"  First pass correct:     {summary['first_pass_correct']}")
print(f"  First pass accuracy:    {summary['first_pass_accuracy_pct']}%")
print(f"  Second pass retry cnt:  {summary['second_pass_retry_count']}")
print(f"  Retry improved correct: {summary['retry_improved_to_correct']}")
print(f"  Retry still incorrect:  {summary['retry_still_incorrect']}")
print(f"  Oracle union eval:      {summary['oracle_union_evaluated']}")
print(f"  Oracle union correct:   {summary['oracle_union_correct']}")
print(f"  Oracle union accuracy:  {summary['oracle_union_accuracy_pct']}%")
print(f"  Oracle both correct:    {summary['oracle_both_correct']}")
print(f"  Oracle stage1 only:     {summary['oracle_stage1_only_correct']}")
print(f"  Oracle stage2 only:     {summary['oracle_stage2_only_correct']}")
print("")
PY
}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

echo "========== G2 Two-Round vLLM Eval (2-node parallel) =========="
echo "RUN_DIR:                      ${RUN_DIR}"
echo "MODEL_PATH:                   ${MODEL_PATH}"
echo "EVAL_DATA:                    ${EVAL_DATA}"
echo "FIRST_PASS_MAX_NEW_TOKENS:    ${FIRST_PASS_MAX_NEW_TOKENS}"
echo "SECOND_PASS_MAX_NEW_TOKENS:   ${SECOND_PASS_MAX_NEW_TOKENS}"
echo "N_SHARDS:                     ${N_SHARDS}"
echo "HEAD GPUs / TP:               ${HEAD_CUDA_VISIBLE_DEVICES} / ${HEAD_VLLM_TP_SIZE}"
echo "WORKER GPUs / TP:             ${WORKER_CUDA_VISIBLE_DEVICES} / ${WORKER_VLLM_TP_SIZE}"
echo "POSTEVAL_WORKER_DISPATCH:     ${POSTEVAL_WORKER_DISPATCH}"
echo "WORKER_SSH_TARGET:            ${WORKER_SSH_TARGET:-<unset>}"
echo "POST_EVAL_MAX_SAMPLES:        ${POST_EVAL_MAX_SAMPLES}"
echo "VLLM_MAX_NUM_SEQS:            ${VLLM_MAX_NUM_SEQS}"
echo "VLLM_PROGRESS_BATCH_SIZE:     ${VLLM_PROGRESS_BATCH_SIZE}"
echo "================================================================"

# Rendezvous setup (only matters for POSTEVAL_WORKER_DISPATCH=rendezvous).
# rdv_cleanup_run_started clears any stale sentinels from a prior failed
# attempt against the same RUN_DIR so we don't short-circuit on them.
# Install a trap so that ANY exit path (including set -e trips, ctrl-C,
# or a failing stage returning non-zero) tells the DLC worker watcher to
# give up its watch loop and let the worker pod exit, instead of hanging
# on the OSS-shared marker until POSTEVAL_RDV_WORKER_TIMEOUT fires.
if [[ "${POSTEVAL_WORKER_DISPATCH}" == "rendezvous" ]]; then
  rdv_init_root "${RUN_DIR}"
  rdv_cleanup_run_started
  _on_exit_rdv() {
    local rc=$?
    if (( rc != 0 )); then
      rdv_mark_master_abort "master exited with rc=${rc}" || true
    fi
  }
  trap _on_exit_rdv EXIT
fi

echo ""
echo "===== Sharding eval data into ${N_SHARDS} parts ====="
shard_dataset "${EVAL_DATA}" "${N_SHARDS}" "${SHARD_DIR}" "${SHARD_DATASET_PATTERN}"

echo ""
echo "===== Stage 1 (parallel): full eval at ${FIRST_PASS_MAX_NEW_TOKENS} tokens ====="
run_stage_parallel "stage1" \
  "${SHARD_DATASET_PATTERN}" \
  "${FIRST_PASS_OUTPUT_PATTERN}" \
  "${FIRST_PASS_LOG_PATTERN}" \
  "${FIRST_PASS_MAX_NEW_TOKENS}"

echo ""
echo "===== Merging stage 1 shards ====="
merge_shards "${FIRST_PASS_OUTPUT_PATTERN}" "${N_SHARDS}" "${FIRST_PASS_OUTPUT_PATH}"
run_analysis "stage1-analysis" "${FIRST_PASS_OUTPUT_PATH}" "${FIRST_PASS_ANALYSIS_REPORT_PATH}" "${FIRST_PASS_ANALYSIS_LOG_PATH}" "${FIRST_PASS_MAX_NEW_TOKENS}"

echo ""
echo "===== Extracting retry subset ====="
extract_retry_subset "${EVAL_DATA}" "${FIRST_PASS_ANALYSIS_REPORT_PATH}" "${SECOND_PASS_DATASET_PATH}" "${SECOND_PASS_METADATA_PATH}"
RETRY_COUNT="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['retry_count'])" "${SECOND_PASS_METADATA_PATH}")"
echo "[retry] ${RETRY_COUNT} prompts to retry"

if (( RETRY_COUNT > 0 )); then
  echo ""
  echo "===== Stage 2 (parallel): retry ${RETRY_COUNT} prompts at ${SECOND_PASS_MAX_NEW_TOKENS} tokens ====="
  shard_retry_subset "${SECOND_PASS_DATASET_PATH}" "${N_SHARDS}" "${SECOND_PASS_DATASET_PATTERN}"
  run_stage_parallel "stage2" \
    "${SECOND_PASS_DATASET_PATTERN}" \
    "${SECOND_PASS_OUTPUT_PATTERN}" \
    "${SECOND_PASS_LOG_PATTERN}" \
    "${SECOND_PASS_MAX_NEW_TOKENS}"

  echo ""
  echo "===== Merging stage 2 shards ====="
  merge_shards "${SECOND_PASS_OUTPUT_PATTERN}" "${N_SHARDS}" "${SECOND_PASS_OUTPUT_PATH}"
  run_analysis "stage2-analysis" "${SECOND_PASS_OUTPUT_PATH}" "${SECOND_PASS_ANALYSIS_REPORT_PATH}" "${SECOND_PASS_ANALYSIS_LOG_PATH}" "${SECOND_PASS_MAX_NEW_TOKENS}"
else
  echo "[stage2] All correct on first pass, skipping retry."
  echo '{"summary":{"total_predictions":0,"evaluated":0,"correct":0,"accuracy_pct":0.0},"records":[]}' > "${SECOND_PASS_ANALYSIS_REPORT_PATH}"
  : > "${SECOND_PASS_OUTPUT_PATH}"; : > "${SECOND_PASS_ANALYSIS_LOG_PATH}"
fi

echo ""
echo "===== Building final merged report ====="
{ build_final_report "${FIRST_PASS_ANALYSIS_REPORT_PATH}" "${SECOND_PASS_ANALYSIS_REPORT_PATH}" "${SECOND_PASS_METADATA_PATH}" "${FINAL_ANALYSIS_REPORT_PATH}"
  echo "[final] Report: ${FINAL_ANALYSIS_REPORT_PATH}"
} 2>&1 | tee "${FINAL_ANALYSIS_LOG_PATH}"

# Post-eval pipeline finished successfully -- tell the DLC worker watcher
# it can exit. In ssh mode this is a no-op (the trap will still also run
# but rdv_mark_complete() with an un-initialized root is a no-op path).
if [[ "${POSTEVAL_WORKER_DISPATCH}" == "rendezvous" && "${POSTEVAL_RDV_MARK_COMPLETE:-true}" == "true" ]]; then
  rdv_mark_complete
  # Replace the abort-on-nonzero trap with a benign one so we don't flag
  # our own successful exit as an abort.
  trap - EXIT
fi

echo ""
echo "========== Done =========="
echo "First pass report:  ${FIRST_PASS_ANALYSIS_REPORT_PATH}"
echo "Retry report:       ${SECOND_PASS_ANALYSIS_REPORT_PATH}"
echo "Final report:       ${FINAL_ANALYSIS_REPORT_PATH}"
echo "Script log:         ${SCRIPT_LOG_PATH}"
echo "Per-shard logs:     ${SHARD_DIR}"
