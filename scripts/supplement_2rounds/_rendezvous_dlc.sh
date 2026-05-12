# shellcheck shell=bash
# ---------------------------------------------------------------------------
# DLC-mode post-eval rendezvous helpers.
#
# Purpose
# -------
# In DLC multi-pod mode there's no sshd inside worker pods, so the "head
# ssh's into worker to run a second shard" pattern used by
# G{2,3}_2node.sh fails with `Connection refused`. This file implements a
# file-based RPC ("rendezvous") that works instead:
#
#   * Master writes a JSON **request** file per stage shard.
#   * Worker (which is parked in a watcher loop after the training
#     ray-join ended) sees the request, runs vLLM, writes an **OK** or
#     **ERR** sentinel.
#   * Master polls for the sentinel, picks up the output (which lives on
#     the OSS-shared RUN_DIR), and continues the pipeline.
#   * When master has merged the final report it writes a **complete**
#     sentinel; worker exits 0.
#
# The rendezvous directory lives under RUN_DIR, which is on OSS and is
# visible from both pods. All file writes use a tmp+rename idiom so a
# reader never observes a torn write (rename is atomic on the underlying
# ossfs2 transport for the whole-object path).
#
# Public API
# ----------
# Master side (sourced from G{2,3}_2node.sh):
#   rdv_init_root            - set/create the rendezvous root for this run
#   rdv_mark_complete        - tell worker the whole post-eval is done
#   rdv_cleanup_run_started  - at start of a new post-eval, delete stale
#                              sentinels from a previous AIMaster-restarted
#                              attempt so we don't pick them up by mistake
#   run_shard_rendezvous ... - drop-in replacement for run_shard_ssh() when
#                              POSTEVAL_WORKER_DISPATCH=rendezvous
#
# Worker side (sourced from the runner's worker bootstrap):
#   rdv_init_root            - same as master
#   posteval_worker_watch    - block in a watcher loop until master signals
#                              complete (or the watcher hits its max-wait)
#
# Required env on both sides:
#   RUN_DIR          - OSS-shared run dir (same value on master and worker)
#   TEACHER_PYTHON_BIN - python inside teacher venv (for vLLM + JSON parsing)
#   PROGRESS_HELPER  - path to scripts/supplement/vllm_generate_progress.py
#   MODEL_PATH       - HF checkpoint dir
#
# Required env on master side (per-call, stored in request file):
#   POST_EVAL_PROMPT_MAX_LEN, POST_EVAL_TEMPERATURE, POST_EVAL_TOP_P,
#   POST_EVAL_REPETITION_PENALTY, POST_EVAL_MAX_SAMPLES,
#   POST_EVAL_BEST_OF_N, VLLM_MAX_NUM_SEQS, VLLM_PROGRESS_BATCH_SIZE,
#   VLLM_SEED, VLLM_ENABLE_PREFIX_CACHING, INPUT_TEMPLATE (optional)
#
# Tunables (env):
#   POSTEVAL_RDV_POLL_SECS       - poll interval (default 5s)
#   POSTEVAL_RDV_MASTER_TIMEOUT  - master max-wait for a shard (default
#                                  7200s = 2h per stage)
#   POSTEVAL_RDV_WORKER_TIMEOUT  - worker max idle wait before giving up
#                                  (default 10800s = 3h total post-eval)
# ---------------------------------------------------------------------------

_RDV_ROOT=""

rdv_init_root() {
  # $1: RUN_DIR; subdir "post_eval_rendezvous" under a dedicated dir so it
  # can't collide with the training-time dlc_rendezvous/worker_*_ip.txt
  # files (which have their own lifecycle / cleanup semantics).
  local run_dir="$1"
  if [[ -z "${run_dir}" ]]; then
    echo "[rdv] ERROR: rdv_init_root requires RUN_DIR arg" >&2
    return 1
  fi
  _RDV_ROOT="${run_dir}/dlc_rendezvous/post_eval"
  mkdir -p "${_RDV_ROOT}"
}

_rdv_request_path()   { printf '%s/%s.request.json' "${_RDV_ROOT}" "$1"; }
_rdv_ok_path()        { printf '%s/%s.done.ok' "${_RDV_ROOT}" "$1"; }
_rdv_err_path()       { printf '%s/%s.done.err' "${_RDV_ROOT}" "$1"; }
_rdv_complete_path()  { printf '%s/all_done.marker' "${_RDV_ROOT}"; }
_rdv_master_abort_path() { printf '%s/master_abort.marker' "${_RDV_ROOT}"; }

# Atomic write: render to a tmp file in the same dir, then rename.
_rdv_atomic_write() {
  local target="$1"
  local payload="$2"
  local dir; dir="$(dirname "${target}")"
  local tmp; tmp="$(mktemp "${dir}/.rdv.XXXXXX")"
  printf '%s' "${payload}" > "${tmp}"
  # best-effort sync so ossfs2 flushes before rename
  sync 2>/dev/null || true
  mv -f "${tmp}" "${target}"
}

rdv_cleanup_run_started() {
  # Remove stale files from a previous attempt of the same RUN_DIR
  # (AIMaster restart of a failed master). We keep the directory itself so
  # worker can still see the dir come up quickly.
  if [[ -z "${_RDV_ROOT}" ]]; then
    return 0
  fi
  find "${_RDV_ROOT}" -maxdepth 1 -type f \( -name '*.request.json' \
      -o -name '*.done.ok' -o -name '*.done.err' \
      -o -name 'all_done.marker' -o -name 'master_abort.marker' \
      -o -name '.rdv.*' \) -delete 2>/dev/null || true
}

rdv_mark_complete() {
  _rdv_atomic_write "$(_rdv_complete_path)" "ok $(date -Is)"
}

# Called by master on hard failure: let worker exit instead of hanging.
rdv_mark_master_abort() {
  local reason="${1:-unknown}"
  _rdv_atomic_write "$(_rdv_master_abort_path)" "aborted: ${reason} @ $(date -Is)"
}

# ---------------------------------------------------------------------------
# Master-side: run_shard_rendezvous
# ---------------------------------------------------------------------------
# Args mirror run_shard_ssh / run_shard_local:
#   stage_name, dataset_path, output_path, log_path, max_new_tokens,
#   cuda_devices, tp_size
#
# Returns 0 on success, non-zero otherwise. Side effect: writes the
# request file and leaves the done sentinel in place for audit. Re-runs
# are safe because stage output resume logic is per-shard (the worker
# side invokes _stage_output_already_complete via the helper).
run_shard_rendezvous() {
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
  if [[ -z "${_RDV_ROOT}" ]]; then
    echo "[${stage_name}] ERROR: rdv_init_root was not called" >&2
    return 2
  fi

  local request_file ok_file err_file
  request_file="$(_rdv_request_path "${stage_name}")"
  ok_file="$(_rdv_ok_path "${stage_name}")"
  err_file="$(_rdv_err_path "${stage_name}")"

  # Clear any sentinel from an earlier attempt at this same stage name so
  # we don't mistake it for the one we're about to produce.
  rm -f "${ok_file}" "${err_file}" 2>/dev/null || true

  # Build the request JSON via the analysis venv's python (always present).
  local request_json
  request_json="$(
    STAGE="${stage_name}" DATASET="${dataset_path}" \
    OUTPUT="${output_path}" LOG_PATH="${log_path}" \
    MAX_NEW="${max_new_tokens}" CUDA="${cuda_devices}" TP="${tp_size}" \
    MODEL="${MODEL_PATH}" PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN}" \
    TEMP="${POST_EVAL_TEMPERATURE}" TOPP="${POST_EVAL_TOP_P}" \
    REP_PEN="${POST_EVAL_REPETITION_PENALTY}" \
    MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES}" \
    BEST_OF="${POST_EVAL_BEST_OF_N}" \
    MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
    PROGRESS_BS="${VLLM_PROGRESS_BATCH_SIZE}" \
    SEED="${VLLM_SEED}" \
    ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}" \
    INPUT_TEMPLATE_VAL="${INPUT_TEMPLATE:-}" \
    PROGRESS_HELPER="${PROGRESS_HELPER}" \
    TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN}" \
    "${ANALYSIS_PYTHON_BIN}" -c '
import json, os, time
req = {
    "schema": "posteval_rendezvous/v1",
    "stage": os.environ["STAGE"],
    "created_at": time.time(),
    "dataset_path": os.environ["DATASET"],
    "output_path": os.environ["OUTPUT"],
    "log_path": os.environ["LOG_PATH"],
    "max_new_tokens": int(os.environ["MAX_NEW"]),
    "cuda_devices": os.environ["CUDA"],
    "tp_size": int(os.environ["TP"]),
    "model_path": os.environ["MODEL"],
    "prompt_max_len": int(os.environ["PROMPT_MAX_LEN"]),
    "temperature": float(os.environ["TEMP"]),
    "top_p": float(os.environ["TOPP"]),
    "repetition_penalty": float(os.environ["REP_PEN"]),
    "max_samples": int(os.environ["MAX_SAMPLES"]),
    "best_of_n": int(os.environ["BEST_OF"]),
    "max_num_seqs": int(os.environ["MAX_NUM_SEQS"]),
    "progress_batch_size": int(os.environ["PROGRESS_BS"]),
    "seed": int(os.environ["SEED"]),
    "enable_prefix_caching": os.environ["ENABLE_PREFIX_CACHING"].lower() == "true",
    "input_template": os.environ.get("INPUT_TEMPLATE_VAL", "") or None,
    "progress_helper": os.environ["PROGRESS_HELPER"],
    "teacher_python_bin": os.environ["TEACHER_PYTHON_BIN"],
}
print(json.dumps(req, ensure_ascii=False, indent=2))
'
  )"
  if [[ -z "${request_json}" ]]; then
    echo "[${stage_name}] ERROR: failed to build rendezvous request JSON" >&2
    return 2
  fi

  echo "[${stage_name}] (rendezvous) dispatching to worker: max_new=${max_new_tokens} tp=${tp_size} cuda=${cuda_devices}"
  echo "[${stage_name}]   request: ${request_file}"
  _rdv_atomic_write "${request_file}" "${request_json}"

  # Wait for worker done sentinel.
  local poll="${POSTEVAL_RDV_POLL_SECS:-5}"
  local max_wait="${POSTEVAL_RDV_MASTER_TIMEOUT:-7200}"
  local waited=0
  while (( waited < max_wait )); do
    if [[ -f "${err_file}" ]]; then
      echo "[${stage_name}] (rendezvous) worker reported FAILURE: $(cat "${err_file}")" >&2
      return 1
    fi
    if [[ -f "${ok_file}" ]]; then
      echo "[${stage_name}] (rendezvous) worker reported OK after ${waited}s"
      return 0
    fi
    sleep "${poll}"
    waited=$(( waited + poll ))
  done

  echo "[${stage_name}] (rendezvous) ERROR: worker did not respond in ${max_wait}s" >&2
  rdv_mark_master_abort "worker-timeout@${stage_name}"
  return 1
}

# ---------------------------------------------------------------------------
# Worker-side: posteval_worker_watch
# ---------------------------------------------------------------------------
# Blocks in a loop. On each iteration it scans for new *.request.json
# files, fulfills each one by invoking the progress helper, then writes
# the matching .done.ok or .done.err. Exits 0 when it sees
# all_done.marker (success) or master_abort.marker (master aborted).
# Exits 1 on watcher-wide timeout.
#
# If _stage_output_already_complete returns 0 for the shard, we skip
# vLLM and go straight to writing .done.ok (mirrors the master side
# logic).
posteval_worker_watch() {
  if [[ -z "${_RDV_ROOT}" ]]; then
    echo "[posteval-worker] ERROR: rdv_init_root was not called" >&2
    return 2
  fi

  local poll="${POSTEVAL_RDV_POLL_SECS:-5}"
  local max_wait="${POSTEVAL_RDV_WORKER_TIMEOUT:-10800}"
  local waited=0
  local processed_list=""  # space-separated list of stage names already handled

  echo "[posteval-worker] entering watch loop (root=${_RDV_ROOT}, max_wait=${max_wait}s, poll=${poll}s)"
  while true; do
    # Exit conditions first.
    if [[ -f "$(_rdv_complete_path)" ]]; then
      echo "[posteval-worker] saw all_done.marker; exiting"
      return 0
    fi
    if [[ -f "$(_rdv_master_abort_path)" ]]; then
      echo "[posteval-worker] saw master_abort.marker; exiting"
      cat "$(_rdv_master_abort_path)" 2>/dev/null || true
      return 0
    fi

    # Pick up any new request.
    shopt -s nullglob
    local req
    for req in "${_RDV_ROOT}"/*.request.json; do
      [[ -f "${req}" ]] || continue
      local stage_name
      stage_name="$(basename "${req}" .request.json)"
      # Skip requests whose done sentinel is already in place (e.g. we
      # successfully processed them earlier in this same worker lifetime,
      # or a previous worker attempt wrote it and died after).
      if [[ -f "$(_rdv_ok_path "${stage_name}")" || -f "$(_rdv_err_path "${stage_name}")" ]]; then
        continue
      fi
      # Avoid double-processing within one loop iteration.
      case " ${processed_list} " in
        *" ${stage_name} "*) continue ;;
      esac
      processed_list+=" ${stage_name}"

      echo "[posteval-worker] fulfilling stage=${stage_name}"
      _posteval_worker_fulfill "${req}" "${stage_name}"
    done
    shopt -u nullglob

    sleep "${poll}"
    waited=$(( waited + poll ))
    if (( waited >= max_wait )); then
      echo "[posteval-worker] ERROR: idle watch loop timed out after ${max_wait}s" >&2
      return 1
    fi
  done
}

# Internal: run one shard from a request file. Writes .done.ok on success
# (including the "already complete" resume case) or .done.err on failure.
_posteval_worker_fulfill() {
  local req_file="$1"
  local stage_name="$2"

  # Load request into local vars via python (one shell eval line).
  local spec_json
  spec_json="$(cat "${req_file}")"
  local cuda_devices tp_size dataset_path output_path log_path
  local max_new_tokens model_path prompt_max_len
  local temperature top_p repetition_penalty max_samples best_of_n
  local max_num_seqs progress_batch_size seed
  local enable_prefix_caching input_template progress_helper teacher_python_bin
  local assignments
  assignments="$(SPEC="${spec_json}" "${TEACHER_PYTHON_BIN}" -c '
import json, os, shlex
d = json.loads(os.environ["SPEC"])
def emit(k, v):
    print(f"{k}={shlex.quote(str(v))}")
emit("cuda_devices", d["cuda_devices"])
emit("tp_size", d["tp_size"])
emit("dataset_path", d["dataset_path"])
emit("output_path", d["output_path"])
emit("log_path", d["log_path"])
emit("max_new_tokens", d["max_new_tokens"])
emit("model_path", d["model_path"])
emit("prompt_max_len", d["prompt_max_len"])
emit("temperature", d["temperature"])
emit("top_p", d["top_p"])
emit("repetition_penalty", d["repetition_penalty"])
emit("max_samples", d["max_samples"])
emit("best_of_n", d["best_of_n"])
emit("max_num_seqs", d["max_num_seqs"])
emit("progress_batch_size", d["progress_batch_size"])
emit("seed", d["seed"])
emit("enable_prefix_caching", 1 if d.get("enable_prefix_caching") else 0)
emit("input_template", d.get("input_template") or "")
emit("progress_helper", d["progress_helper"])
emit("teacher_python_bin", d["teacher_python_bin"])
')" || {
    _rdv_atomic_write "$(_rdv_err_path "${stage_name}")" "request parse failed"
    return 1
  }
  eval "${assignments}"

  # Resume-skip: output already has enough rows → write .done.ok without
  # running vLLM. Keeps total post-eval wall time minimal on retries.
  if _stage_output_already_complete "${stage_name}-worker-resume" "${output_path}" "${dataset_path}"; then
    _rdv_atomic_write "$(_rdv_ok_path "${stage_name}")" "resume-skipped $(date -Is)"
    return 0
  fi

  mkdir -p "$(dirname "${log_path}")" "$(dirname "${output_path}")"

  # Build the vllm CLI and run it with the same retry helper the local
  # shard gets on the master side.
  local -a vllm_cmd=(
    env "CUDA_VISIBLE_DEVICES=${cuda_devices}"
    "${teacher_python_bin}" "${progress_helper}"
    --pretrain "${model_path}"
    --dataset "${dataset_path}"
    --input_key question
    --output_path "${output_path}"
    --prompt_max_len "${prompt_max_len}"
    --max_new_tokens "${max_new_tokens}"
    --temperature "${temperature}"
    --top_p "${top_p}"
    --repetition_penalty "${repetition_penalty}"
    --max_samples "${max_samples}"
    --best_of_n "${best_of_n}"
    --tp_size "${tp_size}"
    --max_num_seqs "${max_num_seqs}"
    --progress_batch_size "${progress_batch_size}"
    --seed "${seed}"
  )
  [[ -n "${input_template}" ]] && vllm_cmd+=(--input_template "${input_template}")
  [[ "${enable_prefix_caching}" == "1" ]] && vllm_cmd+=(--enable_prefix_caching)

  echo "[posteval-worker ${stage_name}] max_new=${max_new_tokens} tp=${tp_size} cuda=${cuda_devices}"
  local rc=0
  if declare -F run_vllm_generation_with_retry >/dev/null; then
    run_vllm_generation_with_retry "${stage_name}-worker" "${log_path}" \
        "${output_path}" "${dataset_path}" "${vllm_cmd[@]}" || rc=$?
  else
    "${vllm_cmd[@]}" 2>&1 | tee -a "${log_path}" || rc=$?
  fi

  if (( rc == 0 )); then
    _rdv_atomic_write "$(_rdv_ok_path "${stage_name}")" "ok $(date -Is)"
  else
    _rdv_atomic_write "$(_rdv_err_path "${stage_name}")" "vllm rc=${rc}"
  fi
  return "${rc}"
}
