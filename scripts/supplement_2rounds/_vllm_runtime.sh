# shellcheck shell=bash
# ---------------------------------------------------------------------------
# Shared vLLM runtime helpers for scripts/supplement_2rounds/{baseline,G1,G2,
# G3,Teacher}.sh. Source this from the worker scripts after they have set
# REPO_ROOT, MODEL_CUDA_VISIBLE_DEVICES, VLLM_TP_SIZE and PROGRESS_HELPER.
#
# What this file owns:
#   1) DLC / virtualized-A100 stability env vars
#        - VLLM_DISABLE_CUSTOM_ALL_REDUCE   (custom-all-reduce racy on DLC)
#        - NCCL_P2P_LEVEL=NVL                (allow NVLink P2P, ban PCIe P2P)
#        - NCCL_NET_GDR_DISABLE=1            (no GPUDirect RDMA fallback;
#                                            stops `mlx5:1 async fatal QP /
#                                            local access violation` on
#                                            cross-NUMA RoCE loopback)
#        - VLLM_RPC_TIMEOUT                  (10 min; vLLM default 10s
#                                            way too short for 27B teachers)
#        - /dev/shm remount to 16 GiB        (default 64 MB blocks shm_broadcast)
#        - VLLM_ENFORCE_EAGER                (escape hatch; honored by
#                                            vllm_generate_progress.py)
#   2) Pre-flight cluster topology dump
#        - hostname / IPs
#        - nvidia-smi -L / nvlink --status / topo -m
#        - IB devices listing
#        - WARN if TP>1 but no NVLink edges (= GPUs spread across hosts)
#   3) run_vllm_generation_with_retry() helper
#        - relaunch vLLM up to VLLM_MAX_FATAL_RETRIES times if exit log
#          shows transient cluster faults (NCCL IB QP fatal,
#          shm_broadcast timeout, RPC timeout, NCCL watchdog, OOM that
#          followed a NCCL fatal, ...)
#        - kills orphan multiproc TP workers + waits for GPU memory to
#          drain before each retry, so the new attempt doesn't OOM into
#          GPUs still held by the dead-but-running workers from the
#          previous attempt
#        - hard faults bubble up immediately (no infinite retry)
# ---------------------------------------------------------------------------

# 1) DLC stability env vars ------------------------------------------------
export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"

# Allow P2P over NVLink only. The previous default of NCCL_P2P_DISABLE=1
# was too coarse: on a single-host 8x A100-SXM4 box (NVSwitch full mesh)
# disabling NVLink P2P forced NCCL to fall back to RoCE GDRDMA between
# the two NUMA halves (GPUs 0-3 vs 4-7), and that cross-NUMA RoCE
# loopback periodically tripped `mlx5:1 async fatal event on QP /
# local access violation work queue error` after ~32 long-tail prompts.
# NCCL_P2P_LEVEL=NVL keeps NVLink/NVSwitch as the fast path while still
# refusing the (separately-flaky) PCIe-P2P path that the old default
# was actually trying to ban.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
# If NCCL_P2P_DISABLE=1 is inherited from somewhere upstream (it was the
# previous default in this file), it would override NCCL_P2P_LEVEL.
# Drop it so NVLink P2P actually engages. To restore the old behavior
# explicitly, export NCCL_P2P_DISABLE=1 *after* sourcing this file.
[[ "${NCCL_P2P_DISABLE:-}" == "1" ]] && unset NCCL_P2P_DISABLE

# Defense in depth: even if a future topology change pushes some path
# back to NET, keep GPUDirect RDMA off. RoCE staged through host RAM is
# slower per-step but eliminates the GDR-page-unmap window that fires
# the QP-fatal asynchronously while a DMA is in flight.
export NCCL_NET_GDR_DISABLE="${NCCL_NET_GDR_DISABLE:-1}"

export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600000}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_MAX_FATAL_RETRIES="${VLLM_MAX_FATAL_RETRIES:-2}"
NCCL_FATAL_RETRY_BACKOFF_SECS="${NCCL_FATAL_RETRY_BACKOFF_SECS:-60}"
# How long to wait for GPU memory to drain before a retry. After we
# pkill the orphan workers from the previous attempt, NCCL/CUDA cleanup
# can take a few seconds. If we relaunch immediately, the new vLLM
# tries to allocate KV cache on still-held memory and CUDA-OOMs, and
# OOM looks different enough from the original NCCL fatal that the
# retry helper would otherwise (a) retry into the same OOM, or (b)
# misclassify it as non-transient and give up. Polling avoids both.
GPU_DRAIN_MAX_WAIT_SECS="${GPU_DRAIN_MAX_WAIT_SECS:-90}"
GPU_DRAIN_THRESHOLD_MIB="${GPU_DRAIN_THRESHOLD_MIB:-2048}"

# /dev/shm remount: vLLM v1 multiproc_executor uses shm_broadcast; default
# DLC /dev/shm is 64 MB which 8 TP workers fill instantly -> the broadcast
# queue blocks and the (default 60s, here 600s) RPC timeout fires every
# step. Bumping to 16 GiB is a virtual cap (tmpfs, doesn't pre-allocate
# RAM). Failing remount is non-fatal; DSW already has a huge /dev/shm.
_shm_kb=$(df --output=avail -k /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
if [[ -n "${_shm_kb}" && "${_shm_kb}" -lt 4194304 ]]; then  # < 4 GiB
  echo "[shm] /dev/shm has ${_shm_kb} KiB; remounting to 16 GiB"
  mount -o remount,size=16g /dev/shm 2>/dev/null \
    && echo "[shm] OK -> $(df -h /dev/shm | tail -1)" \
    || echo "[shm] remount failed (no CAP_SYS_ADMIN); ask DLC to set shm-size=16Gi"
fi

# 2) Pre-flight cluster topology dump --------------------------------------
echo "================================================================"
echo "[pre-flight] cluster topology"
echo "[pre-flight] hostname:    $(hostname) ($(hostname -i 2>/dev/null || echo 'no -i'))"
echo "[pre-flight] kernel:      $(uname -r)"
nvidia-smi -L 2>/dev/null | sed 's/^/[pre-flight] /' || true
echo "[pre-flight] --- NVLink status (NA / 0 = no NVLink; healthy single-node 8GPU shows NVLink) ---"
nvidia-smi nvlink --status 2>/dev/null | head -40 | sed 's/^/[pre-flight] /' \
  || echo "[pre-flight] (nvidia-smi nvlink not available)"
echo "[pre-flight] --- GPU topology (NV# = NVLink direct, PXB/PHB = host bridge, NS = no support) ---"
nvidia-smi topo -m 2>/dev/null | head -30 | sed 's/^/[pre-flight] /' || true
echo "[pre-flight] --- IB devices (active in cross-node NCCL) ---"
if command -v ibv_devices >/dev/null 2>&1; then
  ibv_devices 2>/dev/null | sed 's/^/[pre-flight] /'
elif [[ -d /sys/class/infiniband ]]; then
  ls /sys/class/infiniband/ 2>/dev/null | sed 's/^/[pre-flight] IB device: /'
else
  echo "[pre-flight] (no IB tooling and no /sys/class/infiniband; pure-host setup)"
fi
echo "[pre-flight] --- CUDA visible: ${MODEL_CUDA_VISIBLE_DEVICES:-?} (TP=${VLLM_TP_SIZE:-?}) ---"
echo "[pre-flight] --- NCCL routing controls (effective env) ---"
echo "[pre-flight] NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-(unset, defaults to SYS)}"
echo "[pre-flight] NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-(unset)}"
echo "[pre-flight] NCCL_NET_GDR_DISABLE=${NCCL_NET_GDR_DISABLE:-(unset)}"
echo "[pre-flight] NCCL_IB_HCA=${NCCL_IB_HCA:-(unset)}"
echo "[pre-flight] VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-(unset)}"
echo "[pre-flight] VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-(unset)}ms"
echo "[pre-flight] VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-(unset)}"
# Heuristic: TP>1 with no NVLink (NV#) edges = GPUs likely span 2+ hosts
# -> NCCL falls back to IB GDR -> hits the mlx5 QP fatal we saw on PAI-DLC.
_topo_dump="$(nvidia-smi topo -m 2>/dev/null || true)"
if [[ "${VLLM_TP_SIZE:-1}" -gt 1 ]] && ! grep -q 'NV[0-9]' <<<"${_topo_dump}"; then
  echo "[pre-flight] WARN: TP=${VLLM_TP_SIZE} but no NVLink (NV#) edges visible in nvidia-smi topo."
  echo "[pre-flight] WARN: GPUs likely span 2+ hosts. NCCL will fall back to IB GDR which has"
  echo "[pre-flight] WARN: previously hit 'mlx5:1 async fatal event on QP' on this cluster"
  echo "[pre-flight] WARN: (engine dies after ~32 prompts). Re-submit with 1 worker x ${VLLM_TP_SIZE} GPU"
  echo "[pre-flight] WARN: and verify GPUs land on a single physical node."
fi
echo "================================================================"

# 3) Retry helper ----------------------------------------------------------

# Decide whether the tail of the vLLM log shows a transient cluster fault
# (worth retrying) vs a code/config bug (re-running won't help).
#
# Includes CUDA OOM and NCCL Cuda failure: in the cleanup-orphans-and-
# retry world, the second attempt can legitimately race with leftover
# memory that's still being freed by the dying multiproc workers, and
# get a transient OOM. Treat that as retryable so the next attempt
# (after _wait_for_gpu_memory_drain) gets a clean slate.
_is_transient_vllm_fault() {
  local log="$1"
  [[ -f "${log}" ]] || return 1
  tail -n 4000 "${log}" 2>/dev/null | grep -qE \
    'NCCL WARN.*async fatal event on QP|NCCL WARN.*local access violation|RPC call to .* timed out|EngineCore encountered a fatal error|EngineDeadError|TimeoutError: RPC call|shm_broadcast.*timeout|Watchdog caught collective operation timeout|c10::DistBackendError|torch\.cuda\.OutOfMemoryError|CUDA out of memory|NCCL WARN Cuda failure|NCCL error.*unhandled cuda error|RuntimeError: NCCL'
}

# Force-kill any orphan vLLM worker tree that the previous attempt left
# behind. When EngineCore raises EngineDeadError, only the parent Python
# process exits; the 8 multiproc TP workers and the EngineCore subprocess
# stay hung on a dead NCCL collective and continue holding all GPU
# memory + KV cache. Without this, a retry will CUDA-OOM at LLM(...)
# load time on every GPU.
#
# We can't use "kill the bash pgid" because the previous attempt was
# launched as `cmd 2>&1 | tee -a log` so $! gives us the tee, not vLLM.
# Pattern-match by name instead (only one vLLM run per pod is the
# expected pattern, see baseline.sh / G*.sh — they're sequential).
_force_kill_orphan_vllm_workers() {
  echo "[runtime] killing orphan vLLM workers..."
  # Order: kill TP/multiproc workers first so they can't try to respawn,
  # then EngineCore, then the launcher script.
  pkill -9 -f 'multiproc_executor' 2>/dev/null || true
  pkill -9 -f 'vllm.v1.engine.core' 2>/dev/null || true
  pkill -9 -f 'EngineCore' 2>/dev/null || true
  pkill -9 -f 'vllm_generate_progress' 2>/dev/null || true
  sleep 2
  # Second sweep for any straggler that respawned during the first.
  pkill -9 -f 'multiproc_executor' 2>/dev/null || true
  pkill -9 -f 'vllm_generate_progress' 2>/dev/null || true
  sleep 1
}

# Wait for visible GPUs to drop below GPU_DRAIN_THRESHOLD_MIB. Returns
# 0 even on timeout (we still try to relaunch; the next attempt will
# either work or hit OOM, which is now in the transient signature).
_wait_for_gpu_memory_drain() {
  local cuda_csv="${1:-}"
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local waited=0 max_used=0
  while (( waited < GPU_DRAIN_MAX_WAIT_SECS )); do
    if [[ -n "${cuda_csv}" ]]; then
      max_used="$(nvidia-smi --id="${cuda_csv}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk 'BEGIN{m=0} {if($1+0>m) m=$1+0} END{print m+0}')"
    else
      max_used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk 'BEGIN{m=0} {if($1+0>m) m=$1+0} END{print m+0}')"
    fi
    if (( max_used <= GPU_DRAIN_THRESHOLD_MIB )); then
      echo "[runtime] GPU memory drained: max ${max_used} MiB <= ${GPU_DRAIN_THRESHOLD_MIB} MiB after ${waited}s"
      return 0
    fi
    sleep 3
    waited=$(( waited + 3 ))
  done
  echo "[runtime] WARN: GPU memory still ${max_used} MiB after ${GPU_DRAIN_MAX_WAIT_SECS}s; retrying anyway"
  return 0
}

# Resume / crash-recovery helper.
#
# If a previous run crashed AFTER vLLM had written its full output (e.g.
# bash itself died because someone edited the script while it was paused
# inside an 11-hour vLLM call, see G1.sh / 2026-04-28 incident), we want
# the next invocation to skip the stage instead of redoing 11 hours of
# work. We declare a stage "already complete" iff the output_path exists
# AND has at least min(dataset_lines, POST_EVAL_MAX_SAMPLES) lines.
#
# If you ever need to force a re-run despite a complete output (e.g. the
# old jsonl was corrupted), set FORCE_RERUN_VLLM=1 or just `rm` the file.
_stage_output_already_complete() {
  local stage_name="$1"
  local output_path="$2"
  local dataset_path="$3"
  if [[ "${FORCE_RERUN_VLLM:-0}" == "1" ]]; then
    return 1
  fi
  [[ -f "${output_path}" ]] || return 1
  [[ -s "${output_path}" ]] || return 1
  local actual expected dataset_lines max_samples
  actual="$(wc -l < "${output_path}" 2>/dev/null || echo 0)"
  if [[ -f "${dataset_path}" ]]; then
    dataset_lines="$(wc -l < "${dataset_path}" 2>/dev/null || echo 0)"
  else
    dataset_lines=0
  fi
  max_samples="${POST_EVAL_MAX_SAMPLES:-0}"
  if (( max_samples > 0 && dataset_lines > 0 )); then
    expected=$(( max_samples < dataset_lines ? max_samples : dataset_lines ))
  elif (( max_samples > 0 )); then
    expected="${max_samples}"
  elif (( dataset_lines > 0 )); then
    expected="${dataset_lines}"
  else
    # Can't determine expected count; play it safe and re-run.
    return 1
  fi
  if (( actual >= expected )); then
    echo "[${stage_name}] resume: ${output_path} already has ${actual} rows (expected ${expected}); skipping vLLM"
    echo "[${stage_name}] (set FORCE_RERUN_VLLM=1 or remove the file to force a re-run)"
    return 0
  fi
  echo "[${stage_name}] ${output_path} has ${actual} rows but expected ${expected}; running vLLM"
  return 1
}

# run_vllm_generation_with_retry STAGE LOG_PATH OUTPUT_PATH DATASET_PATH CMD...
#   STAGE         - human label, e.g. "stage1"
#   LOG_PATH      - file to tee to (also scanned for transient-fault signature)
#   OUTPUT_PATH   - vLLM --output_path. If already complete we skip CMD.
#   DATASET_PATH  - vLLM --dataset (jsonl). Used to compute expected
#                   row count for the resume check.
#   CMD...        - the actual command to run (with all its args)
# Returns 0 on eventual success (or skipped); non-zero on either
# non-transient error or after VLLM_MAX_FATAL_RETRIES exhausted retries.
run_vllm_generation_with_retry() {
  local stage_name="$1"; shift
  local log_path="$1"; shift
  local output_path="$1"; shift
  local dataset_path="$1"; shift
  if _stage_output_already_complete "${stage_name}" "${output_path}" "${dataset_path}"; then
    return 0
  fi
  local total_attempts=$(( VLLM_MAX_FATAL_RETRIES + 1 ))
  local attempt=0
  local rc=0
  while (( attempt < total_attempts )); do
    attempt=$(( attempt + 1 ))
    if (( attempt == 1 )); then
      echo "[${stage_name}] launching vLLM"
    else
      echo "[${stage_name}] retry ${attempt}/${total_attempts} after transient fault"
      _force_kill_orphan_vllm_workers
      _wait_for_gpu_memory_drain "${MODEL_CUDA_VISIBLE_DEVICES:-}"
      echo "[${stage_name}] sleeping ${NCCL_FATAL_RETRY_BACKOFF_SECS}s for NCCL/IB driver to settle..."
      sleep "${NCCL_FATAL_RETRY_BACKOFF_SECS}"
    fi
    set +o pipefail
    "$@" 2>&1 | tee -a "${log_path}"
    rc=${PIPESTATUS[0]}
    set -o pipefail
    if (( rc == 0 )); then
      echo "[${stage_name}] vLLM completed (attempt ${attempt})"
      return 0
    fi
    if _is_transient_vllm_fault "${log_path}"; then
      echo "[${stage_name}] WARN: vLLM exited rc=${rc} with transient fault signature (attempt ${attempt})"
      # Always sweep orphans here too, even on non-final attempt: the
      # next iteration's _force_kill at top will re-do this, but doing
      # it now means GPU memory starts draining during the backoff
      # sleep instead of after it.
      _force_kill_orphan_vllm_workers
      if (( attempt >= total_attempts )); then
        echo "[${stage_name}] ERROR: ${total_attempts} attempts all hit transient faults; giving up"
        return "${rc}"
      fi
      continue
    fi
    echo "[${stage_name}] ERROR: vLLM exited rc=${rc} with NON-transient fault; not retrying"
    return "${rc}"
  done
  return "${rc}"
}
