#!/usr/bin/env bash
set -uo pipefail

# AoPS Triplet Experiment Orchestration
# Runs 3 groups sequentially: baseline -> dist-only -> dist+ema
# Each group: smoke first, then formal (if smoke passes)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/mnt/workspace/EBFT/ebft_only_bundle_march19/repo/ebft_openrlhf_u1_ncfm}"
RESULTS_DIR="/mnt/workspace/runs"
SUMMARY_FILE="${RESULTS_DIR}/triplet_summary.json"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Initialize summary
echo '{"experiments": [], "start_time": "'$(date -u '+%Y-%m-%d %H:%M:%S UTC)'"}' > "${SUMMARY_FILE}"

log_info() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] INFO: $1"
}

log_error() {
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ERROR: $1" >&2
}

check_smoke_success() {
    local log_file="$1"
    local group="$2"
    local checks_passed=0
    
    log_info "Checking smoke success for Group ${group}..."
    
    # Check 1: Global step exists
    if grep -q "train/global_step" "${log_file}" 2>/dev/null; then
        log_info "  ✓ Global step found"
        ((checks_passed++))
    else
        log_error "  ✗ Global step NOT found"
    fi
    
    # Check 2: Evaluation completed
    if grep -q "Evaluation completed" "${log_file}" 2>/dev/null; then
        log_info "  ✓ Evaluation completed"
        ((checks_passed++))
    else
        log_error "  ✗ Evaluation NOT completed"
    fi
    
    # Check 3: No critical errors
    if ! grep -qE "(FileNotFoundError|KeyError|RuntimeError|ActorDiedError)" "${log_file}" 2>/dev/null; then
        log_info "  ✓ No critical errors"
        ((checks_passed++))
    else
        log_error "  ✗ Critical errors found"
    fi
    
    # Group 3 additional checks
    if [[ "${group}" == "3" ]]; then
        log_info "  Checking Group 3 specific metrics..."
        
        # Check 4: critic_opt_lr_group1 non-zero
        if grep -q "critic_opt_lr_group1" "${log_file}" 2>/dev/null; then
            log_info "    ✓ critic_opt_lr_group1 logged"
            ((checks_passed++))
        else
            log_error "    ✗ critic_opt_lr_group1 NOT found"
        fi
        
        # Check 5: feature_adapter_drift non-zero
        if grep -q "feature_adapter_drift" "${log_file}" 2>/dev/null; then
            log_info "    ✓ feature_adapter_drift logged"
            ((checks_passed++))
        else
            log_error "    ✗ feature_adapter_drift NOT found"
        fi
        
        # Group 3 needs 5 checks
        if [[ ${checks_passed} -ge 5 ]]; then
            return 0
        fi
    else
        # Group 1/2 needs 3 checks
        if [[ ${checks_passed} -ge 3 ]]; then
            return 0
        fi
    fi
    
    return 1
}

run_smoke() {
    local group="$1"
    local script_name="$2"
    local max_samples="$3"
    local num_episodes="$4"
    local eval_max="$5"
    
    log_info "=========================================="
    log_info "Group ${group}: SMOKE TEST"
    log_info "=========================================="
    
    local log_file="${RESULTS_DIR}/aops_g${group}_smoke.log"
    
    # Group 3 smoke uses 0 critic warmup to ensure adapter trains
    if [[ "${group}" == "3" ]]; then
        MAX_SAMPLES="${max_samples}" NUM_EPISODES="${num_episodes}" \
            EVAL_MAX_SAMPLES="${eval_max}" EVAL_STEPS=5 \
            ACTOR_NUM_GPUS_PER_NODE=1 CRITIC_NUM_GPUS_PER_NODE=1 \
            CRITIC_LR_WARMUP_RATIO=0.0 \
            bash "${SCRIPT_DIR}/${script_name}" > "${log_file}" 2>&1
    else
        MAX_SAMPLES="${max_samples}" NUM_EPISODES="${num_episodes}" \
            EVAL_MAX_SAMPLES="${eval_max}" EVAL_STEPS=5 \
            ACTOR_NUM_GPUS_PER_NODE=1 CRITIC_NUM_GPUS_PER_NODE=1 \
            bash "${SCRIPT_DIR}/${script_name}" > "${log_file}" 2>&1
    fi
    
    local exit_code=$?
    
    if [[ ${exit_code} -eq 0 ]] && check_smoke_success "${log_file}" "${group}"; then
        log_info "Group ${group} SMOKE: PASSED"
        return 0
    else
        log_error "Group ${group} SMOKE: FAILED (exit code: ${exit_code})"
        log_error "Check log: ${log_file}"
        return 1
    fi
}

run_formal() {
    local group="$1"
    local script_name="$2"
    
    log_info "=========================================="
    log_info "Group ${group}: FORMAL RUN (8 GPUs)"
    log_info "=========================================="
    
    # 8 GPU allocation: 2+2+2+2 or 2+4+1+1
    local actor_gpus=2
    local critic_gpus=2
    local ref_gpus=2
    local reward_gpus=2
    
    # Group 3 uses more GPUs for critic (adapter training)
    if [[ "${group}" == "3" ]]; then
        actor_gpus=2
        critic_gpus=4
        ref_gpus=1
        reward_gpus=1
    fi
    
    log_info "GPU Allocation: actor=${actor_gpus}, critic=${critic_gpus}, ref=${ref_gpus}, reward=${reward_gpus}"
    
    MAX_SAMPLES=-1 NUM_EPISODES=3 EVAL_MAX_SAMPLES=-1 EVAL_STEPS=100 \
        ACTOR_NUM_GPUS_PER_NODE=${actor_gpus} \
        CRITIC_NUM_GPUS_PER_NODE=${critic_gpus} \
        REF_NUM_GPUS_PER_NODE=${ref_gpus} \
        REWARD_NUM_GPUS_PER_NODE=${reward_gpus} \
        bash "${SCRIPT_DIR}/${script_name}"
    
    return $?
}

# ==========================================
# MAIN EXECUTION
# ==========================================

log_info "Starting AoPS Triplet Experiment"
log_info "Results will be saved to: ${RESULTS_DIR}"

# Group 1: Baseline
log_info ""
log_info ">>> GROUP 1: BASELINE <<<"
if run_smoke "1" "run_aops_group1_baseline.sh" "128" "1" "32"; then
    if run_formal "1" "run_aops_group1_baseline.sh"; then
        log_info "Group 1 Formal: SUCCESS"
    else
        log_error "Group 1 Formal: FAILED"
    fi
else
    log_error "Group 1 Smoke failed, skipping formal run"
fi

# Group 2: Dist-Only
log_info ""
log_info ">>> GROUP 2: DIST-ONLY <<<"
if run_smoke "2" "run_aops_group2_distonly.sh" "128" "1" "32"; then
    if run_formal "2" "run_aops_group2_distonly.sh"; then
        log_info "Group 2 Formal: SUCCESS"
    else
        log_error "Group 2 Formal: FAILED"
    fi
else
    log_error "Group 2 Smoke failed, skipping formal run"
fi

# Group 3: Dist+EMA (with stricter smoke)
log_info ""
log_info ">>> GROUP 3: DIST+EMA <<<"
if run_smoke "3" "run_aops_group3_distema.sh" "256" "1" "32"; then
    if run_formal "3" "run_aops_group3_distema.sh"; then
        log_info "Group 3 Formal: SUCCESS"
    else
        log_error "Group 3 Formal: FAILED"
    fi
else
    log_error "Group 3 Smoke failed, skipping formal run"
fi

# Generate summary
log_info ""
log_info "=========================================="
log_info "GENERATING SUMMARY"
log_info "=========================================="

python3 "${SCRIPT_DIR}/summarize_aops_results.py" --results_dir "${RESULTS_DIR}"

log_info ""
log_info "Triplet experiment completed!"
log_info "Summary: ${RESULTS_DIR}/triplet_summary.txt"
