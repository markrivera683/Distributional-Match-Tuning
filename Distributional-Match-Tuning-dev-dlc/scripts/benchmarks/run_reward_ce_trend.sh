#!/usr/bin/env bash
# Run the paper's standalone reward / CE evaluation across every actor checkpoint
# of `paperqa_ebft_trend_seed43`, plus the base-model baseline.
#
# Output: outputs/paperqa_ebft_trend_seed43/offline_reward_ce_trend/step{N}.json
# Each JSON contains:
#   metrics.reward_pass1_gt          (alignment reward, pass@1)
#   metrics.reward_pass1_diversity   (diversity reward, pass@1)
#   metrics.reward_pass1_effective   (combined, pass@1)
#   metrics.reward_passk_*           (same, pass@k)
#   metrics.full_ce_loss / full_perplexity      (CE on all tokens)
#   metrics.answer_ce_loss / answer_perplexity  (CE only on answer tokens)
#
# Usage:
#   bash scripts/benchmarks/run_reward_ce_trend.sh                # smoke (32 chunks)
#   N_CHUNKS=256 bash scripts/benchmarks/run_reward_ce_trend.sh   # full
#   GPU=2 STEP=971 bash scripts/benchmarks/run_reward_ce_trend.sh # one ckpt
set -euo pipefail

REPO_ROOT="/root/code/Distributional-Match-Tuning"
PAPER_REPO="$REPO_ROOT/reproduce/ebft_openrlhf-main"
PYBIN="$REPO_ROOT/.venv/bin/python"

RUN_DIR="$REPO_ROOT/outputs/paperqa_ebft_trend_seed43"
CKPT_DIR="$RUN_DIR/checkpoints"
OUT_DIR="$RUN_DIR/offline_reward_ce_trend"
mkdir -p "$OUT_DIR"

BASE_MODEL="${BASE_MODEL:-/root/model}"
EVAL_DATASET="${EVAL_DATASET:-sjelassi/opencode-instruct_100k_200tok}"

# --- Match training-time eval setup (from run_paper_qa_ebft_trend.sh) --------
N_SAMPLES="${N_SAMPLES:-4}"
TEMP="${TEMP:-1.0}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
GEN_LEN="${GEN_LEN:-8}"
CTX_LEN="${CTX_LEN:-8}"
STRIDE="${STRIDE:-8}"
N_CHUNKS="${N_CHUNKS:-32}"   # number of packed (1024-token) chunks
BATCH="${BATCH:-1}"
SEED="${SEED:-43}"

# When STEP is set, run only that one checkpoint. Otherwise run baseline + all 5.
DEFAULT_STEPS=(0 38 97 194 388 971)
if [[ -n "${STEP:-}" ]]; then
  STEPS=("$STEP")
else
  STEPS=("${DEFAULT_STEPS[@]}")
fi

# Pin to GPU if requested (otherwise let CUDA_VISIBLE_DEVICES inherit from env)
if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "[reward_ce_trend] repo:   $REPO_ROOT"
echo "[reward_ce_trend] python: $PYBIN"
echo "[reward_ce_trend] base:   $BASE_MODEL"
echo "[reward_ce_trend] N_CHUNKS=$N_CHUNKS  N_SAMPLES=$N_SAMPLES  TEMP=$TEMP  GPU=${CUDA_VISIBLE_DEVICES:-(all)}"
echo "[reward_ce_trend] steps:  ${STEPS[*]}"
echo

cd "$PAPER_REPO"

for step in "${STEPS[@]}"; do
  if [[ "$step" -eq 0 ]]; then
    actor_path="$BASE_MODEL"
    out_file="$OUT_DIR/step0_baseline.json"
    log_file="$OUT_DIR/step0_baseline.log"
    label="step0 (base Qwen2.5-1.5B)"
  else
    actor_path="$CKPT_DIR/global_step${step}_hf"
    out_file="$OUT_DIR/step${step}.json"
    log_file="$OUT_DIR/step${step}.log"
    label="step${step}"
  fi

  if [[ ! -d "$actor_path" && ! -f "$actor_path" ]]; then
    echo "[skip] $label -- actor path not found: $actor_path"
    continue
  fi

  if [[ -f "$out_file" && -z "${FORCE:-}" ]]; then
    echo "[skip] $label -- $out_file already exists (set FORCE=1 to redo)"
    continue
  fi

  echo "==== $label  =>  $out_file ===="
  PYTHONPATH="$PAPER_REPO" "$PYBIN" scripts/evaluate_reward_ce.py \
      --actor_checkpoint "$actor_path" \
      --critic_checkpoint "$BASE_MODEL" \
      --eval_dataset "$EVAL_DATASET" \
      --eval_split test \
      --input_key question \
      --label_key answer \
      --eval_batch_size "$BATCH" \
      --eval_max_samples "$N_CHUNKS" \
      --n_samples_per_prompt "$N_SAMPLES" \
      --temperature "$TEMP" \
      --top_p 1.0 \
      --prompt_max_len "$PROMPT_LEN" \
      --generate_max_len "$GEN_LEN" \
      --context_max_len "$CTX_LEN" \
      --stride "$STRIDE" \
      --embed_method last_token \
      --hidden_state_method concat \
      --use_whitening \
      --output_file "$out_file" \
      --seed "$SEED" \
      2>&1 | tee "$log_file"
done

echo
echo "[reward_ce_trend] done."
