#!/usr/bin/env bash
# Plan-B helper: prompt-template ablation on a small AOPS subset.
#
# Why this script exists
# ----------------------
# stage1 of supplement_2rounds (G1/G2/G3/...) runs the model with `--input_key
# question` and NO prompt template, i.e. the model sees raw question text. On
# the g1_rebase_0427_1553 run this produced ~16% "pure_eos" outputs (output is
# literally just <eos>) -- the model immediately gives up. One competing
# hypothesis is that this is purely a training/inference distribution mismatch:
# during ebft training the model saw question tokens preceded by <eos> from
# the previous packed sample, but at inference the question stands alone.
# Adding a tiny prompt template like "Problem: {}\n\nSolution: " might break
# that mismatch. This script tests a fixed list of templates on a small
# subset (default 300 samples) and prints a side-by-side comparison so you
# can decide whether to a) add a default INPUT_TEMPLATE to G1/G2/G3.sh,
# b) leave it alone (it's a real ebft-distill artifact), or c) revisit
# training to fix the root cause.
#
# Usage
# -----
#   bash scripts/supplement_2rounds/_template_ablation.sh /path/to/run_dir
#   RUN_DIR=/path/to/run_dir MAX_SAMPLES=300 bash scripts/supplement_2rounds/_template_ablation.sh
#
# Tunables (env vars)
#   MAX_SAMPLES       (default 300)   -- # of test prompts per template
#   MAX_NEW_TOKENS    (default 4096)  -- smaller cap so each template finishes fast
#   PROMPT_MAX_LEN    (default 512)
#   TP_SIZE           (default 8)
#   TEMPERATURE       (default 0.6)   -- match G1.sh defaults so results are comparable
#   MODEL_PATH        (default ${RUN_DIR}/model)
#   EVAL_DATA         (default /mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl)
#   TEMPLATES_FILE    (default = built-in 4-template list below; one
#                      "name|template" line per template, '{}' = question slot)
#
# Outputs go to ${RUN_DIR}/template_ablation_<TS>/.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"

_DEFAULT_TEACHER_VENV="/mnt/workspace/venvs/.teacherVenv"
[[ -d "${_DEFAULT_TEACHER_VENV}" ]] || _DEFAULT_TEACHER_VENV="${REPO_ROOT}/.teacherVenv"
TEACHER_VENV="${TEACHER_VENV:-${_DEFAULT_TEACHER_VENV}}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"

_DEFAULT_ANALYSIS_VENV="/mnt/workspace/venvs/.venv"
[[ -d "${_DEFAULT_ANALYSIS_VENV}" ]] || _DEFAULT_ANALYSIS_VENV="${REPO_ROOT}/.venv"
ANALYSIS_VENV="${ANALYSIS_VENV:-${_DEFAULT_ANALYSIS_VENV}}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"
PROGRESS_HELPER="${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py"

RUN_DIR="${RUN_DIR:-${1:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: RUN_DIR=/path/to/run bash $0"
  echo "   or: bash $0 /path/to/run"
  exit 1
fi
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model}"

MAX_SAMPLES="${MAX_SAMPLES:-300}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-512}"
MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _VISIBLE_GPUS <<< "${MODEL_CUDA_VISIBLE_DEVICES}"
TP_SIZE="${TP_SIZE:-${#_VISIBLE_GPUS[@]}}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-1234}"

[[ -x "${TEACHER_PYTHON_BIN}" ]]   || { echo "[ERROR] Not executable: ${TEACHER_PYTHON_BIN}"; exit 1; }
[[ -x "${ANALYSIS_PYTHON_BIN}" ]]  || { echo "[ERROR] Not executable: ${ANALYSIS_PYTHON_BIN}"; exit 1; }
[[ -d "${RUN_DIR}" ]]              || { echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"; exit 1; }
[[ -e "${MODEL_PATH}" ]]           || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -e "${EVAL_DATA}" ]]            || { echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"; exit 1; }
[[ -f "${PROGRESS_HELPER}" ]]      || { echo "[ERROR] PROGRESS_HELPER not found: ${PROGRESS_HELPER}"; exit 1; }

# Built-in template list. Format: name|template (with literal {} as question slot).
# Newlines and special characters are produced by `printf %b` below to make
# it easy to keep the list inline; if you want to override, set TEMPLATES_FILE
# to a file with one "name|template" line per row.
read -r -d '' DEFAULT_TEMPLATES <<'TEMPLATES_EOF' || true
raw|{}
problem_solution|Problem: {}\n\nSolution: 
question_answer|Question: {}\n\nAnswer: 
stepbystep|{}\n\nLet's solve this step by step.\n\n
TEMPLATES_EOF

if [[ -n "${TEMPLATES_FILE:-}" ]]; then
  [[ -f "${TEMPLATES_FILE}" ]] || { echo "[ERROR] TEMPLATES_FILE not found: ${TEMPLATES_FILE}"; exit 1; }
  TEMPLATES_DATA="$(cat "${TEMPLATES_FILE}")"
else
  TEMPLATES_DATA="${DEFAULT_TEMPLATES}"
fi

TS="${TS:-$(date +%m%d_%H%M)}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/template_ablation_${TS}}"
mkdir -p "${LOG_DIR}"

# Header for the human eyeballing the run.
echo "========== Plan-B template ablation =========="
echo "RUN_DIR:           ${RUN_DIR}"
echo "MODEL_PATH:        ${MODEL_PATH}"
echo "EVAL_DATA:         ${EVAL_DATA}"
echo "MAX_SAMPLES:       ${MAX_SAMPLES}"
echo "MAX_NEW_TOKENS:    ${MAX_NEW_TOKENS}"
echo "PROMPT_MAX_LEN:    ${PROMPT_MAX_LEN}"
echo "TP_SIZE:           ${TP_SIZE}"
echo "GPUs:              ${MODEL_CUDA_VISIBLE_DEVICES}"
echo "Output dir:        ${LOG_DIR}"
echo "Templates to test:"
while IFS='|' read -r name tpl; do
  [[ -z "${name}" ]] && continue
  printf '  - %s :: %q\n' "${name}" "${tpl}"
done <<< "${TEMPLATES_DATA}"
echo "================================================"

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

cd "${REPO_ROOT}"

run_one() {
  local name="$1"
  local template="$2"

  local output_jsonl="${LOG_DIR}/eval_${name}.jsonl"
  local vllm_log="${LOG_DIR}/vllm_${name}.log"
  local analysis_json="${LOG_DIR}/analysis_${name}.json"
  local analysis_log="${LOG_DIR}/analysis_${name}.log"
  local progress_json="${LOG_DIR}/progress_${name}.json"

  echo ""
  echo "===== [${name}] template=$(printf '%q' "${template}") ====="

  local -a vllm_cmd=(
    env "CUDA_VISIBLE_DEVICES=${MODEL_CUDA_VISIBLE_DEVICES}"
    "${TEACHER_PYTHON_BIN}" "${PROGRESS_HELPER}"
    --pretrain "${MODEL_PATH}"
    --dataset "${EVAL_DATA}"
    --input_key question
    --output_path "${output_jsonl}"
    --prompt_max_len "${PROMPT_MAX_LEN}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --repetition_penalty 1.0
    --max_samples "${MAX_SAMPLES}"
    --best_of_n 1
    --tp_size "${TP_SIZE}"
    --max_num_seqs 256
    --progress_batch_size 256
    --progress_json_path "${progress_json}"
    --seed "${SEED}"
  )
  # Only pass --input_template when the template is non-empty AND not literally "{}".
  # The "raw" baseline uses bare question text (matches current G1.sh behaviour).
  if [[ -n "${template}" && "${template}" != "{}" ]]; then
    vllm_cmd+=(--input_template "${template}")
  fi

  "${vllm_cmd[@]}" 2>&1 | tee "${vllm_log}"

  "${ANALYSIS_PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_eval_results.py" \
    --eval_results "${output_jsonl}" \
    --eval_dataset "${EVAL_DATA}" \
    --input_key question --label_key answer \
    --report_path "${analysis_json}" \
    --tokenizer_path "${MODEL_PATH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    2>&1 | tee "${analysis_log}"
}

while IFS='|' read -r name tpl; do
  [[ -z "${name}" ]] && continue
  # Expand \n etc. so the user-facing "{}\n\n..." actually contains newlines.
  expanded_tpl="$(printf '%b' "${tpl}")"
  run_one "${name}" "${expanded_tpl}"
done <<< "${TEMPLATES_DATA}"

echo ""
echo "================================================"
echo "  Summary table"
echo "================================================"
"${ANALYSIS_PYTHON_BIN}" - "${LOG_DIR}" <<'PY'
import json, os, sys, glob

log_dir = sys.argv[1]
rows = []
for js in sorted(glob.glob(os.path.join(log_dir, 'analysis_*.json'))):
    name = os.path.basename(js).replace('analysis_', '').replace('.json', '')
    try:
        s = json.load(open(js))['summary']
    except Exception as exc:
        print(f"[WARN] failed to read {js}: {exc}")
        continue
    rows.append({
        'template': name,
        'n': s.get('total_predictions', 0),
        'pure_eos_pct': s.get('pure_eos_pct', 0.0),
        'accuracy_pct': s.get('accuracy_pct', 0.0),
        'avg_tokens': s.get('avg_output_length_tokens') or 0.0,
        'hit_cap_pct': s.get('hit_max_new_tokens_pct') or 0.0,
        'reasoning_incomplete_pct': (
            s.get('categories', {}).get('reasoning_incomplete', 0)
            / max(1, s.get('total_predictions', 1)) * 100.0
        ),
    })

if not rows:
    print('(no analysis_*.json files found in', log_dir, ')')
    sys.exit(0)

print(f"{'template':<22} {'n':>5} {'pure_eos%':>10} {'reas_inc%':>10} "
      f"{'correct%':>10} {'avg_tok':>10} {'hit_cap%':>10}")
print('-' * 90)
for r in rows:
    print(f"{r['template']:<22} {r['n']:>5} {r['pure_eos_pct']:>10.1f} "
          f"{r['reasoning_incomplete_pct']:>10.1f} {r['accuracy_pct']:>10.1f} "
          f"{r['avg_tokens']:>10.0f} {r['hit_cap_pct']:>10.1f}")

print()
print('Interpretation hints:')
print('  - If "raw" pure_eos% >> non-raw -> prompt-template mismatch is real')
print('    (low-effort fix: set INPUT_TEMPLATE in G1/G2/G3.sh and re-run)')
print('  - If pure_eos% is roughly equal across all templates -> ebft-distill')
print('    artifact (no prompt fix; revisit training or move to distrib metrics)')
print('  - correct% / reas_inc% movement is also informative; pure_eos% is most diagnostic')
PY

echo ""
echo "Per-template artefacts under: ${LOG_DIR}"
