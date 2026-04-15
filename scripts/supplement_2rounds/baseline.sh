#!/usr/bin/env bash
# G2 two-round post-eval via vLLM: 16k first pass, 32k retry on incorrect prompts.
# Usage:
#   bash scripts/supplement_2rounds/G2.sh /mnt/data/teacher_model/models/qwen3.5-0.8b
#   RUN_DIR=/mnt/data/teacher_model/models/qwen3.5-0.8b bash scripts/supplement_2rounds/baseline.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/code/Distributional-Matching-Tuning}"
DEFAULT_EVAL_DATA="/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl"
EVAL_DATA="${EVAL_DATA:-${DEFAULT_EVAL_DATA}}"

TEACHER_VENV="${TEACHER_VENV:-${REPO_ROOT}/.teacherVenv}"
TEACHER_PYTHON_BIN="${TEACHER_PYTHON_BIN:-${TEACHER_VENV}/bin/python}"
ANALYSIS_VENV="${ANALYSIS_VENV:-${REPO_ROOT}/.venv}"
ANALYSIS_PYTHON_BIN="${ANALYSIS_PYTHON_BIN:-${ANALYSIS_VENV}/bin/python}"
PROGRESS_HELPER="${PROGRESS_HELPER:-${REPO_ROOT}/scripts/supplement/vllm_generate_progress.py}"

RUN_DIR="${RUN_DIR:-${1:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "Usage: RUN_DIR=/path/to/run bash scripts/supplement_2rounds/baseline.sh"
  echo "   or: bash scripts/supplement_2rounds/baseline.sh /path/to/run"
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model}"
EVAL_TAG="${EVAL_TAG:-2rounds_vllm}"
SCRIPT_NAME="$(basename "$0" .sh)"
TS="${TS:-$(date +%m%d_%H%M)}"

MODEL_CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _VISIBLE_GPUS <<< "${MODEL_CUDA_VISIBLE_DEVICES}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-${#_VISIBLE_GPUS[@]}}"
POST_EVAL_MAX_SAMPLES="${POST_EVAL_MAX_SAMPLES:-5328}"
POST_EVAL_PROMPT_MAX_LEN="${POST_EVAL_PROMPT_MAX_LEN:-512}"
FIRST_PASS_MAX_NEW_TOKENS="${FIRST_PASS_MAX_NEW_TOKENS:-16384}"
SECOND_PASS_MAX_NEW_TOKENS="${SECOND_PASS_MAX_NEW_TOKENS:-32768}"
POST_EVAL_TEMPERATURE="${POST_EVAL_TEMPERATURE:-0.6}"
POST_EVAL_TOP_P="${POST_EVAL_TOP_P:-1.0}"
POST_EVAL_REPETITION_PENALTY="${POST_EVAL_REPETITION_PENALTY:-1.0}"
POST_EVAL_BEST_OF_N="${POST_EVAL_BEST_OF_N:-1}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
VLLM_SEED="${VLLM_SEED:-1234}"
INPUT_TEMPLATE="${INPUT_TEMPLATE:-}"

LOG_DIR="${LOG_DIR:-${RUN_DIR}/supplement_logs}"
SCRIPT_LOG_PATH="${SCRIPT_LOG_PATH:-${LOG_DIR}/${SCRIPT_NAME}_${EVAL_TAG}_${TS}.log}"

FIRST_PASS_OUTPUT_PATH="${LOG_DIR}/eval_results_${EVAL_TAG}_stage1_${TS}.jsonl"
FIRST_PASS_LOG_PATH="${LOG_DIR}/eval_${EVAL_TAG}_stage1_${TS}.log"
FIRST_PASS_ANALYSIS_REPORT_PATH="${LOG_DIR}/eval_analysis_${EVAL_TAG}_stage1_${TS}.json"
FIRST_PASS_ANALYSIS_LOG_PATH="${LOG_DIR}/eval_analysis_${EVAL_TAG}_stage1_${TS}.log"

SECOND_PASS_DATASET_PATH="${LOG_DIR}/eval_retry_subset_${EVAL_TAG}_${TS}.jsonl"
SECOND_PASS_METADATA_PATH="${LOG_DIR}/eval_retry_subset_meta_${EVAL_TAG}_${TS}.json"
SECOND_PASS_OUTPUT_PATH="${LOG_DIR}/eval_results_${EVAL_TAG}_stage2_${TS}.jsonl"
SECOND_PASS_LOG_PATH="${LOG_DIR}/eval_${EVAL_TAG}_stage2_${TS}.log"
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

for _bin in "${TEACHER_PYTHON_BIN}" "${ANALYSIS_PYTHON_BIN}"; do
  [[ -x "${_bin}" ]] || { echo "[ERROR] Not executable: ${_bin}"; exit 1; }
done
[[ -d "${RUN_DIR}" ]] || { echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"; exit 1; }
[[ -e "${MODEL_PATH}" ]] || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 1; }
[[ -e "${EVAL_DATA}" ]] || { echo "[ERROR] EVAL_DATA not found: ${EVAL_DATA}"; exit 1; }
[[ -f "${PROGRESS_HELPER}" ]] || { echo "[ERROR] PROGRESS_HELPER not found: ${PROGRESS_HELPER}"; exit 1; }

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${SCRIPT_LOG_PATH}") 2>&1
cd "${REPO_ROOT}"

run_vllm_generation() {
  local stage_name="$1"
  local dataset_path="$2"
  local output_path="$3"
  local log_path="$4"
  local max_new_tokens="$5"

  echo "[${stage_name}] vLLM generating with max_new_tokens=${max_new_tokens}"
  local -a vllm_cmd=(
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
    --tp_size "${VLLM_TP_SIZE}"
    --max_num_seqs "${VLLM_MAX_NUM_SEQS}"
    --seed "${VLLM_SEED}"
  )
  [[ -n "${INPUT_TEMPLATE}" ]] && vllm_cmd+=(--input_template "${INPUT_TEMPLATE}")
  [[ "${VLLM_ENABLE_PREFIX_CACHING}" == "true" ]] && vllm_cmd+=(--enable_prefix_caching)

  CUDA_VISIBLE_DEVICES="${MODEL_CUDA_VISIBLE_DEVICES}" \
  "${vllm_cmd[@]}" 2>&1 | tee "${log_path}"
}

run_analysis() {
  local stage_name="$1"
  local eval_results_path="$2"
  local report_path="$3"
  local log_path="$4"

  echo "[${stage_name}] Analyzing results"
  "${ANALYSIS_PYTHON_BIN}" "${REPO_ROOT}/scripts/analyze_eval_results.py" \
    --eval_results "${eval_results_path}" \
    --eval_dataset "${EVAL_DATA}" \
    --input_key question --label_key answer \
    --report_path "${report_path}" \
    2>&1 | tee "${log_path}"
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

build_final_report() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, os, sys
fp, sp, mp, op = sys.argv[1:5]
fr = json.load(open(fp)); sr = json.load(open(sp)); rm = json.load(open(mp))
frc = fr.get("records",[]); src = sr.get("records",[]); final = list(frc); ri = set()
for r in src:
    si = r.get("source_idx") or r.get("idx")
    if si is None: continue
    si = int(si); ri.add(si)
    if 0<=si<len(final):
        m=dict(final[si]); m["first_pass"]=dict(final[si]); m["second_pass"]=dict(r)
        m["prompt"]=r.get("prompt",m.get("prompt",""))
        m["model_output"]=r.get("model_output",m.get("model_output",""))
        m["gold_answer"]=r.get("gold_answer",m.get("gold_answer"))
        m["is_correct"]=r.get("is_correct"); m["category"]=r.get("category")
        m["detail"]=r.get("detail"); m["retry_applied"]=True; final[si]=m
for i,r in enumerate(final): r.setdefault("retry_applied",i in ri)
ev=[r for r in final if r.get("is_correct") is not None]
cor=sum(1 for r in ev if r.get("is_correct"))
acc=round(cor/len(ev)*100,2) if ev else 0.0
imp=sum(1 for i in ri if 0<=i<len(frc) and 0<=i<len(final) and frc[i].get("is_correct") is not True and final[i].get("is_correct") is True)
stw=sum(1 for i in ri if 0<=i<len(final) and final[i].get("is_correct") is not True)
out={"summary":{"total_predictions":len(final),"evaluated":len(ev),"correct":cor,"accuracy_pct":acc,
     "first_pass_correct":fr.get("summary",{}).get("correct"),"first_pass_accuracy_pct":fr.get("summary",{}).get("accuracy_pct"),
     "second_pass_retry_count":rm.get("retry_count",len(ri)),"retry_improved_to_correct":imp,"retry_still_incorrect":stw},
     "first_pass_report_path":fp,"second_pass_report_path":sp,"retry_metadata":rm,"records":final}
os.makedirs(os.path.dirname(os.path.abspath(op)),exist_ok=True)
with open(op,"w") as f: json.dump(out,f,indent=2,ensure_ascii=False); f.write("\n")
PY
}

echo "========== G2 Two-Round vLLM Eval =========="
echo "RUN_DIR:                      ${RUN_DIR}"
echo "MODEL_PATH:                   ${MODEL_PATH}"
echo "EVAL_DATA:                    ${EVAL_DATA}"
echo "FIRST_PASS_MAX_NEW_TOKENS:    ${FIRST_PASS_MAX_NEW_TOKENS}"
echo "SECOND_PASS_MAX_NEW_TOKENS:   ${SECOND_PASS_MAX_NEW_TOKENS}"
echo "MODEL_CUDA_VISIBLE_DEVICES:   ${MODEL_CUDA_VISIBLE_DEVICES}"
echo "VLLM_TP_SIZE:                 ${VLLM_TP_SIZE}"
echo "VLLM_MAX_NUM_SEQS:            ${VLLM_MAX_NUM_SEQS}"
echo "POST_EVAL_MAX_SAMPLES:        ${POST_EVAL_MAX_SAMPLES}"
echo "============================================="

echo ""
echo "===== Stage 1: Full eval at ${FIRST_PASS_MAX_NEW_TOKENS} tokens ====="
run_vllm_generation "stage1" "${EVAL_DATA}" "${FIRST_PASS_OUTPUT_PATH}" "${FIRST_PASS_LOG_PATH}" "${FIRST_PASS_MAX_NEW_TOKENS}"
run_analysis "stage1-analysis" "${FIRST_PASS_OUTPUT_PATH}" "${FIRST_PASS_ANALYSIS_REPORT_PATH}" "${FIRST_PASS_ANALYSIS_LOG_PATH}"

echo ""
echo "===== Extracting retry subset ====="
extract_retry_subset "${EVAL_DATA}" "${FIRST_PASS_ANALYSIS_REPORT_PATH}" "${SECOND_PASS_DATASET_PATH}" "${SECOND_PASS_METADATA_PATH}"
RETRY_COUNT="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['retry_count'])" "${SECOND_PASS_METADATA_PATH}")"
echo "[retry] ${RETRY_COUNT} prompts to retry"

if (( RETRY_COUNT > 0 )); then
  echo ""
  echo "===== Stage 2: Retry ${RETRY_COUNT} prompts at ${SECOND_PASS_MAX_NEW_TOKENS} tokens ====="
  run_vllm_generation "stage2" "${SECOND_PASS_DATASET_PATH}" "${SECOND_PASS_OUTPUT_PATH}" "${SECOND_PASS_LOG_PATH}" "${SECOND_PASS_MAX_NEW_TOKENS}"
  run_analysis "stage2-analysis" "${SECOND_PASS_OUTPUT_PATH}" "${SECOND_PASS_ANALYSIS_REPORT_PATH}" "${SECOND_PASS_ANALYSIS_LOG_PATH}"
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

echo ""
echo "========== Done =========="
echo "First pass report:  ${FIRST_PASS_ANALYSIS_REPORT_PATH}"
echo "Retry report:       ${SECOND_PASS_ANALYSIS_REPORT_PATH}"
echo "Final report:       ${FINAL_ANALYSIS_REPORT_PATH}"
echo "Script log:         ${SCRIPT_LOG_PATH}"
