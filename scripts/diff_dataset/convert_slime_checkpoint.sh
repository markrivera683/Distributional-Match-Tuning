#!/usr/bin/env bash
# Auto-convert a HuggingFace checkpoint to Megatron torch_dist using slime's official converter.
#
# Usage:
#   bash scripts/diff_dataset/convert_slime_checkpoint.sh /path/to/hf_model
#   bash scripts/diff_dataset/convert_slime_checkpoint.sh mcore_to_hf /path/to/slime_torch_dist
#
# Optional env:
#   MCORE_CHECKPOINT=/path/to/output_torch_dist
#   HF_CHECKPOINT=/path/to/original_hf_model
#   HF_OUTPUT=/path/to/output_hf_model
#   SLIME_ROOT=/mnt/data/distribution-matching-slime/code/slime-0.2.4
#   MEGATRON_PATH=/root/slime_runtime/Megatron-LM
#   MODEL_ARGS_SCRIPT=/mnt/data/distribution-matching-slime/code/slime-0.2.4/slime/scripts/models/qwen3.5-4B.sh

set -euo pipefail

MODE="${MODE:-hf_to_mcore}"
if [[ "${1:-}" == "hf_to_mcore" || "${1:-}" == "hf_to_torch_dist" || "${1:-}" == "mcore_to_hf" || "${1:-}" == "torch_dist_to_hf" ]]; then
  MODE="$1"
  shift
fi

case "${MODE}" in
  hf_to_mcore|hf_to_torch_dist)
    HF_CHECKPOINT="${1:-${MODEL_PATH:-${HF_CHECKPOINT:-}}}"
    if [[ -z "${HF_CHECKPOINT}" ]]; then
      echo "[ERROR] missing model path. Usage: $0 /path/to/hf_model" >&2
      exit 2
    fi
    HF_CHECKPOINT="$(cd "${HF_CHECKPOINT}" && pwd)"
    MODEL_NAME="$(basename "${HF_CHECKPOINT%/}")"
    ;;
  mcore_to_hf|torch_dist_to_hf)
    MCORE_CHECKPOINT="${1:-${MCORE_CHECKPOINT:-}}"
    HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_PATH:-/mnt/data/models/Qwen3.5-4B}}"
    if [[ -z "${MCORE_CHECKPOINT}" ]]; then
      echo "[ERROR] missing Megatron checkpoint path. Usage: $0 mcore_to_hf /path/to/slime_torch_dist" >&2
      exit 2
    fi
    MCORE_CHECKPOINT="$(cd "${MCORE_CHECKPOINT}" && pwd)"
    HF_CHECKPOINT="$(cd "${HF_CHECKPOINT}" && pwd)"
    MODEL_NAME="$(basename "${HF_CHECKPOINT%/}")"
    ;;
  *)
    echo "[ERROR] unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

if [[ -z "${SLIME_ROOT:-}" ]]; then
  for candidate in "/mnt/data/distribution-matching-slime/code/slime-0.2.4" "/root/slime_runtime/slime"; do
    if [[ -f "${candidate}/tools/convert_hf_to_torch_dist.py" ]]; then
      SLIME_ROOT="${candidate}"
      break
    fi
  done
fi
SLIME_ROOT="${SLIME_ROOT:-/mnt/data/distribution-matching-slime/code/slime-0.2.4}"

if [[ -z "${MEGATRON_PATH:-}" ]]; then
  for candidate in "/root/slime_runtime/Megatron-LM" "/mnt/data/utils/slime_runtime/Megatron-LM" "/root/Megatron-LM"; do
    if [[ -d "${candidate}/megatron" ]]; then
      MEGATRON_PATH="${candidate}"
      break
    fi
  done
fi
MEGATRON_PATH="${MEGATRON_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MCORE_ROOT="${MCORE_ROOT:-/root/slime_runtime/checkpoints}"
MCORE_CHECKPOINT="${MCORE_CHECKPOINT:-${MCORE_ROOT}/${MODEL_NAME}_torch_dist}"
HF_OUTPUT="${HF_OUTPUT:-${MCORE_CHECKPOINT%/}_hf}"
DRY_RUN="${DRY_RUN:-false}"

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || { echo "[ERROR] required directory not found: ${path}" >&2; exit 1; }
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || { echo "[ERROR] required file not found: ${path}" >&2; exit 1; }
}

require_dir "${HF_CHECKPOINT}"
require_file "${HF_CHECKPOINT}/config.json"
require_dir "${SLIME_ROOT}"
case "${MODE}" in
  hf_to_mcore|hf_to_torch_dist)
    require_file "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py"
    ;;
  mcore_to_hf|torch_dist_to_hf)
    require_dir "${MCORE_CHECKPOINT}"
    require_file "${MCORE_CHECKPOINT}/common.pt"
    require_file "${SLIME_ROOT}/tools/convert_torch_dist_to_hf.py"
    ;;
esac

if [[ -n "${MEGATRON_PATH}" ]]; then
  require_dir "${MEGATRON_PATH}"
  export PYTHONPATH="${MEGATRON_PATH}:${SLIME_ROOT}:${PYTHONPATH:-}"
else
  echo "[warn] MEGATRON_PATH not found; relying on the active Python environment to provide megatron" >&2
  export PYTHONPATH="${SLIME_ROOT}:${PYTHONPATH:-}"
fi

if [[ "${MODE}" == "mcore_to_hf" || "${MODE}" == "torch_dist_to_hf" ]]; then
  echo "[convert] mode=${MODE}"
  echo "[convert] mcore_checkpoint=${MCORE_CHECKPOINT}"
  echo "[convert] origin_hf=${HF_CHECKPOINT}"
  echo "[convert] hf_output=${HF_OUTPUT}"
  echo "[convert] slime_root=${SLIME_ROOT}"
  echo "[convert] megatron_path=${MEGATRON_PATH}"

  CMD=(
    "${PYTHON_BIN}" "${SLIME_ROOT}/tools/convert_torch_dist_to_hf.py"
    --input-dir "${MCORE_CHECKPOINT}"
    --output-dir "${HF_OUTPUT}"
    --origin-hf-dir "${HF_CHECKPOINT}"
    --force
  )

  if [[ -n "${HF_VOCAB_SIZE:-}" ]]; then
    CMD+=(--vocab-size "${HF_VOCAB_SIZE}")
  fi

  cd "${SLIME_ROOT}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[dry-run] '
    printf '%q ' "${CMD[@]}"
    echo
    exit 0
  fi

  "${CMD[@]}"
  exit 0
fi

MODEL_ARGS=()

try_model_script() {
  local candidate="$1"
  if [[ -f "${candidate}" ]]; then
    # shellcheck disable=SC1090
    source "${candidate}"
    echo "[model-args] using script: ${candidate}" >&2
    return 0
  fi
  return 1
}

if [[ -n "${MODEL_ARGS_SCRIPT:-}" ]]; then
  require_file "${MODEL_ARGS_SCRIPT}"
  # shellcheck disable=SC1090
  source "${MODEL_ARGS_SCRIPT}"
  echo "[model-args] using MODEL_ARGS_SCRIPT=${MODEL_ARGS_SCRIPT}" >&2
else
  # Prefer official slime model arg scripts when the model directory name matches one.
  shopt -s nullglob
  wanted_names=("${MODEL_NAME,,}" "${MODEL_NAME//_/-}")
  for candidate in "${SLIME_ROOT}"/scripts/models/*.sh "${SLIME_ROOT}"/slime/scripts/models/*.sh; do
    candidate_name="$(basename "${candidate%.sh}")"
    candidate_name="${candidate_name,,}"
    for wanted_name in "${wanted_names[@]}"; do
      if [[ "${candidate_name}" == "${wanted_name,,}" ]]; then
        try_model_script "${candidate}"
        break 3
      fi
    done
  done
  shopt -u nullglob

  # Also try a few direct paths for exact-case names.
  for candidate in \
    "${SLIME_ROOT}/scripts/models/${MODEL_NAME}.sh" \
    "${SLIME_ROOT}/scripts/models/${MODEL_NAME,,}.sh" \
    "${SLIME_ROOT}/scripts/models/${MODEL_NAME//_/-}.sh" \
    "${SLIME_ROOT}/slime/scripts/models/${MODEL_NAME}.sh" \
    "${SLIME_ROOT}/slime/scripts/models/${MODEL_NAME,,}.sh" \
    "${SLIME_ROOT}/slime/scripts/models/${MODEL_NAME//_/-}.sh"; do
    if [[ ${#MODEL_ARGS[@]} -eq 0 ]] && try_model_script "${candidate}"; then
      break
    fi
  done
fi

if [[ ${#MODEL_ARGS[@]} -eq 0 ]]; then
  echo "[model-args] no official model script matched; inferring Megatron args from config.json" >&2
  mapfile -t MODEL_ARGS < <("${PYTHON_BIN}" - "${HF_CHECKPOINT}" <<'PYARGS'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
cfg = data.get("text_config") if isinstance(data.get("text_config"), dict) else data


def pick(*names, default=None):
    for name in names:
        if name in cfg and cfg[name] is not None:
            return cfg[name]
        if name in data and data[name] is not None:
            return data[name]
    return default


model_type = str(pick("model_type", default="")).lower()
root_model_type = str(data.get("model_type", "")).lower()
hidden = int(pick("hidden_size"))
heads = int(pick("num_attention_heads"))
kv_heads = int(pick("num_key_value_heads", default=heads))
head_dim = pick("head_dim", default=None)
kv_channels = int(head_dim) if head_dim else hidden // heads
rope_parameters = pick("rope_parameters", default={}) or {}
rope_theta = pick("rope_theta", default=None) or rope_parameters.get("rope_theta") or 1000000
rotary_percent = rope_parameters.get("partial_rotary_factor", pick("rotary_percent", default=None))
norm_eps = pick("rms_norm_eps", "layer_norm_epsilon", default=1e-6)
intermediate = int(pick("intermediate_size"))
vocab = int(pick("vocab_size"))
layers = int(pick("num_hidden_layers"))
tie_word_embeddings = bool(pick("tie_word_embeddings", default=True))
attention_bias = bool(pick("attention_bias", default=False))
qk_layernorm = bool(pick("qk_layernorm", "qk_layer_norm", default=False))
if "qwen" in model_type:
    qk_layernorm = True

args = []
if root_model_type in {"qwen3_5", "qwen3_5_moe"}:
    args += ["--spec", "slime_plugins.models.qwen3_5", "get_qwen3_5_spec"]
args += ["--swiglu"]
args += ["--num-layers", str(layers)]
args += ["--hidden-size", str(hidden)]
args += ["--ffn-hidden-size", str(intermediate)]
args += ["--num-attention-heads", str(heads)]
if kv_heads != heads:
    args += ["--group-query-attention", "--num-query-groups", str(kv_heads)]
if root_model_type in {"qwen3_5", "qwen3_5_moe"}:
    args += ["--use-gated-attention"]
    args += ["--position-embedding-type", "rope"]
    args += ["--apply-layernorm-1p"]
    if rotary_percent is not None:
        args += ["--rotary-percent", str(rotary_percent)]
else:
    args += ["--use-rotary-position-embeddings"]
if not attention_bias:
    args += ["--disable-bias-linear"]
args += ["--normalization", "RMSNorm"]
args += ["--norm-epsilon", str(norm_eps)]
args += ["--rotary-base", str(int(rope_theta))]
args += ["--vocab-size", str(vocab)]
args += ["--kv-channels", str(kv_channels)]
if qk_layernorm:
    args += ["--qk-layernorm"]
if not tie_word_embeddings:
    args += ["--untie-embeddings-and-output-weights"]
if root_model_type in {"qwen3_5", "qwen3_5_moe"}:
    args += ["--attention-output-gate"]

for item in args:
    print(item)
PYARGS
  )
fi

mkdir -p "${MCORE_CHECKPOINT}"

echo "[convert] mode=${MODE}"
echo "[convert] hf_checkpoint=${HF_CHECKPOINT}"
echo "[convert] save=${MCORE_CHECKPOINT}"
echo "[convert] slime_root=${SLIME_ROOT}"
echo "[convert] megatron_path=${MEGATRON_PATH}"
echo "[convert] model_args=${MODEL_ARGS[*]}"

CMD=(
  "${PYTHON_BIN}" -c
  "import runpy; import sglang.srt.utils.hf_transformers_utils; runpy.run_path('tools/convert_hf_to_torch_dist.py', run_name='__main__')"
  "${MODEL_ARGS[@]}"
  --hf-checkpoint "${HF_CHECKPOINT}"
  --save "${MCORE_CHECKPOINT}"
)

cd "${SLIME_ROOT}"
if [[ "${DRY_RUN}" == "true" ]]; then
  printf '[dry-run] '
  printf '%q ' "${CMD[@]}"
  echo
  exit 0
fi

"${CMD[@]}"
