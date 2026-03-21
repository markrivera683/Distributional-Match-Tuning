#!/usr/bin/env bash
set -euo pipefail

# Stable EBFT-only transfer bundle for moving this project to another machine.
# This bundles:
# - code repo (excluding heavy runs/)
# - current validated base conda environment
# - required model weights
# - required dataset caches
# - selected experiment logs
# - environment metadata

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/transfer}"
BUNDLE_NAME="${BUNDLE_NAME:-ebft_only_bundle_${STAMP}}"
BUNDLE_DIR="${OUT_ROOT}/${BUNDLE_NAME}"
ARCHIVE_PATH="${OUT_ROOT}/${BUNDLE_NAME}.tar.gz"

REPO_SRC="/root/autodl-tmp/Energy/ebft_openrlhf_u1_ncfm"
MODEL_SRC="/root/autodl-tmp/modelscope_cache_test/Qwen/Qwen2___5-1___5B"
DATASET1_SRC="/root/.cache/huggingface/datasets/sjelassi___opencode-instruct_100k_200tok"
DATASET2_SRC="/root/.cache/huggingface/datasets/openai___openai_humaneval"
CONDA_SRC="/root/miniconda3"

SELECTED_RUNS=(
  "step1_paperqa_identity_qwen25_1p5b_seed43_gc1_fullfix"
  "step2_u1_cf_l1oo_smoke_seed43_fg"
  "step2_u1_cf_l1oo_vicinal_smoke_seed43_fg"
  "step2c_u1_cf_tokencloud_smoke_seed43_fg"
  "step2lite_cf_adapter_smoke_seed43_fix1"
  "step2lite_cf_adapter_directdisc_smoke_seed43"
  "step2lite_cf_adapter_directdisc_smoke_seed43_fix1"
)

mkdir -p "${BUNDLE_DIR}"/{repo,model,datasets,env,results}

echo "[1/6] Copying repo (excluding runs/)..."
rsync -a --delete --exclude 'runs/' "${REPO_SRC}/" "${BUNDLE_DIR}/repo/ebft_openrlhf_u1_ncfm/"

echo "[2/6] Copying model weights..."
rsync -a "${MODEL_SRC}/" "${BUNDLE_DIR}/model/Qwen2___5-1___5B/"

echo "[3/6] Copying dataset caches..."
rsync -a "${DATASET1_SRC}/" "${BUNDLE_DIR}/datasets/sjelassi___opencode-instruct_100k_200tok/"
rsync -a "${DATASET2_SRC}/" "${BUNDLE_DIR}/datasets/openai___openai_humaneval/"

echo "[4/6] Copying base conda environment..."
rsync -a "${CONDA_SRC}/" "${BUNDLE_DIR}/env/miniconda3/"

echo "[5/6] Exporting environment metadata and selected run logs..."
conda env export -n base > "${BUNDLE_DIR}/env/base.environment.yml"
conda list --explicit > "${BUNDLE_DIR}/env/base.explicit.txt"
python -m pip freeze > "${BUNDLE_DIR}/env/base.pip-freeze.txt"

for run in "${SELECTED_RUNS[@]}"; do
  src_run="${REPO_SRC}/runs/${run}"
  if [[ -d "${src_run}" ]]; then
    mkdir -p "${BUNDLE_DIR}/results/${run}"
    if [[ -f "${src_run}/train.log" ]]; then
      cp -a "${src_run}/train.log" "${BUNDLE_DIR}/results/${run}/"
    fi
    if [[ -d "${src_run}/tensorboard" ]]; then
      mkdir -p "${BUNDLE_DIR}/results/${run}/tensorboard"
      rsync -a --include '*.tfevents*' --exclude '*' "${src_run}/tensorboard/" "${BUNDLE_DIR}/results/${run}/tensorboard/" || true
    fi
  fi
done

cat > "${BUNDLE_DIR}/RESTORE.md" <<'EOF'
# EBFT Transfer Bundle Restore

Recommended restore path assumptions:
- Restore to `/`
- Keep the conda path as `/root/miniconda3`
- Keep the project path as `/root/autodl-tmp/Energy/ebft_openrlhf_u1_ncfm`

## Restore

```bash
cd /
tar -xzf /path/to/ebft_only_bundle_*.tar.gz
```

## Repoint the project repo into editable install

```bash
source /root/miniconda3/bin/activate base
cd /root/autodl-tmp/Energy/ebft_openrlhf_u1_ncfm
python -m pip install -e .
```

## Quick sanity checks

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import deepspeed, ray, transformers; print(deepspeed.__version__, ray.__version__, transformers.__version__)"
cd /root/autodl-tmp/Energy/ebft_openrlhf_u1_ncfm
bash scripts/run_step2lite_cf_adapter_directdisc_smoke.sh
```
EOF

echo "[6/6] Creating compressed archive..."
mkdir -p "${OUT_ROOT}"
tar -C "${OUT_ROOT}" -czf "${ARCHIVE_PATH}" "${BUNDLE_NAME}"

echo
echo "Bundle created:"
echo "  ${ARCHIVE_PATH}"
du -sh "${ARCHIVE_PATH}"
