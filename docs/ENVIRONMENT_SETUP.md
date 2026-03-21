# Environment Setup

This document summarizes the verified environment used to run:

- vanilla paper-aligned EBFT reproduction
- `cf_l1oo`
- `cf_l1oo` with `vicinal` target
- `cf_tokencloud_l1oo`

on the current machine.

## 1. Verified Runtime

- OS: Linux
- Python: `3.12.3`
- Conda env: `base`
- GPU: `NVIDIA A800 80GB PCIe`
- NVIDIA driver: `550.54.14`
- CUDA toolkit: `12.4`
- PyTorch: `2.5.1+cu124`
- Torch CUDA: `12.4`

## 2. Verified Python Packages

These are the important packages actually imported during our runs.

- `torch==2.5.1+cu124`
- `transformers==4.53.3`
- `datasets==4.6.1`
- `accelerate==1.12.0`
- `deepspeed==0.18.8`
- `ray==2.47.0`
- `flash-attn==2.8.3`
- `trl==0.9.6`
- `peft==0.18.1`
- `modelscope==1.34.0`

Packages listed in the repo but not required for our verified runs:

- `bitsandbytes` was not installed
- `vllm` was not installed

## 3. Repo Dependency Declaration

The repo requirement file is:

- [requirements.txt](/root/autodl-tmp/Energy/ebft_openrlhf_u1_ncfm/requirements.txt)

Important note:

- the repo declares `ray[default]==2.48.0`
- the verified runtime here used `ray==2.47.0`

So for maximum fidelity, I recommend trying `2.47.0` first if you want to match our currently verified environment exactly.

## 4. Important Install Pitfall

On the current machine, `openrlhf` is installed in editable mode, but it points to:

- `/root/autodl-tmp/Energy/ebft_openrlhf`

That means on a new machine you must re-install editable mode from the repo you actually want to run, for example:

- `/root/autodl-tmp/Energy/ebft_openrlhf_stepwise`
- or `/root/autodl-tmp/Energy/ebft_openrlhf_u1_ncfm`

Do not rely on an old editable install path.

## 5. Recommended Rebuild Procedure

### Option A: reproduce our verified environment most closely

1. Install Miniconda or Anaconda.
2. Create a Python 3.12 environment.
3. Install PyTorch for CUDA 12.4.
4. Install repo requirements.
5. Install the target repo in editable mode.

Example:

```bash
conda create -n ebft python=3.12.3 -y
conda activate ebft

pip install --upgrade pip wheel setuptools

pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

cd /path/to/ebft_openrlhf_u1_ncfm
pip install -r requirements.txt
pip install ray[default]==2.47.0
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
```

### Option B: clone from current machine snapshot

We already saved an environment snapshot at:

- [conda_env_export_base.yml](/root/autodl-tmp/Energy/backups/ebft_openrlhf_20260317_233420/env/conda_env_export_base.yml)
- [conda_explicit_base.txt](/root/autodl-tmp/Energy/backups/ebft_openrlhf_20260317_233420/env/conda_explicit_base.txt)
- [pip_freeze.txt](/root/autodl-tmp/Energy/backups/ebft_openrlhf_20260317_233420/env/pip_freeze.txt)

This is useful if you want a very close mirror of the current machine.

## 6. Required Runtime Environment Variables

The main verified run script is:

- [run_step1_paper_qa_feature_map.sh](/root/autodl-tmp/Energy/ebft_openrlhf_u1_ncfm/scripts/run_step1_paper_qa_feature_map.sh)

It relies on these runtime defaults:

- `CUDA_VISIBLE_DEVICES=0`
- `HF_HOME=/root/.cache/huggingface`
- `HF_DATASETS_CACHE=/root/.cache/huggingface/datasets`
- `HF_HUB_CACHE=/root/.cache/huggingface/hub`
- `HF_HUB_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`
- `HF_HUB_DISABLE_XET=1`
- `TOKENIZERS_PARALLELISM=false`
- `RAY_memory_usage_threshold=0.995`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES=8589934592`

These settings matter for stability on a single GPU machine.

## 7. Required Local Assets

### Model

Our paper-aligned runs used a full local copy of:

- `/root/autodl-tmp/modelscope_cache_test/Qwen/Qwen2___5-1___5B`

The run script expects:

- `MODEL_PATH` to point to a complete local checkpoint
- weight files such as `*.safetensors` to actually exist

### Datasets

The verified runs used:

- `sjelassi/opencode-instruct_100k_200tok`
- `openai/openai_humaneval`

The script defaults to offline mode, so on a new machine you either need:

- a warm Hugging Face cache
- or you must temporarily set:
  - `HF_HUB_OFFLINE=0`
  - `HF_DATASETS_OFFLINE=0`

to let datasets download once.

## 8. Single-GPU Compatibility Notes

The original paper setting is multi-component and memory-heavy. On one 80GB GPU, the following engineering choices were important:

- `--colocate_all_models`
- `--adam_offload`
- `--flash_attn`
- `--gradient_checkpointing` for stable long runs
- Ray object store capped to `8 GiB`

Without these compatibility adjustments, we previously hit:

- Ray CPU-memory OOM during eval
- actor-training CUDA OOM
- gradient-checkpointing + 4D mask attention bugs

## 9. Recommended Bring-Up Checklist

On a new machine, verify these before full runs:

1. `python -V`
2. `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
3. `python -c "import flash_attn, deepspeed, ray, transformers; print('ok')"`
4. `python -m pip show openrlhf`
5. Confirm editable install points to the intended repo
6. Confirm `MODEL_PATH` contains actual model weights
7. Run a small smoke command first, not a long paper-aligned run

## 10. Practical Recommendation

If your goal is to reproduce our current results on another machine, the safest order is:

1. rebuild the Python environment close to the versions above
2. install the chosen repo with `pip install -e .`
3. warm local model and dataset caches
4. run the paper-aligned vanilla smoke or short run first
5. only then run `cf_l1oo` / `cf_tokencloud_l1oo`
