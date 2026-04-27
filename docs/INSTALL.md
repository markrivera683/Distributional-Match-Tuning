# 安装与下载指南（Environment & Asset Setup）

本文档梳理把这个仓库 **从零跑起来** 所需的全部一次性准备工作，包括：

1. 系统先决条件
2. Python 双虚拟环境（学生 `.venv` + 教师 `.teacherVenv`）
3. 模型权重下载与放置
4. 训练 / 评测数据集下载与放置
5. HuggingFace 缓存与离线模式
6. 安装验证清单
7. 路径覆盖速查
8. 常见问题（FAQ）
9. 与历史文档的关系

> 训练 / 评测脚本本身的用法见 `[docs/SCRIPTS_OVERVIEW.md](./SCRIPTS_OVERVIEW.md)`，G2 阶段的设计背景见 `[docs/G2/](./G2)`。

---

## 0. TL;DR —— 4 步跑通

```bash
# 1. 克隆仓库
git clone <repo-url> /root/code/Distributional-Match-Tuning
cd /root/code/Distributional-Match-Tuning

# 2. 一键创建两个 venv（用 uv，自动安装 uv 本身）
bash scripts/setup_env.sh

# 3. 准备模型与数据（详见 §3、§4）
#    - 学生模型: /mnt/data/teacher_model/models/Qwen3.5-0.8B
#    - 教师模型: /mnt/data/models/qwen3.5-27b           （仅 G2/G3 主线需要）
#    - 训练数据: /mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict
#    - 评测数据: /mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl

# 4. 跑一个最小冒烟（G1，无教师，单机 4 卡）
TARGET_STEPS=20 bash scripts/run_G1_rebase.sh
```

---

## 1. 系统先决条件


| 项           | 推荐                                          | 必要  | 说明                                                                       |
| ----------- | ------------------------------------------- | --- | ------------------------------------------------------------------------ |
| OS          | Linux x86_64（kernel 5.10+）                  | ✓   | macOS / WSL 仅支持 CPU 调试，不能跑 vLLM 教师                                       |
| Python      | 3.12.12                                     | ✓   | `setup_env.sh` 默认值；可通过 `PYTHON_VERSION` 覆盖                               |
| CUDA driver | 12.4+                                       | ✓   | 学生侧 PyTorch 锁定 `2.5.1+cu124`；教师侧 PyTorch 2.10 与 vLLM 0.19 已在 cu126 驱动上验证 |
| GPU         | 8×A100/A800 80GB（G2 主线）                     | —   | G1 / G2 消融可以 4 卡；G3 主线需要 2 节点 × 8 卡 = 16 卡                               |
| 系统内存        | ≥ 128GB                                     | ✓   | Ray object store 默认 8GB；vLLM kv-cache 还会占据相当显存                           |
| 磁盘          | ≥ 500GB SSD（含模型 + 数据 + 输出 + teacher cache）  | ✓   | Qwen3.5-27B bf16 ≈ 55GB；teacher_cache_shared 单次 100k prompt 训练可达 10–30GB |
| 工具          | `git` / `curl` / `tmux` / `ssh`（多节点）/ `tar` | ✓   | `setup_env.sh` 在缺 `uv` 时会用 `curl` 自动安装                                   |


---

## 2. 创建双虚拟环境

仓库内的 **每一个** 训练 / 评测 / 论文复现脚本都按下面两个固定路径硬编码：


| Venv | 路径（可被 `STUDENT_VENV` / `TEACHER_VENV` 覆盖） | 角色                                              |
| ---- | ----------------------------------------- | ----------------------------------------------- |
| 学生   | `${REPO_ROOT}/.venv/bin/python`           | OpenRLHF / Ray / DeepSpeed 训练；本仓库 editable 安装目标 |
| 教师   | `${REPO_ROOT}/.teacherVenv/bin/vllm`      | 远程 / 本地 vLLM 教师服务                               |


### 2.1 一键安装（推荐）

```bash
bash scripts/setup_env.sh
```

`scripts/setup_env.sh` 会：

1. 检测 / 自动安装 `uv`（写入 `~/.local/bin/uv`）。
2. 在 `${REPO_ROOT}/.venv` 创建学生环境，按下表锁定版本安装并 `pip install -e .`。
3. 在 `${REPO_ROOT}/.teacherVenv` 创建教师环境，按下表锁定版本安装。
4. 两侧都会跑一段内联 Python 校验，打印 `torch.cuda.is_available()` 等关键状态。

**学生环境（`.venv`）锁定版本**


| 包                                                                   | 版本                               | 备注                                                                            |
| ------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------- |
| `python`                                                            | 3.12.12                          | `PYTHON_VERSION` 可覆盖                                                          |
| `torch` / `torchvision` / `torchaudio`                              | 2.5.1 / 0.20.1 / 2.5.1（`+cu124`） | 来自 `https://download.pytorch.org/whl/cu124`；可改 `STUDENT_TORCH_INDEX_URL`      |
| `transformers`                                                      | git ref `db9f18c3…`              | 来自 `huggingface/transformers`；可改 `STUDENT_TRANSFORMERS_REF` 切到任意 commit / tag |
| `deepspeed`                                                         | 0.18.9                           | ZeRO Stage 2 训练                                                               |
| `ray[default]`                                                      | 2.48.0                           | actor / critic / ref / reward 编排                                              |
| `flash-attn`                                                        | 2.8.3                            | 失败时自动 `--no-build-isolation` 重试一次                                             |
| `accelerate` / `datasets` / `peft` / `tokenizers` / `safetensors` … | 见 `setup_env.sh`                 | 全部按 `scripts/stash/recreate_current_env.sh` 快照锁定                              |
| 本仓库 `openrlhf`                                                      | editable                         | `pip install -e .`                                                            |


**教师环境（`.teacherVenv`）锁定版本**


| 包                                      | 版本                       | 备注                                     |
| -------------------------------------- | ------------------------ | -------------------------------------- |
| `python`                               | 3.12.12                  | 与学生侧同                                  |
| `torch` / `torchvision` / `torchaudio` | 2.10.0 / 0.25.0 / 2.10.0 | vLLM 0.19 的硬依赖                         |
| `vllm`                                 | 0.19.0                   | OpenAI-compatible HTTP server          |
| `flashinfer-python`                    | 0.6.6                    | vLLM kv-cache attention backend        |
| `huggingface_hub`                      | 1.9.0                    | 显式装在 vllm 之后避免被降级                      |
| `transformers`                         | 5.5.0（`--no-deps`）       | vLLM 0.19 的兼容版本，`--no-deps` 防止覆盖 torch |


### 2.2 部分安装

```bash
SKIP_TEACHER=1 bash scripts/setup_env.sh   # 只装学生（训练机）
SKIP_STUDENT=1 bash scripts/setup_env.sh   # 只装教师（推理 / 教师机）
```

### 2.3 完全自定义版本

任何 `*_VERSION` / `STUDENT_TORCH_INDEX_URL` / `STUDENT_TRANSFORMERS_REF` 等都可被环境变量覆盖，例如：

```bash
PYTHON_VERSION=3.12.10 \
STUDENT_VENV=/data/envs/dmt_student \
TEACHER_VENV=/data/envs/dmt_teacher \
STUDENT_FLASH_ATTN_VERSION=2.7.4post1 \
bash scripts/setup_env.sh
```

如需更高级的能力（apt 系统包、git checkout 仓库、从快照锁文件还原依赖、记录 `pip freeze` 等），用：

```bash
bash scripts/stash/recreate_current_env.sh --help
```

> ❗ 不要再用 `scripts/setup_env_deprecated.sh`（旧版单 conda env 安装，与所有 `run_G*.sh` / `reproduction_*.sh` 路径都不兼容）。

---

## 3. 模型权重下载

所有脚本默认假设权重已经躺在本地路径上；脚本本身 **不会** 触发下载。下面给出模型清单与推荐放置位置。

### 3.1 学生模型（必装）


| 默认路径                                                                 | 说明                                                              | 谁会用                                                      |
| -------------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------- |
| `/mnt/data/teacher_model/models/Qwen3.5-0.8B`（大小写均接受 `qwen3.5-0.8b`） | bf16 权重，约 1.6GB；本仓库内部命名 "Qwen3.5"，对应 HF 上的 Qwen2.5-0.8B-Base 系列 | 全部 `run_G{1,2,3}_*.sh`、`smoketests/`*、`supplement*/*.sh` |


下载示例（HuggingFace 在线 + 显式 cache 目录）：

```bash
mkdir -p /mnt/data/teacher_model/models
HF_HUB_OFFLINE=0 \
HF_HOME=/root/.cache/huggingface \
huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir /mnt/data/teacher_model/models/Qwen3.5-0.8B \
  --local-dir-use-symlinks False
```

ModelScope 镜像示例（国内推荐）：

```bash
.venv/bin/pip install modelscope
.venv/bin/python - <<'PY'
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B",
                  local_dir="/mnt/data/teacher_model/models/Qwen3.5-0.8B")
PY
```

下载后必须满足：

```bash
ls /mnt/data/teacher_model/models/Qwen3.5-0.8B/config.json
ls /mnt/data/teacher_model/models/Qwen3.5-0.8B/*.safetensors
```

> 论文复现脚本（`reproduction_sftEbft.sh` / `reproduction_ebft_100k.sh` / `run_paper_qa_ebft_trend.sh`）默认 `MODEL_PATH=/root/model`，可通过 `MODEL_PATH=...` 覆盖到上面的真实路径，或者把权重做软链：
>
> ```bash
> ln -sfn /mnt/data/teacher_model/models/Qwen3.5-0.8B /root/model
> ```

### 3.2 教师模型（仅 G2 / G3 主线需要）


| 默认路径                                                 | 说明                                                                    | 谁会用                                                                                                                                                     |
| ---------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/mnt/data/models/qwen3.5-27b`（大小写均接受 `Qwen3.5-27B`） | bf16 权重，约 55GB；本仓库内部命名 "Qwen3.5-27B"，对应 HF 上的 Qwen2.5-32B-Instruct 系列 | `run_G2_rebase.sh`、`run_G2_rebase_pointwise.sh`、`run_G3_rebase_2node_once.sh`、`smoketests/launch_teacher_*.sh`、`supplement/teacher_qwen35_vllm_eval.sh` |


下载示例：

```bash
mkdir -p /mnt/data/models
HF_HUB_OFFLINE=0 \
huggingface-cli download Qwen/Qwen2.5-32B-Instruct \
  --local-dir /mnt/data/models/qwen3.5-27b \
  --local-dir-use-symlinks False
```

> 不跑 G2/G3 主线、只用 `run_G1_rebase.sh` / `run_G2_rebase_no_teacher_*.sh` / `run_G3_rebase_no_teacher_vicinal.sh` / `reproduction_*.sh` 的话，**可以完全跳过教师权重和 `.teacherVenv`**。

### 3.3 教师服务的两种部署方式

教师权重就绪后，G2 / G3 脚本支持两种教师部署：

- **本地拉起**（默认）：`run_G2_rebase.sh` / `run_G3_rebase_2node_once.sh` 会用 `.teacherVenv/bin/vllm serve` 在脚本内启教师 worker，等待 `/v1/models` health check 通过再开训。
- **远端复用**：把教师起好的 OpenAI 兼容服务放到另一台机器，然后传：
  ```bash
  LAUNCH_TEACHER=false \
  TEACHER_API_BASE=http://<teacher-host>:8004/v1,http://<teacher-host>:8005/v1 \
  TEACHER_API_KEY=teacher-local \
  TEACHER_MODEL_NAME=qwen3.5-27b \
  bash scripts/run_G2_rebase.sh
  ```
  此时训练机本身不需要 `.teacherVenv`、不需要教师权重。

---

## 4. 数据集下载

### 4.1 主线 AOPS 数据（G1 / G2 / G3 全部消融）


| 默认路径                                                            | 类型                                                                           | 说明                                                                             |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict` | HuggingFace `DatasetDict`（目录里含 `dataset_info.json`、`state.json`、各 split 子目录） | 训练用，由 `DeepStudentLlama/AoPS-Instruct` 切分预处理而来，键为 `input` / `output` / `label` |
| `/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl`   | JSONL，单行一个样本                                                                 | 评测用；`run_G*` 训练完成后自动调用 `batch_inference` 跑这份评测集                                |


**从 HuggingFace 重新生成 AOPS DatasetDict**：

```bash
mkdir -p /mnt/data/ebft-teacher-distribution/data/aops
.venv/bin/python - <<'PY'
from datasets import load_dataset, DatasetDict
ds = load_dataset("DeepStudentLlama/AoPS-Instruct")
ds.save_to_disk("/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict")
PY
```

**评测 JSONL** 通常是从上面 dataset 的 test split 抽样而来，类似：

```bash
.venv/bin/python - <<'PY'
from datasets import load_from_disk
ds = load_from_disk("/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict")
ds["test"].to_json("/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl")
PY
```

### 4.2 论文复现数据（`reproduction_*.sh` / `run_paper_qa_ebft_trend.sh`）


| 默认路径 / 名称                          | 默认值                                      | 说明                                                  |
| ---------------------------------- | ---------------------------------------- | --------------------------------------------------- |
| `TRAIN_DATA` (sftEbft / ebft_100k) | `/root/OpenCode`                         | 任意磁盘上的 OpenCode 100k 子集，要求字段 `input` / `output`     |
| `TRAIN_DATA` (paper_qa_ebft_trend) | `sjelassi/opencode-instruct_100k_200tok` | HuggingFace Hub 上的官方版本，会被 `resolve_dataset_spec` 解析 |
| `DOWNSTREAM_HUMANEVAL_DATASET`     | `openai/openai_humaneval`                | 下游 benchmark                                        |
| `DOWNSTREAM_MBPP_DATASET`          | `google-research-datasets/mbpp`          | 下游 benchmark                                        |
| `DOWNSTREAM_MULTIPLE_DATASET`      | `nuprl/MultiPL-E`                        | 下游 benchmark（多语言 humaneval）                         |


最小预热（首次需要联网）：

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
.venv/bin/python - <<'PY'
from datasets import load_dataset
load_dataset("sjelassi/opencode-instruct_100k_200tok")
load_dataset("openai/openai_humaneval", split="test")
load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
load_dataset("nuprl/MultiPL-E", "humaneval-cpp", split="test")
PY
```

预热完成后再切回 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` 即可全离线运行。

### 4.3 数据集自定义路径

每个脚本都暴露相同的覆盖入口，按需传入：

```bash
TRAIN_DATA=/data/my_train_dict \
EVAL_DATA=/data/my_test.jsonl \
bash scripts/run_G2_rebase.sh
```

---

## 5. HuggingFace 缓存与离线模式

训练 / 评测脚本均默认 **离线**，因此首次跑之前必须把模型 / 数据 / tokenizer 全部缓存好。


| 变量                                       | 默认                           | 用途                       |
| ---------------------------------------- | ---------------------------- | ------------------------ |
| `HF_HOME`                                | `${HOME}/.cache/huggingface` | 缓存根                      |
| `HF_HUB_CACHE`                           | `${HF_HOME}/hub`             | 模型 / tokenizer 缓存        |
| `HF_DATASETS_CACHE`                      | `${HF_HOME}/datasets`        | 数据集缓存                    |
| `HF_HUB_OFFLINE`                         | `1`                          | 训练脚本 export，强制不访问 HF Hub |
| `HF_DATASETS_OFFLINE`                    | `1`                          | 同上                       |
| `HF_HUB_DISABLE_XET`                     | `1`                          | 关闭 Xet 协议，避开 LFS 带宽限制    |
| `TOKENIZERS_PARALLELISM`                 | `false`                      | 避免 fork 警告               |
| `RAY_memory_usage_threshold`             | `0.995`                      | Ray 内存溢出阈值               |
| `PYTORCH_CUDA_ALLOC_CONF`                | `expandable_segments:True`   | 减少 CUDA OOM 碎片           |
| `OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES` | `8589934592`（8GB）            | Ray object store 上限      |


首次预热阶段建议：

```bash
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
HF_ENDPOINT=https://hf-mirror.com \    # 国内可加速
bash scripts/run_G1_rebase.sh           # 让训练脚本顺便把缺的资产下载下来
```

预热结束后正式训练时不需要再设置 `HF_*_OFFLINE`，脚本顶部会自动 export `=1`。

---

## 6. 安装验证清单

### 6.1 venv 自检

```bash
# 学生 venv
.venv/bin/python - <<'PY'
import torch, transformers, deepspeed, ray, openrlhf
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("deepspeed", deepspeed.__version__, "ray", ray.__version__)
print("openrlhf", getattr(openrlhf, "__version__", "editable"))
PY

# 教师 venv
.teacherVenv/bin/python - <<'PY'
import torch, transformers, vllm, shutil, sys
from pathlib import Path
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("vllm_cli", Path(sys.executable).with_name("vllm"))
print("vllm_cli_on_path", shutil.which("vllm"))
PY
```

预期输出（关键字段）：

```
torch 2.5.1+cu124 cuda 12.4 available True
transformers 4.x.dev0  (git ref db9f18c3)
deepspeed 0.18.9 ray 2.48.0
openrlhf editable
---
torch 2.10.0 cuda 12.x available True
transformers 5.5.0
vllm 0.19.0
```

### 6.2 资产自检

```bash
# 学生模型
ls /mnt/data/teacher_model/models/Qwen3.5-0.8B/config.json
ls /mnt/data/teacher_model/models/Qwen3.5-0.8B/*.safetensors

# 教师模型（仅 G2/G3 主线）
ls /mnt/data/models/qwen3.5-27b/config.json
ls /mnt/data/models/qwen3.5-27b/*.safetensors

# 训练 / 评测数据
ls /mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict/dataset_info.json
ls /mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl
```

### 6.3 端到端冒烟

最便宜的端到端验证（4 卡，无教师，约 5–10 分钟）：

```bash
TARGET_STEPS=20 EVAL_STEPS=20 SAVE_STEPS=20 \
bash scripts/run_G1_rebase.sh
```

教师服务自检（仅 G2/G3）：

```bash
bash scripts/smoketests/smoketest_teacher_qwen35_1gpu.sh
# 预期：vLLM 启动后能拿到 /v1/models 列表，模型名为 qwen3.5-27b
```

---

## 7. 路径覆盖速查

所有可被环境变量覆盖的关键路径：


| 变量                   | 默认                                                                                          | 影响                                |
| -------------------- | ------------------------------------------------------------------------------------------- | --------------------------------- |
| `REPO_ROOT`          | `/root/code/Distributional-Match-Tuning`                                                    | 解析 `.venv` / `.teacherVenv` / 输出根 |
| `STUDENT_VENV`       | `${REPO_ROOT}/.venv`                                                                        | 学生 Python 解释器目录                   |
| `TEACHER_VENV`       | `${REPO_ROOT}/.teacherVenv`                                                                 | 教师 vLLM CLI 目录                    |
| `STUDENT_PYTHON_BIN` | `${STUDENT_VENV}/bin/python`                                                                | 部分 reproduction 脚本显式引用            |
| `TEACHER_VLLM_BIN`   | `${TEACHER_VENV}/bin/vllm`                                                                  | G2/G3 启动教师                        |
| `MODEL_PATH`         | `/mnt/data/teacher_model/models/Qwen3.5-0.8B`（主线）/ `/root/model`（论文复现）                      | 学生模型                              |
| `TEACHER_MODEL_PATH` | `/mnt/data/models/qwen3.5-27b`                                                              | 教师模型                              |
| `TRAIN_DATA`         | `/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict`（主线）/ `/root/OpenCode`（论文复现） | 训练集                               |
| `EVAL_DATA`          | `/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl`                               | 训练后评测集                            |
| `OUTPUT_ROOT`        | `/root/outputs`（主线）/ 各脚本自定义                                                                 | 输出根                               |
| `RUN_NAME`           | 各脚本自定义                                                                                      | 子目录名                              |


---

## 8. 常见问题（FAQ）

### Q1. `setup_env.sh` 卡在 `flash-attn` 编译

`flash-attn==2.8.3` 走 PyPI wheel 时偶尔会 fall back 到源码编译，可能耗时 10–20 分钟、需 GCC 11+。两个绕过方法：

- 提前下载好 wheel：在 `dlc_eval/` 内有 `STUDENT_FLASH_ATTN_WHEEL=/mnt/.../flash_attn-2.8.3+cu124torch2.5-cp312-cp312-linux_x86_64.whl` 的用法可参考。
- 临时跳过：`STUDENT_FLASH_ATTN_VERSION=2.7.4post1 bash scripts/setup_env.sh`，运行时缺 `flash_attn` 的脚本会回落到 SDPA 注意力（性能略差但能跑）。

### Q2. 教师 vLLM 启动后 `/v1/models` 一直 404

- 看 `${RUN_DIR}/teacher_*.log` 里 vLLM 是否报 `RuntimeError: CUDA out of memory` —— 调小 `TEACHER_GPU_MEMORY_UTIL`（默认 0.96 → 试 0.85）或 `TEACHER_MAX_NUM_SEQS`。
- 看是否端口被占：`ss -tlnp | grep 800`，必要时 `TEACHER_BASE_PORT=9100 bash scripts/run_G2_rebase.sh`。
- 看 `TEACHER_WAIT_SECONDS`（默认 1800s）是否够 27B 加载，必要时调大。

### Q3. `STUDENT_PYTHON_BIN not executable` / `TEACHER_VLLM_BIN not executable`

意味着默认路径下没找到 venv。两种可能：

1. 没跑过 `setup_env.sh`，或装到了别的位置 —— 执行 `bash scripts/setup_env.sh`。
2. 仓库克隆到了非默认目录 —— 在调用脚本时传 `REPO_ROOT=/your/path`，或 `STUDENT_VENV=...` / `TEACHER_VENV=...` 覆盖。

### Q4. `pip show openrlhf` 指向旧仓库

editable 安装是 **last-write-wins**：在新仓库再跑一次 `setup_env.sh`，或单独：

```bash
.venv/bin/pip install -e /root/code/Distributional-Match-Tuning
.venv/bin/pip show openrlhf | grep Location
```

### Q5. 离线模式下 `datasets` 加载失败

按 §5 把 `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0` 临时设上、跑一次预热，再恢复成 `=1`。注意 `HF_HOME` / `HF_HUB_CACHE` / `HF_DATASETS_CACHE` 在预热阶段和正式运行阶段必须 **指向同一目录**，否则缓存命中失败。

### Q6. 想验证某个脚本到底要什么

每个脚本顶部都会用 `VAR="${VAR:-default}"` 列出所有可覆盖的变量，包括路径。最简单的办法：

```bash
bash -n scripts/run_G2_rebase.sh                       # 语法检查
sed -n '1,150p' scripts/run_G2_rebase.sh               # 看顶部 100 多行的所有默认配置
```

或直接读 `[docs/SCRIPTS_OVERVIEW.md](./SCRIPTS_OVERVIEW.md)` 的"GPU 布局 / 教师 / 数据"列。

---

## 9. 与历史文档的关系


| 文档                                                                                | 状态        | 说明                                                  |
| --------------------------------------------------------------------------------- | --------- | --------------------------------------------------- |
| `[docs/INSTALL.md](./INSTALL.md)`                                                 | **当前权威**  | 本文                                                  |
| `[docs/SCRIPTS_OVERVIEW.md](./SCRIPTS_OVERVIEW.md)`                               | 当前        | 安装完成后看这里挑脚本                                         |
| `[docs/G2/ENVIRONMENT_AND_DEPENDENCIES.md](./G2/ENVIRONMENT_AND_DEPENDENCIES.md)` | 历史（G2 阶段） | 写于"单 conda env"年代，包名版本可参考但目录布局已被本文取代                |
| `[docs/previous/ENVIRONMENT_SETUP.md](./previous/ENVIRONMENT_SETUP.md)`           | 历史（早期）    | 路径指向 `/root/autodl-tmp/Energy/...`，与当前仓库无关，仅作历史快照参考 |
| `scripts/setup_env.sh`                                                            | **当前权威**  | 双 venv 一键安装                                         |
| `scripts/setup_env_deprecated.sh`                                                 | **已废弃**   | 旧版单 conda env，已不可用                                  |
| `scripts/stash/recreate_current_env.sh`                                           | 高级参考      | 完整可参数化版本，含 apt deps、git checkout、snapshot 还原        |


