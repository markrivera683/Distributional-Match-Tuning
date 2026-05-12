# `scripts/` 顶层脚本说明

本文档梳理 `scripts/` **根目录**下的全部 `.sh` 脚本（不含 `benchmarks/`、`dlc/`、`dlc_eval/`、`postversion/`、`previous/`、`smoketests/`、`snapshot/`、`stash/`、`supplement/`、`supplement_2rounds/`、`deprecated/` 等子文件夹），并按照实验组（baseline / G1 / G2 / G3）+ 评测工具 + 论文复现 三条主线进行分类，方便快速定位。

> 实验命名约定：`G1` < `G2` < `G3` 表示三个递进的训练配置，控制变量沿"奖励类型 → 教师目标 → 训练栈"逐步打开。

---

## 0. 总览速查表

| 类别 | 脚本 | 是否需要教师 | 奖励类型 (`distribution_reward_type`) | 目标测度 (`cf_target_mode`) | G3 训练栈（EMA / adapter / critic-head） |
|---|---|---|---|---|---|
| 环境 | `setup_env.sh` | — | — | — | — |
| 环境（已废弃） | `setup_env_deprecated.sh` | — | — | — | — |
| **Baseline** | `run_baseline.sh` | 否（不训练） | — | — | — |
| Baseline / G1 | `run_G1_rebase.sh` | 否 | `pointwise` | `single` | 否 |
| **G2 主** | `run_G2_rebase.sh` | **在线 vLLM teacher** | `cf_l1oo` | `teacher` | 否 |
| G2 消融 | `run_G2_rebase_pointwise.sh` | 在线 vLLM teacher | `pointwise` | （未传 `cf_*`） | 否 |
| G2 消融 | `run_G2_rebase_no_teacher_vicinal.sh` | 否 | `cf_l1oo` | `vicinal` | 否 |
| G2 消融 | `run_G2_rebase_no_teacher_single_2rounds.sh` | 否 | `cf_l1oo` | `single` | 否 |
| **G3 主** | `run_G3_rebase_2node_once.sh` | **在线 vLLM teacher（双节点）** | `cf_l1oo` | `teacher` | **是** |
| G3 消融 | `run_G3_rebase_no_teacher_vicinal.sh` | 否 | `cf_l1oo` | `vicinal` | 是 |
| Post-eval | `run_g2_posteval_retry16k_32k.sh` | — | — | — | — |
| Post-eval（二轮工具集） | `supplement_2rounds/{baseline,G1,G2,G3,Teacher}.sh` | — | — | — | — |
| 论文复现 | `run_paper_qa_ebft_trend.sh` | 否 | `pointwise` | `single` | EMA 开 |
| 论文复现 | `reproduction_sftEbft.sh` | 否 | `pointwise` | `single` | 否 |
| 论文复现 | `reproduction_ebft_100k.sh` | 否 | `pointwise` | `single` | 否 |

---

## 1. 阅读约定

所有训练脚本（除 `setup_env.sh` 外）都遵循同一套写法：

- **变量集中在脚本顶部**，使用 `VAR="${VAR:-default}"` 形式，因此可以直接通过环境变量覆盖。
- 默认数据：`/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict`（训练）、`.../test_qa.jsonl`（评测）。
- 默认学生模型：`/mnt/data/teacher_model/models/Qwen3.5-0.8B`，默认教师模型（如启用）：`/mnt/data/models/qwen3.5-27b`。
- 默认输出根目录：`/root/outputs/<RUN_NAME>`，含 `model/`、`tensorboard/`、`train.log`、`eval.log`、`status.txt` 等。
- Python 环境：学生用 `${REPO_ROOT}/.venv`（`STUDENT_PYTHON_BIN`），教师 vLLM 用 `${REPO_ROOT}/.teacherVenv`（`TEACHER_VLLM_BIN`）。

最常用的覆盖方式：

```bash
TARGET_STEPS=500 bash scripts/run_G2_rebase.sh
MAX_SAMPLES=10000 bash scripts/run_G1_rebase.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 ACTOR_GPUS=2 CRITIC_GPUS=2 bash scripts/run_G2_rebase_no_teacher_vicinal.sh
```

`G1 / G2 / G3` 的核心控制变量差异（与 `run_G1_rebase.sh` 顶部注释一致）：

```text
G1: distribution_reward_type=pointwise, cf_target_mode=single, no teacher
G2: distribution_reward_type=cf_l1oo,   cf_target_mode=teacher, online teacher
G3 = G2 + enable_ema + feature_adapter + trainable critic-head + direct discrepancy
```

---

## 2. 环境准备

> 完整的"零到能跑"安装与下载指南（含模型 / 数据集 / 缓存 / FAQ）见 [`docs/INSTALL.md`](./INSTALL.md)。本节仅给出脚本级别的最短摘要。

仓库的所有训练 / 评测脚本都按"双 venv"约定硬编码：
- 学生侧：`${REPO_ROOT}/.venv/bin/python` —— 跑 OpenRLHF / Ray / DeepSpeed 训练
- 教师侧：`${REPO_ROOT}/.teacherVenv/bin/vllm` —— 起远程 vLLM 教师服务

### 2.1 `setup_env.sh` —— 双 venv 安装入口（推荐）

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/setup_env.sh` |
| 用途 | 用 `uv` 在仓库根创建 `.venv` 与 `.teacherVenv` 两个虚拟环境，并按 `scripts/stash/recreate_current_env.sh`（commit `0a9b59b9` 快照）的锁定版本安装两套依赖 |
| 学生环境 | Python 3.12.12；`torch 2.5.1+cu124`、`transformers @ db9f18c3`（git ref）、`deepspeed 0.18.9`、`ray[default] 2.48.0`、`flash-attn 2.8.3`；最后 `pip install -e .` 安装本仓库 |
| 教师环境 | Python 3.12.12；`torch 2.10.0`、`vllm 0.19.0`、`flashinfer-python 0.6.6`、`transformers 5.5.0`、`huggingface_hub 1.9.0` |
| 切换 | `SKIP_STUDENT=1` 只装教师；`SKIP_TEACHER=1` 只装学生；`PYTHON_VERSION` / `STUDENT_VENV` / `TEACHER_VENV` / 各 `*_VERSION` 均可被环境变量覆盖 |
| 调用 | `bash scripts/setup_env.sh`（首次配置环境时执行一次） |

> 完整可参数化版本（含 apt deps、git checkout、snapshot 等）参见 `scripts/stash/recreate_current_env.sh`。

### 2.2 `setup_env_deprecated.sh` —— 早期单 conda 环境（已废弃）

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/setup_env_deprecated.sh` |
| 状态 | **不要再用**。原始单 conda env (`openrlhf`) 安装脚本，保留作为历史对照 |
| 与现仓库不兼容的原因 | (1) 只创建 `~/anaconda3/envs/openrlhf/` 单环境，不会生成 `${REPO_ROOT}/.venv` 与 `${REPO_ROOT}/.teacherVenv`；(2) torch / vllm / flash-attn 版本均与 `.venv`/`.teacherVenv` 快照不一致；(3) 直接执行后所有 `run_G*.sh`、`reproduction_*.sh`、`supplement*/*.sh` 都会立刻报 `STUDENT_PYTHON_BIN not executable` 或 `TEACHER_VLLM_BIN not executable` |

---

## 3. Baseline / G1 系列

### 3.1 `run_baseline.sh` —— 纯基线（未训练模型 + 16k/32k 二轮评测）

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/run_baseline.sh` |
| 定位 | **不训练**，直接对预训练 / 基线模型跑 G1/G2/G3 同口径的 16k → 32k 二轮 vLLM 评测，给出对照表里的"未训练基线"那一行 |
| GPU 布局 | 全部可见 GPU 都给 vLLM（`MODEL_CUDA_VISIBLE_DEVICES`，默认 `0..7`，`VLLM_TP_SIZE` 默认等于可见 GPU 数） |
| 教师 | 无（与"baseline = 无 RL 介入"语义一致） |
| 评测协议 | 委托 `scripts/supplement_2rounds/baseline.sh`：第一轮 `FIRST_PASS_MAX_NEW_TOKENS=16384` 全量跑 → 抽取错题 → 第二轮 `SECOND_PASS_MAX_NEW_TOKENS=32768` 复跑 → 合并出 final report 与 oracle-union 统计 |
| 默认数据 / 模型 | 与 `run_G1_rebase.sh` 完全一致：`MODEL_PATH=/mnt/data/models/gemma-4-E4B/`、`EVAL_DATA=/mnt/data/.../test_qa.jsonl`，便于直接对齐对比 |
| 输出 | `/root/outputs/baseline_<MMDD_HHMM>/supplement_logs/`（含 `eval_results_*_stage{1,2}_*.jsonl`、`eval_analysis_*_final_*.json`、`status.txt`） |
| 启动 | `bash scripts/run_baseline.sh` |
| 常用覆盖 | `MODEL_PATH=/mnt/data/models/qwen3.5-0.8b bash scripts/run_baseline.sh`<br>`POST_EVAL_MAX_SAMPLES=512 bash scripts/run_baseline.sh`<br>`MODEL_CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_baseline.sh` |

> 与 `run_G1_rebase.sh` 的关系：两者**评测段完全同口径**（同 worker、同 16k/32k、同 prompt、同 oracle-union 统计），区别只在前者"不训练 + 模型即原样"，后者"先训 G1 → 再用同协议评测"。所以 `baseline / G1 / G2 / G3` 四个数字可以放在同一张表里。

### 3.2 `run_G1_rebase.sh` —— 严格消融基线

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/run_G1_rebase.sh` |
| 定位 | "退化版 G2"——**与 G2 共享数据、模型、batch、optimizer、seed**，仅切换三处控制变量来构成对比基线 |
| GPU 布局 | 4 卡（`CUDA_VISIBLE_DEVICES=0,1,2,3`），actor/ref 2 卡 + critic/reward 2 卡 |
| 教师 | **不启动**任何远程教师 |
| 奖励配置 | `distribution_reward_type=pointwise`、`cf_target_mode=single`、`cf_teacher_lambda=0.0` |
| critic | 冻结（`critic_learning_rate=0`、`critic_lr_head=0`） |
| 训练规模 | 默认 `TARGET_STEPS=500`，按公式回算 `MAX_SAMPLES`；`EVAL_STEPS=25`、`SAVE_STEPS=25` |
| Post-train eval | 单卡 `torch.distributed.run` 跑 `openrlhf.cli.batch_inference` 生成 + `analyze_eval_results.py` 分析 |
| 启动 | `bash scripts/run_G1_rebase.sh` |

> 与 G2 / G3 的差异由脚本顶部"CONTROLLED VARIABLES vs G2 / G3"注释给出，是写消融对比表时的参考点。

---

## 4. G2 系列（cf_l1oo 分布匹配）

### 4.1 `run_G2_rebase.sh` —— G2 主版本（在线教师）

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/run_G2_rebase.sh` |
| 定位 | G2 阶段的**首选生产脚本**：远程 vLLM 教师 + cf_l1oo 分布匹配 |
| GPU 布局 | 单机 8 卡，教师 6 卡（`0..5`） + 学生 2 卡（`6,7`），学生侧 actor/ref 1 卡 + critic/reward 1 卡 |
| 教师 | 在脚本内 `vllm serve` 启动 `qwen3.5-27b`（`TEACHER_TP_SIZE=1` × 多 worker），等 health check 通过后再训练 |
| 奖励配置 | `distribution_reward_type=cf_l1oo`、`cf_target_mode=teacher`、`cf_teacher_lambda=0.6`、`cf_teacher_n_samples=8` |
| 教师采样 | `temperature=0.7`、`top_p=0.95`、`max_new_tokens=768`，开启 `--teacher_cache_enable`，cache 目录 `/root/outputs/teacher_cache_shared` |
| 训练规模 | `TARGET_STEPS=500`、`EVAL_STEPS=100`、`SAVE_STEPS=50` |
| Post-train eval | 训练结束后 8 卡 `batch_inference` 跑 5328 个 prompt，`max_new_tokens=8192`，再调用分析脚本 |
| 启动 | `bash scripts/run_G2_rebase.sh` |

### 4.2 `run_G2_rebase_pointwise.sh` —— G2 消融：保留教师，切回 pointwise 奖励

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/run_G2_rebase_pointwise.sh` |
| 定位 | 在 G2 主脚本基础上**仅改两处**：(1) `--distribution_reward_type pointwise`；(2) **不传任何 `--cf_*`**。其它一切（教师、batch、数据、seed）与 `run_G2_rebase.sh` 完全一致 |
| 用途 | 干净消融——衡量"分布级 cf_l1oo"相对"逐点 pointwise + diversity"的增益 |
| GPU 布局 / 教师 | 与 `run_G2_rebase.sh` 相同 |
| 启动 | `bash scripts/run_G2_rebase_pointwise.sh` |

### 4.3 `run_G2_rebase_no_teacher_vicinal.sh` —— G2 消融：去教师 + vicinal target

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/run_G2_rebase_no_teacher_vicinal.sh` |
| 定位 | 在 G2 上**移除整套教师服务**，目标测度从 `teacher` 切到 `vicinal`（`cf_target_num_refs=8`、`cf_target_std=0.05`） |
| GPU 布局 | 单机全部 8 卡都用于学生（actor/ref 4 + critic/reward 4） |
| 奖励配置 | `distribution_reward_type=cf_l1oo`、`cf_target_mode=vicinal`、`cf_teacher_lambda=0.0` |
| 用途 | 衡量"vicinal 邻域目标"是否能在没有教师时近似 cf_l1oo 的分布信号 |
| 启动 | `bash scripts/run_G2_rebase_no_teacher_vicinal.sh` |

### 4.4 `run_G2_rebase_no_teacher_single_2rounds.sh` —— G2 消融：去教师 + single GT + 两轮 post-eval

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/run_G2_rebase_no_teacher_single_2rounds.sh` |
| 定位 | 与上一个脚本相比，仅把 `cf_target_mode` 从 `vicinal` 切到 `single`、`cf_target_num_refs` 从 8 改成 1；训练完成后**自动调用** `scripts/supplement_2rounds/G2.sh` 跑 16k 首次 + 32k 重试两轮 post-eval |
| GPU 布局 | 单机 8 卡（actor/ref 4 + critic/reward 4） |
| 奖励配置 | `distribution_reward_type=cf_l1oo`、`cf_target_mode=single` |
| 用途 | 给 G2 在"无教师 + 单 GT 目标"下提供与 16k/32k 长上下文评测对齐的对照点 |
| 启动 | `bash scripts/run_G2_rebase_no_teacher_single_2rounds.sh` |

---

## 5. G3 系列（G2 + EMA + feature_adapter + 可训 critic-head）

### 5.1 `run_G3_rebase_2node_once.sh` —— G3 主版本（双节点在线教师）

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/run_G3_rebase_2node_once.sh` |
| 定位 | G3 阶段的生产脚本，**双节点 16 卡**，"head 节点 + 一台 worker 节点"模式，head 起 ray、ssh 拉起 worker |
| 节点要求 | 两节点各 8 卡；`HEAD_NODE` 上执行本脚本；要求 head→worker 免密 SSH；要求 `curl`/`ray`/`ssh` 可用 |
| GPU 布局 | 每节点：教师占 6 卡（`0..5`） + 学生占 2 卡（`6,7`）。学生侧 actor 1 节点×2 卡、critic 1 节点×2 卡 |
| 教师 | 头/工节点各起 6 个 vLLM worker（`qwen3.5-27b`、`TEACHER_TP_SIZE=1`），共 12 个端点 |
| 奖励配置 | `distribution_reward_type=cf_l1oo`、`cf_target_mode=teacher`、`cf_teacher_lambda=0.6`、`cf_teacher_n_samples=8` |
| **G3 专属** | `--enable_ema`（`EMA_BETA=0.99`）、`--feature_adapter_enable`（`residual_bottleneck`、`rank=64`）、`critic_lr_head=5e-5`、`critic_direct_discrepancy_coef=0.1`、`critic_direct_discrepancy_target=ema_gt` |
| Post-train eval | 自动调用 `scripts/supplement/G3_eval.sh` 在两节点上跑 `nproc=16` 的批量推理（5328 prompt × 8192 tokens） |
| 归档 | 默认 `ARCHIVE_OUTPUTS_AFTER_RUN=true`，训练完毕将 `RUN_DIR` 移动到 `/mnt/data/ebft-teacher-distribution/outputs_g3_0.99/` |
| 启动 | 在 head 节点：`HEAD_NODE=<head> WORKER_NODE=<worker> bash scripts/run_G3_rebase_2node_once.sh` |

### 5.2 `run_G3_rebase_no_teacher_vicinal.sh` —— G3 消融：去教师 + vicinal target

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/run_G3_rebase_no_teacher_vicinal.sh` |
| 定位 | 保留 G3 全套训练栈（EMA / feature_adapter / 可训 critic-head / direct discrepancy），仅把 `cf_target_mode` 从 `teacher` 切到 `vicinal` 并移除所有教师依赖 |
| GPU 布局 | 单机 8 卡（actor/ref 4 + critic/reward 4） |
| 奖励配置 | `distribution_reward_type=cf_l1oo`、`cf_target_mode=vicinal`、`cf_teacher_lambda=0.0` |
| 与 `run_G2_rebase_no_teacher_vicinal.sh` 的差异 | 多了 G3 训练栈；其它相同 |
| 启动 | `bash scripts/run_G3_rebase_no_teacher_vicinal.sh` |

---

## 6. Post-training 评测工具

### 6.1 `run_g2_posteval_retry16k_32k.sh` —— 对已训好的 G2 ckpt 跑两轮长上下文评测

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/run_g2_posteval_retry16k_32k.sh` |
| 定位 | **不训练**，仅对已有 G2 训练产物（默认 `MODEL_PATH=/root/outputs/g2_online_teacher_8gpu_0411_0652/model`）跑两阶段 vLLM 评测：第一轮 `max_new_tokens=16384`，对答错的 prompt 第二轮放大到 `32768` |
| 环境 | 用 `.teacherVenv` 跑生成，用 `.venv` 跑分析 |
| GPU | 8 卡（`VLLM_TP_SIZE=8`） |
| 实际工作 | `exec` 转交给 `scripts/dlc_eval/dlc_baseline_eval.sh` 完成 |
| 启动 | `bash scripts/run_g2_posteval_retry16k_32k.sh`（建议在 g3train 的 tmux 结束后再跑） |

### 6.2 `scripts/supplement_2rounds/` —— 二轮长上下文评测工具集（按训练组提供的 5 个等价入口）

`supplement_2rounds/` 下统一实现了一套 **"16k 全量 → 抽取错题 → 32k 重跑 → 合并 + oracle union 统计"** 的二轮 vLLM 评测协议，每个训练组（baseline / G1 / G2 / G3）以及独立教师评测各有一个同名入口：

| 脚本 | 入口约束 | 默认 `MODEL_PATH` | 默认 `RUN_DIR` | 默认 `VLLM_MAX_NUM_SEQS` | 与其他四个的差异 |
|---|---|---|---|---|---|
| `baseline.sh` | 必传 `RUN_DIR=` 或第一个位置参数 | `${RUN_DIR}/model` | 用户传入 | 32 | 多了 `VLLM_GPU_MEMORY_UTILIZATION` 透传 + `VLLM_TP_SIZE ≤ 可见 GPU 数` 强校验，是这个目录里最稳健的版本 |
| `G1.sh` | 同上 | `${RUN_DIR}/model` | 用户传入 | 32 | 与 G2/G3 行为完全等价，仅日志 header 写 `G1` |
| `G2.sh` | 同上 | `${RUN_DIR}/model` | 用户传入 | 32 | 与 G1/G3 等价；**`run_G2_rebase_no_teacher_single_2rounds.sh` 训练完会自动调用本脚本**（见 §4.4） |
| `G3.sh` | 同上 | `${RUN_DIR}/model` | 用户传入 | 32 | 与 G1/G2 等价；G3 主脚本不调用它（G3 主脚本走 `scripts/supplement/G3_eval.sh`），仅手动复跑用 |
| `Teacher.sh` | 必传 `MODEL_PATH=` 或第一个位置参数；不需要 `RUN_DIR` 已存在 | 自动探测 `/mnt/data/{,teacher_model/}models/{q,Q}wen3.5-27b{,B}` 四个候选 | `/root/outputs/teacher_2rounds_eval_${TS}` 自动创建 | **128**（教师吞吐高） | 直接评测教师 27B 自身在 AOPS 测试集上的能力，作为各训练组的上限对照 |

**共同的两轮评测协议（所有 5 个脚本完全一致）**：

1. **Stage 1（16k 全量）**：用 `.teacherVenv/bin/python scripts/supplement/vllm_generate_progress.py` 在 `EVAL_DATA`（默认 `/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl`，前 `POST_EVAL_MAX_SAMPLES=5328` 条）上跑一次推理；`prompt_max_len=512`、`max_new_tokens=16384`、`temperature=0.6`、`top_p=1.0`、`tp_size=可见 GPU 数`、`seed=1234`。
2. **Stage 1 分析**：用 `.venv/bin/python scripts/analyze_eval_results.py` 解析输出，生成 `eval_analysis_*_stage1_*.json`，每条记录带 `is_correct` 与 `source_idx`。
3. **抽取错题子集**：内联 Python 把 stage1 中 `is_correct != True` 的样本按 `source_idx` 拼成 `eval_retry_subset_*.jsonl`，并写 metadata（含 retry_count 与 source_indices）。
4. **Stage 2（32k 重跑）**：仅对错题子集再跑一次 vLLM，`max_new_tokens=32768`，其他超参与 stage 1 相同；若 stage 1 全对则跳过这一步并写空报告。
5. **Stage 2 分析**：再调一次 `analyze_eval_results.py`，生成 `eval_analysis_*_stage2_*.json`。
6. **合并最终报告**：内联 Python 用 stage 2 结果覆盖 stage 1 的对应 `source_idx` 条目，输出 `eval_analysis_*_final_*.json`，`summary` 字段包含：

   - `total_predictions / evaluated / correct / accuracy_pct`
   - `first_pass_correct / first_pass_accuracy_pct`
   - `second_pass_retry_count / retry_improved_to_correct / retry_still_incorrect`
   - **oracle union 统计**：`oracle_union_evaluated / oracle_union_correct / oracle_union_accuracy_pct`、`oracle_both_correct`、`oracle_stage1_only_correct`、`oracle_stage2_only_correct` —— 表示"两轮中任一轮答对就算对"的上限准确率，用于判断"32k 是否真的解锁了 16k 解不出的题"。

**关键可覆盖参数（5 个脚本通用）**：

| 变量 | 默认 | 说明 |
|---|---|---|
| `EVAL_DATA` | `/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl` | 评测集 |
| `POST_EVAL_MAX_SAMPLES` | `5328` | 取前 N 个 prompt |
| `POST_EVAL_PROMPT_MAX_LEN` | `512` | 截断输入 |
| `FIRST_PASS_MAX_NEW_TOKENS` / `SECOND_PASS_MAX_NEW_TOKENS` | `16384` / `32768` | 两轮的生成长度 |
| `POST_EVAL_TEMPERATURE` / `POST_EVAL_TOP_P` / `POST_EVAL_REPETITION_PENALTY` / `POST_EVAL_BEST_OF_N` | `0.6` / `1.0` / `1.0` / `1` | 采样配置 |
| `MODEL_CUDA_VISIBLE_DEVICES` / `VLLM_TP_SIZE` | `0,1,2,3,4,5,6,7` / 可见 GPU 数 | 8 卡 TP=8 |
| `VLLM_MAX_NUM_SEQS` / `VLLM_ENABLE_PREFIX_CACHING` / `VLLM_SEED` | `32`（Teacher 为 `128`）/ `false` / `1234` | vLLM 调度参数 |
| `EVAL_TAG` | `2rounds_vllm` | 写入文件名 |
| `INPUT_TEMPLATE` | 空 | 非空时套 chat template |
| `LOG_DIR` | `${RUN_DIR}/supplement_logs` | 全部产物落到此目录 |

**产物（按 `EVAL_TAG=2rounds_vllm` 与 `TS=MMDD_HHMM` 计算的文件命名）**：

```
${LOG_DIR}/
├── <script_name>_2rounds_vllm_<TS>.log              # 主控制日志
├── eval_results_2rounds_vllm_stage1_<TS>.jsonl      # stage 1 原始生成
├── eval_2rounds_vllm_stage1_<TS>.log                # stage 1 vLLM 子日志
├── eval_analysis_2rounds_vllm_stage1_<TS>.json/.log # stage 1 单阶段分析
├── eval_retry_subset_2rounds_vllm_<TS>.jsonl        # 错题子集
├── eval_retry_subset_meta_2rounds_vllm_<TS>.json    # 错题元数据
├── eval_results_2rounds_vllm_stage2_<TS>.jsonl      # stage 2 原始生成
├── eval_2rounds_vllm_stage2_<TS>.log                # stage 2 vLLM 子日志
├── eval_analysis_2rounds_vllm_stage2_<TS>.json/.log # stage 2 单阶段分析
└── eval_analysis_2rounds_vllm_final_<TS>.json/.log  # 合并最终报告（含 oracle union）
```

**与其他评测入口的关系**：

| 关系 | 说明 |
|---|---|
| ⇄ §4.4 `run_G2_rebase_no_teacher_single_2rounds.sh` | 该训练脚本在训练完成后**自动调用** `supplement_2rounds/G2.sh`，把 16k+32k 二轮评测纳入主流程 |
| ⇄ §6.1 `run_g2_posteval_retry16k_32k.sh` | 后者只是 G2 老 ckpt 的快捷复评入口，最终走的是 `dlc_eval/dlc_baseline_eval.sh` 的实现；`supplement_2rounds/` 是这套二轮协议的**通用、按训练组分的**版本 |
| ⇄ `scripts/supplement/G{1,2,3}_eval.sh` 与 `Teacher_qwen35_vllm_eval.sh` | `supplement/` 下的同名脚本是**单轮 8k**评测；`supplement_2rounds/` 是它们的两轮 16k+32k 升级版 |

**典型调用方式**：

```bash
# 对一个已训好的 run 跑 G2 二轮评测
bash scripts/supplement_2rounds/G2.sh /root/outputs/g2_online_teacher_8gpu_0411_0652

# 也可以 RUN_DIR= 形式
RUN_DIR=/root/outputs/g3_2node_0420_1530 bash scripts/supplement_2rounds/G3.sh

# 对教师本身做一次 27B 的二轮上限评测（自动选默认权重路径）
bash scripts/supplement_2rounds/Teacher.sh

# 单卡或少卡场景，覆盖 GPU
MODEL_CUDA_VISIBLE_DEVICES=0,1 VLLM_TP_SIZE=2 \
bash scripts/supplement_2rounds/baseline.sh /root/outputs/raw_qwen35_eval

# 缩小评测规模做冒烟
POST_EVAL_MAX_SAMPLES=100 SECOND_PASS_MAX_NEW_TOKENS=16384 \
bash scripts/supplement_2rounds/G1.sh /root/outputs/g1_rebase_0405_2259
```

> 5 个脚本的 stage 1 / stage 2 / 抽取 / 合并这四段核心代码完全等价（甚至 `analyze_eval_results.py` 的调用参数都一样），分成 5 个文件主要是为了在 `${RUN_DIR}/supplement_logs/` 与训练组日志风格一致，方便 grep 与归档。

---

## 7. 论文复现脚本

### 7.1 `run_paper_qa_ebft_trend.sh` —— OpenCode 100k 上的 EBFT 趋势复现

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/run_paper_qa_ebft_trend.sh` |
| 定位 | 论文 paper-QA 段落的 EBFT 训练趋势复现：在 `sjelassi/opencode-instruct_100k_200tok` 上跑 `MAX_SAMPLES=100000`，按 `SAVE_EPOCH_FRACTIONS=0.02,0.05,0.1,0.2,0.5` 保存多个中间 checkpoint，并在 `STOP_AFTER_EPOCH_FRACTION=0.5` 时停止 |
| GPU 布局 | 单机 8 卡（actor/ref 4 + critic/reward 4） |
| 奖励配置 | `distribution_reward_type=pointwise`、`cf_target_mode=single`，开启 `--enable_ema` |
| 下游评测 | 走 `scripts/benchmarks/run_code_generation_benchmarks.py`，默认评测 HumanEval / MBPP（greedy temperature=0、单样本，pass@k 关闭） |
| 输出 | `${REPO_ROOT}/outputs/paperqa_ebft_trend_seed43/` 下含 `final_model/`、`checkpoints/`、`tensorboard/`、`offline_benchmarks/` |
| 启动 | `bash scripts/run_paper_qa_ebft_trend.sh` |

### 7.2 `reproduction_sftEbft.sh` —— SFT warm-start + EBFT 两阶段复现

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/reproduction_sftEbft.sh` |
| 定位 | 完整复现论文流程：**Stage 1 SFT 预热**（`bs=64`、`max_len=2048`、`lr=1e-5`、`cosine_with_min_lr`、1 epoch）+ **Stage 2 EBFT**（`rollout_batch_size=16`、`train_batch_size=64`、`prompt_max_len=1024`、`generate_max_len=8`、`lr=1e-6`、`temperature=0.6`、`init_kl_coef=0`） |
| GPU 布局 | 单机 8 卡（actor/ref 4 + critic/reward 4） |
| EBFT 预算 | 通过 `EBFT_EQUIV_EPOCH=0.25`（基于 `EBFT_BASE_SAMPLE_COUNT=100000`）回算 `EBFT_MAX_SAMPLES` |
| 下游评测 | 两阶段后均可触发（`RUN_POST_STAGE1_BENCHMARKS` / `RUN_POST_STAGE2_BENCHMARKS`），默认 HumanEval + MBPP + MultiPL-E（C++/JS/TS/Rust/C#/Go/PHP/Java），含 greedy + pass@1/4/16 |
| 输出 | `${OUTPUT_ROOT}/${RUN_NAME}/{stage1_sft_model,stage2_ebft_model,...}` |
| 启动 | `bash scripts/reproduction_sftEbft.sh` |

### 7.3 `reproduction_ebft_100k.sh` —— 单独跑 EBFT 100k（假设 SFT 已完成）

| 项目 | 内容 |
|---|---|
| 路径 | `scripts/reproduction_ebft_100k.sh` |
| 定位 | `reproduction_sftEbft.sh` 中**仅保留 Stage 2 EBFT**：跳过 SFT，直接从已有 warm-start checkpoint 起步 |
| 必填变量 | `SFT_SAVE_PATH=/path/to/stage1_sft_model` |
| 训练超参 | 与 `reproduction_sftEbft.sh` 的 EBFT 阶段完全一致 |
| 下游评测 | 与 7.2 相同（HumanEval / MBPP / MultiPL-E + greedy + pass@1/4/16） |
| 启动 | `SFT_SAVE_PATH=... bash scripts/reproduction_ebft_100k.sh` |

---

## 8. 速查："我应该用哪个脚本？"

| 场景 | 推荐脚本 |
|---|---|
| 首次安装训练环境 | `setup_env.sh` |
| 拿"未训练基线"那一格的数字（直接评测预训练模型，不做 RL） | `run_baseline.sh` |
| 跑严格消融基线（无教师 / pointwise / 单 GT） | `run_G1_rebase.sh` |
| 跑 G2 主线（在线教师 + cf_l1oo） | `run_G2_rebase.sh` |
| 测"分布级奖励 vs pointwise 奖励"在有教师下的差异 | `run_G2_rebase_pointwise.sh` |
| 测"无教师 + vicinal target"是否能近似 cf_l1oo 分布信号 | `run_G2_rebase_no_teacher_vicinal.sh` |
| 测"无教师 + single GT"，并跑 16k/32k 两轮 post-eval | `run_G2_rebase_no_teacher_single_2rounds.sh` |
| 跑 G3 主线（双节点 + 在线教师 + EMA + adapter + 可训 critic-head） | `run_G3_rebase_2node_once.sh` |
| 测 G3 训练栈在"无教师 + vicinal target"下的表现 | `run_G3_rebase_no_teacher_vicinal.sh` |
| 对已训好的 G2 模型跑 16k → 32k 两轮长上下文评测 | `run_g2_posteval_retry16k_32k.sh` |
| 对任意训练组 ckpt 复跑统一的 16k+32k 二轮评测（含 oracle union 统计） | `supplement_2rounds/{baseline,G1,G2,G3}.sh` |
| 评测教师 27B 自身在 AOPS 测试集上的二轮上限 | `supplement_2rounds/Teacher.sh` |
| 论文 paper-QA 的 EBFT 趋势复现（含多 ckpt + 下游 benchmark） | `run_paper_qa_ebft_trend.sh` |
| 论文 SFT + EBFT 两阶段完整复现 | `reproduction_sftEbft.sh` |
| 已有 SFT warm-start，仅复现 EBFT 100k 阶段 | `reproduction_ebft_100k.sh` |
