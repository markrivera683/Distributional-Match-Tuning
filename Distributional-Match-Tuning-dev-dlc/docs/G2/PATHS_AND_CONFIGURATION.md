# G2 路径与超参数配置

本文档列出 G2 训练中**所有需要根据你的机器和实验调整的变量**，包括它们在 shell 脚本中的位置和映射到的 CLI 参数。

> 推荐参考脚本：`scripts/run_g2_8gpu_remote_teacher.sh`（结构最清晰，所有变量集中在顶部 8 个区块）。

---

## 1. 模型与数据路径

| 脚本变量 | 默认值 | CLI 参数 | 说明 |
|---------|--------|---------|------|
| `MODEL_PATH` | `/mnt/data/Qwen3.5-2B` | `--pretrain` / `--critic_pretrain` | 本地模型权重目录，需包含 `config.json` 和 `*.safetensors` |
| `TRAIN_DATA` 或 `PROMPT_DATA` | `/mnt/data/data/aops/aops_qa_hf_dict` | `--prompt_data` | 训练集路径 |
| `EVAL_DATA` 或 `EVAL_DATASET` | `/mnt/data/data/aops/test_qa.jsonl` | `--eval_dataset` | 评测集路径（JSONL 格式） |
| `INPUT_KEY` | `question` | `--input_key` | 数据集中问题字段名 |
| `LABEL_KEY` | `answer` | `--label_key` | 数据集中答案字段名 |
| `OUTPUT_KEY` | `answer` | `--output_key` | 输出字段名 |

### 命名差异说明

不同脚本对同一配置使用了不同变量名：

- `run_g2_8gpu_remote_teacher.sh` 和 `run_g2_baseline_8gpu_rerun.sh` 使用 `TRAIN_DATA` / `EVAL_DATA`。
- `run_train_qwen35_2b_aops_g2_remote_teacher.sh` 和 smoke 脚本使用 `PROMPT_DATA` / `EVAL_DATASET`。

它们最终都映射到同一个 CLI 参数 `--prompt_data` / `--eval_dataset`。

### HuggingFace 数据集两种落盘格式

训练数据可以是以下任一种形式，代码（`ebft_trainer.py` 中 `prepare_datasets()`）会自动识别：

| 目录 | 格式 | `load_from_disk` 返回类型 | 备注 |
|------|------|--------------------------|------|
| `aops_qa_hf/` | 单个 `Dataset` | `datasets.Dataset` | 目录内直接是 `data-*.arrow` + `dataset_info.json` |
| `aops_qa_hf_dict/` | `DatasetDict` | `datasets.DatasetDict` | 含 `dataset_dict.json` + `train/` 子目录 |

使用 `DatasetDict` 时，代码会自动按 `--prompt_split`（默认 `train`）取对应 split。

---

## 2. 远程教师端点

| 脚本变量 | 默认值 | CLI 参数 | 说明 |
|---------|--------|---------|------|
| `TEACHER_API_BASE` | `http://172.17.0.26:8000/v1` | `--teacher_api_base` | OpenAI 兼容 API 地址（vLLM / TGI 等） |
| `TEACHER_MODEL` | `qwen-122b` | `--teacher_model_name` | 教师模型在服务端的 `model` 名 |
| `TEACHER_API_KEY` | `teacher-local` | `--teacher_api_key` | Bearer token；若服务端无鉴权可设为 `EMPTY` |
| `TEACHER_API_STYLE` | `chat_completions` | `--teacher_api_style` | `chat_completions` 或 `completions` |
| `TEACHER_BACKEND` | `remote` | `--teacher_backend` | `remote`=HTTP API，`local`=Ray actor（G2 固定用 `remote`） |

在脚本顶部 `EDIT THESE FIRST` 区块修改这三个关键项：

```bash
TEACHER_API_BASE=http://your-teacher-host:8000/v1
TEACHER_MODEL=your-teacher-model-name
TEACHER_API_KEY=your-api-key
```

---

## 3. 教师生成参数与稳健性

| 脚本变量 | 默认值 | CLI 参数 | 说明 |
|---------|--------|---------|------|
| `TEACHER_TEMPERATURE` | `0.7` | `--teacher_temperature` | 教师采样温度 |
| `TEACHER_TOP_P` | `0.95` | `--teacher_top_p` | 教师 top-p 采样 |
| `TEACHER_MAX_NEW_TOKENS` | `512` | `--teacher_max_new_tokens` | 教师单次生成最大 token 数（完整答案长度） |
| `TEACHER_TIMEOUT` | `180` | `--teacher_timeout` | 单次 HTTP 请求超时（秒） |
| `TEACHER_MAX_RETRIES` | `3` | `--teacher_max_retries` | 失败重试次数 |
| `TEACHER_REMOTE_BATCH_SIZE` | `4` | `--teacher_remote_batch_size` | 并发 HTTP 请求数 |

---

## 4. 教师目标测度

目标测度公式：**nu_c = (1-lambda) * delta(GT) + lambda * (1/M) * sum(delta(teacher_i))**

| 脚本变量 | 默认值 | CLI 参数 | 说明 |
|---------|--------|---------|------|
| `CF_TEACHER_LAMBDA` | `0.5` | `--cf_teacher_lambda` | 混合权重 lambda：0=纯 GT，1=纯教师，0.5=等权 |
| `CF_TEACHER_N_SAMPLES` | `2` | `--cf_teacher_n_samples` | M：每题请求教师的独立回答数量 |

---

## 5. 奖励函数（CF-L1OO 分布匹配）

| 脚本变量 | 默认值 | CLI 参数 | 说明 |
|---------|--------|---------|------|
| `DISTRIBUTION_REWARD_TYPE` | `cf_l1oo`（固定） | `--distribution_reward_type` | 分布奖励类型 |
| `CF_TARGET_MODE` | `teacher`（固定） | `--cf_target_mode` | 目标构建模式（`single` / `vicinal` / `teacher`） |
| `CF_NUM_FREQS` | `128` | `--cf_num_freqs` | 随机傅里叶频率数量 |
| `CF_SIGMA` | `1.0` | `--cf_sigma` | RFF 核带宽 |
| `CF_ALPHA` | `0.5` | `--cf_alpha` | CF loss 中振幅权重 |
| `CF_BETA` | `0.5` | `--cf_beta` | CF loss 中相位权重 |
| `CF_REWARD_SCALE` | `1.0` | `--cf_reward_scale` | 奖励缩放因子 |
| `CF_SEED` | `43` | `--cf_seed` | 随机种子 |
| `FEATURE_MAP_TYPE` | `identity` | `--feature_map_type` | 特征映射：`identity`=原始隐藏态，`rff`=随机傅里叶特征 |
| `RFF_NUM_FEATURES` | `128` | `--rff_num_features` | RFF 特征维度 |
| `RFF_SIGMA` | `1.0` | `--rff_sigma` | RFF 带宽 |
| `RFF_SEED` | `43` | `--rff_seed` | RFF 随机种子 |

以下 vicinal 参数仅在 `cf_target_mode=vicinal` 时生效，G2 teacher 模式下不影响结果：

| 脚本变量 | 默认值 | CLI 参数 |
|---------|--------|---------|
| `CF_TARGET_NUM_REFS` | `1` | `--cf_target_num_refs` |
| `CF_TARGET_STD` | `0.05` | `--cf_target_std` |
| `CF_TARGET_SEED` | `43` | `--cf_target_seed` |

---

## 6. GPU 分配

| 脚本变量 | 默认值（8卡） | CLI 参数 | 说明 |
|---------|-------------|---------|------|
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 环境变量 | 可见 GPU 列表 |
| `ACTOR_GPUS` | `4` | `--actor_num_gpus_per_node` | Actor 模型 GPU 数 |
| `CRITIC_GPUS` | `4` | `--critic_num_gpus_per_node` | Critic 模型 GPU 数 |
| `REF_GPUS` | `4` | `--ref_num_gpus_per_node` | Reference 模型 GPU 数（与 actor 共置） |
| `REWARD_GPUS` | `4` | `--reward_num_gpus_per_node` | Reward 模型 GPU 数（与 critic 共置） |

8 卡推荐分配：actor+ref 共置 4 卡，critic+reward 共置 4 卡，通过 `--colocate_actor_ref --colocate_critic_reward` 启用。

冒烟测试可用单卡：所有值设为 1，加 `--colocate_all_models`。

---

## 7. 训练预算与 Batch

| 脚本变量 | 默认值 | CLI 参数 | 说明 |
|---------|--------|---------|------|
| `N_SAMPLES_PER_PROMPT` | `4` | `--n_samples_per_prompt` | 每题 rollout 样本数 |
| `ROLLOUT_BATCH_SIZE` | `64` | `--rollout_batch_size` | 每步 rollout 的 prompt 数 |
| `TRAIN_BATCH_SIZE` | `256` | `--train_batch_size` | 全局训练 batch size |
| `MICRO_TRAIN_BATCH_SIZE` | `4` | `--micro_train_batch_size` | 每 GPU 微 batch |
| `MAX_SAMPLES` | `46000` | `--max_samples` | 训练用总 prompt 数 |
| `NUM_EPISODES` | `1` | `--num_episodes` | 训练轮次 |
| `PROMPT_MAX_LEN` | `256` | `--prompt_max_len` | prompt 最大 token 长度 |
| `CONTEXT_MAX_LEN` | `8` | `--context_max_len` | block 上下文长度 |
| `GENERATE_MAX_LEN` | `8` | `--generate_max_len` | block 生成长度 |
| `STRIDE` | `8` | `--stride` | block 滑动步长 |

### Batch 约束（必须同时满足）

1. **DeepSpeed 约束**：`train_batch_size % (micro_train_batch_size x actor_gpus) == 0`
2. **ED 约束**：`train_batch_size == n_samples_per_prompt x rollout_batch_size`

默认值：`256 == 4 x 64`，`256 % (4 x 4) == 0`，两个约束均满足。

---

## 8. 损失系数与优化器

| 脚本变量 | 默认值 | CLI 参数 | 说明 |
|---------|--------|---------|------|
| `CE_LOSS_COEF` | `0.03` | `--ce_loss_coef` | 交叉熵辅助损失权重 |
| `DIVERSITY_REW_COEF` | `0.5` | `--diversity_rew_coef` | 多样性奖励系数 |
| `ALIGNMENT_REW_COEF` | `1.0` | `--alignment_rew_coef` | 对齐奖励系数 |
| `EMA_BETA` | `0.9` | `--ema_beta` | EMA 平滑系数 |
| `ACTOR_LR` | `1e-6` | `--actor_learning_rate` | Actor 学习率 |
| `CRITIC_LR` | `0` | `--critic_learning_rate` | Critic 学习率（G2 中冻结 critic） |

---

## 9. 输出目录与 Teacher Cache

### 输出目录结构

```
/root/outputs/
└── g2_8gpu_remote_teacher_20260324_153000/   ← RUN_ROOT（时间戳自动生成）
    ├── train.log
    ├── model/
    │   └── ckpt/
    ├── tensorboard/
    └── teacher_cache/                         ← 教师回答缓存
        └── teacher_cache.db                   ← SQLite 文件
```

| 脚本变量 | 默认值 | 说明 |
|---------|--------|------|
| `OUTPUT_ROOT` | `/root/outputs` | 所有 run 的父目录 |
| `RUN_TAG` | `g2_8gpu_remote_teacher_$(date +%Y%m%d_%H%M%S)` | 自动带时间戳 |
| `RUN_ROOT` | `${OUTPUT_ROOT}/${RUN_TAG}` | 本次 run 根目录 |
| `CACHE_DIR` | `${RUN_ROOT}/teacher_cache` | 教师缓存目录 |

### Teacher Cache 机制

- 缓存实现：`openrlhf/utils/teacher_provider.py` 中的 `TeacherCache` 类，SQLite 格式。
- CLI 参数：`--teacher_cache_enable`（开关）、`--teacher_cache_dir`（路径）。
- **每次新 run 因 `RUN_TAG` 带时间戳，cache 目录自动为新的空目录**，等同于 cold-start。
- 若需复用旧缓存：将 `CACHE_DIR` 显式指向历史 run 的 `teacher_cache/` 路径。
- 若需禁用缓存：设置 `TEACHER_CACHE_ENABLE=false`。

### Eval 配置

| 脚本变量 | 默认值 | CLI 参数 | 说明 |
|---------|--------|---------|------|
| `EVAL_STEPS` | `-1` | `--eval_steps` | 评测间隔步数；`-1`=禁用 |
| `EVAL_MAX_SAMPLES` | `50` | `--eval_max_samples` | 评测用最大样本数 |
| `EVAL_GENERATE_MAX_LEN` | `8` | `--eval_generate_max_len` | 评测生成长度 |

> 默认禁用 eval 是因为 GatedDeltaNet fallback 在长生成时会 OOM。如已安装兼容的 flash-linear-attention，可将 `EVAL_STEPS` 设为正值。

---

## 10. 在脚本中的位置速查

以推荐脚本 `scripts/run_g2_8gpu_remote_teacher.sh` 为例，变量按区块组织：

| 区块 | 行号范围 | 内容 |
|------|---------|------|
| 1. GPU ALLOCATION | ~20-27 | `CUDA_VISIBLE_DEVICES`、各组件 GPU 数 |
| 2. REMOTE TEACHER ENDPOINT | ~30-36 | API 地址、模型名、密钥、接口风格 |
| 3. TEACHER TARGET DISTRIBUTION | ~39-54 | lambda、M、生成参数、超时、cache |
| 4. REWARD FUNCTION | ~57-76 | cf_l1oo 相关参数 |
| 5. MODEL & DATA PATHS | ~79-87 | 模型、训练/评测数据路径 |
| 6. TRAINING BUDGET | ~90-107 | batch size、样本数、episode、block 参数 |
| 7. LOSS COEFFICIENTS | ~110-123 | 损失权重、学习率 |
| 8. OUTPUT & LOGGING | ~126-138 | 输出路径、eval 配置 |

其他脚本结构类似，变量均集中在文件顶部的 `EDIT THESE FIRST` 区块。
