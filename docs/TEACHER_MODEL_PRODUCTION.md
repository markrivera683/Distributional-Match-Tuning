# Teacher Model 生产运行指南（Production 模式）

本文档说明正式跑全量数据时的 teacher model 配置，目标是最大化 throughput、实现 training loop 中零等待。

---

## 1. Teacher Server 配置（Production）

### 硬件
- 机器：独立节点，8×A100 80GB
- 模型：`qwen-122b`（122B 参数）
- TP=8，所有卡用于一个模型实例

### 启动命令

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 8 \
  --dtype "${DTYPE:-auto}" \
  --api-key "${API_KEY:-teacher-local}" \
  --generation-config "${GEN_CONFIG:-vllm}" \
  --max-num-seqs 384 \
  --max-num-batched-tokens 98304 \
  --max-model-len 4096 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.93 \
  --disable-log-requests
```

### 参数说明

| 参数 | Production 值 | 说明 |
|---|---|---|
| `--tensor-parallel-size` | `8` | 122B 模型必须 8 卡切分 |
| `--max-num-seqs` | `384` | 同时处理的最大序列数；提高并发能力，让 GPU 满负荷跑 continuous batching |
| `--max-num-batched-tokens` | `98304` | 每个 decode step 最多处理的 token 数（= 384 × 256）；打满 GPU 矩阵乘法 |
| `--max-model-len` | `4096` | AoPS 题目 + 答案实际 ≤ 1536 token，设 4096 有余量且大幅节省 KV cache 显存 |
| `--enable-chunked-prefill` | 开启 | 长 prompt 的 prefill 分块执行，避免阻塞其他请求的 decode，降低 tail latency |
| `--gpu-memory-utilization` | `0.93` | 93% 显存给 KV cache，比默认 0.9 多出约 18GB，可并发更多序列 |
| `--disable-log-requests` | 开启 | 关闭每请求日志，高并发 warmup 时避免 I/O 瓶颈 |

> **关于 `--max-model-len=4096`**：vLLM 默认读模型 config 中的 `max_position_embeddings`（Qwen2-122B 默认 32768），会预分配大量 KV cache 块白白浪费显存。设为 4096 可以大幅增加实际可并发序列数。启动日志中 `Number of GPU blocks` 行可以确认实际分配了多少 KV cache。

---

## 2. 生成参数（Production）

以下参数同时出现在 **warmup 脚本**（`run_warmup_teacher_cache.sh`）和**训练脚本**（`run_train_qwen35_2b_aops_g2_remote_teacher.sh`）中，**两边必须完全一致**，任何一个不同都会导致 cache key 不匹配，旧 cache 全部作废。

| 参数 | Production 值 | 说明 |
|---|---|---|
| `CF_TEACHER_N_SAMPLES` | `2` | 每道题生成的 teacher completion 数量（目标分布支撑点数 M）；M=2 是目前的稳定配置 |
| `TEACHER_TEMPERATURE` | `0.7` | 采样温度；0.7 在数学推理任务上平衡答案质量和多样性，使 M 个 completion 有实质差异 |
| `TEACHER_TOP_P` | `0.95` | Nucleus sampling 截断阈值；过滤概率最低的 5% token，防止生成错误词汇 |
| `TEACHER_MAX_NEW_TOKENS` | `512` | 每个 completion 最多生成 token 数；AoPS 完整解答通常需要 200-500 token，512 是安全上限 |

> **Cache key 组成**：`SHA256(prompt + model_name + n_samples + temperature + top_p + max_new_tokens)`
> 改动以上任何一个参数，所有旧 cache 条目对新 key 不可见，需要重新跑 warmup。

---

## 3. 全量 Cache Warmup 流程（Production）

Warmup 的目标是在训练开始前把全部 training prompts 的 teacher completions 写入 SQLite cache，使训练过程中每个 step 的 teacher 调用都是 cache hit，延迟降为零。

### 步骤

**Step 1：确认 server 已启动并可达**

```bash
curl -s -H "Authorization: Bearer teacher-local" \
  http://172.17.0.26:8000/v1/models | python3 -m json.tool
```

**Step 2：运行高并发 warmup**

```bash
WARMUP_BATCH_SIZE=64 \
CF_TEACHER_N_SAMPLES=2 \
TEACHER_MAX_NEW_TOKENS=512 \
MAX_SAMPLES=46000 \
CACHE_DIR=/root/outputs/teacher_cache_shared \
bash scripts/run_warmup_teacher_cache.sh
```

### Warmup 参数说明

| 参数 | Production 值 | 说明 |
|---|---|---|
| `WARMUP_BATCH_SIZE` | `64` | 并发 HTTP 请求数；这是 `ThreadPoolExecutor` 的 `max_workers`，也是同时在途的请求数。vLLM 的 continuous batching 能吃下这个并发量，64 可以充分打满 server |
| `MAX_SAMPLES` | `46000` | 全量 training prompts 数量；与训练脚本 `MAX_SAMPLES` 保持一致，确保 100% coverage |
| `TEACHER_TIMEOUT` | `180` | 单次请求超时秒数；122B 模型生成 512 token 约需 20-60s，180s 留有足够余量 |
| `TEACHER_MAX_RETRIES` | `3` | 失败重试次数；网络抖动时自动重试，指数退避（2s, 4s, 8s） |
| `CACHE_DIR` | `/root/outputs/teacher_cache_shared` | SQLite cache 目录；**必须在本地文件系统**（不能在 ossfs/NFS），SQLite 需要文件锁 |

**Step 3：确认 coverage**

Warmup 结束时日志中会打印：
```
Final cache coverage:   46000 / 46000 (100.0%)
```
若 coverage 不足 100%，有 `warmup_failed.txt` 记录失败的 prompt，可以针对性重跑。

**Step 4（可选）：导出为 HF Dataset**

导出后训练时可切换到 `teacher_backend=dataset`，彻底脱离网络依赖：

```bash
python scripts/export_teacher_cache_to_dataset.py \
  --prompt_data /mnt/data/data/aops/aops_qa_hf_dict \
  --input_key question \
  --split train \
  --cache_dir /root/outputs/teacher_cache_shared \
  --model_name qwen-122b \
  --n_samples 2 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_new_tokens 512 \
  --output_dir /mnt/data/data/aops/teacher_dataset_n_samples_2
```

---

## 4. 训练脚本中的 Teacher 参数（Production）

```bash
# run_train_qwen35_2b_aops_g2_remote_teacher.sh 中的关键参数

# ── Teacher 端点 ──
TEACHER_API_BASE=http://172.17.0.26:8000/v1   # teacher server 地址
TEACHER_MODEL=qwen-122b                        # 必须与 --served-model-name 一致
TEACHER_API_KEY=teacher-local                  # 必须与 server --api-key 一致
TEACHER_API_STYLE=chat_completions             # chat_completions 或 completions

# ── 目标分布参数（必须与 warmup 完全一致）──
CF_TEACHER_N_SAMPLES=2                         # M：teacher 支撑点数
TEACHER_TEMPERATURE=0.7                        # 采样温度
TEACHER_TOP_P=0.95                             # Nucleus sampling
# TEACHER_MAX_NEW_TOKENS 在训练脚本中默认 512，需确认

# ── 训练侧并发参数 ──
TEACHER_REMOTE_BATCH_SIZE=32                   # 训练时同时发出的 teacher 请求数（建议从 4 提高到 32）
TEACHER_TIMEOUT=180                            # 请求超时
TEACHER_MAX_RETRIES=3                          # 失败重试

# ── 目标分布混合 ──
CF_TEACHER_LAMBDA=0.5                          # λ：target = (1-λ)·GT + λ·teacher
CF_TARGET_MODE=teacher                         # 启用 teacher target mode
TEACHER_BACKEND=remote                         # 使用 HTTP API（cache 命中时不实际发请求）

# ── Cache ──
# --teacher_cache_enable                       # 开启 SQLite cache
# --teacher_cache_dir /root/outputs/teacher_cache_shared  # 与 warmup 相同目录
```

> **关于 `TEACHER_REMOTE_BATCH_SIZE`**：当前脚本默认值为 4，建议提高到 32。
> 这是训练进程发 teacher API 请求时的并发数，低并发会导致即使 cache miss 也要串行等待。
> Cache 全部命中时此参数无影响，但 warmup 未完成时直接影响 training step 延迟。

---

## 5. 推荐工作流

```
1. 启动 teacher server（带上 production 参数）
2. 运行 warmup（WARMUP_BATCH_SIZE=64，MAX_SAMPLES=46000）
3. 确认 cache coverage = 100%
4. （可选）导出 HF Dataset，切换 teacher_backend=dataset
5. 启动训练
   - 若用 remote backend：训练时 100% cache hit，teacher 延迟 ≈ 0
   - 若用 dataset backend：teacher 延迟 = 0，server 可以关闭
```

---

## 6. 多 n_samples 配置管理

不同 `CF_TEACHER_N_SAMPLES` 配置的 cache **不能混用**，需分目录管理：

| n_samples | cache 目录 | HF dataset 目录 |
|---|---|---|
| 2 | `/root/teacher_cache_n_samples_2/` | `/mnt/data/data/aops/teacher_dataset_n_samples_2` |
| 4 | `/root/teacher_cache_n_samples_4/` | `/mnt/data/data/aops/teacher_dataset_n_samples_4` |
| 8 | `/root/teacher_cache_n_samples_8/` | `/mnt/data/data/aops/teacher_dataset_n_samples_8`（已有 28,009 条）|

切换实验配置时，只需改 `CF_TEACHER_N_SAMPLES` 和对应的 `CACHE_DIR`，不影响其他配置的 cache。

---

## 7. 多 Worker 部署（水平扩展）

当单台 8×A100 server 的 throughput 不足（例如 warmup 耗时过长，或训练时 teacher 调用偶发 cache miss 导致延迟），可以部署多个 vLLM worker 实例，由客户端做 round-robin 负载均衡。

### 架构

```
训练进程
  └─ MultiWorkerTeacherProvider
        ├─ RemoteTeacherProvider → Worker 0  http://172.17.0.26:8000/v1  (8×A100)
        ├─ RemoteTeacherProvider → Worker 1  http://172.17.0.26:8001/v1  (8×A100)
        └─ RemoteTeacherProvider → Worker 2  http://172.17.0.27:8000/v1  (8×A100)
```

每个 worker 是完全独立的 vLLM 进程，各自管理自己的 GPU、KV cache 和 SQLite cache sub-dir。
请求按 round-robin 分配，所有 worker 同时工作，总 throughput ≈ N × 单 worker throughput。

### 启动多个 Worker

每台机器（或同一机器的不同端口）分别运行一个 server：

```bash
# Worker 0 — 端口 8000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve /mnt/data/models/teacher/qwen-122b \
  --served-model-name qwen-122b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 \
  --dtype auto \
  --api-key teacher-local \
  --max-num-seqs 384 \
  --max-num-batched-tokens 98304 \
  --max-model-len 4096 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.93 \
  --disable-log-requests &

# Worker 1 — 端口 8001（同一机器需要不同端口，或另一台机器用 8000）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve /mnt/data/models/teacher/qwen-122b \
  --served-model-name qwen-122b \
  --host 0.0.0.0 --port 8001 \
  --tensor-parallel-size 8 \
  ...
```

> 注意：同一台机器上运行两个 TP=8 的 122B 实例需要 16 张 A100（每个实例独占 8 张）。
> 如果只有 8 张卡，需要用两台独立机器。

### 训练脚本配置（多 Worker）

`--teacher_api_base` 支持逗号分隔的多个 URL，自动启用 `MultiWorkerTeacherProvider`：

```bash
# 2 个 worker 示例
TEACHER_API_BASE=http://172.17.0.26:8000/v1,http://172.17.0.26:8001/v1

# 3 个 worker 示例（跨两台机器）
TEACHER_API_BASE=http://172.17.0.26:8000/v1,http://172.17.0.27:8000/v1,http://172.17.0.28:8000/v1

# 并发数建议设为 worker 数 × 单 worker 推荐并发
# 2 workers × 32 = 64
TEACHER_REMOTE_BATCH_SIZE=64
```

完整训练脚本参数段：

```bash
TEACHER_API_BASE=http://172.17.0.26:8000/v1,http://172.17.0.26:8001/v1 \
TEACHER_REMOTE_BATCH_SIZE=64 \
TEACHER_TIMEOUT=180 \
bash scripts/run_train_qwen35_2b_aops_g2_remote_teacher.sh
```

### Warmup 分片（多 Worker 并行填充 Cache）

多 worker 部署时，warmup 也可以分片并行执行，把数据集按 worker 数量切分，每个 worker 独立负责自己那份：

```bash
# 共 46000 条，3 个 worker 各负责约 1/3
# Shard 0：第 0-15333 条
MAX_SAMPLES=15334 \
CACHE_DIR=/root/teacher_cache_shard_0 \
TEACHER_API_BASE=http://172.17.0.26:8000/v1 \
WARMUP_BATCH_SIZE=64 \
bash scripts/run_warmup_teacher_cache.sh &

# Shard 1：第 15334-30666 条（需要 warmup 脚本支持 --offset，或手动跳过）
# 目前 warmup 脚本不支持 offset，推荐方式：
# 1. 每个 worker 跑全量 warmup，重叠部分互相覆盖写入（SQLite INSERT OR REPLACE 安全）
# 2. 或者统一用一个高并发 warmup 进程指向负载均衡地址
```

**最简单的多 worker warmup 方式**：warmup 脚本的 `TEACHER_API_BASE` 也支持逗号分隔，
一个 warmup 进程自动把请求分散到所有 worker，所有结果写入同一个 cache 目录：

```bash
TEACHER_API_BASE=http://172.17.0.26:8000/v1,http://172.17.0.26:8001/v1 \
WARMUP_BATCH_SIZE=128 \
CACHE_DIR=/root/outputs/teacher_cache_shared \
bash scripts/run_warmup_teacher_cache.sh
```

> 注意：多 worker 时 cache 按 worker 分成子目录（`worker_0/`, `worker_1/`...）。
> warmup 写入的是每个 worker 自己的子目录，训练时各 worker 从自己的子目录读取。
> 两者使用相同的 `CACHE_DIR` 根目录即可自动对齐。

### Cache 目录结构（多 Worker）

```
/root/outputs/teacher_cache_shared/
  worker_0/
    teacher_cache.db    ← Worker 0 的 SQLite cache
  worker_1/
    teacher_cache.db    ← Worker 1 的 SQLite cache
  worker_2/
    teacher_cache.db    ← Worker 2 的 SQLite cache
```

每个 worker 只读写自己的 sub-dir，避免 SQLite 写锁竞争。
Round-robin 分配保证相同 prompt 总是被路由到同一个 worker（取模），cache 命中率不受影响。

### 吞吐量估算

| 配置 | 单 worker 估算 | 2 workers | 3 workers |
|---|---|---|---|
| qwen-122b, 8×A100, n=2, 512 tokens | ~30-60 prompts/min | ~60-120 prompts/min | ~90-180 prompts/min |
| 46000 prompts warmup 时间 | ~8-25h | ~4-13h | ~3-9h |

实际速度取决于网络延迟、prompt 长度和 vLLM KV cache 命中率。

### 健康检查（多 Worker）

```python
from openrlhf.utils.teacher_provider import MultiWorkerTeacherProvider

provider = MultiWorkerTeacherProvider(
    api_bases=[
        "http://172.17.0.26:8000/v1",
        "http://172.17.0.26:8001/v1",
    ],
    model_name="qwen-122b",
    api_key="teacher-local",
)
print(provider.health_check())
# 期望输出: {'http://172.17.0.26:8000/v1': 'ok', 'http://172.17.0.26:8001/v1': 'ok'}
```

---

## 8. 监控 Teacher 是否真正参与训练

在训练日志中搜索以下关键字确认 teacher 信号正常：

**单 worker：**
```
[RemoteTeacher] Init: api_base=http://172.17.0.26:8000/v1, ...
```

**多 worker：**
```
[MultiWorkerTeacher] Init: 2 workers, total_concurrency=64, per_worker_concurrency=32
  workers: http://172.17.0.26:8000/v1, http://172.17.0.26:8001/v1
[MultiWorkerTeacher] Requesting 64 prompts x 2 samples across 2 workers
[MultiWorkerTeacher] Done: 64 prompts in 12.3s (5.2 prompts/s)
```

**Teacher 信号正常：**
```
[TEACHER-VERIFY] ENTER teacher branch          ← teacher 分支被激活
[Teacher-FullQ] Requesting N unique questions  ← teacher 被调用
[TEACHER-VERIFY] teacher_embedding built: shape=... mean=...  ← embedding 构建成功，mean 不为 0
teacher_in_reward = True                       ← teacher 信号进入 reward 计算
```

**异常信号：**
```
[Teacher-Diag] WARNING: zero tokens replaced by teacher  ← teacher completion 为空，信号丢失
[TEACHER-VERIFY] _build_teacher_embedding returned None  ← teacher 整体失败，回退到 GT only
teacher_in_reward = False                                ← teacher 未参与 reward（检查 cf_target_mode）
```

```
[TEACHER-VERIFY] ENTER teacher branch          ← teacher 分支被激活
[Teacher-FullQ] Requesting N unique questions  ← teacher 被调用（cache miss 或 dataset lookup）
[TEACHER-VERIFY] teacher_embedding built: shape=... mean=...  ← embedding 构建成功，mean 不为 0
teacher_in_reward = True                       ← teacher 信号进入 reward 计算
```

**异常信号：**

```
[Teacher-Diag] WARNING: zero tokens replaced by teacher  ← teacher completion 为空，信号丢失
[TEACHER-VERIFY] _build_teacher_embedding returned None  ← teacher 整体失败，回退到 GT only
teacher_in_reward = False                                ← teacher 未参与 reward（检查 cf_target_mode）
```
