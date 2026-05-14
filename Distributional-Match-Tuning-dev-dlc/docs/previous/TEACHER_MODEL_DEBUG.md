# Teacher Model 调试指南（Debug 模式）

本文档说明如何在**小规模、快速验证**场景下配置和测试 teacher model，用于确认整条调用链是否通畅，不消耗大量 GPU 时间和 API 配额。

---

## 1. Teacher Server 配置（Debug）

Debug 模式的目标是验证 API 可达、参数传递正确、cache 命中逻辑正常。

### 启动命令

```bash
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --dtype "${DTYPE:-auto}" \
  --api-key "${API_KEY:-teacher-local}" \
  --generation-config "${GEN_CONFIG:-vllm}" \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --max-model-len 2048 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.90
```

### 参数说明

| 参数 | Debug 值 | 说明 |
|---|---|---|
| `--max-num-seqs` | 32 | 同时处理的最大序列数，debug 不需要高并发 |
| `--max-num-batched-tokens` | 4096 | 每个 decode step 的 token 上限，debug 用小值 |
| `--max-model-len` | 2048 | 最大 context 长度，debug 题目 + 答案通常在 1536 token 以内 |
| `--enable-chunked-prefill` | 开启 | 防止长 prompt 阻塞 decode，debug 也建议开启 |
| `--gpu-memory-utilization` | 0.90 | 留更多显存余量，debug 环境更稳定 |

---

## 2. 生成参数（Debug）

以下参数在 warmup 脚本和训练脚本中必须完全一致，否则 cache key 不匹配，旧 cache 全部失效。

| 参数 | Debug 推荐值 | 说明 |
|---|---|---|
| `CF_TEACHER_N_SAMPLES` | `2` | 每道题生成的 teacher completion 数量（目标分布支撑点数 M）；debug 用最小值 |
| `TEACHER_TEMPERATURE` | `0.7` | 采样温度；控制答案多样性，0.7 是数学推理经验值，debug 保持不变 |
| `TEACHER_TOP_P` | `0.95` | Nucleus sampling 截断阈值；保留概率累积到 95% 的 token 集合，过滤低概率噪声词 |
| `TEACHER_MAX_NEW_TOKENS` | `128` | 每个 completion 最多生成 token 数；debug 用短答案，节省时间 |

> **重要**：`TEACHER_MAX_NEW_TOKENS=128` 只用于 debug，正式跑数据必须改回 512，否则答案被截断。

---

## 3. Cache Warmup（Debug）

```bash
# debug 只跑少量样本验证流程
CF_TEACHER_N_SAMPLES=2 \
TEACHER_MAX_NEW_TOKENS=128 \
MAX_SAMPLES=50 \
WARMUP_BATCH_SIZE=4 \
CACHE_DIR=/root/teacher_cache_debug \
bash scripts/run_warmup_teacher_cache.sh
```

### Warmup 参数说明

| 参数 | Debug 值 | 说明 |
|---|---|---|
| `MAX_SAMPLES` | `50` | 只处理 50 道题，几分钟内跑完 |
| `WARMUP_BATCH_SIZE` | `4` | 并发 HTTP 请求数，debug 用小值避免打爆 server |
| `TEACHER_TIMEOUT` | `60` | 单次请求超时秒数，debug 用短 timeout 快速发现问题 |
| `TEACHER_MAX_RETRIES` | `2` | 失败重试次数，debug 快速失败 |
| `CACHE_DIR` | `/root/teacher_cache_debug` | **与正式 cache 分开**，避免污染生产 cache |

---

## 4. 验证调用链是否通畅

### Step 1：确认 API 可达

```bash
curl -s -H "Authorization: Bearer teacher-local" \
  http://172.17.0.26:8000/v1/models | python3 -m json.tool
```

预期输出：返回 `qwen-122b` 的模型信息。

### Step 2：手动发一个 completion 请求

```bash
curl -s -X POST http://172.17.0.26:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer teacher-local" \
  -d '{
    "model": "qwen-122b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "n": 2,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 64
  }' | python3 -m json.tool
```

预期：`choices` 里有 2 个 completion，每个有实质内容。

### Step 3：运行 debug warmup

```bash
bash scripts/run_warmup_teacher_cache.sh
```

观察日志中：
- `Already cached: 0 / 50` → 首次运行正常
- `Fetched this run: 50` → 全部成功
- `Final cache coverage: 50 / 50 (100.0%)` → cache 写入正常

### Step 4：验证 cache 命中

```bash
# 再跑一次 warmup，应该全部命中 cache，不发任何 API 请求
bash scripts/run_warmup_teacher_cache.sh
```

预期日志：`Already cached: 50 / 50 → All questions already cached. Nothing to do.`

### Step 5：小规模训练验证

```bash
MAX_SAMPLES=50 \
ROLLOUT_BATCH_SIZE=8 \
TRAIN_BATCH_SIZE=32 \
CF_TEACHER_N_SAMPLES=2 \
TEACHER_REMOTE_BATCH_SIZE=4 \
bash scripts/run_train_qwen35_2b_aops_g2_remote_teacher.sh
```

在训练日志中确认：
- `[TEACHER-VERIFY] ENTER teacher branch` → teacher 分支被激活
- `[Teacher-FullQ] Requesting N unique questions x M=2` → teacher 被调用
- `[TEACHER-VERIFY] teacher_embedding built: shape=...` → embedding 构建成功
- `teacher_in_reward = True` → teacher 信号进入了 reward 计算

---

## 5. 多 Worker Debug

多 worker 部署时，在单 worker 调通之后再扩展，逐步验证。

### Step 1：分别验证每个 worker 可达

```bash
# Worker 0
curl -s -H "Authorization: Bearer teacher-local" \
  http://172.17.0.26:8000/v1/models | python3 -m json.tool

# Worker 1
curl -s -H "Authorization: Bearer teacher-local" \
  http://172.17.0.26:8001/v1/models | python3 -m json.tool
```

### Step 2：用 health_check 一次性验证所有 worker

```python
from openrlhf.utils.teacher_provider import MultiWorkerTeacherProvider

provider = MultiWorkerTeacherProvider(
    api_bases=["http://172.17.0.26:8000/v1", "http://172.17.0.26:8001/v1"],
    model_name="qwen-122b",
    api_key="teacher-local",
)
print(provider.health_check())
# 期望: {'http://172.17.0.26:8000/v1': 'ok', 'http://172.17.0.26:8001/v1': 'ok'}
```

### Step 3：多 worker 小规模 warmup

```bash
TEACHER_API_BASE=http://172.17.0.26:8000/v1,http://172.17.0.26:8001/v1 \
CF_TEACHER_N_SAMPLES=2 \
TEACHER_MAX_NEW_TOKENS=128 \
MAX_SAMPLES=20 \
WARMUP_BATCH_SIZE=8 \
CACHE_DIR=/root/teacher_cache_debug_multiworker \
bash scripts/run_warmup_teacher_cache.sh
```

观察日志确认：
- `[MultiWorkerTeacher] Init: 2 workers` → 多 worker 模式生效
- 两个 worker 都有 `[RemoteTeacher] Done` 日志 → 负载均衡正常工作
- `Final cache coverage: 20 / 20 (100.0%)` → cache 正常写入

### 多 Worker Cache 目录结构

多 worker 时 cache 按 worker 分成子目录：

```
/root/teacher_cache_debug_multiworker/
  worker_0/teacher_cache.db
  worker_1/teacher_cache.db
```

确认两个子目录都有 `.db` 文件且非空，说明两个 worker 都在写入 cache。

---

## 6. 常见 Debug 问题

### API 返回 404 / Connection refused
- 检查 `TEACHER_API_BASE` IP 和端口是否正确
- 确认 vLLM server 已启动：`ps aux | grep vllm`

### Cache 不命中（训练时仍然调 API）
- 检查训练脚本和 warmup 脚本的以下参数是否完全一致：
  - `CF_TEACHER_N_SAMPLES`
  - `TEACHER_TEMPERATURE`
  - `TEACHER_TOP_P`
  - `TEACHER_MAX_NEW_TOKENS`
  - `TEACHER_MODEL`（即 `--teacher_model_name`）
  - `CACHE_DIR`

### `[Teacher-Diag] WARNING: zero tokens replaced by teacher`
- teacher completion 为空字符串，说明 cache miss 且 API 返回空
- 先检查 API 是否正常，再检查 cache key 是否对齐

### `teacher_in_reward = False`
- 确认训练脚本中 `CF_TARGET_MODE=teacher` 且 `TEACHER_BACKEND=remote`
- 确认 `CF_TEACHER_LAMBDA > 0`
