# G2 环境与依赖

## 1. 硬件要求


| 资源     | 推荐配置                               | 备注                                    |
| ------ | ---------------------------------- | ------------------------------------- |
| GPU    | 8x NVIDIA A100 80GB（或同等 A800）      | 冒烟测试可单卡                               |
| GPU 显存 | 80GB/卡                             | 4 卡 actor+ref 共置，4 卡 critic+reward 共置 |
| 系统内存   | >= 128GB                           | Ray object store 默认分配 8GB             |
| 磁盘     | >= 50GB 空闲（模型权重 + 数据 + 输出 + cache） | teacher cache 为 SQLite 文件             |


## 2. 软件运行栈

```
Linux (kernel 5.10+)
├── Python 3.12
├── PyTorch 2.5.1+cu124
├── DeepSpeed 0.18.x          ← 分布式训练 ZeRO Stage 2
├── Ray 2.47~2.48             ← 多 actor group 编排
├── transformers >= 4.53       ← 模型加载、tokenizer
├── datasets >= 4.6           ← HF Dataset / DatasetDict 加载
├── flash-attn 2.8.x          ← 注意力加速（可选但推荐）
└── openrlhf（本仓库 editable install）
```

远程教师端需要一个 **OpenAI 兼容的 HTTP 服务**（例如 vLLM `--served-model-name qwen-122b`），训练机本身不需要安装 vLLM。

## 3. 依赖声明

本仓库的依赖文件：

- `**requirements.txt`**（仓库根目录）

关键条目：


| 包              | 声明版本       | 说明                                                  |
| -------------- | ---------- | --------------------------------------------------- |
| `ray[default]` | `==2.48.0` | 实际验证通过的版本为 2.47.0，两者均可                              |
| `deepspeed`    | 未锁版        | 实际验证 0.18.8                                         |
| `transformers` | 未锁版        | 需 >= 4.53（支持 Qwen3.5 架构）                            |
| `datasets`     | 未锁版        | 需 >= 4.6                                            |
| `flash-attn`   | 注释状态       | 需单独安装：`pip install flash-attn --no-build-isolation` |
| `torch`        | 未锁版        | 需匹配 CUDA 版本，推荐 `pip install torch==2.5.1+cu124`     |


## 4. 安装步骤（推荐）

```bash
conda create -n g2 python=3.12.3 -y
conda activate g2

pip install --upgrade pip wheel setuptools

# PyTorch（匹配你的 CUDA 版本）
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# 仓库依赖
cd /root/code/data/Distributional-Match-Tuning
pip install -r requirements.txt

# Ray 如需精确匹配验证版本
pip install ray[default]==2.47.0

# Flash Attention
# Time-consuming without flash-atten and oom risk
pip install flash-attn==2.8.3 --no-build-isolation    

# 本仓库 editable 安装
pip install -e .
```

## 5. 与 docs/ENVIRONMENT_SETUP.md 的关系

仓库中已有的 `docs/ENVIRONMENT_SETUP.md` 记录的是**旧机器上的验证快照**：

- 路径指向 `/root/autodl-tmp/Energy/...`，与当前仓库位置（`/root/code/data/Distributional-Match-Tuning`）不同。
- 模型为 `Qwen2.5-1.5B`，G2 阶段已切换到 `Qwen3.5-2B`。
- 数据集为 `sjelassi/opencode-instruct_100k_200tok`，G2 使用 AoPS 数据。

**结论**：该文档中的**版本号列表**仍有参考价值（如确认 PyTorch/CUDA 兼容性），但文件路径和数据配置请以本 G2 文档为准。

## 6. Editable 安装注意事项

`openrlhf` 通过 `pip install -e .` 安装。一台机器上若存在多个仓库副本，editable 安装只会指向最后一次 `pip install -e` 的目录。

验证当前指向：

```bash
python -m pip show openrlhf | grep Location
```

如果输出的路径不是 `/root/code/data/Distributional-Match-Tuning`，需要重新执行 `pip install -e .`。

## 7. 环境变量

G2 训练脚本会自动 export 以下环境变量（一般不需要手动修改）：


| 变量                                       | 默认值                        | 用途                       |
| ---------------------------------------- | -------------------------- | ------------------------ |
| `CUDA_VISIBLE_DEVICES`                   | `0,1,2,3,4,5,6,7`          | 可见 GPU                   |
| `HF_HUB_OFFLINE`                         | `1`                        | 离线模式，不访问 HuggingFace Hub |
| `HF_DATASETS_OFFLINE`                    | `1`                        | 数据集离线加载                  |
| `TOKENIZERS_PARALLELISM`                 | `false`                    | 避免 tokenizer fork 警告     |
| `RAY_memory_usage_threshold`             | `0.995`                    | Ray 内存溢出阈值               |
| `PYTORCH_CUDA_ALLOC_CONF`                | `expandable_segments:True` | 减少 CUDA OOM 碎片           |
| `OPENRLHF_RAY_OBJECT_STORE_MEMORY_BYTES` | `8589934592`（8GB）          | Ray object store 上限      |
| `PYTHONUNBUFFERED`                       | `1`                        | 日志即时输出                   |


首次运行前若模型或数据尚未下载到本地，需临时设置 `HF_HUB_OFFLINE=0` 和 `HF_DATASETS_OFFLINE=0`。

## 8. 快速验证清单

在启动正式训练前，按顺序检查：

```bash
# 1. Python 版本
python -V

# 2. PyTorch + CUDA
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# 3. 关键包
python -c "import deepspeed, ray, transformers, datasets; print('ok')"

# 4. flash-attn（可选）
python -c "import flash_attn; print(flash_attn.__version__)"

# 5. openrlhf editable install 指向
python -m pip show openrlhf | grep Location

# 6. 模型权重存在
ls /mnt/data/Qwen3.5-2B/config.json

# 7. 训练数据存在
ls /mnt/data/data/aops/aops_qa_hf/dataset_info.json

# 8. 远程教师可达
curl -s -m 5 http://172.17.0.26:8000/v1/models && echo "teacher OK"
```

