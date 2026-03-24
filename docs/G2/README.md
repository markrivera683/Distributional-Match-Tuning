# G2：远程教师目标测度 + cf_l1oo 分布匹配训练

本目录记录 **G2 阶段**（distribution-level discrepancy with remote teacher-augmented target measure）的环境、配置、代码结构与脚本用途。

## 阅读顺序

| # | 文档 | 内容 |
|---|------|------|
| 1 | [ENVIRONMENT_AND_DEPENDENCIES.md](ENVIRONMENT_AND_DEPENDENCIES.md) | 运行环境、Python/CUDA 版本、依赖包安装 |
| 2 | [PATHS_AND_CONFIGURATION.md](PATHS_AND_CONFIGURATION.md) | 路径与超参数在哪里改、变量→CLI 对应关系、输出目录与 teacher cache |
| 3 | [G2_PHASE_SUMMARY.md](G2_PHASE_SUMMARY.md) | G2 阶段做了什么、数据流、关键代码文件与函数索引 |
| 4 | [SCRIPTS_CATALOG.md](SCRIPTS_CATALOG.md) | 各脚本分类（正式训练 / 冒烟 / 单机开发 / 历史环境 / 辅助工具） |

## 与仓库其他文档的关系

- [docs/ENVIRONMENT_SETUP.md](../ENVIRONMENT_SETUP.md) 记录的是**早期验证环境**的快照（含旧机器路径如 `/root/autodl-tmp/`），仅供参考版本号；G2 以本目录文档为准。
- [docs/STEP2_U1_CF_L1OO.md](../STEP2_U1_CF_L1OO.md) 和 [docs/STEP2D_TEACHER_TARGET_INTEGRATION.md](../STEP2D_TEACHER_TARGET_INTEGRATION.md) 包含 cf_l1oo 和教师目标集成的设计推导，是 G2 的理论基础。
- [docs/STEP3A_TEACHER_SAMPLING_PIPELINE.md](../STEP3A_TEACHER_SAMPLING_PIPELINE.md) 讨论教师采样管线的演进方向。

## 一句话概括 G2

> 训练 student（Qwen3.5-2B）时，用外部远程大模型（如 qwen-122b via vLLM）为每道完整数学题生成 M 条独立答案，与 ground-truth 混合构成目标测度 nu_c = (1-lambda) * delta(GT) + lambda * (1/M) * sum(delta(teacher_i))，然后通过 cf_l1oo（特征函数 leave-one-out）计算分布级奖励信号来指导 student 训练。
