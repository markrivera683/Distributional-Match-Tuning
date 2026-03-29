# Distributional Match Tuning：项目完整非代码介绍

> 本文档基于仓库内的真实文件内容综合撰写，包括 README.md、EBFT SFT 进展汇报、docs/ 目录下所有设计文档、inference_loss/README.md、data/case_study_results/ 审计报告、configs/ 配置文件、scripts/ 脚本命名与分组、以及 openrlhf/ 目录结构。所有重要判断均标注依据文件；无法确认的内容单独注明。

---

## 一、项目一句话概括

这个项目是一个**语言模型后训练方法研究项目**，核心目标是把原始论文提出的"基于能量的特征匹配微调"（EBFT，Energy-Based Feature-matching Fine-Tuning）从"单点特征对齐"逐步升级为"条件特征分布匹配"，并在此基础上引入外部教师模型作为目标分布的来源，最终探索一种更具原则性的、兼顾能力迁移与多样性建模的语言模型优化范式。

项目以 AoPS 竞赛数学问答为主要实验场景，建立在 OpenRLHF 分布式训练框架之上，是一项从论文实现出发、向更广泛分布级训练方法演进的中期研究工作。

**依据文件：** `README.md`（ArXiv 论文引用 arXiv:2603.12248），`EBFT SFT 进展汇报`，`docs/ebft_upgrade_master_plan.md`

---

## 二、项目面向的问题与目标

### 2.1 核心问题

现有语言模型后训练方法（SFT、RLHF、标准蒸馏等）大多从"单点监督"角度建模训练目标：给定一个 prompt，让模型的输出靠近某一条参考答案。这种训练范式有几个根本性限制：

- **目标稀疏：** 把训练目标压缩成单一参考点，忽略了输出空间的多模态结构。同一道数学题可能有多种合理推理路径，而单点对齐只能引导模型靠向其中一条。
- **奖励粒度粗：** 奖励信号是"这个样本好不好"的 pointwise 打分，而不是"这一组生成结果在分布层面是否更接近期望分布"，缺乏对分布形态的感知。
- **多样性外挂：** 传统方法中，多样性要么被忽略，要么作为独立的惩罚项附加，而不是从训练目标本身内生出来。

**依据文件：** `docs/ebft_upgrade_master_plan.md`（第 3 节），`docs/ebft_ncfm_upgrade_notes.md`

### 2.2 针对的任务与场景

- **任务层面：** 主要是 AoPS 竞赛数学问答（本项目主实验场景），以及原论文验证的代码问答（Q&A code）、非结构化代码生成、翻译等任务。
- **模型层面：** 针对语言模型的后训练阶段（RL-style fine-tuning），而非预训练。具体模型包括 Qwen2.5-1.5B（原论文）和 Qwen3.5-2B Base（本项目主要实验模型）。
- **训练机制：** 基于 on-policy rollout 的策略梯度训练（PPO/RLOO 变体），reward 的来源和形式是该项目的核心改造对象。

**依据文件：** `EBFT SFT 进展汇报`（第 3.2 节），`configs/qa_code.yaml`，`configs/local_qwen35_2b_train.yaml`

### 2.3 为什么值得做

**第一层：理论更有原则性。** 把训练目标从"单点对齐"升级为"条件分布匹配"，是从第一性原理出发的方法演进。`ebft_upgrade_master_plan.md` 明确写道，vanilla EBFT 可以被理解为"在冻结特征几何中做条件一阶矩匹配"，而这个项目要把它推进到更丰富的条件分布差异族。

**第二层：NCFM 的可移植洞察。** 项目从图像合成领域的 NCFM 工作中引入了一个关键概念：用特征函数（characteristic function）来刻画整体分布之间的差异，而不仅仅比较均值或方差。这使得多样性成为分布差异本身的内生属性，而不需要作为额外惩罚项附加。

**第三层：教师蒸馏的自然结合。** 一旦训练目标变成分布级匹配，目标分布的来源就可以从"单条 GT 答案"扩展到"来自更强教师模型的多条输出"，把后训练和知识蒸馏统一到同一个框架下。

**依据文件：** `docs/ebft_upgrade_master_plan.md`（第 1、5 节），`docs/ebft_ncfm_upgrade_notes.md`，`docs/teacher_replacing_gt_lit_review.md`

### 2.4 项目目标的多个层次

| 层次 | 目标 | 主要依据文件 |
|------|------|----------|
| 能力提升层 | 在 AoPS 数学问答上，通过更好的训练信号提升小模型数学推理能力 | `EBFT SFT 进展汇报` 第 3.1 节 |
| 训练范式探索层 | 把 EBFT 推进到 distribution-level、teacher-extendable 的方法家族 | `docs/ebft_upgrade_master_plan.md` |
| 评测框架建设层 | 发现并修复 AoPS 评测管线的系统性缺陷，建立可信基准 | `data/case_study_results/N2_case_study_and_eval_audit_report.md` |
| 工程基础设施层 | 构建 teacher warmup 缓存、远程 API 调用、多 worker 部署等工程配套 | `docs/TEACHER_MODEL_PRODUCTION.md`，`docs/G2/G2_PHASE_SUMMARY.md` |
---

## 三、项目整体故事线 / 主线叙事

### 3.1 起点：一篇论文的工程实现

项目起源于 2026 年 3 月的 ArXiv 论文《Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models》（arXiv:2603.12248），作者包括 Samy Jelassi、Mujin Kwun、Rosie Zhao、Yuanzhi Li、Nicolo Fusi、Yilun Du、Sham M. Kakade、Carles Domingo-Enrich。

EBFT 的核心思想：不再用 token-level 交叉熵训练模型，而是通过一个 critic 模型提取的隐状态特征来定义奖励信号：

- 让多条生成样本的特征表示与 ground-truth 的特征在特征空间中对齐（**alignment reward**）；
- 同时用多样性奖励（**diversity reward**）防止生成结果退化成同一个答案；
- 训练过程基于 on-policy rollout，类似 PPO 策略梯度框架，但 reward 定义在特征空间而非 token 空间。

论文在代码生成和翻译任务上用 Qwen2.5-1.5B 验证了这一方法，代码实现基于 OpenRLHF 框架。本项目继承这套实现，并在此基础上进行深度改造和方法演进。

**依据文件：** `README.md`

### 3.2 第一步挑战：暴露了两个前提性问题

**问题一：评测管线存在系统性缺陷。**

`data/case_study_results/N2_case_study_and_eval_audit_report.md` 记录了一次系统审计，发现：

- AoPS 答案大量为 LaTeX 格式，不经 `\boxed{}` 包裹直接解析时，约 **19% 的答案无法解析**，另有 **21.8% 会被错误解析**（例如 `2^{2022}` 被解析为 `2`）；
- 训练器 eval 路由存在 bug：按路径名是否含 "gsm8k" 或 "math" 判断评测分支，而 AoPS 数据路径 `LiveAoPSBench-2024.jsonl` 不含这两个子串，导致被**错误路由到翻译 BLEU 评测**；
- EBFT 与 SFT 的 eval 路径对 gold answer 解析处理不一致，制造系统性偏差。

结论："至少 19-22% 的评测结果可能不可信"。

**问题二：任务难度与模型能力存在错配。**

Case study 对 30 个 Qwen3.5-2B Base 输出样本归因，发现：27/30（90%）是"推理在中途被截断"——模型能正确启动推理，但 512 token 生成长度对竞赛数学题根本不够；无一个样本"完全不会做"。原 AoPS 论文使用 DeepSeek-Math-7B-Instruct 等数学专用模型训练约 18,000 steps，而本项目使用 Qwen3.5-2B Base 训练 500 steps。

**依据文件：** `data/case_study_results/N2_case_study_and_eval_audit_report.md`，`EBFT SFT 进展汇报`（第 4 节）

### 3.3 方法演进：G1 -> G2 -> G3 的三级递进

**G1（paper-faithful baseline）：** 尽量忠实复现原论文 EBFT，作为干净的参考基线。使用原始 pointwise alignment reward + diversity reward，目标是回答"EBFT 本身在 AoPS 场景下是否能稳定工作"。

**G2（reward 层面的分布升级）：** 在 G1 基础上，把训练信号从"单样本与单目标点的对齐"改成"一组生成样本的经验分布与目标分布之间的 discrepancy"。核心变化：

1. 引入基于特征函数的 **leave-one-out 分布差异奖励**（`cf_l1oo`）：不再问"样本 j 是否接近 GT"，而是问"把样本 j 从生成集合中移除后，整组生成的经验分布与目标分布之间的 discrepancy 如何变化"，即该样本对分布匹配的边际贡献；
2. 引入外部远程教师模型（如 Qwen-122B），为每道题生成 M 条独立答案，与 GT 混合构成目标分布：`nu_c = (1-lambda)*delta(GT) + lambda*(1/M)*sum(delta(teacher_i))`。

G2 核心判断：**多样性不再是外加项，而是由分布差异本身内生决定。**

**G3（representation/geometry 层面的升级）：** G2 仍假设 critic 提供的特征几何是合适的。G3 的判断是：feature geometry 本身也应该允许被轻量调整。做法：在 critic 特征流上增加轻量 **residual bottleneck adapter**，通过 **EMA 目标网络**稳定这一过程——backbone 尽量冻结，只训练小规模 head/adapter，EMA 分支提供较慢变化的 target geometry，防止 discrepancy learning 因 target 漂移而失稳。

**依据文件：** `EBFT SFT 进展汇报`（第 2 节），`docs/G2/G2_PHASE_SUMMARY.md`，`docs/STEP4_FEATURE_ADAPTER_2LITE.md`，`docs/ebft_upgrade_master_plan.md`（第 5 节）

### 3.4 工程主线：围绕 teacher 的完整基础设施

- **教师模型服务化：** Qwen-122B 通过 vLLM 部署为独立 HTTP 服务（8xA100，TP=8），支持多 worker 水平扩展，提供 OpenAI 兼容 API；
- **Warmup 缓存：** 训练前通过 64 并发请求把全部约 46,000 道训练题的 teacher completions 写入 SQLite 缓存，使训练中每步 teacher 调用延迟降为近零；
- **完整题目查询：** 教师接收每道题的完整原始题目，生成完整答案后再 tokenize 对齐到 student 的 block 结构；
- **缓存导出：** 支持把 SQLite 缓存导出为 HuggingFace Dataset，切换到 `teacher_backend=dataset` 后训练完全脱网。

**依据文件：** `docs/TEACHER_MODEL_PRODUCTION.md`，`docs/G2/G2_PHASE_SUMMARY.md`（第 3 节）

### 3.5 当前实验状态

根据 `docs/STEPX_PROGRESS_CHECKPOINT.md` 和 `EBFT SFT 进展汇报`：

- G1/G2/G3 均已在 8xA100 上完成 500 step 训练（每组超过 7 小时）；
- 当前结果（修复评测前）显示所有组准确率与 Base 模型相近，无显著提升，SFT 大幅下降；

| 模型 | 总样本 | 正确数 | 准确率 |
|------|--------|--------|--------|
| Base | 5328 | 447 | 8.39% |
| G1   | 5328 | 448 | 8.41% |
| G2   | 5328 | 427 | 8.02% |
| G3   | 5328 | 430 | 8.07% |
| SFT  | 5328 | 240 | 4.50% |

- 评测管线存在系统性缺陷，上表数字诊断价值受限；
- 当前主线任务：修复评测管线 -> 获得可信 baseline -> 重新评估各组差异 -> 决定是否进一步调整超参或延长训练。

**依据文件：** `EBFT SFT 进展汇报`（第 4 节），`docs/STEPX_PROGRESS_CHECKPOINT.md`
---

## 四、项目的方法论框架（非代码版）

### 4.1 外行版：用大白话解释这个项目在做什么

想象你在教一个学生做数学题。传统方法是：每道题给他一个标准答案，如果他的答案和标准答案一样就给满分。这很直接，但有两个问题：

第一，数学题往往有多种解法。如果只提供一种标准答案，学生可能学到的是"模仿那一种写法"，而不是"理解数学的本质"。

第二，这种方式鼓励学生每次都写同一种答案。但真正优秀的学生应该能用多种方式解决问题，展示对知识的灵活掌握。

这个项目的思路是：**不看单次答案是否和标准答案一样，而是看这个学生整体上是否在以一种"接近好学生的方式"思考和解题**。具体来说，是把学生的多次解题尝试看成一个"解题风格的分布"，然后问这个分布是否接近好学生（教师模型）的"解题风格的分布"。这就是"分布匹配"：不是比较单次输出，而是比较整体输出模式。

### 4.2 研究同学版：方法体系的核心概念

**EBFT 在这个项目里扮演什么角色？**

EBFT（Energy-Based Feature-matching Fine-Tuning）是这个项目的起点和基础方法。它的关键创新是：把奖励信号从 token 空间搬到特征空间。用一个 critic 模型提取生成序列的隐状态特征，然后计算生成特征与 GT 特征之间的相似度作为奖励。这让训练信号在语义层面而非 token 层面对齐，对生成的表面形式更鲁棒。训练时采用 on-policy rollout + 类 PPO 的策略梯度更新，这已经比纯 SFT 更能利用多样生成路径的信息。

**distribution matching / distributional reward 是什么意思？**

原始 EBFT 的奖励是 pointwise 的：每个样本独立与 GT 特征对齐。这个项目的 G2 把它升级为 distributional：不再问"样本 j 是否接近 GT"，而是问"把样本 j 从生成集合中移除后，整组生成的经验分布与目标分布之间的 discrepancy 如何变化"。这个差值就是样本 j 的奖励，即 leave-one-out 边际贡献。这种奖励自然地编码了多样性——如果样本 j 是重复的，移除它不会让整体分布变差，奖励就低；如果它覆盖了目标分布中其他样本没覆盖的区域，移除它会让 discrepancy 大幅上升，奖励就高。

**teacher target / target measure 是什么意思？**

`target measure`（目标测度）是分布匹配的"对齐对象"。在 G1 中，这个对象是单点 GT 特征（Dirac 测度）；在 G2 中，它被扩展为 GT 特征与教师模型多条输出的混合经验分布：`nu_c = (1-lambda)*delta(phi(GT)) + lambda*(1/M)*sum(delta(phi(teacher_i)))`。这里 `phi` 是 critic 模型提取的特征映射，`M` 是教师采样数，`lambda` 控制 GT 与教师的混合比例。这个目标测度比单点 Dirac 更丰富，能够捕捉条件输出分布的多模态结构。

**依据文件：** `docs/G2/G2_PHASE_SUMMARY.md`（第 2 节），`docs/STEP2D_TEACHER_TARGET_INTEGRATION.md`（第 3 节），`docs/STEP2E_DISTILLATION_FORMULATION.md`

**discrepancy / feature geometry / EMA target geometry 是什么意思？**

`discrepancy`（差异度）是衡量两个分布之间距离的量。本项目使用基于**特征函数（characteristic function，CF）**的差异度：CF 通过对随机频率向量的内积取期望来表示整个分布，两个分布的 CF 之间的 L1 距离给出一个 distribution-level 的差异量。这个差异度天然捕捉分布的整体形态（不只是均值），且通过固定随机频率向量在计算上保持可行。

`feature geometry` 是指 critic 模型所定义的特征空间的结构——哪些序列在特征空间里相近，哪些相远。这个几何直接决定了 discrepancy 的含义和训练信号的质量。在 G1/G2 中，feature geometry 是**冻结的**，保证奖励信号稳定。在 G3 中，允许通过轻量 adapter 做局部调整。

`EMA target geometry` 是 G3 中用于稳定 discrepancy learning 的机制：通过指数移动平均（EMA）更新的 critic 副本，作为较慢变化的参照几何用于 reward 计算，而 online 分支负责更新。这种在线/目标非对称设计借鉴自自监督学习（MoCo、BYOL、DINO 等）的稳定化经验。

**依据文件：** `docs/ebft_upgrade_master_plan.md`（第 2、5 节），`docs/STEP2_U1_CF_L1OO.md`，`docs/STEP4_FEATURE_ADAPTER_2LITE.md`，`docs/feature_network_unfreeze_lit_review.md`

### 4.3 项目内语境版：G1/G2/G3 变化发生在哪一层

用 `docs/ebft_upgrade_master_plan.md` 的统一数学框架描述，三次升级对应三个正交轴：

对每个 context `c`，定义：
- `q_c(y)`：目标条件测度
- `z_eta(c,y) = phi_eta(c:y)`：特征几何（由 critic 参数 eta 决定）
- `mu_{theta,c}`：student rollout 在特征空间中诱导的经验分布
- `nu_c`：目标在特征空间中诱导的分布

整体目标函数族：`L(theta; eta, psi) = E_c [ D_psi(mu_{theta,c}, nu_c) ] + regularizers`

| 版本 | 改变的轴 | 具体改变 |
|------|---------|--------|
| G1 | — | vanilla pointwise alignment + diversity penalty，接近弱形式的条件一阶矩匹配 |
| G2 | 改变 `D_psi`（discrepancy 族）+ 改变 `q_c`（目标测度）| CF leave-one-out discrepancy 替代 pointwise reward；GT + teacher 混合经验分布替代单点 Dirac 目标 |
| G3 | 改变 `z_eta`（特征几何）| critic feature stream 上加轻量 residual bottleneck adapter；EMA target geometry 稳定 discrepancy learning |

这三个轴的关系是递进而非并列的：先把 discrepancy 做对（G2），再把 target 做丰富（G2 teacher），最后才考虑让 feature geometry 自适应（G3）。这个顺序在 `ebft_upgrade_master_plan.md` 第 6 节中有明确的方法论论证。

**依据文件：** `docs/ebft_upgrade_master_plan.md`（第 2、6 节），`EBFT SFT 进展汇报`（第 2 节）
---

## 五、项目结构总览（按目录与职责）

### 5.1 根目录关键文件

- `README.md`：项目对外介绍，面向论文复现，了解原始论文方法的入口。
- `EBFT SFT 进展汇报`：项目核心进展文档，记录 G1/G2/G3 的方法设计、实验设置、当前结果与解释。了解"项目在做什么、做到哪"的最重要单文件。
- `STEPWISE_WORKTREE.md`：说明当前仓库是逐步实验的副本，与原始工作仓库分离，用于增量测试。
- `README_zh.md` / `README_openrlhf.md`：OpenRLHF 框架说明，面向框架用户而非方法研究。

### 5.2 `docs/` 目录——方法设计与研究规划的核心

`docs/` 是整个项目最重要的文档中心，记录了从方法设计到工程执行的完整研究思路。

**方法设计主文档：**
- `ebft_upgrade_master_plan.md`：项目方法升级的"内部真相来源"，定义三个升级轴的统一数学框架、执行顺序逻辑、go/no-go 标准。整个项目方法论最权威的文件。
- `ebft_ncfm_upgrade_notes.md`：从 NCFM 论文提炼可移植洞察，说明为什么用 CF-based distribution discrepancy 替代 pointwise alignment。

**逐步执行计划（STEP 系列）：**
- `STEP0_BASELINE_CHECKLIST.md`：基线最低验收标准，列出训练侧/评测侧必须存在的所有 metric。
- `STEP1_PAPER_QA_ALIGNMENT.md`：paper-aligned 对照实验设计，只比较 `identity` vs `rff` feature map。
- `STEP1_U1_EXECUTION_PLAN.md`：Upgrade 1 的执行梯子，Stage A(长预算对照) -> Stage B(窄 RFF 调参) -> Stage C(重复性检验)。
- `STEP2_U1_CF_L1OO.md`：`cf_l1oo` reward 的第一个最小实现——用固定频率 CF discrepancy 替代 pointwise reward。
- `STEP2B_TARGET_MEASURE_DESIGN.md`：深度分析 target measure 现有局限，对比 single/vicinal/multi-reference/teacher 四种方案。
- `STEP2C_TOKEN_CLOUD_TARGET.md`：token-cloud 经验目标设计，用 GT 的全部 token 特征云作为目标经验分布，无需 teacher。
- `STEP2D_TEACHER_TARGET_INTEGRATION.md`：teacher-augmented target measure 的集成方案，定义 tensor 形状设计和代码集成点。
- `STEP2E_DISTILLATION_FORMULATION.md`：把 teacher target 模式理解为显式蒸馏，梳理与 SFT/KD/RLHF/vanilla EBFT 的关系。
- `STEP3_TOKEN_LEVEL_PREP.md`：token-level CF reward 的预研，说明当前代码中已有的 token 支持路径。
- `STEP3A_TEACHER_SAMPLING_PIPELINE.md`：teacher 数据路径的端到端管线，从 prompt 到 embedding 的完整流程。
- `STEP3B_GT_TO_DISTRIBUTION_TRANSITION.md`：目标侧过渡梯子：Dirac -> Vicinal -> Empirical Measure。
- `STEP4_FEATURE_ADAPTER_2LITE.md`：G3 的 feature adapter 2-lite 设计，residual bottleneck adapter + 冻结 backbone 的第一个稳定版本。
- `STEPX_PROGRESS_CHECKPOINT.md`：跨 Step 的进度检查点，记录各 Step 的完成状态和当前主线。

**Teacher 相关文档：**
- `TEACHER_DISTILLATION_NOTES.md`：API teacher vs open-weight teacher 的对比，以及与 EMA teacher 的区别。
- `TEACHER_MODEL_PRODUCTION.md`：生产环境下 teacher server 的完整配置，包括 vLLM 启动参数、warmup 流程、多 worker 部署。
- `TEACHER_MODEL_DEBUG.md`：小规模调试场景下的 teacher 配置和验证流程。

**文献调研文档：**
- `feature_network_unfreeze_lit_review.md`：feature network 解冻方向的文献综述，总结 DQN/MoCo/BYOL/VICReg 等工作对"moving target"问题的启示。
- `teacher_replacing_gt_lit_review.md`：用教师模型替换 GT 的文献综述，梳理 MiniLLM/OPCD/DASD/DLCoT 等工作。

**G2 子目录 `docs/G2/`：** G2 阶段的完整文档包，含 `G2_PHASE_SUMMARY.md`（阶段总结、数据流图、代码变更汇总）、`SCRIPTS_CATALOG.md`（脚本分类目录）、环境配置说明等。
### 5.3 `openrlhf/` 目录——训练框架的核心实现

**`openrlhf/cli/`：训练入口**
- `train_ebft_ray.py`：EBFT 的主训练入口，定义所有 CLI 参数，初始化 Ray 集群，启动分布式训练 actors。
- 其他 `train_*.py`：OpenRLHF 框架原有的训练入口（PPO、SFT、DPO、KTO 等），说明框架的基础能力范围。

**`openrlhf/trainer/`：训练循环与逻辑**
- `ebft_trainer.py`：EBFT 的中央训练循环，协调 rollout 生成、embedding 提取、reward 计算、advantage 估计、策略更新，并运行周期性评测。
- `ebft_eval_mixin.py`：评测逻辑的 mixin，处理下游任务（GSM8K/MATH/AoPS）的答案解析和准确率计算。
- `ray/ebft_actor.py`：作为 Ray actor 的策略模型包装，负责 log-probabilities 计算和策略梯度。
- `ray/ebft_critic.py`：作为 Ray actor 的 critic 模型包装，提取 hidden-state embeddings，用于 reward 计算。
- `ppo_utils/ebft_experience_maker.py`：生成和存储 rollout 经验，包含 teacher 采样路径（本地 actor 或远程 HTTP provider）、embedding 构建、CF reward 计算的完整逻辑。这是 G2 工程改造的核心文件。
- `ppo_utils/ebft_replay_buffer.py`：EBFT 的经验回放缓冲区。

**`openrlhf/models/`：模型与损失函数**
- `actor.py`、`critic.py`：Actor/Critic 模型的基础实现，包含本项目对 Qwen3.5 等模型的兼容性补丁。
- `loss.py`：`EBFTPolicyLoss`（PPO/GSPO 策略梯度损失）、`ClassifierLoss`（可选的 critic 分类器损失）。
- `utils.py`：`build_strided_attention_mask_and_positions`——构建 EBFT 训练的核心 4D attention mask 和 position IDs，实现 strided-block 并行生成。

**`openrlhf/utils/`：工具库**
- `embedding_utils.py`：CF reward 计算的核心，包含 `compute_cf_l1oo_reward`、`_build_cf_target_embedding`（支持 single/vicinal/teacher 三种 target 模式）。
- `teacher_provider.py`：远程教师 HTTP 客户端（`RemoteTeacherProvider`）、SQLite 缓存（`TeacherCache`）、多 worker 负载均衡（`MultiWorkerTeacherProvider`）。
- `math_verifier.py`：数学答案验证工具，用于下游评测。

**`openrlhf/datasets/`：数据集**
- `qa_dataset.py`：核心数据集类 `QADataset`，负责把 QA 对 tokenize 后打包为固定长度 chunks，生成 `doc_ids` 和 `answer_masks` 张量，并保存原始问题文本列表（用于教师查询）。

### 5.4 `inference_loss/` 目录——独立评测模块

`inference_loss/` 是一个轻量、独立的评测模块，把 EBFT 的核心评测逻辑从完整训练系统中提取出来，不依赖 Ray、DeepSpeed 或训练基础设施，可以单独运行：

- `StridedActorModel`：加载 causal LM，用 strided attention mask 生成样本并计算 log-probabilities；
- `StridedCriticModel`：加载模型（via AutoModel），用 strided blocks 提取 hidden-state embeddings；
- `EvaluationMetrics`：计算 alignment rewards、diversity rewards、perplexity/cross-entropy，报告 Pass@1 和 Pass@k 变体。

这个模块的存在说明：项目有在训练系统之外独立评测 reward 信号质量的需求，属于评测基础设施建设的一部分。

**依据文件：** `inference_loss/README.md`

### 5.5 `scripts/` 目录——实验脚本的完整分层

`scripts/` 是项目实验运行的操作层，包含 60+ 个脚本，按用途分为以下几类：

**G1/G2/G3 正式训练脚本（8xA100）：**
- `run_g1_baseline_8gpu_rerun.sh`：G1 baseline 复跑，使用 pointwise reward，无 teacher。
- `run_g2_8gpu_remote_teacher.sh`：G2 推荐的 8 卡正式训练脚本，所有变量集中在顶部，支持环境变量覆盖。
- `run_g3_2full_8gpu.sh`：G3 的 8 卡完整训练脚本。

**冒烟测试脚本（快速验证链路）：**
- `run_step0_vanilla_qwen3_smoke.sh`：Step 0 vanilla EBFT 冒烟，验证基线路径可运行。
- `dry_run_8gpu_real_v2/v3/v4.sh`：多版本的 8 卡 dry run。

**辅助工具脚本：**
- `warmup_teacher_cache.py` / `run_warmup_teacher_cache.sh`：teacher 缓存的 warmup 工具。
- `export_teacher_cache_to_dataset.py`：把 SQLite 缓存导出为 HuggingFace Dataset。
- `mock_teacher_server.py`：本地假教师服务，用于无真实 teacher API 时的端到端测试。
- `evaluate_reward_ce.py`：独立的 reward 与交叉熵评估脚本。
- `case_study_aops.py`：AoPS 基座模型 case study 和评测审计脚本。
- `summarize_aops_results.py`：汇总 AoPS 结果。

**历史/异构环境脚本：** `run_aops_g1_8gpu.sh`、`run_aops_g2_8gpu.sh`、`run_aops_g3_8gpu.sh` 等，来自早期开发阶段，路径和模型与当前环境不匹配，仅供了解历史实验配置，不推荐直接使用。

**依据文件：** `docs/G2/SCRIPTS_CATALOG.md`

### 5.6 `configs/` 目录——超参配置

- `qa_code.yaml`：原论文的 Q&A code 任务超参配置，用于 SLURM 阵列作业的超参扫描，定义了 sweep 网格（`ce_loss_coef` 和 `diversity_rew_coef` 的组合）。
- `translation.yaml`、`unstructured_code.yaml`：原论文其他任务的超参配置。
- `local_qwen35_2b_train.yaml`：本项目单机调试用的 Qwen3.5-2B 配置，distribution matching 主线（`distribution_reward_type: cf_l1oo`，`cf_target_mode: single`），适合单卡本地开发。

### 5.7 `data/` 目录——数据与实验结果

- `LiveAoPSBench-2024.jsonl`：AoPS 测试集，5328 条竞赛数学题，用于评测 actor 的最终答题准确率。
- `data/case_study_results/`：
  - `case_study_base.json`：对 Base 模型输出进行 case study 的原始数据。
  - `N2_case_study_and_eval_audit_report.md`：评测审计报告，发现了评测管线的系统性缺陷，是当前项目进度判断的关键文件。

### 5.8 `examples/` 目录

`examples/scripts/` 下是 OpenRLHF 框架自带的训练示例（PPO、DPO、GRPO、SFT、Rejection Sampling 等标准流程）。这些脚本与 EBFT 无直接关系，但说明了框架的基础能力范围，可作为对比参考。

### 5.9 `dockerfile/` 目录

包含 `Dockerfile` 和 `docker-entrypoint.sh`，提供了基于 NVIDIA PyTorch 镜像的容器化运行环境，便于在集群环境中部署训练。
---

## 六、实验路线与版本演进

### 6.1 Step 系列：方法开发的阶梯

在 G1/G2/G3 的大版本框架之下，项目内部还有一套更细粒度的 Step 执行序列，记录在 `docs/STEP*.md` 系列文档中。这套序列反映了方法开发的实际推进节奏：

**Step 0（baseline 建立）：**
目标是获得一个稳定、可诊断的 vanilla EBFT baseline。验收标准包括：训练能端到端完成不崩溃、train-side 和 eval-side 的所有 metric 都能正常输出、TensorBoard 日志存在。对应脚本 `run_step0_vanilla_qwen3_smoke.sh`。

**Step 1（Upgrade 1：richer discrepancy）：**
锁定所有超参，只比较 `feature_map_type=identity`（原始 EBFT）vs `feature_map_type=rff`（随机傅里叶特征）。这是在确认方法升级可训练、可稳定之前的最保守一步。对应脚本 `run_step1_paper_qa_feature_map.sh`、`run_step1_feature_map_qwen3.sh`。

**Step 2（cf_l1oo：第一个最小分布匹配实现）：**
用固定频率 CF discrepancy 替代 pointwise reward，保持其他一切不变。这是 G2 的奖励核心。目标是证明 distributional reward 在当前 EBFT pipeline 下可以稳定训练。对应脚本 `run_step2lite_cf_adapter_smoke.sh`、`run_step2lite_cf_adapter_directdisc_smoke.sh`。

**Step 2B/2C/2D/2E（target measure 升级系列）：**
在 cf_l1oo 稳定后，系统地升级 target 侧：
- Step 2B：诊断 single/vicinal target 的局限，确立 teacher-augmented target 的优先方向；
- Step 2C：实现 token-cloud 经验目标（`cf_tokencloud_l1oo`），不需要 teacher 就能获得比 Dirac 更丰富的目标；
- Step 2D：集成 teacher-augmented target measure，打通本地/远程两条 teacher 路径；
- Step 2E：把 teacher target 模式正式解读为分布级蒸馏。

**Step 3/3A/3B（teacher pipeline 硬化）：**
- Step 3：token-level CF reward 的预研（暂缓）；
- Step 3A：teacher 采样管线的端到端硬化，确保 teacher embeddings 的形状对齐和失效模式可被诊断；
- Step 3B：目标侧过渡梯子的形式化，确立 Dirac -> Vicinal -> Empirical Measure 的演进路径。

**Step 4（feature adapter 2-lite，即 G3）：**
在 cf_l1oo + teacher target 稳定后，在 critic feature stream 上加轻量 residual bottleneck adapter，EMA 目标网络稳定训练。这是 G3 的工程实现。

### 6.2 G1/G2/G3 的详细对比

**哪些东西在三个版本中保持不变：**
- 基座模型：Qwen3.5-2B Base（本项目）；
- 训练框架：OpenRLHF + Ray + DeepSpeed；
- 训练任务：AoPS 竞赛数学问答；
- 训练预算：500 global steps，8xA100；
- 评测协议：统一使用 actor-only final-answer accuracy benchmark，统一测试集 `LiveAoPSBench-2024.jsonl`；
- On-policy rollout 结构：strided-block 生成，每个 prompt 生成 N 条样本；
- Critic 架构：与 actor 同架构，frozen（G1/G2）或 adapter-tuned（G3）。

**关键变量在三个版本中如何变化：**

| 维度 | G1 | G2 | G3 |
|------|----|----|----|
| reward 类型 | pointwise（alignment + diversity penalty）| cf_l1oo（distributional leave-one-out）| cf_l1oo（同 G2）|
| target measure | 单点 GT（Dirac）| GT + teacher 混合经验分布 | GT + teacher 混合经验分布 |
| critic feature geometry | 完全冻结 | 完全冻结 | 轻量 adapter 可训练，EMA 目标网络稳定 |
| 教师模型 | 无 | Qwen-122B（远程 HTTP API + SQLite 缓存）| Qwen-122B（同 G2）|
| 多样性来源 | 显式 diversity penalty 项 | 由 CF discrepancy 内生 | 由 CF discrepancy 内生 |

**为什么这种对比设计是合理的：**

`docs/ebft_upgrade_master_plan.md` 第 11 节明确说明："如果我们同时改变 discrepancy geometry、target source 和 reward substrate，我们就不知道方法为什么提升或失败了。因此，这些阶段不只是为了写作更清晰，而是为了科学性。" 每个版本只改变一个轴，保持其他轴不变，这是保持因果归因清晰性的基本要求。

**依据文件：** `EBFT SFT 进展汇报`（第 2、3 节），`docs/ebft_upgrade_master_plan.md`（第 9-11 节），`docs/STEPX_PROGRESS_CHECKPOINT.md`

### 6.3 baseline 层的细化

除了 G1/G2/G3 的方法对比，项目还设计了一套 baseline 内部的细化比较：

- `cf_target_mode=single`：单点 GT 目标，最简单的 cf_l1oo baseline；
- `cf_target_mode=vicinal`：GT 特征加局部高斯扰动，作为低成本的平滑 baseline；
- `cf_target_mode=teacher`：GT + teacher 混合目标，G2 的主线配置。

这三者构成了 target measure 设计的内部对照组，使得 teacher 带来的增益可以与 target measure 的平滑化效果分离。

**依据文件：** `docs/STEP2B_TARGET_MEASURE_DESIGN.md`，`docs/STEP3B_GT_TO_DISTRIBUTION_TRANSITION.md`
---

## 七、训练与实验运行流程（非代码视角）

### 7.1 数据从哪里来

训练数据来自 `DeepStudentLlama/AoPS-Instruct`，共约 647,255 个数学 QA 对，格式为 `{question, answer}`，已处理为 HuggingFace Arrow 格式存储在 `/mnt/data/data/aops/aops_qa_hf`。测试集来自 `jojo23333/LiveAoPSBench-2024`，共 5328 条，存储为 `data/LiveAoPSBench-2024.jsonl`。

数据预处理的核心步骤是"packing"：`QADataset` 把 QA 对 tokenize 后，按 strided-block 结构打包为固定长度的 chunks，每个 chunk 包含多个 question-answer block，并生成 `doc_ids`（标记每个 block 来自哪道题）和 `answer_masks`（标记哪些 token 位置是答案区域）。这种打包结构是 EBFT strided 生成机制的数据基础。

**依据文件：** `docs/G2/G2_PHASE_SUMMARY.md`（第 5 节），`data/case_study_results/N2_case_study_and_eval_audit_report.md`

### 7.2 各角色分工

EBFT 训练系统中有五个主要角色，通过 Ray 分布在不同 GPU 上：

- **Actor（策略模型）：** 负责 on-policy rollout——接收 packed prompt，用 strided-block 结构生成多条候选答案，计算 log-probabilities，接受策略梯度更新。Actor 是最终要优化的对象，也是评测时单独使用的模型。
- **Critic（特征提取模型）：** 与 Actor 同架构，但用于提取 hidden-state embeddings 而非生成。Critic 把完整的 packed sequence（prompt + generation）通过 strided attention 处理，提取每个 block 的特征表示。在 G1/G2 中 Critic 完全冻结；在 G3 中允许通过 feature adapter 做轻量调整。
- **Reference model（参考模型）：** Actor 的初始化副本，冻结不更新，用于计算 KL 散度惩罚，防止训练中 actor 与初始策略漂移过远。
- **Reward model：** 在 EBFT 中，reward 不是一个独立的神经网络，而是由 Critic 提取的 embeddings 经过 CF discrepancy 计算得到的。Reward 计算逻辑封装在 `ebft_experience_maker.py` 和 `embedding_utils.py` 中。
- **Teacher model（G2/G3）：** 外部更强的模型，负责为每道题生成 M 条独立答案，丰富目标分布的支撑。Teacher 通过 HTTP API 提供服务（远程路径）或作为本地 actor group 加载（本地路径）。Teacher 不参与策略梯度更新，只提供目标分布的样本。

**依据文件：** `README.md`（"Key components" 部分），`docs/TEACHER_DISTILLATION_NOTES.md`

### 7.3 训练前的准备

在正式训练开始前，G2 需要完成以下准备步骤：

1. **启动 teacher server：** 在独立节点上用 vLLM 部署 Qwen-122B，配置 TP=8，设置合适的 `max-num-seqs` 和 `max-model-len`（生产环境为 384 和 4096）；
2. **Warmup teacher cache：** 运行 `run_warmup_teacher_cache.sh`，以 64 并发请求把全部约 46,000 道训练题的 teacher completions（每题 M=2 条，temperature=0.7，top_p=0.95，max_new_tokens=512）写入 SQLite 缓存；
3. **确认 cache coverage：** warmup 结束时确认 `Final cache coverage: 46000 / 46000 (100.0%)`；
4. **（可选）导出为 HF Dataset：** 用 `export_teacher_cache_to_dataset.py` 导出，切换到 `teacher_backend=dataset` 彻底脱网。

cache key 由 `SHA256(prompt + model_name + n_samples + temperature + top_p + max_new_tokens)` 构成，warmup 和训练脚本中的生成参数必须完全一致，否则 cache 不命中。

**依据文件：** `docs/TEACHER_MODEL_PRODUCTION.md`（第 3 节）

### 7.4 Warmup / Cache 的意义

warmup 的核心价值是**把 teacher 调用的延迟从"训练时在途"变为"训练前离线"**。在 AoPS 场景中，每道训练题需要 teacher 生成 2 条完整解答，122B 模型生成 512 token 约需 20-60 秒。如果训练时实时调用，每个 training step 都要等待 teacher 响应，会导致 GPU 利用率极低。通过 warmup 预填充缓存，训练时每次 teacher 调用都是 cache hit，延迟降为近零，GPU 利用率不受影响。

### 7.5 Online teacher 和 Offline teacher 的区别

- **Online teacher（远程 HTTP 路径）：** `teacher_backend=remote`，每次需要 teacher 样本时向 HTTP API 发请求，如果 cache hit 则直接返回，否则实时生成。适合开发调试阶段或 cache 未完全预热时。
- **Offline teacher（dataset 路径）：** `teacher_backend=dataset`，teacher completions 已经全部导出为 HuggingFace Dataset，训练时直接从本地读取，完全不需要 teacher server 在线。适合生产训练阶段，延迟为零，可重复性最强。
- **本地 open-weight teacher（本地 actor 路径）：** `teacher_backend=local`，teacher 模型作为本地 actor group 加载，在训练环境内部生成 teacher completions。适合方法验证阶段，完整控制生成参数和可重复性，但占用大量本地 GPU。

**依据文件：** `docs/TEACHER_DISTILLATION_NOTES.md`（第 1-2 节），`docs/TEACHER_MODEL_PRODUCTION.md`（第 5 节）

### 7.6 训练时主要在比较什么

每个 training step 的核心计算链路：
1. Actor 对当前 batch 的 prompts 执行 strided-block rollout，生成 N 条候选答案；
2. Critic 对所有生成序列（和 GT 序列）提取 block-level embeddings；
3. Teacher provider 提供 M 条 teacher completions 的 embeddings（G2/G3）；
4. `_build_cf_target_embedding` 把 GT embeddings 和 teacher embeddings 混合，构建目标测度；
5. `compute_cf_l1oo_reward` 计算每个生成样本对整体分布匹配的边际贡献，作为该样本的 reward；
6. 用 RLOO/PPO 的 advantage 估计把 reward 转化为策略梯度，更新 actor 参数。

训练中监控的核心 metric 包括：`reward`、`effective_reward`、`diversity_reward`、`gt_reward`、`feature_map_reward`、`std_reward`、`actor_grad_norm` 等。

**依据文件：** `docs/STEP0_BASELINE_CHECKLIST.md`（"Metrics that must exist" 部分）

### 7.7 最终如何评测

评测采用 **actor-only 协议**：
- 只使用训练后的 actor 模型，不依赖 critic 参与打分；
- 在统一测试集 `LiveAoPSBench-2024.jsonl` 上运行；
- 统一 prompt / generation / scoring 协议；
- 评测答案时使用 `parse(f"\\boxed{{{answer}}}")` 路径（修复后），确保 LaTeX 格式答案可被正确解析。

"Actor-only evaluation" 的设计意义在于：如果 EBFT 真的提升了模型的生成能力，这种提升应该体现在 actor 模型的生成质量上，而不依赖于评测时 critic 的参与。这使评测结果更干净，也更接近实际部署场景。

**依据文件：** `EBFT SFT 进展汇报`（第 3.4 节），`data/case_study_results/N2_case_study_and_eval_audit_report.md`
---

## 八、当前实验发现与诊断

### 8.1 当前数字结果

根据 `EBFT SFT 进展汇报`，在 AoPS 测试集（5328 条）上，500 step 训练后的初步结果：

| 模型 | 正确数 | 准确率 |
|------|--------|--------|
| Base（Qwen3.5-2B）| 447 | 8.39% |
| G1（vanilla EBFT）| 448 | 8.41% |
| G2（cf_l1oo + teacher）| 427 | 8.02% |
| G3（cf_l1oo + adapter）| 430 | 8.07% |
| SFT（监督微调）| 240 | 4.50% |

**核心观察：** G1/G2/G3 准确率与 Base 相近（约 8%），无显著提升；SFT 大幅下降（4.5%）。

### 8.2 结果的可信度分析

上表数字在当前阶段的诊断价值受限，原因有三：

**原因一：评测管线存在系统性缺陷（主要原因）。** 审计显示至少 19-22% 的评测结果可能不可信。在修复评测管线之前，无法对上表数字做可靠的方法比较。

**原因二：任务难度与训练预算的错配。** 原 AoPS 论文使用数学专用模型训练约 18,000 steps；本项目使用通用 2B 基座模型训练 500 steps。500 steps 可能根本不足以让任何方法产生可测量的改善。

**原因三：Case study 揭示的主要失败模式是 truncation。** 30 个样本中 90% 是因为生成长度不够（512 token）而截断，并非因为模型"不会做"这道题。生成长度从 512 增加到 1024/2048 可能比方法差异更重要。

### 8.3 SFT 大幅下降的解释

SFT 从 8.39% 下降到 4.50%。进展汇报的解释是：SFT 使模型的生成风格被拉向了训练集的答案格式，但评测时的答案解析与 SFT 训练时的答案格式不一致，导致系统性格式不匹配。这更多反映的是"评测格式不对齐"，而不一定是"SFT 破坏了推理能力"。

### 8.4 当前主线任务（按优先级）

1. **修复 AoPS eval 路由 bug：** 确保训练器 eval 分支正确路由到 AoPS 数学评测路径，而不是 BLEU 路径；
2. **修复 AoPS 答案解析：** 统一使用 `parse(f"\\boxed{{{answer}}}")` 路径解析 GT 答案，消除 19-22% 的系统性解析误差；
3. **获得可信 baseline 数字：** 在修复后的评测管线上重新运行评测，获得可以进行方法比较的数字；
4. **诊断 G1/G2/G3 实际差异：** 确认修复后各组的准确率变化，决定是否延长训练或调整超参；
5. **（可能）增加生成长度：** 把 `max_new_tokens` 从 512 增加到 1024 或更多，以解决 truncation 导致的失败模式。

**依据文件：** `docs/STEPX_PROGRESS_CHECKPOINT.md`，`EBFT SFT 进展汇报`（第 5 节）
---

## 九、项目在方法谱系中的位置

### 9.1 与 RLHF / PPO 的关系

这个项目是 RLHF 范式的变体，而不是对 RLHF 的替代。它保留了 RLHF 的核心结构：on-policy rollout + advantage estimation + policy gradient update + KL penalty。它改变的只是 reward 的来源和计算方式——从"reward model 的 pointwise 打分"改为"feature space 中的 distributional discrepancy"。因此，这个项目的定位是"更好的 reward signal 设计"，而不是"新的训练算法"。

**依据文件：** `README.md`，`docs/STEP2E_DISTILLATION_FORMULATION.md`

### 9.2 与标准知识蒸馏（KD）的关系

传统 KD 让 student 学习 teacher 的 token-level 输出分布（soft labels）。本项目的 G2 teacher target 模式可以被理解为一种"feature-space 蒸馏"——student 不学习 teacher 的 token 分布，而是学习让自己的 hidden-state feature 分布接近 teacher 的 hidden-state feature 分布。`docs/STEP2E_DISTILLATION_FORMULATION.md` 明确写道："G2 teacher target 模式是一种显式蒸馏，只是蒸馏发生在 feature space 而非 token space。"

**依据文件：** `docs/STEP2E_DISTILLATION_FORMULATION.md`，`docs/teacher_replacing_gt_lit_review.md`

### 9.3 与 GAN 和 Energy-Based Model 的关系

"Energy-Based Feature-matching" 这个名字来自能量模型（EBM）的框架。EBFT 的 alignment reward 可以被理解为一种隐式的能量函数：critic 的特征空间中，与 GT 特征更接近的生成样本得到更高的奖励（更低的能量）。但与 GAN 不同的是，EBFT 的 critic 是冻结的（G1/G2），不参与对抗训练，因此没有 GAN 的训练不稳定性。G3 的 feature adapter 使 critic 部分可训练，通过 EMA 和 adapter-only 训练维持稳定性。

**依据文件：** `docs/ebft_upgrade_master_plan.md`（第 3 节），`docs/feature_network_unfreeze_lit_review.md`

### 9.4 与 GRPO / DAPO 等 RL 变体的关系

近期 LLM 后训练领域出现了以 RLVR（Reinforcement Learning with Verifiable Rewards）为核心的方法（DeepSeek 的 GRPO、字节跳动的 DAPO 等）。这些方法的 reward 通常是"最终答案是否正确"的 0/1 信号。EBFT 与这些方法的根本区别在于：reward 不依赖于"答案是否正确"，而是依赖于"生成样本的 feature 是否与目标分布对齐"。这使 EBFT 在没有可靠自动验证器的任务（如开放式生成、翻译）上也可以工作。

**依据文件：** `README.md`，`docs/ebft_upgrade_master_plan.md`（第 4 节）

### 9.5 与 MiniLLM / GKD 等 seq-level 蒸馏的关系

`docs/teacher_replacing_gt_lit_review.md` 综述了 MiniLLM（用 reverse KL + on-policy rollout 做 seq-level 蒸馏）、OPCD（online policy + teacher soft label + contrastive）等工作。这些工作表明：在数学推理等需要多样性的任务上，用教师生成的多条高质量轨迹作为目标比单条 GT 更好。本项目的 G2 teacher target 从不同角度得出了相同的方向：用教师模型的多条输出构建更丰富的目标分布，而不是只对齐单条 GT。

**依据文件：** `docs/teacher_replacing_gt_lit_review.md`

---

## 十、项目未解决的问题与开放挑战

以下问题在当前进度下尚未得到明确答案，是后续研究的开放方向：

1. **评测管线修复后，G1/G2/G3 是否有可测量的差异？** 当前 19-22% 的评测误差可能掩盖了方法之间的真实差异，修复后才能知道。

2. **500 step 是否足够？** 原论文在简单任务上用较短的训练预算就能看到效果；AoPS 场景是否需要更长的训练才能产生可测量的改善，尚不清楚。

3. **Qwen3.5-2B Base 是否是合适的基座模型？** 一个数学专用的或更大的基座模型是否更适合这个实验场景，是一个有价值的对照实验。

4. **CF discrepancy 的超参是否需要针对 AoPS 调参？** 原论文的超参在代码任务上确定，AoPS 任务的特征空间可能需要不同的频率分布。

5. **teacher 的 M 个样本数量是否足够？** 当前 M=2，目标分布的支撑非常有限。M=8 或 M=16 是否能显著改善目标分布的质量，需要实验验证。

6. **G3 的 EMA target geometry 是否能在 AoPS 场景下稳定？** feature adapter + EMA 的组合在图像自监督学习中有充分的理论和实验支撑，但在 language model 的 RL fine-tuning 场景下的行为尚无系统研究。

7. **distribution matching 的 scaling 行为如何？** 随着模型规模（2B -> 7B -> 70B）增大，feature-space distribution matching 的 reward signal 是否仍然有效，目标分布的质量是否也随之提升，是一个有价值的 scaling 实验。

---

## 十一、一句话总结各文件的价值

| 文件 | 一句话总结 |
|------|----------|
| `README.md` | 了解论文方法的入口，包含完整的 arXiv 引用和脚本说明 |
| `EBFT SFT 进展汇报` | 了解"项目在做什么、做到哪"最重要的单文件，包含 G1/G2/G3 的方法对比和当前实验结果 |
| `docs/ebft_upgrade_master_plan.md` | 项目方法论最权威的文件，定义了三个升级轴的统一数学框架和执行顺序逻辑 |
| `docs/ebft_ncfm_upgrade_notes.md` | 解释了为什么要从 pointwise alignment 升级到 CF-based distribution discrepancy |
| `data/case_study_results/N2_case_study_and_eval_audit_report.md` | 发现了评测管线的系统性缺陷，是当前实验结果可信度分析的关键文件 |
| `docs/STEPX_PROGRESS_CHECKPOINT.md` | 快速了解项目整体进度和当前主线任务的入口 |
| `docs/TEACHER_MODEL_PRODUCTION.md` | 了解 teacher server 完整部署方案，包括 vLLM 配置、warmup 流程、多 worker 部署 |
| `docs/G2/G2_PHASE_SUMMARY.md` | G2 阶段的完整总结，包含目标测度公式、数据流图、代码变更汇总 |
| `openrlhf/utils/embedding_utils.py` | CF reward 计算的核心实现文件，包含 `compute_cf_l1oo_reward` 和 `_build_cf_target_embedding` |
| `openrlhf/utils/teacher_provider.py` | teacher 服务的完整客户端实现，包含 HTTP 客户端、SQLite 缓存、多 worker 负载均衡 |
| `openrlhf/trainer/ppo_utils/ebft_experience_maker.py` | 训练核心逻辑所在，包含 teacher 采样、embedding 构建、CF reward 计算的完整链路 |
| `scripts/run_g2_8gpu_remote_teacher.sh` | G2 推荐的生产训练脚本，包含完整的超参配置和环境变量说明 |
| `docs/feature_network_unfreeze_lit_review.md` | G3 方法设计的文献依据，总结了自监督学习中 moving target 问题的解决经验 |
| `docs/teacher_replacing_gt_lit_review.md` | G2 teacher target 方法设计的文献依据，梳理了 seq-level 蒸馏领域的相关工作 |
| `inference_loss/README.md` | 独立评测模块的说明，了解如何在训练系统之外评测 reward signal 质量 |

---

*本文档由项目内文件内容综合撰写，最后更新基于 2026 年 3 月底的仓库状态。*
