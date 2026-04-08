# 实验汇报（EBFT Distributional Matching）

---

## 1. 背景与目标

这阶段工作的核心目标，是在 AoPS 数学问答场景下，探索一条从 paper-faithful EBFT baseline 出发、逐步引入更强分布建模能力的训练路线，并观察其是否能够转化为最终的数学答题能力提升。

这条线的出发点有两个：

1. 原始 EBFT / distributional reward 的思路，本质上不是只看单点答案是否正确，而是希望从“分布匹配”的角度去建模生成行为。
2. 我们进一步考虑到，在数学推理任务里，模型的输出并不总是一个单点对象，而更像是一簇可能的生成路径、局部 token 轨迹和中间表征分布。因此，单纯的 pointwise reward 可能不足以表达我们真正希望模型靠近的目标分布。

在这个意义上，我们的方法设计受到 **distribution matching / distribution-level target** 思想的启发，而这部分更直接的灵感来源，是师兄关于 **NCFM** 的工作。我们不是简单把 reward 看成一个标量打分问题，而是尝试把训练目标理解为：**让模型生成分布在某种特征空间或目标测度意义下，更接近我们想要的分布。**

构建了 G1 / G2 / G3 三个逐步递进的版本。

**G1 在固定 feature geometry 中做 pointwise 对齐，G2 在同一固定 geometry 中把训练目标升级为 teacher-augmented 的条件分布匹配，G3 再在此基础上用 adapter + EMA 让 feature geometry 本身开始受控学习。**

---

## 2. 方法设计：G1 / G2 / G3

### 2.1 G1：baseline

G1 是一个严格的 control，也是必须存在的 sanity check。当前 `run_G1_rebase.sh` 对 G1 的定义非常克制：使用 pointwise reward、冻结 critic、不开 teacher，仅让 actor 在一个固定的 feature geometry 中接受最保守的 EBFT 信号。

G1 把问题拆开。它隔离了 teacher target、distributional discrepancy、geometry learning 这些

### 2.2 G2：distributional reward

因此，G2 把训练信号**从“单样本对单目标点的相似度”升级为“生成样本集与目标测度之间的 discrepancy”**。在当前 rebase 脚本里，这个 discrepancy 由 `cf_l1oo` 实现，其直觉并不复杂：先用 characteristic-function 风格的 witness 去比较两组经验分布，再通过 leave-one-out 的方式把 group-level discrepancy 还原为 sample-level reward。

G2 让 reward 从 pointwise alignment 变成 set-level matching，当前 rebase 进一步把 target measure 定义为 GT 与 teacher completion 的混合经验分布：`ν_c = (1 - λ) δ(φ(y_gt)) + λ (1/M) Σ_i δ(φ(y_i^T))`。这里 `λ=0.6` 的作用是让目标既不完全被 teacher 风格牵着走，也不退回到单点 GT；`M=4` 则让目标分布至少拥有一个最小但非退化的样本支撑。

### 2.3 G3：feature geometry learning

G3 : **critic backbone 继续冻结，只在 feature stream 上增加一个小型 residual bottleneck adapter，并只训练 adapter 与小规模 head。**

EMA target geometry ：online 分支更新得更快，EMA 分支变化更慢，reward 与 direct discrepancy 都参考这条慢速分支，从而抑制 actor 和 critic 同步漂移导致的 reward ranking 崩塌。换句话说，EMA 在这里不是附加技巧，而是为了维持“可学习几何”与“可置信奖励”之间的平衡。

需要额外指出的是，当前 `run_G3_rebase.sh` 除了方法上引入 adapter + EMA 之外，还把训练侧 `prompt_max_len` 从 256 提高到 512，并把 post-eval 预算扩大到 `384 prompt / 768 new tokens`。

### 2.4 G1 / G2 / G3 对比总结

**G1 在固定 feature geometry 中做 pointwise 对齐，G2 在同一固定 geometry 中把训练目标升级为 teacher-augmented 的条件分布匹配，G3 再在此基础上用 adapter + EMA 让 feature geometry 本身开始受控学习。**

---

## 3. 实验设置

### 3.1 实验目标

本轮实验的 evaluation target 是非常明确的：只看 actor-only 的最终答题准确率，而不让 critic 在推理时参与任何 reranking 或额外打分。原因也很直接。如果 distributional matching 真能改进模型的解题能力，那么这种改进必须最终体现在 actor 自己生成答案的能力上，而不是依赖训练期 critic 在测试时继续充当外部裁判。换句话说，我们要评估的是“训练目标是否真正转化成了 actor 的可部署能力”。

### 3.2 数据与模型

训练数据来自 AoPS 数学问答数据，脚本路径为 `/mnt/data/ebft-teacher-distribution/data/aops/aops_qa_hf_dict`；评测数据为 `/mnt/data/ebft-teacher-distribution/data/aops/test_qa.jsonl`，共 5328 道题。在当前 rebase 脚本中，为了把预算固定在 500 step，训练侧实际使用 `max_samples=16000` 的截断子集，而不是把整套 AoPS 训练语料完全跑满。因此，这一轮更应被视为方法 feasibility run，而不是大预算收敛实验。

需要特别说明的是，当前 rebase artifacts 所用的 student base model 是 `Qwen3.5-0.8B Base`，并非早期内部文档中反复讨论的 `Qwen3.5-2B Base`。这一点对结果解释非常关键，因为它意味着本轮实验的 capacity mismatch 实际上比“2B base 对 AoPS”还要更强。无论是 2B 还是 0.8B，这类通用 base 模型都不是 math-specialized，也不是 instruct-tuned；但对当前 rebase run 来说，真实脚本口径甚至比旧文档里的叙事更保守。G2/G3 所使用的 teacher 则是远程 `Qwen3.5-27B`，其角色只是为目标测度提供更丰富的样本支撑，而不进入最终评测。

### 3.3 训练预算

在当前 rebase artifacts 中，G1 没有 teacher，只使用 student 侧 4 张 A100（2 actor + 2 critic），500 step 的训练本体约 9.1 小时。G2 与 G3 都需要同时维持 student 与 remote teacher：student 侧仍是 4 张 A100，而 teacher 服务额外占用 4 张 A100，总占用达到 8 张 A100。就训练本体而言，G2 在 500 step 处约耗时 17.8 小时，G3 约耗时 21.8 小时；此外，G2/G3 的全量 actor-only post-eval 还要再花近 3 小时做 5328 题 generation。也就是说，对于 teacher-augmented distribution matching，真正昂贵的不是单步 PPO 更新，而是“目标分布采样 + 最终评测”这一整条链路。

### 3.4 评测协议

评测统一采用 actor-only final-answer accuracy protocol：先对测试集做 generation，再用 `scripts/analyze_eval_results.py` 结合 `math_verify` 进行答案抽取与校验。修复后的分析脚本会优先通过 `parse(f"\\boxed{{gold}}")` 路径处理金标答案，以避免 AoPS LaTeX answer 的系统性解析错误；这一步很关键，因为项目内部审计曾指出，旧评测在 answer parsing 与 eval routing 上都存在明显偏差，至少会引入约 19% 到 22% 的不可信度。

但即便进入修复后的 actor-only 分析，也仍要注意“可复核”与“可比较”不是同一个概念。当前 outputs 中，G1 的 post-eval 预算是 `256 prompt / 512 new tokens`，而 G2 与 G3 都是 `384 prompt / 768 new tokens`。因此，G2 与 G3 之间的比较相对更直接，而 G1 只能作为趋势性的低预算参考；要得到真正闭合的 ablation ladder，仍需要在统一评测预算下补齐 Base、SFT 以及统一预算版的 G1 重评。

---

## 4. 实验结果与解释

### 4.1 结果表

先给出当前能拿到的两类结果：一类是**修复前的全量历史对比**，它的价值在于五路模型都齐；另一类是**修复后可复核的分析**，它的价值在于协议更可信

**表 1：修复前的历史全量对比（用于记录现象，不用于做最终结论）**

| 模型 | 总样本 | 正确数 | 准确率 |
| --- | --- | --- | --- |
| Base | 5328 | 447 | 8.39% |
| G1 | 5328 | 448 | 8.41% |
| G2 | 5328 | 427 | 8.02% |
| G3 | 5328 | 430 | 8.07% |
| SFT | 5328 | 240 | 4.50% |

**表 2：修复后当前可复核的 rebase 结果（更可信，但尚非全量闭合对比）**

| 版本 | 评测样本 | 正确数 | 准确率 | 备注 |
| --- | --- | --- | --- | --- |
| G1 rebase | 5317 | 230 | 4.33% | `reasoning_incomplete` 占 65.6%，post-eval 为 `256 prompt / 512 new tokens` |
| G2 rebase | 5317 | 330 | 6.21% | `reasoning_incomplete` 占 51.1%，`wrong_answer` 占 27.7%，post-eval 为 `384 prompt / 768 new tokens` |
| G3 rebase | 5317 | 415 | 7.81% | `wrong_answer` 占 46.6%，`reasoning_incomplete` 降到 36.0%，post-eval 为 `384 prompt / 768 new tokens` |

### 4.2 现象总结

如果先看修复前的五路完整对比，最直观的现象是：G1 基本没有超过 Base，G2/G3 也没有打出清晰增益，SFT 甚至出现了显著下降。换句话说，在那一版完整 ladder 里，所有 fine-tuned 版本都没有形成一个令人信服的超越 Base 的信号。这至少说明一件事：在 AoPS 上，方法升级并不会像在较简单任务上一样自动转化成 final-answer 提升。

但如果再看修复后可复核的 rebase 梯度，现象会变得更有结构。G1 的 actor-only 准确率是 4.33%，且 65.6% 的失败属于 `reasoning_incomplete`；G2 提升到 6.21%，`reasoning_incomplete` 降到 51.1%，同时 `wrong_answer` 上升到 27.7%；G3 进一步到 7.81%，`reasoning_incomplete` 继续降到 36.0%，而 `wrong_answer` 上升到 46.6%。也就是说，修复后的 rebase ladder 呈现出一个相当清楚的趋势：`G1 -> G2 -> G3` 并不是毫无差异，而是在逐步把失败模式从“推理做不完”推向“推理更完整但最后仍会做错”。

因此，本轮现象不能再被简单概括成“distributional 方法没效果”，但也仍然不能被简化成“G3 有提升所以方法已经成立”。更准确的总结应该是：在修复前的完整表里，没有任何版本表现出可靠优势；而在修复后的 rebase 证据里，我们第一次看到了一个方向上单调的梯度，但这个梯度仍叠加了评测预算差异和未补齐的 Base/SFT 重评，所以它更像是“方法信号开始浮现”，而不是已经足够宣告胜负的最终证据。

### 4.3 第一性原理解释（核心）

第一类原因是**任务难度问题**。AoPS 与 coding、translation 的根本差异，在于它不是“局部 token 正确就会逐渐逼近答案”的任务，而是一个长程、组合式、强符号约束的推理问题。对于这类任务，reward 在 feature space 中做得再精细，也未必能沿着一条短路径映射到 exact final answer。distributional reward 也许更适合改善“候选轨迹的覆盖”和“推理展开的完整性”，但 AoPS 的最终 metric 却只奖励最后一跳是否精确命中。也就是说，任务本身就在放大 reward 与 metric 之间的距离。

第二类原因是**模型能力问题**。早期项目讨论常把问题表述成“2B base 对比 math-specialized model”，但当前 rebase 脚本实际上使用的是更小的 `Qwen3.5-0.8B Base`。这意味着即便按“2B 也偏弱”的口径来看，本轮真实实验的基座能力还要再向下一级。与此同时，AoPS 论文常见的工作区间却是 `DeepSeek-Math-7B-Instruct`、`Mathstral-7B`、`Llama-3.2 Instruct` 这类更强的数学或指令模型，甚至还会用更大的 teacher 做答案重写，并配合更长的训练预算。于是，本轮实验所面对的其实不是“一个已经有较强数学归纳能力的模型，再比较不同后训练目标”，而更像是“让一个较弱的通用 base 先跨过 AoPS 的能力门槛，再期待目标函数差异显现”。如果底座尚未进入 AoPS 的有效工作区间，那么方法差异很可能被 capacity ceiling 直接淹没。

第三类原因是**目标函数问题**。G2/G3 优化的是 feature-space distribution matching，而最终评测看的是 actor-only final-answer accuracy；这两者之间隔着一条很长的因果链：`reward discrepancy -> policy update -> 轨迹展开质量 -> 最终答案抽取 -> 精确答案验证`。只要这条链上的任意一环没有被打通，distributional reward 的改进就可能停留在中间层，而无法体现在最终 accuracy 上。修复后的 `G1 -> G2 -> G3` 梯度恰好说明了这一点：correct 从 230 提升到 330，再到 415，但更显著的变化其实是 failure mode 的迁移，即 `reasoning_incomplete` 持续下降，而 `wrong_answer` 持续上升。这表明 distributional reward 和 geometry learning 很可能先改善了“推理是否能展开、是否能形成更完整轨迹”，但还没有完全打通“更完整的轨迹如何稳定转化为正确终答案”的最后一段路径。这也是为什么 feature geometry learning 即便在概念上成立，也仍需证明它改善的是与数学 correctness 相关的几何，而不是仅仅让风格、篇幅或局部结构更像 target。

因此，**当前结果无法直接说明方法无效，而更可能是任务难度、基座能力与目标函数之间存在明显的 mismatch**：AoPS 需要的是长程符号推理，小型通用 base 还没有进入这个任务的有效能力区间，而 distributional reward 当前优化的又是 feature-space discrepancy，而不是 final-answer correctness 本身。在这种三重错配没有被解除之前，负结果更应该被理解为“实验栈尚未把方法信号与任务/模型噪声分离开”，而不是“distributional matching 方向已经被证伪”。

此外，还必须把评测因素单独记住。修复前的五路结果之所以只能作为现象记录，正是因为旧评测协议在 AoPS 路由和答案解析上存在系统噪声；修复后的局部结果虽然更可信，却又叠加了不一致的 eval budget。也就是说，我们目前面临的并不是一个简单的“方法输赢”问题，而是一个 measurement、capacity 与 objective 三者相互纠缠的问题。

---

## 5. 结论

这轮实验最稳妥的结论，不是“distributional matching 没效果”，而是：**目前证据开始显示方法信号，但仍不足以完成最终验证**。修复前的完整五路对比没有出现清晰赢家；修复后的 rebase 结果则形成了 `G1 4.33 -> G2 6.21 -> G3 7.81` 的单调梯度，说明方法升级并非完全没有方向性作用。但由于 Base/SFT 仍未在同一修复协议下补齐，且 G1 与 G2/G3 的 eval budget 不一致，这组结果还不足以支持“方法已被验证有效”的强结论。

换句话说，本轮实验的真正产出，不是一个已经定型的方法胜负，而是一组更清晰的研究判断：AoPS 上的负结果很可能首先暴露的是任务难度、模型能力和评测协议的错配，而不是单纯地否定 distributional reward 这一路线。就研究推进而言，这仍然是有价值的，因为它告诉我们下一步该优先解决什么，而不是继续在一个尚未闭合的实验面上堆更多技巧。

---

## 6. 后续计划（必须写，且要具体）

1. **先把比较面闭合，再谈方法优劣。** 需要在统一的 post-eval 预算下，补齐 Base、SFT 以及统一预算版的 G1 actor-only 重评，尤其要让 `prompt_max_len`、`max_new_tokens` 与答案解析路径完全一致；否则当前 `G1 4.33 -> G2 6.21 -> G3 7.81` 这条梯度仍然只能被视为方向性证据，而不是最终结论。
2. **先做 model scaling，再看方法 scaling。** 下一轮优先把 student 从当前 base 升到更有希望进入 AoPS 工作区间的模型，例如 `7B` 级别或至少 instruct/math-specialized 的基座；如果连底座都不具备稳定完成 AoPS 推理的能力，那么 reward 设计的差异大概率仍然会被 capacity ceiling 吞掉。
3. **显著增加训练 steps。** 当前 500 step 更像 feasibility run，而不是足以让方法差异充分显现的预算。后续至少应把训练步数拉到数千步量级，并向 AoPS 常见的更长预算靠拢；同时保留 `max_samples` 与 wall-clock 的精确记录，区分“方法无效”和“预算尚未进入可观测区间”。
4. **系统扫 CF 参数，而不是沿用默认值。** 目前 `cf_num_freqs=128`、`cf_sigma=1.0` 更像一个保守起点，而不是针对 AoPS 几何调过的配置。后续应围绕 `freq / sigma` 做成体系的 sweep，例如比较更稀疏与更密的频率分辨率、以及更窄与更宽的 witness 尺度，看 reward 是否对数学推理的 feature variation 足够敏感。
5. **调大 `n_samples_per_prompt` 与 teacher target 的样本密度。** 当前 student 侧 `n_samples_per_prompt=4`、teacher 侧 `cf_teacher_n_samples=4` 只是一个小样本经验分布，leave-one-out reward 的统计稳定性仍然有限。后续应测试 `4 -> 8` 甚至更高的 rollout/sample 配置，并观察 variance、reward ranking 与最终 accuracy 之间的关系。
6. **引入更强、更贴近数学任务的 teacher。** 当前 G2/G3 使用的是 `Qwen3.5-27B` remote teacher，它已经足够强于 student，但不一定是最适合 AoPS 的 target source。后续可以比较更强 teacher、math-specialized teacher，甚至更高质量的多样本 target，看 target measure 的提升是否比 reward 公式本身更关键。
7. **显式检查 `reward -> accuracy` 的路径是否真的打通。** 这一步不能再只看最终 accuracy，而应增加中间诊断：例如 reward 与 answer correctness 的相关性、`reasoning_incomplete -> wrong_answer -> correct` 的类别迁移、不同 step 上 reward ranking 的稳定性，以及 G3 中 feature geometry drift 与 final-answer gain 的关系。如果这条路径不被显式验证，我们就无法判断 distributional reward 到底是在优化“正确性”，还是只是在优化“更像一段完整推理”。
8. **继续审计 teacher path 的稳定性，而不是只看最终分数。** 虽然当前 G2 已经完成并可纳入分析，但训练日志里仍能看到 remote teacher fallback 与 cache/请求稳定性问题。下一轮需要把 teacher 请求失败率、fallback 比例、cache 命中与最终 accuracy 一起纳入诊断，否则我们很难区分“G2 本身的目标函数上限”和“teacher 管线噪声对结果的污染”。 