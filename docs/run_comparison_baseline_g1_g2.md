# Baseline / G1 / G2 三组实验对比汇报

对应三个 run 目录：

- `outputs/baseline_0430_1443`
- `outputs/g1_rebase_0427_1553`
- `outputs/g2_online_teacher_8gpu_0501_2148_resume_step200`（前置：`outputs/g2_online_teacher_8gpu_0501_2148`）

评测脚本一致（`scripts/supplement_2rounds/G{1,2,baseline}_2rounds_vllm`），评测数据集一致（aops `test_qa.jsonl`，5328 题）。

---

## 1. 各自是什么

| run | 类型 | 起始模型 | 是否训练 | 训练步数 | 训练用卡 |
| --- | --- | --- | --- | --- | --- |
| baseline | 直接评测 base 模型 | `Gemma-4-E4B`（原始权重） | 否 | 0 | 仅评测 8 卡 |
| g1 | 离线 RL，无 teacher | `Gemma-4-E4B` | 是 | 0 → 500 | 4 卡 |
| g2 | 在线 teacher 蒸馏，断点续训 | g2 前段 `global_step200_hf` | 是 | 前段 200 + 本次 300 = 500 | 前段 8 卡 / 本次 1 卡 actor |

g2 的两段训练详情：

- 前段 run `g2_online_teacher_8gpu_0501_2148` 计划 500 步，**实际只训到 step 203 就停了**（进度 41%），ckpt 保存到 `global_step200_hf`。
- 本次 resume run `g2_online_teacher_8gpu_0501_2148_resume_step200` 把 `global_step200_hf` 当成新的 `pretrain` 加载（不是 deepspeed universal ckpt resume），训练 step counter 重新从 1 开始；本次跑完 300 步。
- 累计训练量 = 200 + 300 = **500 步**，与 g1 持平。

---

## 2. 训练相关的关键超参对比（仅对 g1 / g2）

| 参数 | g1 | g2（续训段） |
| --- | --- | --- |
| 起始模型 | `gemma-4-E4B` | g2 前段 `global_step200_hf` |
| 训练数据 | `aops_qa_hf_dict`，max_samples 16000 | `aops_qa_hf_dict`，max_samples 9600 |
| 累计步数 | 500 | 500（200 + 300） |
| ZeRO stage | 2 | 2 |
| Adam offload | 是 | 是 |
| Gradient checkpointing | 否 | 是 |
| Actor / Critic / Ref / Reward 卡数 | 4 / 4 / 4 / 4 | 1 / 1 / 1 / 1 colocate |
| Train batch | 128 | 128 |
| Micro train batch | 4 | 4 |
| Rollout batch | 32 | 32 |
| n_samples_per_prompt | 4 | 4 |
| prompt / context / generate max len | 384 / 8 / 8 | 384 / 8 / 8 |
| stride | 8 | 8 |
| Actor LR | 1e-6 | 1e-6 |
| Critic LR | 0 | 0 |
| Critic LR head | 0 | 0 |
| LR scheduler | cosine_with_min_lr，warmup 3% | 同上 |
| KL loss | k2 estimator，coef 0 | 同上 |
| ce_loss_coef | 0.01 | 0.01 |
| Reward 类型 | `pointwise`（无 teacher） | `cf_l1oo`（带 teacher） |
| cf_target_mode | `single`（仅 GT） | `teacher`（GT + teacher 混合） |
| cf_teacher_lambda | 0（不参与） | 0.6 |
| cf_teacher_n_samples | 4（不生效） | 4 |
| Teacher backend | local（实际未启用） | remote（6 个 vLLM 端口） |
| Teacher 模型 | 无 | qwen3.5-27b |
| Teacher API base | 无 | `127.0.0.1:8004 .. 8009` |
| Teacher 缓存 | 无 | `/mnt/workspace/teacher_cache_shared` |
| EMA | 关 | 关 |
| Feature adapter | 关 | 关 |
| Critic classifier loss coef | 1.0 | 1.0 |
| Critic direct discrepancy coef | 0 | 0 |
| Diversity / Alignment reward coef | 1.0 / 1.0 | 1.0 / 1.0 |
| Save 策略 | 每 25 步存一个 hf ckpt | 每 50 步存一个 hf ckpt |

注：因为 g1 / g2 的 `train_batch_size`、`micro_train_batch`、`n_samples_per_prompt`、`rollout_batch` 都一样，所以"每 step 看到的 prompt 数量"和"每 step 的梯度更新量"一致。500 步意味着相同的优化信号总量，训练量对比是公平的。卡数差异主要影响每步耗时和 rollout 数据多样性，不改变 effective step 数。

---

## 3. 评测设置（三个 run 完全一致）

| 项目 | 取值 |
| --- | --- |
| 评测数据集 | `aops/test_qa.jsonl` |
| 评测样本数 | 5328 |
| 评测后端 | vLLM 0.19.0，TP 8，A100 80G x 8 |
| 第一轮最大新 token 数 | 16384 |
| 第二轮最大新 token 数 | 32768 |
| Temperature / top_p / repetition_penalty | 0.6 / 1.0 / 1.0 |
| best_of_n | 1 |
| max_num_seqs / progress batch | 256 / 256 |
| Prompt 模板 | 不套 chat template，纯文本 |
| 评测流程 | Stage 1 全跑 5328 题；Stage 2 仅 retry Stage 1 判错的题 |

---

## 4. Stage 1 结果（16384 token 上限，5328 题全跑）

| 指标 | baseline | g1（v3） | g2 |
| --- | --- | --- | --- |
| Correct（题数 / 可解析样本） | 242 / 5317 | 251 / 5317 | 207 / 5317 |
| Stage 1 准确率 | 4.6% | 4.7% | 3.9% |
| 平均输出 token 数 | 3217 | 2046 | 3105 |
| 平均输出字符数 | 7027 | 4210 | 6157 |
| Token 中位数 / p90 / p99 / max | 286 / 16384 / 16402 / 26175 | 379 / 5191 / 16390 / 24900 | 403 / 16387 / 16910 / 24125 |
| 命中 16384 上限的样本 | 914 / 5328（17.2%） | 464 / 5328（8.7%） | 828 / 5328（15.5%） |
| 直接吐 EOS 不写答案的样本 | 816 / 5328（15.3%） | 853 / 5328（16.0%） | 1010 / 5328（19.0%） |

Stage 1 类别明细（占 5328 的比例，列含义见 `docs/analysis.md`；g1 用 v3 严判口径）：

| 类别 | baseline | g1（v3） | g2 |
| --- | --- | --- | --- |
| wrong_answer_fallback | 39.4% | 49.4% | 42.3% |
| calculation_error | 20.5% | 12.6% | 13.9% |
| reasoning_incomplete | 15.6% | 15.2% | 17.8% |
| pure_eos | 15.3% | 16.0% | 19.0% |
| correct（严判） | 2.5% | 2.4% | 2.1% |
| wrong_answer | 2.5% | 1.6% | 2.4% |
| no_reasoning | 2.0% | 0.2% | 0.6% |
| correct_raw_match | 1.0% | 0.8% | 0.7% |
| correct_fallback | 0.6% | 0.7% | 0.6% |
| correct_raw_match_fallback | 0.5% | 0.8% | 0.6% |
| missing_gold | 0.2% | 0.2% | 0.2% |
| empty_output | 0.0% | 0.0%（2） | 0.0%（2） |

注意 g1 / g2 之间在分类口径上的两个差异：

- g1 用的是 v3 严判（更严格的 fallback 接受标准），所以 wrong_answer_fallback 比 g2 更高（49.4% vs 42.3%）—— 不是 g1 模型更差，而是 v3 把 g2 在旧口径下被算成 fallback correct 的部分判回 wrong。
- baseline 也用旧口径，比较时把 g1 的 49.4% 看作"v3 严口径"，把 baseline 的 39.4% 与 g2 的 42.3% 看作"旧口径"。g1 v3 严判的 stage1 correct 反而比 g2 更高（251 vs 207），是趋势是可比的；只是细分桶的绝对比例不能直接 cross-version 比。

---

## 5. Stage 2 结果（32768 token 上限，仅 retry Stage 1 错题）

| 指标 | baseline | g1（v3） | g2 |
| --- | --- | --- | --- |
| Stage 2 retry 题数 | 5086 | 5158 | 5121 |
| Stage 2 单独 Correct 数 | 186 / 5075 | 214 / 5147 | 182 / 5110 |
| Stage 2 单独准确率 | 3.7% | 4.2% | 3.6% |
| 平均输出 token 数 | n/a | 3423 | n/a |
| 平均输出字符数 | n/a | 6900 | n/a |
| Token 中位数 / p90 / p99 / max | n/a | 373 / 4992 / 32774 / 34174 | n/a |
| 命中 32768 上限的样本 | 840 / 5086（16.5%） | 427 / 5158（8.3%） | 769 / 5121（15.0%） |
| 直接 EOS | 799 / 5086（15.7%） | 861 / 5158（16.7%） | 1046 / 5121（20.4%） |

Stage 2 类别明细（占各自 retry 题数的比例）：

| 类别 | baseline（占 5086） | g1 v3（占 5158） | g2（占 5121） |
| --- | --- | --- | --- |
| wrong_answer_fallback | 39.3% | 50.7% | 41.8% |
| calculation_error | 21.2% | 12.9% | 13.5% |
| reasoning_incomplete | 15.1% | 13.6% | 17.8% |
| pure_eos | 15.7% | 16.7% | 20.4% |
| correct（严判） | n/a 单列 | 2.1% | n/a 单列 |
| wrong_answer | 2.6% | 1.5% | 2.0% |
| no_reasoning | 2.2% | 0.2% | 0.6% |
| correct_raw_match | n/a 单列 | 0.7% | 0.6% |
| correct_fallback | 0.5% | 0.7% | 0.6% |
| correct_raw_match_fallback | 0.4% | 0.7% | 0.5% |
| missing_gold | 0.2% | 0.2% | 0.2% |
| empty_output | 0% | 0.0%（1） | n/a |

注意 stage 2 的"Stage 2 单独 Correct 数 / 准确率"：

- baseline / g2 用旧版分析脚本，stage2 自己的 log 直接给 186 / 5075 和 182 / 5110，对应"stage1 错的题里 stage2 又有多少道做对"。
- g1 用 v3 新版分析脚本，stage2 自己的 log 给 214 / 5147，比"net improved 201"多 13 —— 那 13 道是 v3 严判 stage1 wrong 但 raw stage1 correct 的样本，stage2 也判对了，所以在 stage2 自身视角是 correct，但合并到 final 时不算"新增"。

---

## 6. 最终合并结果（Stage 1 正确的留下，错的换成 Stage 2 的判定）

| 指标 | baseline | g1（v3） | g2 |
| --- | --- | --- | --- |
| Total predictions | 5328 | 5328 | 5328 |
| 可评估样本 | 5317 | 5317 | 5317 |
| Stage 1 答对 | 242（4.55%） | 251（4.72%） | 207（3.89%） |
| Stage 2 retry 中新增答对 | 186 | 201（其中 13 与 stage1 重叠，净增 133） | 182 |
| 最终 Correct | 428 | 384 | 389 |
| 最终 Accuracy | 8.05% | 7.22% | 7.32% |
| Oracle Union（Stage 1 ∪ Stage 2 任一对就算对） | 428（8.05%） | 452（8.5%） | 389（7.32%） |

说明：

- baseline 和 g2 用的是旧版分析脚本：Stage 2 只 retry Stage 1 判错的题，所以 stage1 ∩ stage2 = 0，oracle union 必然等于 final，且 oracle "both correct" = 0。
- g1 用的是 v3 新版分析脚本：会用更严格的判定标准重判 Stage 1，retry 的子集和 Stage 1 correct 集合在新口径下会有少量重叠（13 道在 v3 严判下 stage1 wrong 但 raw stage1 correct，stage2 也判对），所以 oracle union 略高于 final。
- 因此 g1 的 oracle 8.5% 与 baseline / g2 的 oracle 数字不完全可比，更严格的对比口径还是看 final accuracy。

---

## 7. 主要观察 / 结论

1. **g1 和 g2 的实际训练量都是 500 步**，对比是公平的。
2. **Stage 1 单轮看，baseline 和 g1 接近（4.6% vs 4.7%），g2 反而比 baseline 更低（3.9%）**。两组 RL 训练在 16k token 上限下都没显著超过 base 模型，g2 续训甚至轻微退化。
3. **g2 的"直接吐 EOS"比例最高（19.0% vs baseline 15.3%）**。在线 teacher 训练后，模型有更明显的"开口就放 EOS 不答题"倾向，这是 stage1 准确率降低的主要原因。
4. **g1 的输出长度明显更短**：平均 2046 tokens，对比 baseline 3217 / g2 3105。命中 16k 上限的比例也低很多（8.7% vs 17.2% vs 15.5%）。也就是 g1 学会了更早结束（但没退化到秒 EOS）。
5. **两轮合并后 final accuracy**：baseline 8.05%、g1 7.22%、g2 7.32%。**baseline 反而最高**。原因：
   - Stage 2 把上限扩到 32k 后 baseline 多救回 186 题；
   - g1 / g2 因为 EOS 早停问题，stage 2 多给 token 也救不回多少（g1 多救 201，但其中 13 是 v3 重判口径产生的重叠，净增 133；g2 多救 182）；
   - 加上 stage1 起点更低，所以 final 都低于 baseline。
6. **g1 的 oracle 8.5% 是三组里唯一比 baseline 8.05% 高的指标**，但这是"两轮里任一对就算对"的最宽松口径，而且只有 g1 用了能产生重叠的分析脚本，并不是直接可比。
7. **总体结论**：在当前 reward / 训练配方下，无 teacher 的 g1 和有 teacher 的 g2 相对 baseline 都没拿到稳定收益。**主要瓶颈不是缺训练量，而是 RL 后模型 EOS 早停过强**（pure_eos 比例都在 15% - 19%）。后续要么改 reward 形状抑制 EOS、要么继续训更长配合更强的 stage2，再来对比。
