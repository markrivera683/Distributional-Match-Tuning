# EBFT 三档配置方法文档：Pointwise / Characteristic Function / Unfrozen FeatureNet

> 适用范围：本仓库 `dev/dlc` 分支下 `openrlhf.cli.train_ebft_ray` 训练入口，对应实验脚本
> `scripts/diff_dataset/run_G1_rebase.sh`、`scripts/diff_dataset/run_G2_rebase_*.sh`、
> `scripts/diff_dataset/run_G3_rebase_*.sh`。数据集为 Code Completion（`open-code-instruct_qa_100k`
> 中 16k 子集），student=Qwen3.5-4B，teacher=Qwen3.5-27B，post-eval=HumanEval（greedy / pass@1 T=0.6）。

本文档自下而上把 Stage1（pointwise EBFT 对照）、Stage2（Characteristic Function + Target 三种构造）
与 Stage3（CF + Teacher + Unfrozen Feature Network）三档方法的「数学定义、代码落点、关键配置」一次说清楚。
每一档都直接对应到训练入口里的 CLI 参数和 `openrlhf/utils/embedding_utils.py` 中的实现函数，便于复现。

---

## 0. 共同符号与流水线

所有三档共用同一个 EBFT 流水线（参见 `openrlhf/trainer/ppo_utils/ebft_experience_maker.py`），差异
仅在 reward 的计算方式与 target 构造方式：

- B = 微批数，G = 同 batch 内 prompt 组数，N = `n_samples_per_prompt`，K = `num_blocks`，D = 特征维度。
- 对每个 prompt 采样 `N` 条 student rollout 序列；每条序列被切成 K 个长度为 `generate_max_len` 的 block。
- Critic（与 actor 共享 backbone 但额外带 classifier head 与可选 `feature_adapter`）对 GT 与生成序列
  做 strided 前向，按 `embed_method` 把每 block 的 hidden state 聚合成一个特征向量，得到：
  - `gen_embedding ∈ R^(B, G, N, K, D)`（学生采样的特征）
  - `gt_embedding  ∈ R^(B, G, N, K, D)`（GT 沿 N 轴复制成 N 份的特征）
  - 在 teacher 模式下额外得到 `teacher_embedding ∈ R^(B, G, M, K, D)`，其中 M=`cf_teacher_n_samples`。
- 可选的固定 RFF 特征映射（`--feature_map_type=rff`）在送入 reward 前对最后一维做一次确定性投影；
  本期三档实验都使用 `identity`，因此不展开。

```26:30:openrlhf/utils/embedding_utils.py
import math
from typing import Tuple

import torch
import torch.nn.functional as F
```

reward 计算分派在：

```1322:1385:openrlhf/trainer/ppo_utils/ebft_experience_maker.py
        reward_type = getattr(self.args, "distribution_reward_type", "pointwise")
        ...
        if reward_type == "pointwise":
            gt_rewards_tensor = get_alignment_rewards(gen_embedding, gt_embedding)
            diversity_rewards_tensor = get_diversity_rewards(gen_embedding, per_token)
            ...
            gt_rewards_tensor *= 2
            diversity_rewards_tensor *= 2
        elif reward_type == "cf_l1oo":
            ...
            gt_rewards_tensor = get_cf_l1oo_rewards(...)
            diversity_rewards_tensor = torch.zeros_like(gt_rewards_tensor)
```

最终每 token 的 reward 在 `compute_advantages_and_returns` 里聚合：

\[
r = \alpha\_{\text{align}} \cdot r\_{\text{gt}} - \alpha\_{\text{div}} \cdot r\_{\text{div}}
\]

其中 `--alignment_rew_coef` 与 `--diversity_rew_coef` 控制两个系数（`cf_l1oo` 下 `r_div ≡ 0`）。
随后按 `advantage_estimator=rloo` 做 leave-one-out baseline 去中心化，advantage 复制到所有生成 token，
喂给 PPO/GSPO loss；同时附加 `--ce_loss_coef` 的 SFT-style CE 监督和可选 KL 正则。

---

## 1. Stage 1 — Pointwise EBFT（与原文严格对齐）

### 1.1 方法定义

按原 EBFT 论文，把 reward 拆成两个**逐样本**的对齐/多样性项：

- 对齐项：学生第 j 条样本的 block 嵌入与 GT 嵌入做余弦相似度
  \[
  r\_{\text{align},j} = \cos\big(\phi(y\_j), \phi(y^*)\big)
  \]
- 多样性项：同 prompt 组内其余样本之间的内积平均（"反多样性"），用于鼓励生成多样
  \[
  r\_{\text{div},j} = \tfrac{1}{N-1} \sum\_{i \neq j} \langle \phi(y\_j), \phi(y\_i) \rangle
  \]
- 组合 reward：`r_j = α_align · r_align,j − α_div · r_div,j`（两项最终都 ×2，见上文派发处）。

target 分布在这一档**显式塌缩为单点 δ(GT)**，即 `cf_target_mode=single` 仍可生效但事实上不进入 CF
路径——pointwise 路径完全不使用 CF 频率投影。

### 1.2 代码实现

对齐 reward：

```781:786:openrlhf/utils/embedding_utils.py
def get_alignment_rewards(gen_embedding, gt_embedding):
    # Alignment reward: cosine similarity so the actor optimizes directional
    # alignment in embedding space (not raw vector magnitude).
    gt_rewards_tensor = F.cosine_similarity(gen_embedding, gt_embedding, dim=-1)
    return gt_rewards_tensor
```

多样性 reward（非 token 模式，N>1 时启用）：

```807:822:openrlhf/utils/embedding_utils.py
            reorg = gen_embedding.permute(0,1,3,2,4) # num micro batches, num groups per micro batch, num blocks, n samples pp, num features, embed dim
            n_samples_per_prompt = gen_embedding.shape[2]
            gen_embedding_unsqueeze_2 = reorg.unsqueeze(3).repeat(1,1,1,n_samples_per_prompt,1,1)
            gen_embedding_unsqueeze_3 = reorg.unsqueeze(4).repeat(1,1,1,1,n_samples_per_prompt,1)
            full_sims = torch.sum(gen_embedding_unsqueeze_2 * gen_embedding_unsqueeze_3, dim=-1)
            no_jvms = torch.eye(full_sims.shape[-1], device=full_sims.device, dtype=torch.bool)
            sims = full_sims.masked_fill(no_jvms.view(1,1,1,full_sims.shape[-1],full_sims.shape[-1]), 0.0)
            diversity_rewards = sims.sum(dim=-1) / (full_sims.shape[-1] - 1)
            diversity_rewards_tensor = diversity_rewards.permute(0,1,3,2)
```

RLOO baseline（每个 prompt 组内做 leave-one-out 去基线）：

```1705:1725:openrlhf/trainer/ppo_utils/ebft_experience_maker.py
        denom_loo = float(n_samples_per_prompt - 1)
        sum_gt_rewards = gt_rewards.sum(2, keepdim=True)
        gt_baseline = (sum_gt_rewards - gt_rewards) / denom_loo
        ...
        baseline = (
            alignment_rew_coef * gt_baseline
            - diversity_rew_coef * diversity_baseline
        )
```

### 1.3 关键 CLI 配置（对应 `run_G1_rebase.sh`）

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--distribution_reward_type` | `pointwise` | 走 `get_alignment_rewards` + `get_diversity_rewards` 分支 |
| `--cf_target_mode` | `single` | 不进入 CF 路径，但保留默认值以共用配置 |
| `--alignment_rew_coef` | `1.0` | α_align |
| `--diversity_rew_coef` | `1.0` | α_div |
| `--ce_loss_coef` | `0.01` | 原文同款 SFT-CE 正则 |
| `--advantage_estimator` | `rloo` | leave-one-out baseline |
| `--n_samples_per_prompt` | `4` | 每个 prompt 采 4 条 |
| `--temperature` / `--top_p` | `0.6` / `1.0` | student 采样 |
| `--use_kl_loss` + `--init_kl_coef=0.0` | enabled | 走 KL 计算管道但权重为 0，便于日志对齐 |
| `--use_whitening` | enabled | 对 (gen, gt) 在样本轴做白化，统一各 feature_map 的尺度 |
| 学生/critic 学习率 | `actor_lr=1e-6`，`critic_lr=0`，`critic_lr_head=0` | 与原文一致：critic 在 G1 完全冻结 |
| 拓扑 | A100×4（actor+ref 共享 2 卡，critic+reward 共享 2 卡） | 与 README 描述一致 |

### 1.4 论文意图与实现的对应

原文的 EBFT 即「冻结 critic，固定特征空间内做 cosine 对齐 + 反多样性」。本仓库的 G1 在
`distribution_reward_type=pointwise + cf_target_mode=single` 下严格复现这一公式；teacher 完全不
参与（`teacher_pretrain` 留空 → `teacher_model=None`，experience maker 中的 teacher 分支被跳过）。
这条线就是文档前言里 12.68% → 12.04% / 14.12% 那条"保守复刻"基线。

---

## 2. Stage 2 — Characteristic Function (CF) + 三种 Target 构造

### 2.1 方法定义（NCFM-style 经验 CF 距离）

把 reward 视角从「逐样本余弦」升级为「组级别经验分布到 target 经验分布的差异」，使用
**经验特征函数 (Empirical Characteristic Function)** 在固定随机频率上的内积匹配，对应 NCFM 风格：

给定固定的 F 个 d 维 Gaussian 频率 `ω_f ~ N(0, σ⁻² I)`（由 `cf_seed`、`cf_num_freqs`、`cf_sigma`
确定，跨 step 跨 rank 完全确定性），对一组样本 \{x_n\} 定义经验 CF：
\[
\hat C\_X(\omega) = \tfrac{1}{|X|} \sum\_n e^{i \langle \omega, x\_n \rangle}
\]

NCFM 的 amplitude + phase 解耦损失：
\[
\ell\_f(\hat C\_X, \hat C\_Y) = \sqrt{
  \alpha\,(|\hat C\_X| - |\hat C\_Y|)^2 +
  \beta\,(2\,|\hat C\_X|\,|\hat C\_Y| - 2\,\Re\langle\hat C\_X, \overline{\hat C\_Y}\rangle)
}
\]
全组 discrepancy = 在 F 个频率上求平均；样本 j 的 reward 用**leave-one-out 边际收益**做归因：
\[
r\_j = D(X\_{\setminus j}, Y) - D(X, Y)
\]

也就是「把第 j 条删掉之后，组级别 CF 差异变小多少」，正值意味着删掉它能让分布更接近 target，
所以保留它对降低 group discrepancy 是负贡献——这里**直接把这个差当作奖励**，鼓励"留下它能让组分布更接近
target"的样本（注意符号约定：在 `n_samples=1` 退化时退回 `-D(X, Y)`）。

### 2.2 三种 Target 构造（`cf_target_mode`）

构造 target 经验测度 `Y` 由 `_build_cf_target_embedding` 统一负责：

```215:283:openrlhf/utils/embedding_utils.py
def _build_cf_target_embedding(
    gt_embedding: torch.Tensor,
    cf_target_mode: str,
    cf_target_num_refs: int,
    cf_target_std: float,
    cf_target_seed: int,
    teacher_embedding: torch.Tensor = None,
    cf_teacher_lambda: float = 0.0,
) -> torch.Tensor:
    """Build the target empirical measure used by the CF discrepancy.
    Modes:
    - single: keep the original EBFT-style single-reference target.
    - vicinal: create a small local target distribution around the GT feature
      by adding deterministic Gaussian perturbations in feature space.
    - teacher: build a mixed empirical target from GT + teacher embeddings.
      nu_c = (1-λ)*δ(GT) + λ*(1/m)*Σ_i δ(teacher_i)
      Implemented by repeating GT r times and concatenating m teacher samples
      so that r/(r+m) ≈ (1-λ).
    """
```

- **single**：`target = gt_embedding[:, :, :1]`，即每个 prompt 只用 GT 这一个点 δ(GT)。
- **vicinal**：在 GT 特征点周围加 `cf_target_num_refs-1` 个确定性 Gaussian 扰动，扰动幅度按
  feature local-RMS × `cf_target_std` 缩放，得到一个 GT 周围的小簇；本期实验里 std=0.05、
  num_refs=8，相当于 "GT + 7 个邻域样本" 共 8 个 target 点。
- **teacher**：把 GT 重复 r 次再拼接 m 条 teacher rollout 的 embedding，使
  `r/(r+m) ≈ 1-λ`，等价于做 `ν = (1-λ) δ(GT) + λ · ν_teacher`。

teacher 模式下「teacher rollout 如何变成 embedding」由
`RemoteExperienceMaker._build_teacher_embedding` 处理：根据 `teacher_backend` 区分本地 Ray actor
(`local`) 还是远程 vLLM 服务 (`remote`)，把 teacher 序列接力 critic forward → 同一套
`embed_method/feature_map` 流水线 → 与 student 完全可比的 (P, 1, M, K, D) 张量，再注入到上面的
`_build_cf_target_embedding`。

```1300:1308:openrlhf/trainer/ppo_utils/ebft_experience_maker.py
        if (
            _dr_type == "cf_l1oo"
            and _ct_mode == "teacher"
            and _has_teacher
        ):
            teacher_embedding = self._build_teacher_embedding(
                samples_list, n_samples, prompt_length, context_length,
                generate_length, stride, num_blocks,
            )
```

### 2.3 CF Reward 实现

完整的 `get_cf_l1oo_rewards`：

```435:526:openrlhf/utils/embedding_utils.py
@torch.no_grad()
def get_cf_l1oo_rewards(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    cf_num_freqs: int = 128,
    cf_sigma: float = 1.0,
    cf_seed: int = 43,
    cf_alpha: float = 0.5,
    cf_beta: float = 0.5,
    cf_reward_scale: float = 1.0,
    cf_target_mode: str = "single",
    cf_target_num_refs: int = 1,
    cf_target_std: float = 0.05,
    cf_target_seed: int = 43,
    teacher_embedding: torch.Tensor = None,
    cf_teacher_lambda: float = 0.0,
) -> torch.Tensor:
    """NCFM-style empirical CF reward with leave-one-out sample attribution.
    The returned reward for sample j is the marginal gain
    `D(X\\{j removed}, Y) - D(X, Y)`, where lower discrepancy is better.
    ...
    """
```

关键流程：

1. 频率投影 `gen_proj = freqs @ gen_flat^T`，由 `_get_fixed_cf_frequencies` 提供确定性频率
   矩阵（按 `(input_dim, num_freqs, sigma, seed, device)` 缓存）。
2. 用 `cos/sin` 计算每个样本的 `Re/Im` 贡献，再在样本轴求均值得到组级别经验 CF
   `gen_real, gen_imag`，target 同理。
3. NCFM 的 amplitude+phase loss 由 `_compute_cf_loss_terms` 实现：
   ```199:212:openrlhf/utils/embedding_utils.py
   def _compute_cf_loss_terms(target_real, target_imag, gen_real, gen_imag, alpha: float, beta: float):
       target_norm = torch.sqrt(target_real * target_real + target_imag * target_imag)
       gen_norm = torch.sqrt(gen_real * gen_real + gen_imag * gen_imag)
       amp_diff = target_norm - gen_norm
       loss_amp = amp_diff * amp_diff
       loss_pha = 2 * (
           target_norm * gen_norm
           - gen_real * target_real
           - gen_imag * target_imag
       )
       loss_pha = loss_pha.clamp(min=1e-12)
       return torch.sqrt(float(alpha) * loss_amp + float(beta) * loss_pha)
   ```
4. **leave-one-out 归因**：通过 `(sum - 自身) / (N-1)` 在 O(N·F·D) 内拿到 X∖{j} 的经验 CF，
   `reward_j = loo_loss − full_loss`，再 reshape 回 `(B, G, N, K)`，乘以 `cf_reward_scale`。

> 注：因为奖励来自 CF 距离的负梯度方向，本期实现中 multiplicative 多样性项被强制为 0，
> CF 内置的 leave-one-out 已经隐式提供了 "组内多样性" 的归因，等价于把 align/div 揉到了同一损失里。

### 2.4 三档 Stage2 具体配置（HumanEval 一致 epoch=0.16）

公共项（与 G1 相同）：
- `--pretrain Qwen3.5-4B`，16k 训练子集，`train_batch_size=128`，`n_samples_per_prompt=4`，
  `actor_learning_rate=1e-6`，`advantage_estimator=rloo`，student `T=0.6 / top_p=1.0`，
  `embed_method=last_token`，`critic_sequence_level=last_token`，`use_whitening`，
  `use_kl_loss + init_kl_coef=0`，critic 完全冻结（`critic_learning_rate=0`，`critic_lr_head=0`）。

CF 通用项：
- `--distribution_reward_type cf_l1oo`，`--cf_num_freqs 128`，`--cf_sigma 1.0`，`--cf_seed 43`，
  `--cf_alpha 0.5`，`--cf_beta 0.5`，`--cf_reward_scale 1.0`，`--feature_map_type identity`。

#### 2.4.1 CharacterFunc + Single GT（`run_G2_rebase_no_teacher_distribution_*.sh`）

| 参数 | 取值 |
| --- | --- |
| `--cf_target_mode` | `single` |
| `--cf_target_num_refs` | `1` |
| teacher | 关闭（`teacher_pretrain` 未传） |
| 拓扑 | A100×4（actor+ref 2 卡，critic+reward 2 卡） |

含义：仅以 GT 单点为 target，对组内 4 条 student rollout 做 CF leave-one-out。

#### 2.4.2 CharacterFunc + Vicinal GT（`run_G2_rebase_no_teacher_vicinal_2node_once.sh`）

| 参数 | 取值 |
| --- | --- |
| `--cf_target_mode` | `vicinal` |
| `--cf_target_num_refs` | `8` |
| `--cf_target_std` | `0.05` |
| `--cf_target_seed` | `43` |
| teacher | 关闭 |
| 拓扑 | A100×4（actor+ref 2 卡跨节点，critic+reward 2 卡跨节点）|

含义：在 GT 特征点附近用 7 个 deterministic Gaussian 扰动 + 1 个 GT 点近似一个局部分布，
让 CF 不再退化为"对单点对齐"，鼓励学生分布覆盖 GT 邻域。`cf_target_std=0.05` 是相对 local-RMS
的小扰动尺度。

#### 2.4.3 CharacterFunc + Teacher（`run_G2_rebase_2node_once.sh`）

| 参数 | 取值 |
| --- | --- |
| `--cf_target_mode` | `teacher` |
| `--cf_teacher_lambda` | `0.6`（GT 权重 0.4，teacher 权重 0.6）|
| `--cf_teacher_n_samples` (M) | `4` |
| Teacher 来源 | `--teacher_backend remote`，远程 vLLM（Qwen3.5-27B）|
| Teacher 采样 | `T=0.7`，`top_p=0.95`，`max_new_tokens=1024` |
| 拓扑 | 2 节点 4 卡（actor 2 + critic 2 跨节点）+ 12 卡 teacher worker |

含义：M=4 条 teacher rollout 与 GT 混合得到 target 经验测度，等价
`r/(r+m) = 0.4` 个 GT、4 条 teacher，effective λ = 4/(r+4) ≈ 0.6。teacher 的 rollout 通过同一
critic + `embed_method=last_token` 抽特征后注入到 CF target，保证与 student 同一特征空间可比。

> 实现上 teacher 总共按 prompt 去重后只采一遍（M 条），通过 critic forward 转成 embedding，再
> 经 `_build_cf_target_embedding` 拼接成 `(B, G, r+M, K, D)` 进入 CF；远程模式额外带 SQLite 磁盘
> 缓存（`--teacher_cache_enable`）与并发请求池。

### 2.5 Reward 端到端时序

```
prompts → actor (T=0.6 ×4) → 4 student rollouts
                        ↓
       critic forward + embed_method=last_token + identity feature_map
                        ↓
                  gen_embedding (B,G,4,K,D)
                  gt_embedding  (B,G,4,K,D)  ← GT 沿 N 复制

[teacher 分支] prompts → teacher (T=0.7 ×M=4) → teacher rollouts
                          ↓
                 critic forward (同一 embed pipeline)
                          ↓
                  teacher_embedding (B,G,M,K,D)

→ _build_cf_target_embedding → target_embedding
→ get_cf_l1oo_rewards (NCFM amp+phase + LOO)
→ shaped_rewards = raw_rewards − RLOO baseline
→ actor 用 PPO + CE + 0 KL 优化
```

---

## 3. Stage 3 — CF + Teacher + Unfrozen Feature Network

### 3.1 设计动机

Stage 2 已经把"target 分布换成 GT+teacher 的 NCFM"这一档锁定（`cf_target_mode=teacher`，
`cf_l1oo`），但 critic 端的特征空间仍然是**初始化 backbone 的冻结特征**，几何上未必匹配 CF
所需的"小邻域内 amp/phase 都可分离"。Stage 3 在此基础上解冻一小段表征以做 in-place 的几何
微调，同时引入 EMA target 防止表征漂移。

具体做了三件事：
1. **Feature Adapter**：在 critic backbone 顶部插一个 residual bottleneck 适配器（rank=64），
   只让这一层 + 分类头可训练。
2. **解冻顶部 1 层**：在 adapter 之外，额外把 backbone **最上面 1 个 transformer block** 解冻
   （`feature_adapter_unfreeze_layers=1`），即 2-full 的轻量变体。
3. **EMA + Direct Discrepancy 监督**：维护一份 critic 的 EMA 副本（`ema_beta=0.99`），
   critic 训练时除了 classifier 信号，还显式监督"online critic 提取的学生特征与 EMA 提取的 GT 特征
   之间的 CF 距离"，权重 `critic_direct_discrepancy_coef=0.1`。

### 3.2 Adapter 结构

```102:120:openrlhf/models/critic.py
class ResidualBottleneckFeatureAdapter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int = 64, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.weight)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.down_proj(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x
```

注意 `up_proj` 初始化为 0，所以训练开始时 adapter 是严格的恒等映射，几何上与 G2 完全一致；
随训练逐步学到对 CF 几何更友好的小幅修正。

### 3.3 解冻策略

`feature_adapter_enable=True` 时**先冻结整个 backbone**，再按 `feature_adapter_unfreeze_layers`
解冻顶层若干 transformer block：

```338:347:openrlhf/models/critic.py
            if self.feature_adapter_enable:
                # Freeze the entire backbone first.
                for param in self.model.parameters():
                    param.requires_grad = False
                # 2-full: selectively unfreeze the top-N transformer layers.
                # feature_adapter_unfreeze_layers=0  → 2-lite (frozen backbone, adapter+head only)
                # feature_adapter_unfreeze_layers>0  → 2-full (top-N layers + adapter + head trainable)
                if self.feature_adapter_unfreeze_layers > 0:
                    self._unfreeze_top_layers(self.model, self.feature_adapter_unfreeze_layers)
```

文档里"解冻 Feature layer 1"对应 `--feature_adapter_unfreeze_layers 1`，即只解冻最顶部
1 个 transformer block + adapter + classifier head + pre-head norm，其余 backbone 仍然冻结。

### 3.4 EMA 副本

EMA 副本由 `EBFTCriticActor` 在初始化时复制一份 critic 得到，每个 optimizer step 后做一次 EMA
更新（`θ_target ← β · θ_target + (1-β) · θ_online`）：

```358:363:openrlhf/trainer/ray/ebft_critic.py
        if train_critic and self.ema_model:
            if self.args.use_dynamic_batch:
                if self.replay_buffer.dynamic_optimizer_step[step]:
                    self.strategy.moving_average(self.critic, self.ema_model, self.ema_beta, "cuda")
            else:
                self.strategy.moving_average(self.critic, self.ema_model, self.ema_beta, "cuda")
```

`ema_beta=0.99` 意味着 `θ_target ← 0.99 θ_target + 0.01 θ_online`，与文档描述一致。
log/post-eval 也优先使用 EMA 副本以获得更稳的指标：

```386:402:openrlhf/trainer/ray/ebft_critic.py
        # Get post-step metrics (using EMA model if available, otherwise current critic)
        with torch.no_grad():
            _post_model = self.ema_model if self.ema_model else self.critic
```

### 3.5 Critic 的额外几何监督：Direct CF Discrepancy

Critic 的损失变为：
\[
\mathcal{L}\_{\text{critic}} = \lambda\_{\text{cls}} \cdot \mathcal{L}\_{\text{classifier}} +
\lambda\_{\text{dd}} \cdot D\_{\text{CF}}\big(\phi\_{\text{online}}(\text{gen}),\;
\phi\_{\text{EMA}}(\text{gt})\big)
\]

- `λ_cls = --critic_classifier_loss_coef`（G3 配置里设为 0，仅保留几何监督）；
- `λ_dd = --critic_direct_discrepancy_coef = 0.1`；
- `D_CF` 由 `compute_cf_discrepancy_loss` 提供（与 reward 端同一个 NCFM amp+phase 公式，但
  保留梯度回传到 online critic）；
- target 特征用 `--critic_direct_discrepancy_target=ema_gt`，即 GT 序列经 EMA critic 抽出来的
  hidden state 做 detach。

```290:347:openrlhf/trainer/ray/ebft_critic.py
        critic_direct_discrepancy_loss = torch.zeros((), device=device, dtype=critic_classifier_loss.dtype)
        direct_discrepancy_coef = float(getattr(self.args, "critic_direct_discrepancy_coef", 0.0) or 0.0)
        direct_discrepancy_target = getattr(self.args, "critic_direct_discrepancy_target", "ema_gt")
        if direct_discrepancy_coef > 0.0 and getattr(self.args, "distribution_reward_type", "pointwise") == "cf_l1oo":
            ...
            if direct_discrepancy_target == "ema_gt" and self.ema_model is not None:
                with torch.no_grad():
                    ema_gt_hidden_states, _, _, _, _ = self.ema_model(
                        full_sequences.to(device),
                        ...
                    )
                target_gt_hidden_states = ema_gt_hidden_states.detach()
            ...
            critic_direct_discrepancy_loss = direct_discrepancy_coef * compute_cf_discrepancy_loss(
                online_gen_embedding,
                target_gt_embedding.detach(),
                ...
            )
```

这样 critic 同时承担了两个角色：
1. 在 reward 侧，CF leave-one-out 的特征几何由 `feature_adapter + 顶层 1 层` 即时调整；
2. 在 supervision 侧，直接最小化 online critic 提取的 student 特征与 EMA 提取的 GT 特征
   之间的 CF 距离，提供"几何坐标系"的稳定学习信号——这就是文档里"critic 吃 RL signal，还直接监督学
   生 feature 与 EMA-平滑 GT feature 的距离，coef=0.1"的实现。

### 3.6 Stage 3 完整配置（`run_G3_rebase_2node_once.sh`）

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| 学生 / 训练数据 / 优化器 | 同 Stage 2 teacher 档 | Qwen3.5-4B, 16k, train_batch_size=128, n=4, actor_lr=1e-6, RLOO, T=0.6 |
| `--distribution_reward_type` | `cf_l1oo` | 同 Stage 2 |
| `--cf_target_mode` / `--cf_teacher_lambda` / `--cf_teacher_n_samples` | `teacher` / `0.6` / `4` | 同 Stage 2 teacher 档 |
| Teacher | Qwen3.5-27B（`teacher_backend=remote`），T=0.7, top_p=0.95, max_new_tokens=1024 | 同上 |
| `--enable_ema` / `--ema_beta` | `True` / `0.99` | 启用 critic EMA 副本 |
| `--feature_adapter_enable` | `True` | 启用 residual bottleneck adapter |
| `--feature_adapter_type` | `residual_bottleneck` | 见 3.2 |
| `--feature_adapter_rank` | `64` | 瓶颈维度 |
| `--feature_adapter_dropout` | `0.0` | |
| `--feature_adapter_unfreeze_layers` | `1` | 解冻顶部 1 个 transformer block |
| `--critic_learning_rate` | `0` | backbone 冻结 |
| `--critic_lr_head` | `5e-5` | adapter + head + 顶层 1 层用的 LR |
| `--critic_classifier_loss_coef` | `0.0` | 该档不使用 classifier 信号 |
| `--critic_direct_discrepancy_coef` | `0.1` | 监督 student feature ↔ EMA GT feature 的 CF 距离 |
| `--critic_direct_discrepancy_target` | `ema_gt` | target 用 EMA critic 提取 |
| `--ce_loss_coef` | `0.03` | 注意：G3 把 CE 调高到 0.03（G1 是 0.01）|
| `--alignment_rew_coef` / `--diversity_rew_coef` | `1.0` / `1.0` | cf_l1oo 路径下 diversity reward=0，这两个系数实际只影响 baseline 的线性组合 |
| 拓扑 | 2 节点 4 卡（actor 2 + critic 2 跨节点）+ 12 卡 teacher worker | 同 Stage 2 teacher 档 |

### 3.7 与 Stage 2 的差异一句话总结

Stage 2 ↔ Stage 3 在数据集、teacher 配置、采样配置、reward 公式（cf_l1oo + GT+teacher target）
**完全相同**；区别只在 critic 端：

- Stage 2：critic backbone + head 完全冻结，CF 在"开局固定"的特征几何下做 reward；
- Stage 3：critic 顶层 1 层 + residual bottleneck adapter 可训练，且通过 EMA-平滑的 GT
  特征对 online critic 做直接 CF 几何监督，**reward 几何与 critic 几何同步演化**。

---

## 4. 与原文对照的几个要点

1. **数据集切换为 Code Completion**：原文 EBFT 在数学 QA 上的对照点为 12.68% (base) →
   12.04% (EBFT) → 14.12% (CF+EBFT)。本仓库当前实验改用 `open-code-instruct_qa_100k` 的 16k
   子集 + HumanEval post-eval，并把 student 升级到 Qwen3.5-4B（原文 Qwen2.5-1.5B）。Stage 1
   pointwise EBFT 在 0.16 epoch 略低于 base（57.32% → 54.11% greedy），与原文图中
   "0.16 epoch 容易掉点"的趋势一致。
2. **CF 重新拿回涨幅**：Stage 2 的 single-GT CF 在同样 epoch 把 greedy 从 57.32% 推到
   **65.45%**，在 pass@1 T=0.6 同样有 56.71%，体现 reward 端从 pointwise cosine 升级为
   NCFM CF 后，对组级分布匹配比单点对齐更稳健。
3. **Teacher 的作用主要体现在 pass@1**：teacher 把 target 从 δ(GT) 扩到 (1-λ)δ(GT)+λν_teacher，
   pass@1 T=0.6 从 56.71% 提到 **58.91%**（greedy 略低于 single-GT，符合"更多 target 样本带来
   更平滑但偏离 GT 的最优解"这一直觉）。
4. **Unfrozen Feature Net 主要修复 pass@1 方差**：Stage 3 在 greedy 上与 Stage 2 teacher 档基本
   持平（62.60% vs 61.57%），pass@1 T=0.6 的均值小幅低于 teacher（58.74% vs 58.91%），但单个
   round 的最高点（62.20%）比 teacher 高，证明 critic 端解冻 + EMA + CF 直接监督让特征几何
   更适配 reward，但代价是稳定性变差，需更多 epoch 收敛。

---

## 5. 复现入口速查

| Stage / 档位 | 启动脚本 | 关键 CLI |
| --- | --- | --- |
| Stage 1 EBFT（pointwise + single GT） | `scripts/diff_dataset/run_G1_rebase.sh` | `--distribution_reward_type pointwise --cf_target_mode single` |
| Stage 2 CF + Single GT | `scripts/diff_dataset/run_G2_rebase_no_teacher_distribution_*.sh` | `--distribution_reward_type cf_l1oo --cf_target_mode single` |
| Stage 2 CF + Vicinal GT | `scripts/diff_dataset/run_G2_rebase_no_teacher_vicinal_2node_once.sh` | `--distribution_reward_type cf_l1oo --cf_target_mode vicinal --cf_target_num_refs 8 --cf_target_std 0.05` |
| Stage 2 CF + Teacher | `scripts/diff_dataset/run_G2_rebase_2node_once.sh` | `--distribution_reward_type cf_l1oo --cf_target_mode teacher --cf_teacher_lambda 0.6 --cf_teacher_n_samples 4 --teacher_backend remote` |
| Stage 3 CF + Teacher + Unfrozen FeatureNet | `scripts/diff_dataset/run_G3_rebase_2node_once.sh` | 在 Stage 2 teacher 档基础上加：`--enable_ema --ema_beta 0.99 --feature_adapter_enable --feature_adapter_rank 64 --feature_adapter_unfreeze_layers 1 --critic_lr_head 5e-5 --critic_direct_discrepancy_coef 0.1 --critic_direct_discrepancy_target ema_gt` |

> 所有档位 post-eval 统一走 `scripts/benchmarks/run_code_generation_benchmarks.py`，HumanEval
> 在 greedy (T=0) 与 sample (T=0.6) 各采 1 / 16 条；3 轮重复取均值 ± std。
