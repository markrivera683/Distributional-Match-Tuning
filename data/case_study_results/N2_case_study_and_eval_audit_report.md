# N2 Case Study & Eval Audit Report

## 1. 评测协议审计 (Eval Audit)

### 1.1 答案解析缺陷 — 关键发现

对 LiveAoPSBench-2024 全部 5317 条非空答案进行 `math_verify.parse()` 测试，发现：

| 指标 | 数值 | 比例 |
|------|------|------|
| 总非空答案 | 5317 | 100% |
| `parse(raw)` 可解析 | 4305 | 81.0% |
| `parse(\boxed{answer})` 可解析 | 5317 | **100%** |
| **解析缺口** (boxed 可解析但 raw 不可) | **1012** | **19.0%** |
| **危险错解** (raw/boxed 都解析成功但结果不同) | **1160** | **21.8%** |

**危险错解典型案例：**
- `2^{2022}` → raw 解析为 `2`，boxed 解析为 `2**2022`
- `\frac{n(n+1)(2n+1)}{6}` → raw 解析为 `[]`（失败），boxed 正确
- `f'(x) (-\cos x + 1) + f(x) \sin x` → raw 解析为 `1`，boxed 正确

### 1.2 评测路由缺陷

EBFT trainer (`ebft_trainer.py` L262-267) 按 eval_dataset 路径名判断走哪个评测函数：

```python
if "gsm8k" in eval_ds or "math" in eval_ds:
    evaluate_downstream_gsm8k_math(...)  # 数学答案验证
else:
    evaluate_downstream_translation(...)  # 翻译 BLEU 评测
```

AoPS 数据路径 (`LiveAoPSBench-2024.jsonl`) **不含 "gsm8k" 或 "math"** 子串，会被错误路由到翻译评测。

### 1.3 EBFT vs SFT 答案验证路径差异

| 路径 | 对 gold answer 的处理 | 影响 |
|------|----------------------|------|
| EBFT eval (`ebft_eval_mixin.py`) | `get_llm_answer(label)` → `parse(text)` 无 boxed 包裹 | 19% 答案无法解析，22% 答案解析错误 |
| SFT eval (`sft_trainer.py`) | `parse(f"\\boxed{{{answer}}}")` 有 boxed 包裹 | 100% 答案可解析 |

### 1.4 答案格式分布

| 类型 | 数量 | 比例 |
|------|------|------|
| 纯数值 (如 `165`) | 2339 | 43.9% |
| LaTeX 命令 (如 `\frac{1}{2}`) | 2011 | 37.7% |
| 含变量符号 (如 `f(x)=x`) | 1552 | 29.1% |
| 空/无答案 | 11 | 0.2% |
| 极长答案 (>100 字符) | 90 | 1.7% |

### 1.5 审计结论

**至少 19-22% 的评测结果可能不可信**。当前评测管线对 AoPS 的 LaTeX 格式答案处理有系统性缺陷。在修复评测之前，N2 文档中 8.39% 等准确率数字的诊断价值受限。

---

## 2. Case Study — Base 模型输出分析

### 2.1 实验设置

- 模型: Qwen3.5-2B Base
- 样本: 30 条 (按数值/LaTeX/符号 三类均匀抽样)
- 生成长度: max_new_tokens = 512
- 解码: greedy (temperature=0)

### 2.2 结果概览

| 归因类别 | 数量 | 比例 |
|----------|------|------|
| 思路接近但推导中断 (reasoning_interrupted) | **27** | **90%** |
| 计算错误 (calculation_error) | 2 | 7% |
| 正确 (correct) | 1 | 3% |
| 完全不会做 (completely_lost) | 0 | 0% |

### 2.3 关键发现

**压倒性结论：问题是"推理深度不足"，而非"完全不会"。**

1. **模型理解题意并能启动正确推理**：27/30 的样本中，模型正确识别了题目类型，选择了合理的解题方法（建立坐标系、使用代数恒等式、分析定义域等），生成了 `<think>` 标签。

2. **推理在中途被截断**：512 token 的生成长度对于 Olympiad 级别数学题远远不够。模型通常在建立问题框架后 token 就耗尽了，无法完成多步推导。

3. **无"完全不会做"案例**：所有 30 个样本都展现了某种程度的数学推理能力，没有出现严重偏题或乱输出的情况。

### 2.4 典型样本

**[reasoning_interrupted] 推导中断 — 主要模式**
```
Q: 给定 x+y+z=3, xy+yz+zx=4, xyz=5, 求 x³+y³+z³
Gold: 6
Model: 正确引用了 x³+y³+z³-3xyz = (x+y+z)(x²+y²+z²-xy-yz-zx)
       但 512 token 内未能完成计算
```

**[calculation_error] 计算错误**
```
Q: 解方程 √(x+1) + √(2x²-x+1) + 2√(x²-1) - 2√(x²+1) = 0
Gold: x=1
Model: 正确分析了定义域条件 x≥-1 和 x²≥1，但推导过程中计算错误
```

---

## 3. 综合归因与建议

### 3.1 归因优先级

```
归因 1 (最高): 评测管线有系统性缺陷 → 当前准确率数字不完全可信
归因 2 (次高): 基座模型推理深度不足 → 能理解+开始推理但无法完成
归因 3 (待验): 生成长度过短 → 512 token 可能不够 Olympiad 题
```

### 3.2 建议的下一步行动

#### 立即可做 (P0)

1. **修复评测管线**：
   - 修复 `ebft_trainer.py` 的 eval 路由：为 AoPS/数学数据集添加正确的条件分支
   - 统一使用 `parse(f"\\boxed{{{answer}}}")` 路径解析 gold answer
   - 修复后用 Base 模型重跑 5328 条完整评测，获得可信 baseline

2. **修复生成长度**：评测时将 `max_new_tokens` 从 512 提升到 2048-4096

#### 短期 (P1)

3. **可信 baseline 建立后**，重新评估 G1/G2/G3/SFT vs Base 的差异
   - 如果修复评测后 Base 准确率明显提升 → 确认「评测假阴性」是主因
   - 如果修复后差异仍不大 → 确认「基座弱+任务难」是主因

4. **AoPS 数据已处理完成**，可用于后续训练：
   - Train: `/mnt/data/data/aops/aops_qa_hf` (647,255 QA pairs, 1.1GB, HF Arrow 格式)
   - Test: `/mnt/data/data/aops/test_qa.jsonl` (5,328 samples, 14MB, JSONL 格式)
   - 格式: `{question, answer}` — 与训练脚本的 `--input_key question --label_key answer` 对齐
   - 来源: `DeepStudentLlama/AoPS-Instruct` (train), `jojo23333/LiveAoPSBench-2024` (test)

#### 中期 (P2)

5. 若确认基座过弱：用更强模型 (7B instruct/math-specialized) 做对照实验
6. 若评测可信+方法有效：细调 G2/G3 超参，延长训练 step
