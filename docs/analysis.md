来，把每个类别精确解释一遍。先把决策顺序摆清楚（这是 `analyze_eval_results.py:classify_output` 里**先做谁后做谁**），再逐项说。

## 决策树（自上而下，命中即返回）

```
1. gold 是否可解析？
   ├─ gold 为空                    → missing_gold       (None, 跳过 acc 统计)
   └─ math_verify 解析不出 gold     → unparseable_gold   (None, 跳过)

2. 输出本身长什么样？
   ├─ 完全空白 (.strip() == "")     → empty_output        (False)
   └─ strip 后只剩一个 EOS marker   → pure_eos            (False) ←秒 EOS
       (markers: <eos>, </s>, <|endoftext|>, <|im_end|>, <|end|>, <|end_of_text|>)

3. 主路径 get_llm_answer(text) 抠答案 + Plan-B fallback 抠尾段答案

4. 验证 (verify):
   ├─ 主路径答案 vs gold(\boxed包装)  → correct                     (True)
   ├─ 主路径答案 vs gold(直接 parse)  → correct_raw_match            (True)
   ├─ fallback 答案 vs gold(\boxed)   → correct_fallback             (True)
   └─ fallback 答案 vs gold(直接)     → correct_raw_match_fallback   (True)

5. 都没对上时的失败分桶:
   ├─ 主路径&fallback 都没抠到东西
   │   ├─ 输出 < 30 字符              → too_short
   │   └─ 输出 ≥ 30 字符              → no_answer_extracted
   ├─ 输出含 \boxed{...}              → wrong_answer            (commit 了, 错了)
   ├─ fallback 抠到答案了              → wrong_answer_fallback   (commit 了, 错了)
   ├─ 含 reasoning 词 + 等号          → reasoning_incomplete    (没 commit, 但写了推理)
   ├─ 只有等号、没 reasoning 词        → calculation_error
   └─ 三者都没                         → no_reasoning
```

`reasoning 词` = `step\d|first|then|therefore|thus|hence|so we|let `（来自 `analyze_eval_results.py` 的 regex）。

---

## 各类别逐个解释（带你这次的真实计数）

### ✅ 正确类（共 384, **7.22%**）

| 类别 | 计数 | 含义 |
|---|---|---|
| **`correct`** | 232 | **金标路径**。主路径（`get_llm_answer`，靠 math_verify 找最末尾 \boxed 或可解析数学表达式）抠出答案，跟 `\boxed{gold}` verify 通过。 |
| **`correct_raw_match`** | 80 | 主路径答案对**不上** `\boxed{gold}` 包装的形式，但对得上 `parse(gold)` 直接形式。常见于 gold 本身就是 LaTeX 表达式（如 `\frac{n(n+1)(2n+1)}{6}`）而不是纯数字时——`\boxed{...}` 包一下反而 parse 不出原意。 |
| **`correct_fallback`** | 36 | 主路径没对上，**Plan-B fallback** 用正则从尾段抠到 plain-English 答案（"Therefore, ... is X."）verify 通过。 |
| **`correct_raw_match_fallback`** | 36 | Fallback 抠到答案，对 raw gold 也 verify 通过。Fallback × raw 双路径。 |

### ❌ 错误但 committed（共 2690, 50.5%）—— 模型给了一个具体答案，就是不对

| 类别 | 计数 | 含义 |
|---|---|---|
| **`wrong_answer`** | 76 | 输出里有 `\boxed{X}`（模型用了 SFT 风格收尾），但 X 不等于 gold。是"完美格式 + 错误答案"的少数情况。 |
| **`wrong_answer_fallback`** | **2614** | Fallback 抠到 plain-English 答案（"the answer is X" / "Therefore X"），但 X 错。**这是 ebft 模型的最大 bucket（49% 输出在这里）**——模型确实 commit 了一个答案，只是没用 `\boxed` 包，且错了。 |

### ❓ 错误且 not committed（共 1381, 26.0%）—— 写了一堆但没给出明确结论

| 类别 | 计数 | 含义 |
|---|---|---|
| **`reasoning_incomplete`** | 702 | 主路径抠到了某个 math 表达式（**但不是答案**，是中间过程的随便一个），fallback 也没找到 commit 性的答案；输出含 reasoning 关键词（therefore/thus/...）和等号。**写了推理过程但没收尾**。 |
| **`calculation_error`** | 667 | 同上，但**没有 reasoning 关键词**，只有等号。"光列式子不解释"。 |
| **`no_reasoning`** | 12 | 抠到了东西没对上，fallback 也没找到；**既没等号也没 reasoning 词**。罕见。 |

### 🚫 没产出（共 862, 16.2%）

| 类别 | 计数 | 含义 |
|---|---|---|
| **`pure_eos`** | 861 | 输出只有一个 EOS marker。**模型在第一个 token 就 EOS，秒放弃**。这是 ebft + packed-stream 训练的典型副作用。 |
| **`empty_output`** | 1 | 输出完全空白（连 EOS 都没有，应该是 vLLM 边界情况）。 |

### ⚠️ 数据问题（共 11, 0.2%）

| 类别 | 计数 | 含义 |
|---|---|---|
| **`missing_gold`** | 11 | 这条样本的 gold answer 字段是空字符串。**这些样本不计入 accuracy 分母**（is_correct = None）。 |

### 📊 这次没出现的类别（说明一下，便于以后看）

- `unparseable_gold` (0): gold 不空但 math_verify parse 不出来（罕见）
- `too_short` (0): 没抠到东西 + 输出 < 30 字符。这次为 0 是因为 **`pure_eos` 优先级更高，把所有"光发一个 EOS"的样本都先抢走了**。如果模型输出"abc<eos>"这种 4 字符的，就会落在 too_short 里。
- `no_answer_extracted` (0): 没抠到东西 + 输出 ≥ 30 字符。这次为 0 是**意料中**：因为只要输出 ≥ 30 字符，主路径 `parse(text)` 几乎一定能从中找到一个 LaTeX-ish 表达式（哪怕是中间过程的随便一个数）抠出来；fallback 也容易在尾段找到点东西。所以"完全没抠到"很难发生在 ≥ 30 字符的输出上。
- `unmatched` (0): 用 `source_idx` 和 prompt-text 都查不到 gold（这次因为 source_idx 修复 bug 走通了，全部 0）

---

## 你写论文 / 汇报 时的高密度三句话总结

> 在 5317 个 evaluable 样本上：
> - **7.22% 正确**（4 种 correct\* 类合起来；其中 1.4pp 来自 fallback 提取，是模型 commit 了 plain-English 答案恰好对的情况）
> - **50.5% 错误但 committed**（其中 49.1pp 是 plain-English 答案，1.4pp 是 \\boxed 答案）—— 模型确实在尝试给答案，只是错了
> - **26.0% 写了推理但没 commit + 16.2% 秒 EOS** —— 这是 ebft 训练 + 没 SFT 监督 `\\boxed` 的直接结果