"""
Case Study: AoPS Base Model Analysis

Selects representative AoPS samples across difficulty tiers,
runs Base (Qwen3.5-2B) inference via transformers, and categorizes
outputs into the five attribution classes defined in N2 §4.2:

  1. 完全不会做 / 严重偏题
  2. 思路接近但推导中断
  3. 计算错误
  4. 答案格式错误导致判错
  5. 微调后出现明显退化  (N/A for base — placeholder)
"""

import json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from math_verify import parse, verify
from openrlhf.utils.math_verifier import get_llm_answer, verify_llm_answer

MODEL_PATH   = os.environ.get("MODEL_PATH", "/mnt/data/Qwen3.5-2B")
DATA_PATH    = os.environ.get("DATA_PATH",
    str(Path(__file__).resolve().parent.parent / "data" / "LiveAoPSBench-2024.jsonl"))
OUT_DIR      = os.environ.get("OUT_DIR",
    str(Path(__file__).resolve().parent.parent / "data" / "case_study_results"))
NUM_SAMPLES  = int(os.environ.get("NUM_SAMPLES", "30"))
MAX_TOKENS   = int(os.environ.get("MAX_TOKENS", "512"))

os.makedirs(OUT_DIR, exist_ok=True)


def load_data(path, n):
    with open(path) as f:
        rows = [json.loads(l) for l in f]
    non_empty = [r for r in rows if r.get("answer")]

    numeric   = [r for r in non_empty if re.match(r'^-?\d+\.?\d*$', r["answer"].strip())]
    latex_sym = [r for r in non_empty
                 if '\\' in r["answer"]
                 and not re.match(r'^-?\d+\.?\d*$', r["answer"].strip())]
    other     = [r for r in non_empty if r not in numeric and r not in latex_sym]

    per_bucket = max(1, n // 3)
    import random; random.seed(42)
    selected = (
        random.sample(numeric,   min(per_bucket, len(numeric)))
        + random.sample(latex_sym, min(per_bucket, len(latex_sym)))
        + random.sample(other,     min(n - 2*per_bucket, len(other)))
    )
    return selected


def run_inference(samples, model_path, max_tokens):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Loading model from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    outputs = []
    for i, s in enumerate(samples):
        prompt = s["question"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        outputs.append(text)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] generated {len(new_ids)} tokens")

    del model
    torch.cuda.empty_cache()
    return outputs


def classify_output(question, gold_answer, model_output):
    gold_boxed = parse(f"\\boxed{{{gold_answer}}}")
    if not gold_boxed:
        return None, "unparseable_gold", f"Gold answer unparseable: {gold_answer[:80]}"

    pred, resp_type = get_llm_answer(model_output)

    if not pred:
        if len(model_output.strip()) < 20:
            return False, "completely_lost", "Output too short / empty"
        return False, "completely_lost", "No parseable answer extracted"

    try:
        correct = verify(pred, gold_boxed)
    except Exception:
        correct = False

    if correct:
        return True, "correct", ""

    raw_gold = parse(gold_answer)
    if raw_gold:
        try:
            if verify(pred, raw_gold):
                return True, "format_false_negative", \
                    "Correct but only matches raw (non-boxed) gold"
        except Exception:
            pass

    out_lower = model_output.lower()
    has_steps = bool(re.search(
        r'step\s*\d|first|then|therefore|thus|hence|so\s+we|let\s+', out_lower))
    has_eq    = bool(re.search(r'[=<>]', model_output))

    if has_steps and has_eq:
        return False, "reasoning_interrupted", \
            "Shows reasoning steps but wrong/incomplete answer"
    if has_eq and not has_steps:
        return False, "calculation_error", "Has equations but answer is wrong"

    return False, "completely_lost", "No recognisable reasoning towards the answer"


def eval_audit_stats(data_path):
    with open(data_path) as f:
        rows = [json.loads(l) for l in f]

    total = ok_raw = ok_boxed = mismatch = 0
    for r in rows:
        ans = r.get("answer", "")
        if not ans:
            continue
        total += 1
        raw   = parse(ans)
        boxed = parse(f"\\boxed{{{ans}}}")
        if raw:   ok_raw   += 1
        if boxed: ok_boxed += 1
        if raw and boxed:
            try:
                if not verify(raw, boxed):
                    mismatch += 1
            except Exception:
                mismatch += 1

    return {
        "total_non_empty":  total,
        "parseable_raw":    ok_raw,
        "parseable_boxed":  ok_boxed,
        "parse_gap":        ok_boxed - ok_raw,
        "dangerous_mismatch": mismatch,
        "pct_gap":          round((ok_boxed - ok_raw) / total * 100, 1),
        "pct_mismatch":     round(mismatch / total * 100, 1),
    }


def main():
    print("=" * 70)
    print("AoPS Case Study — Base Model (Qwen3.5-2B)")
    print("=" * 70)

    # ── Step A: eval-audit parse-gap stats ──
    print("\n[A] Computing eval-audit parse-gap statistics …")
    audit = eval_audit_stats(DATA_PATH)
    print(json.dumps(audit, indent=2))

    # ── Step B: select samples ──
    print(f"\n[B] Selecting {NUM_SAMPLES} representative samples …")
    samples = load_data(DATA_PATH, NUM_SAMPLES)
    print(f"    Selected {len(samples)} samples")

    # ── Step C: run inference ──
    print(f"\n[C] Running Base model inference on {len(samples)} samples …")
    t0 = time.time()
    outputs = run_inference(samples, MODEL_PATH, MAX_TOKENS)
    elapsed = time.time() - t0
    print(f"    Inference done in {elapsed:.0f}s")

    # ── Step D: classify each output ──
    print("\n[D] Classifying outputs …")
    records = []
    for s, out in zip(samples, outputs):
        correct, cat, detail = classify_output(
            s["question"], s["answer"], out)
        records.append({
            "idx":           s.get("idx"),
            "question":      s["question"][:200],
            "gold_answer":   s["answer"],
            "model_output":  out[:500],
            "is_correct":    correct,
            "category":      cat,
            "detail":        detail,
        })

    # ── Step E: aggregate & report ──
    from collections import Counter
    cats = Counter(r["category"] for r in records)
    n_correct = sum(1 for r in records if r["is_correct"])

    print(f"\n{'='*70}")
    print(f"RESULTS:  {n_correct}/{len(records)} correct "
          f"({n_correct/len(records)*100:.1f}%)")
    print(f"{'='*70}")
    for cat, cnt in cats.most_common():
        print(f"  {cat:30s}  {cnt:3d}  ({cnt/len(records)*100:.0f}%)")

    print(f"\n--- Sample outputs by category ---")
    shown = set()
    for cat_name in ["correct", "format_false_negative", "completely_lost",
                     "reasoning_interrupted", "calculation_error",
                     "unparseable_gold"]:
        for r in records:
            if r["category"] == cat_name and cat_name not in shown:
                shown.add(cat_name)
                print(f"\n[{cat_name}]")
                print(f"  Q: {r['question'][:150]}")
                print(f"  Gold: {r['gold_answer'][:100]}")
                print(f"  Model: {r['model_output'][:300]}")
                break

    out_file = os.path.join(OUT_DIR, "case_study_base.json")
    with open(out_file, "w") as f:
        json.dump({
            "eval_audit": audit,
            "summary": {
                "total": len(records),
                "correct": n_correct,
                "accuracy": round(n_correct / len(records) * 100, 2),
                "categories": dict(cats),
            },
            "records": records,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {out_file}")


if __name__ == "__main__":
    main()
