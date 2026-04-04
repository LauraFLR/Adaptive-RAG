#!/usr/bin/env python3
"""
SQuAD verbosity analysis: checks whether GPT's low SQuAD F1 (0.310) is caused
by answer verbosity rather than factual incorrectness, by comparing GPT
predictions against Flan-T5-XL predictions.

Usage (from repo root):
    python classifier/analysis/squad_verbosity_analysis.py
"""

import csv
import json
import os
import sys

# Add repo root so we can import metrics helpers
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO_ROOT)

from metrics.squad_answer_em_f1 import (
    compute_f1,
    get_tokens,
    metric_max_over_ground_truths,
    normalize_answer,
)

# Also reuse the answer extractor from evaluate_final_acc.py
import re


def answer_extractor(potentially_cot: str) -> str:
    if potentially_cot.startswith('"') and potentially_cot.endswith('"'):
        potentially_cot = potentially_cot[1:-1]
    cot_regex = re.compile(".* answer is:? (.*)\\.?")
    match = cot_regex.match(potentially_cot)
    if match:
        output = match.group(1)
        if output.endswith("."):
            output = output[:-1]
    else:
        output = potentially_cot
    return output


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GOLD_PATH = os.path.join(REPO_ROOT, "processed_data", "squad", "test_subsampled.jsonl")

GPT_NOR_PATH = os.path.join(
    REPO_ROOT, "predictions", "test",
    "nor_qa_gpt_squad____prompt_set_1",
    "prediction__squad_to_squad__test_subsampled.json",
)
GPT_ONER_PATH = os.path.join(
    REPO_ROOT, "predictions", "test",
    "oner_qa_gpt_squad____prompt_set_1___bm25_retrieval_count__6___distractor_count__1",
    "prediction__squad_to_squad__test_subsampled.json",
)
XL_NOR_PATH = os.path.join(
    REPO_ROOT, "predictions", "test",
    "nor_qa_flan_t5_xl_squad____prompt_set_1",
    "prediction__squad_to_squad__test_subsampled.json",
)

CSV_OUT = os.path.join(os.path.dirname(__file__), "squad_verbosity_analysis.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_gold():
    """Return dict: qid -> {"question": str, "golds": [str, ...]}"""
    import jsonlines
    result = {}
    with jsonlines.open(GOLD_PATH) as reader:
        for line in reader:
            qid = line["question_id"]
            golds = line["answers_objects"][0]["spans"]
            question = line["question_text"]
            result[qid] = {"question": question, "golds": golds}
    return result


def load_preds(path):
    """Return dict: qid -> extracted answer string."""
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for qid, pred in raw.items():
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
        result[qid] = answer_extractor(str(pred))
    return result


def contains_gold(pred_norm: str, golds_norm: list[str]) -> int:
    """1 if any normalized gold is a substring of the normalized prediction."""
    return int(any(g in pred_norm for g in golds_norm if g))


def verbosity_ratio(pred_norm: str, gold_norm: str) -> float:
    pred_len = len(pred_norm.split())
    gold_len = len(gold_norm.split())
    if gold_len == 0:
        return float(pred_len) if pred_len else 1.0
    return pred_len / gold_len


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    gold_data = load_gold()
    gpt_nor = load_preds(GPT_NOR_PATH)
    gpt_oner = load_preds(GPT_ONER_PATH)
    xl_nor = load_preds(XL_NOR_PATH)

    qids = sorted(gold_data.keys())

    rows = []
    for qid in qids:
        golds = gold_data[qid]["golds"]
        question = gold_data[qid]["question"]
        golds_norm = [normalize_answer(g) for g in golds]

        gpt_pred = gpt_nor.get(qid, "")
        xl_pred = xl_nor.get(qid, "")

        gpt_pred_norm = normalize_answer(gpt_pred)
        xl_pred_norm = normalize_answer(xl_pred)

        gpt_f1 = metric_max_over_ground_truths(compute_f1, gpt_pred, golds)
        xl_f1 = metric_max_over_ground_truths(compute_f1, xl_pred, golds)

        gpt_cg = contains_gold(gpt_pred_norm, golds_norm)
        xl_cg = contains_gold(xl_pred_norm, golds_norm)

        # Use the best-matching gold for verbosity ratio
        best_gold_norm = golds_norm[0] if golds_norm else ""
        gpt_vr = verbosity_ratio(gpt_pred_norm, best_gold_norm)
        xl_vr = verbosity_ratio(xl_pred_norm, best_gold_norm)

        gpt_pred_len = len(gpt_pred_norm.split())
        xl_pred_len = len(xl_pred_norm.split())
        gold_len = len(best_gold_norm.split()) if best_gold_norm else 0

        rows.append({
            "qid": qid,
            "question": question,
            "gold": golds[0] if golds else "",
            "gpt_nor_pred": gpt_pred,
            "xl_nor_pred": xl_pred,
            "gpt_token_f1": gpt_f1,
            "xl_token_f1": xl_f1,
            "gpt_contains_gold": gpt_cg,
            "xl_contains_gold": xl_cg,
            "gpt_verbosity_ratio": gpt_vr,
            "xl_verbosity_ratio": xl_vr,
            "gpt_pred_len": gpt_pred_len,
            "xl_pred_len": xl_pred_len,
            "gold_len": gold_len,
        })

    # -----------------------------------------------------------------------
    # Aggregate stats
    # -----------------------------------------------------------------------
    n = len(rows)
    gpt_mean_f1 = sum(r["gpt_token_f1"] for r in rows) / n
    xl_mean_f1 = sum(r["xl_token_f1"] for r in rows) / n
    gpt_mean_cg = sum(r["gpt_contains_gold"] for r in rows) / n
    xl_mean_cg = sum(r["xl_contains_gold"] for r in rows) / n
    gpt_mean_vr = sum(r["gpt_verbosity_ratio"] for r in rows) / n
    xl_mean_vr = sum(r["xl_verbosity_ratio"] for r in rows) / n
    gpt_mean_pred_len = sum(r["gpt_pred_len"] for r in rows) / n
    xl_mean_pred_len = sum(r["xl_pred_len"] for r in rows) / n

    # Verbosity-penalized correct answers: contains_gold=1 but F1<0.5
    verb_penalized = [r for r in rows if r["gpt_contains_gold"] == 1 and r["gpt_token_f1"] < 0.5]

    print("=" * 72)
    print("SQuAD Verbosity Analysis — GPT nor_qa vs Flan-T5-XL nor_qa")
    print("=" * 72)
    print(f"{'':>30s}  {'GPT':>8s}  {'XL':>8s}")
    print(f"{'Mean token F1':>30s}  {gpt_mean_f1:8.4f}  {xl_mean_f1:8.4f}")
    print(f"{'Mean contains_gold':>30s}  {gpt_mean_cg:8.4f}  {xl_mean_cg:8.4f}")
    print(f"{'Mean verbosity_ratio':>30s}  {gpt_mean_vr:8.2f}  {xl_mean_vr:8.2f}")
    print(f"{'Mean pred token length':>30s}  {gpt_mean_pred_len:8.1f}  {xl_mean_pred_len:8.1f}")
    print()
    print(f"Questions where GPT contains_gold=1 but F1 < 0.5: {len(verb_penalized)}/{n}")
    print()

    # Breakdown of GPT error types
    correct_concise = sum(1 for r in rows if r["gpt_contains_gold"] == 1 and r["gpt_token_f1"] >= 0.5)
    correct_verbose = len(verb_penalized)
    wrong = sum(1 for r in rows if r["gpt_contains_gold"] == 0)
    print("GPT error breakdown:")
    print(f"  Correct & concise (contains_gold=1, F1>=0.5):  {correct_concise:4d}  ({correct_concise/n:.1%})")
    print(f"  Correct but verbose (contains_gold=1, F1<0.5): {correct_verbose:4d}  ({correct_verbose/n:.1%})")
    print(f"  Factually wrong (contains_gold=0):             {wrong:4d}  ({wrong/n:.1%})")
    print()

    # Sample 5 verbosity-penalized questions
    if verb_penalized:
        print("-" * 72)
        print("Sample verbosity-penalized GPT answers (contains_gold=1, F1<0.5):")
        print("-" * 72)
        for r in sorted(verb_penalized, key=lambda x: x["gpt_token_f1"])[:5]:
            print(f"\n  QID:       {r['qid']}")
            print(f"  Question:  {r['question']}")
            print(f"  Gold:      {r['gold']}")
            print(f"  GPT pred:  {r['gpt_nor_pred']}")
            print(f"  Token F1:  {r['gpt_token_f1']:.4f}")
            print(f"  Verbosity: {r['gpt_verbosity_ratio']:.1f}x")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    fieldnames = [
        "question_id", "gold", "gpt_nor_pred", "xl_nor_pred",
        "gpt_token_f1", "xl_token_f1", "gpt_contains_gold",
        "gpt_verbosity_ratio",
    ]
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "question_id": r["qid"],
                "gold": r["gold"],
                "gpt_nor_pred": r["gpt_nor_pred"],
                "xl_nor_pred": r["xl_nor_pred"],
                "gpt_token_f1": f"{r['gpt_token_f1']:.4f}",
                "xl_token_f1": f"{r['xl_token_f1']:.4f}",
                "gpt_contains_gold": r["gpt_contains_gold"],
                "gpt_verbosity_ratio": f"{r['gpt_verbosity_ratio']:.2f}",
            })

    print(f"\nCSV saved to: {CSV_OUT}")


if __name__ == "__main__":
    main()
