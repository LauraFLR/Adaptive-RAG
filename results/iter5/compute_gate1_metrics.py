"""
Compute Gate 1 (A vs not-A) metrics for IT5 agreement gate.

Derives silver labels from test set by comparing each strategy's token-F1
against gold answers. If nor_qa achieves the best F1 (and F1 > 0), the
question is labelled A; otherwise R (not-A).

Then computes confusion matrix, accuracy, precision, recall, F1, and
retrieval rate for the agreement gate's predictions.

Usage:
    python3 results/iter5/compute_gate1_metrics.py \
        --model_name flan_t5_xl \
        --routed_pred_path predictions/classifier/t5-large/flan_t5_xl/split_agreement/iter5_apr25/ \
        --output_file results/iter5/xl/gate1_metrics.json
"""

import argparse
import json
import os
import re
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'classifier', 'postprocess'))

from postprocess_utils import load_json, save_json

DATASETS = ['musique', 'hotpotqa', '2wikimultihopqa', 'nq', 'trivia', 'squad']
ONER_BM25 = {'flan_t5_xl': 15, 'flan_t5_xxl': 15, 'gpt': 6}
IRCOT_BM25 = {'flan_t5_xl': 6, 'flan_t5_xxl': 6, 'gpt': 3}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


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


def compute_token_f1(prediction_str, gold_answers):
    """Compute best token-level F1 between prediction and any gold answer."""
    pred_tokens = normalize_answer(answer_extractor(str(prediction_str))).split()
    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(str(gold)).split()
        if not pred_tokens and not gold_tokens:
            best_f1 = max(best_f1, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        from collections import Counter
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1


def load_ground_truths(dataset_name):
    """Load ground truth answers from processed_data."""
    import jsonlines
    gt_path = os.path.join('processed_data', dataset_name, 'test_subsampled.jsonl')
    id_to_gt = {}
    with jsonlines.open(gt_path, 'r') as reader:
        for line in reader:
            qid = line['question_id']
            answers = line['answers_objects'][0]['spans']
            id_to_gt[qid] = answers
    return id_to_gt


def derive_silver_labels(model_name):
    """Derive A/R silver labels for test set based on strategy performance."""
    m = model_name
    oner_bm25 = ONER_BM25[m]
    ircot_bm25 = IRCOT_BM25[m]

    silver_labels = {}

    for ds in DATASETS:
        gt = load_ground_truths(ds)

        # Load predictions for each strategy
        nor_file = os.path.join("predictions", "test",
            f"nor_qa_{m}_{ds}____prompt_set_1",
            f"prediction__{ds}_to_{ds}__test_subsampled.json")
        oner_file = os.path.join("predictions", "test",
            f"oner_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__{oner_bm25}___distractor_count__1",
            f"prediction__{ds}_to_{ds}__test_subsampled.json")
        ircot_file = os.path.join("predictions", "test",
            f"ircot_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__{ircot_bm25}___distractor_count__1",
            f"prediction__{ds}_to_{ds}__test_subsampled.json")

        nor_preds = load_json(nor_file)
        oner_preds = load_json(oner_file)
        ircot_preds = load_json(ircot_file)

        for qid in gt:
            gold = gt[qid]
            nor_ans = nor_preds.get(qid, "")
            oner_ans = oner_preds.get(qid, "")
            ircot_ans = ircot_preds.get(qid, "")

            if isinstance(nor_ans, list):
                nor_ans = nor_ans[0] if nor_ans else ""
            if isinstance(oner_ans, list):
                oner_ans = oner_ans[0] if oner_ans else ""
            if isinstance(ircot_ans, list):
                ircot_ans = ircot_ans[0] if ircot_ans else ""

            nor_f1 = compute_token_f1(nor_ans, gold)
            oner_f1 = compute_token_f1(oner_ans, gold)
            ircot_f1 = compute_token_f1(ircot_ans, gold)

            # A if nor_qa is best (or tied for best) and has non-zero F1
            if nor_f1 > 0 and nor_f1 >= oner_f1 and nor_f1 >= ircot_f1:
                silver_labels[qid] = "A"
            else:
                silver_labels[qid] = "R"

    return silver_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        choices=("flan_t5_xl", "flan_t5_xxl", "gpt"))
    parser.add_argument("--routed_pred_path", type=str, required=True,
                        help="Path to IT5 routed predictions (contains {dataset}/{dataset}_option.json)")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    print(f"Deriving silver A/R labels for {args.model_name}...")
    silver = derive_silver_labels(args.model_name)

    # Load IT5 agreement gate predictions (A/B/C from option files)
    gate_preds = {}
    for ds in DATASETS:
        opt_file = os.path.join(args.routed_pred_path, ds, f"{ds}_option.json")
        opts = load_json(opt_file)
        for qid, info in opts.items():
            gate_preds[qid] = info["option"]  # A, B, or C

    # Binarize: A vs not-A
    # TP = predicted A, silver A
    # FP = predicted A, silver R
    # FN = predicted not-A, silver A
    # TN = predicted not-A, silver R
    tp = fp = fn = tn = 0
    for qid in silver:
        pred_is_A = (gate_preds[qid] == "A")
        true_is_A = (silver[qid] == "A")
        if pred_is_A and true_is_A:
            tp += 1
        elif pred_is_A and not true_is_A:
            fp += 1
        elif not pred_is_A and true_is_A:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    total_A_pred = tp + fp
    total_notA_pred = fn + tn
    retrieval_rate = total_notA_pred / total if total else 0

    # R-class metrics
    r_precision = tn / (tn + fn) if (tn + fn) else 0
    r_recall = tn / (tn + fp) if (tn + fp) else 0
    r_f1 = 2 * r_precision * r_recall / (r_precision + r_recall) if (r_precision + r_recall) else 0

    results = {
        "model_name": args.model_name,
        "total_questions": total,
        "silver_A_count": tp + fn,
        "silver_R_count": fp + tn,
        "predicted_A_count": tp + fp,
        "predicted_notA_count": fn + tn,
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "accuracy": round(accuracy, 4),
        "A_precision": round(precision, 4),
        "A_recall": round(recall, 4),
        "A_f1": round(f1, 4),
        "R_precision": round(r_precision, 4),
        "R_recall": round(r_recall, 4),
        "R_f1": round(r_f1, 4),
        "retrieval_rate": round(retrieval_rate, 4),
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nGate 1 Metrics ({args.model_name}):")
    print(f"  Silver labels: A={tp+fn}, R={fp+tn}")
    print(f"  Predictions:   A={tp+fp}, not-A={fn+tn}")
    print(f"  Confusion:     TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  A-Precision:   {precision:.4f}")
    print(f"  A-Recall:      {recall:.4f}")
    print(f"  A-F1:          {f1:.4f}")
    print(f"  R-Recall:      {r_recall:.4f}")
    print(f"  Retrieval rate: {retrieval_rate:.4f}")
    print(f"  Saved to {args.output_file}")


if __name__ == "__main__":
    main()
