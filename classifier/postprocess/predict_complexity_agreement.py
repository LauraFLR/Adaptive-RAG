"""
Iteration 2a: Replace Clf1 with cross-strategy answer agreement.

If nor_qa and oner_qa give the same answer (after normalization) for a question,
route to A (no retrieval). Otherwise, pass to Clf2 to decide B vs C.

Usage:
    python classifier/postprocess/predict_complexity_agreement.py flan_t5_xl \
        --clf2_pred_file classifier/outputs/.../single_vs_multi/epoch/35/.../predict/dict_id_pred_results.json \
        --predict_file   classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json \
        --output_path    predictions/classifier/t5-large/flan_t5_xl/split_agreement/nor_oner_clf2ep35/
"""

import argparse
import json
import os
import re
import string
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from postprocess_utils import load_json, save_json


DATASETS = ['musique', 'hotpotqa', '2wikimultihopqa', 'nq', 'trivia', 'squad']


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
    """Extract answer from potential chain-of-thought response."""
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


def load_strategy_predictions(model_name, split="test"):
    """Load nor_qa and oner_qa predictions for all datasets."""
    nor_preds = {}
    oner_preds = {}
    base = f"predictions/{split}"
    for ds in DATASETS:
        nor_dir = f"nor_qa_{model_name}_{ds}____prompt_set_1"
        oner_dir = f"oner_qa_{model_name}_{ds}____prompt_set_1___bm25_retrieval_count__15___distractor_count__1"
        sub = "test_subsampled" if split == "test" else "dev_500_subsampled"
        nor_file = os.path.join(base, nor_dir, f"prediction__{ds}_to_{ds}__{sub}.json")
        oner_file = os.path.join(base, oner_dir, f"prediction__{ds}_to_{ds}__{sub}.json")
        nor_preds.update(load_json(nor_file))
        oner_preds.update(load_json(oner_file))
    return nor_preds, oner_preds


def compute_agreement(nor_preds, oner_preds, qids):
    """Compute agreement between nor_qa and oner_qa answers.
    Returns dict: qid -> {"agree": bool, "nor_answer": str, "oner_answer": str}
    """
    results = {}
    for qid in qids:
        nor_raw = nor_preds[qid]
        oner_raw = oner_preds[qid]
        # Handle list answers
        if isinstance(nor_raw, list):
            nor_raw = nor_raw[0] if nor_raw else ""
        if isinstance(oner_raw, list):
            oner_raw = oner_raw[0] if oner_raw else ""
        nor_raw = str(nor_raw)
        oner_raw = str(oner_raw)
        # Apply answer extraction (handle CoT)
        nor_extracted = answer_extractor(nor_raw)
        oner_extracted = answer_extractor(oner_raw)
        # Normalize
        nor_norm = normalize_answer(nor_extracted)
        oner_norm = normalize_answer(oner_extracted)
        # Agreement: both non-empty and equal
        agree = bool(nor_norm and oner_norm and nor_norm == oner_norm)
        results[qid] = {
            "agree": agree,
            "nor_answer": nor_raw,
            "oner_answer": oner_raw,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, choices=("flan_t5_xl", "flan_t5_xxl"))
    parser.add_argument("--clf2_pred_file", type=str, required=True)
    parser.add_argument("--predict_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    m = args.model_name

    # Load predict.json for qid -> dataset_name mapping
    predict_data = load_json(args.predict_file)
    qid_to_dataset = {item["id"]: item["dataset_name"] for item in predict_data}
    all_qids = list(qid_to_dataset.keys())

    # Load nor_qa and oner_qa predictions
    print("Loading nor_qa and oner_qa predictions...")
    nor_preds, oner_preds = load_strategy_predictions(m, split="test")

    # Compute agreement
    print("Computing answer agreement...")
    agreement = compute_agreement(nor_preds, oner_preds, all_qids)
    agree_count = sum(1 for v in agreement.values() if v["agree"])
    print(f"Agreement rate: {agree_count}/{len(all_qids)} = {agree_count/len(all_qids):.3f}")

    # Load Clf2 predictions
    clf2 = load_json(args.clf2_pred_file)

    # Merge: agree → A, disagree → Clf2 prediction (B or C)
    merged = {}
    for qid in all_qids:
        if agreement[qid]["agree"]:
            merged[qid] = {
                "prediction": "A",
                "dataset_name": qid_to_dataset[qid],
            }
        else:
            merged[qid] = {
                "prediction": clf2[qid]["prediction"],  # B or C
                "dataset_name": qid_to_dataset[qid],
            }

    label_counts = Counter(v["prediction"] for v in merged.values())
    print(f"Merged A/B/C distribution: {dict(sorted(label_counts.items()))}")

    # Load stepNum for ircot routing
    total_step_num = {}
    consolidated = os.path.join("predictions", "test", f"ircot_qa_{m}", "total", "stepNum.json")
    if os.path.exists(consolidated):
        total_step_num = load_json(consolidated)
    else:
        for ds in DATASETS:
            sn_path = os.path.join(
                "predictions", "test",
                f"ircot_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__6___distractor_count__1",
                "stepNum.json"
            )
            if os.path.exists(sn_path):
                total_step_num.update(load_json(sn_path))

    # Build prediction files for QA strategies
    dataName_to_files = {}
    for ds in DATASETS:
        dataName_to_files[ds] = {
            'C': os.path.join("predictions", "test",
                              f'ircot_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__6___distractor_count__1',
                              f'prediction__{ds}_to_{ds}__test_subsampled.json'),
            'B': os.path.join("predictions", "test",
                              f'oner_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__15___distractor_count__1',
                              f'prediction__{ds}_to_{ds}__test_subsampled.json'),
            'A': os.path.join("predictions", "test",
                              f'nor_qa_{m}_{ds}____prompt_set_1',
                              f'prediction__{ds}_to_{ds}__test_subsampled.json'),
        }

    output_path = args.output_path

    total_steps_all = 0
    per_dataset_stats = {}
    for data_name in DATASETS:
        qid_to_pred = {}
        qid_to_pred_option = {}
        total_steps = 0

        for qid, info in merged.items():
            if info["dataset_name"] != data_name:
                continue

            option = info["prediction"]
            if option == "C":
                step_num = total_step_num.get(qid, 0)
            elif option == "B":
                step_num = 1
            else:
                step_num = 0

            all_preds = load_json(dataName_to_files[data_name][option])
            qid_to_pred[qid] = all_preds[qid]
            qid_to_pred_option[qid] = {
                "prediction": all_preds[qid],
                "option": option,
                "stepNum": step_num,
            }
            total_steps += step_num

        a_count = sum(1 for v in qid_to_pred_option.values() if v["option"] == "A")
        b_count = sum(1 for v in qid_to_pred_option.values() if v["option"] == "B")
        c_count = sum(1 for v in qid_to_pred_option.values() if v["option"] == "C")

        per_dataset_stats[data_name] = {"A": a_count, "B": b_count, "C": c_count, "steps": total_steps}
        total_steps_all += total_steps
        print(f"  {data_name}: A={a_count}, B={b_count}, C={c_count}, steps={total_steps}")

        save_json(os.path.join(output_path, data_name, f"{data_name}.json"), qid_to_pred)
        save_json(os.path.join(output_path, data_name, f"{data_name}_option.json"), qid_to_pred_option)

    # Save routing stats
    routing_stats = {
        "model_name": m,
        "clf2_pred_file": args.clf2_pred_file,
        "method": "cross_strategy_agreement",
        "gate_rule": "nor_qa == oner_qa (normalized) -> A, else Clf2(B/C)",
        "total_questions": len(all_qids),
        "agreement_rate": agree_count / len(all_qids),
        "total_A": label_counts.get("A", 0),
        "total_B": label_counts.get("B", 0),
        "total_C": label_counts.get("C", 0),
        "total_steps": total_steps_all,
        "per_dataset": per_dataset_stats,
    }
    save_json(os.path.join(output_path, "routing_stats.json"), routing_stats)

    print(f"\nRouted predictions saved to {output_path}")
    print(f"Run evaluation with: python evaluate_final_acc.py --pred_path {output_path}")


if __name__ == "__main__":
    main()
