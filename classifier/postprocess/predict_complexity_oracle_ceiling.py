"""
Clf2 Oracle Ceiling: Agreement gate + perfect B/C routing on R-routed questions.

For agree questions: route A (same as Iteration 2a).
For disagree questions: use silver B/C labels (cheapest correct retrieval strategy).
  - If oner_qa correct -> B
  - Else if ircot_qa correct -> C
  - Else -> B (default; neither retrieval works, answer wrong regardless)

This measures the maximum F1 achievable if Clf2 were perfect,
establishing the ceiling for Iteration 3 improvements.

Usage:
    python classifier/postprocess/predict_complexity_oracle_ceiling.py flan_t5_xl
    python classifier/postprocess/predict_complexity_oracle_ceiling.py flan_t5_xxl
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

# BM25 retrieval counts differ per model family
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


def load_strategy_predictions(model_name, split="test"):
    nor_preds = {}
    oner_preds = {}
    oner_bm25 = ONER_BM25[model_name]
    base = f"predictions/{split}"
    for ds in DATASETS:
        nor_dir = f"nor_qa_{model_name}_{ds}____prompt_set_1"
        oner_dir = f"oner_qa_{model_name}_{ds}____prompt_set_1___bm25_retrieval_count__{oner_bm25}___distractor_count__1"
        sub = "test_subsampled" if split == "test" else "dev_500_subsampled"
        nor_file = os.path.join(base, nor_dir, f"prediction__{ds}_to_{ds}__{sub}.json")
        oner_file = os.path.join(base, oner_dir, f"prediction__{ds}_to_{ds}__{sub}.json")
        nor_preds.update(load_json(nor_file))
        oner_preds.update(load_json(oner_file))
    return nor_preds, oner_preds


def compute_agreement(nor_preds, oner_preds, qids):
    results = {}
    for qid in qids:
        nor_raw = nor_preds[qid]
        oner_raw = oner_preds[qid]
        if isinstance(nor_raw, list):
            nor_raw = nor_raw[0] if nor_raw else ""
        if isinstance(oner_raw, list):
            oner_raw = oner_raw[0] if oner_raw else ""
        nor_raw = str(nor_raw)
        oner_raw = str(oner_raw)
        nor_extracted = answer_extractor(nor_raw)
        oner_extracted = answer_extractor(oner_raw)
        nor_norm = normalize_answer(nor_extracted)
        oner_norm = normalize_answer(oner_extracted)
        agree = bool(nor_norm and oner_norm and nor_norm == oner_norm)
        results[qid] = {"agree": agree}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, choices=("flan_t5_xl", "flan_t5_xxl", "gpt"))
    args = parser.parse_args()

    m = args.model_name

    # Load predict.json for qid -> dataset mapping
    predict_data = load_json("classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json")
    qid_to_dataset = {item["id"]: item["dataset_name"] for item in predict_data}
    all_qids = list(qid_to_dataset.keys())

    # Load nor_qa and oner_qa predictions for agreement
    print("Loading nor_qa and oner_qa predictions...")
    nor_preds, oner_preds = load_strategy_predictions(m, split="test")

    # Compute agreement
    print("Computing answer agreement...")
    agreement = compute_agreement(nor_preds, oner_preds, all_qids)
    agree_count = sum(1 for v in agreement.values() if v["agree"])
    disagree_count = len(all_qids) - agree_count
    print(f"Agreement: {agree_count}/{len(all_qids)} = {agree_count/len(all_qids):.3f}")
    print(f"R-routed (disagree): {disagree_count}")

    # Load silver labels for test set (valid.json)
    silver_path = f"classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{m}/silver/valid.json"
    silver_data = load_json(silver_path)
    silver_by_qid = {d["id"]: d for d in silver_data}
    print(f"Silver labels loaded: {len(silver_data)} items")

    # For R-routed questions, determine oracle B/C
    # Priority: B (cheaper) if oner_qa works, else C if ircot_qa works, else B default
    merged = {}
    oracle_stats = {"agree_A": 0, "oracle_B": 0, "oracle_C": 0, "no_correct_retrieval": 0}

    for qid in all_qids:
        ds = qid_to_dataset[qid]
        if agreement[qid]["agree"]:
            merged[qid] = {"prediction": "A", "dataset_name": ds}
            oracle_stats["agree_A"] += 1
        else:
            silver = silver_by_qid.get(qid)
            if silver and "one" in silver["total_answer"]:
                merged[qid] = {"prediction": "B", "dataset_name": ds}
                oracle_stats["oracle_B"] += 1
            elif silver and "multiple" in silver["total_answer"]:
                merged[qid] = {"prediction": "C", "dataset_name": ds}
                oracle_stats["oracle_C"] += 1
            else:
                # No retrieval strategy works (or not in silver labels) -> B default
                merged[qid] = {"prediction": "B", "dataset_name": ds}
                oracle_stats["no_correct_retrieval"] += 1

    print(f"\nOracle routing stats:")
    print(f"  Agree -> A: {oracle_stats['agree_A']}")
    print(f"  Oracle -> B: {oracle_stats['oracle_B']}")
    print(f"  Oracle -> C: {oracle_stats['oracle_C']}")
    print(f"  No correct retrieval (default B): {oracle_stats['no_correct_retrieval']}")

    label_counts = Counter(v["prediction"] for v in merged.values())
    print(f"  Final A/B/C: {dict(sorted(label_counts.items()))}")

    # Load step numbers for ircot
    ircot_bm25 = IRCOT_BM25[m]
    total_step_num = {}
    consolidated = os.path.join("predictions", "test", f"ircot_qa_{m}", "total", "stepNum.json")
    if os.path.exists(consolidated):
        total_step_num = load_json(consolidated)
    else:
        for ds in DATASETS:
            sn_path = os.path.join(
                "predictions", "test",
                f"ircot_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__{ircot_bm25}___distractor_count__1",
                "stepNum.json"
            )
            if os.path.exists(sn_path):
                total_step_num.update(load_json(sn_path))

    # Build QA prediction files per dataset
    oner_bm25 = ONER_BM25[m]
    dataName_to_files = {}
    for ds in DATASETS:
        dataName_to_files[ds] = {
            'C': os.path.join("predictions", "test",
                              f'ircot_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__{ircot_bm25}___distractor_count__1',
                              f'prediction__{ds}_to_{ds}__test_subsampled.json'),
            'B': os.path.join("predictions", "test",
                              f'oner_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__{oner_bm25}___distractor_count__1',
                              f'prediction__{ds}_to_{ds}__test_subsampled.json'),
            'A': os.path.join("predictions", "test",
                              f'nor_qa_{m}_{ds}____prompt_set_1',
                              f'prediction__{ds}_to_{ds}__test_subsampled.json'),
        }

    output_path = f"predictions/classifier/t5-large/{m}/split_agreement_oracle/"
    print(f"\nWriting predictions to {output_path}")

    total_steps_all = 0
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
        total_steps_all += total_steps
        print(f"  {data_name}: A={a_count}, B={b_count}, C={c_count}, steps={total_steps}")

        save_json(os.path.join(output_path, data_name, f"{data_name}.json"), qid_to_pred)
        save_json(os.path.join(output_path, data_name, f"{data_name}_option.json"), qid_to_pred_option)

    save_json(os.path.join(output_path, "oracle_stats.json"), oracle_stats)
    print(f"\nTotal retrieval steps: {total_steps_all}")
    print(f"Run evaluation: python evaluate_final_acc.py --pred_path {output_path}")


if __name__ == "__main__":
    main()
