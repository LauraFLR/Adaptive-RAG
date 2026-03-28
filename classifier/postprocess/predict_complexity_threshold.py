"""
End-to-end QA routing with a custom Clf1 threshold.

Loads the Clf1 model, runs inference on predict.json at a given threshold,
merges with Clf2 argmax predictions, routes to QA strategy answers,
and saves routed predictions for evaluation.

Usage:
    python classifier/postprocess/predict_complexity_threshold.py flan_t5_xl \
        --clf1_model_path classifier/outputs/.../no_ret_vs_ret/epoch/20/.../ \
        --clf2_pred_file  classifier/outputs/.../single_vs_multi/epoch/35/.../predict/dict_id_pred_results.json \
        --predict_file    classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json \
        --threshold 0.35 \
        --output_path     predictions/classifier/t5-large/flan_t5_xl/split_thresh035/
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from postprocess_utils import load_json, save_json


def get_clf1_predictions_with_threshold(model_path, predict_file, threshold, labels, batch_size, device):
    """Run Clf1 inference and apply threshold for A prediction."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    raw = load_dataset("json", data_files={"validation": predict_file})
    examples = raw["validation"]

    def tokenize(batch):
        inputs = tokenizer(
            [q.strip() for q in batch["question"]],
            max_length=384, truncation=True, padding="max_length",
        )
        targets = tokenizer(
            text_target=[a if a else "A" for a in batch["answer"]],  # predict.json has empty answers
            max_length=30, truncation=True, padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    ds = examples.map(tokenize, batched=True, remove_columns=examples.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    label_token_ids = [tokenizer(label).input_ids[0] for label in labels]
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            out = model.generate(
                input_ids=batch_gpu["input_ids"],
                attention_mask=batch_gpu["attention_mask"],
                return_dict_in_generate=True,
                output_scores=True,
                max_length=30,
            )
            scores = out.scores[0]
            logits = torch.stack([scores[:, tid] for tid in label_token_ids], dim=1)
            probs = torch.nn.functional.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    # Apply threshold: if P(A) >= threshold, predict A; else predict R
    predictions = {}
    for i in range(len(examples)):
        qid = examples[i]["id"]
        dataset_name = examples[i]["dataset_name"]
        p_a = all_probs[i, 0]
        pred = "A" if p_a >= threshold else "R"
        predictions[qid] = {
            "prediction": pred,
            "prob_A": float(p_a),
            "dataset_name": dataset_name,
        }

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, choices=("flan_t5_xl", "flan_t5_xxl"))
    parser.add_argument("--clf1_model_path", type=str, required=True)
    parser.add_argument("--clf2_pred_file", type=str, required=True)
    parser.add_argument("--predict_file", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = args.model_name

    # Step 1: Run Clf1 with threshold
    print(f"Running Clf1 inference at threshold {args.threshold}...")
    clf1_preds = get_clf1_predictions_with_threshold(
        args.clf1_model_path, args.predict_file, args.threshold,
        labels=["A", "R"], batch_size=args.batch_size, device=device,
    )
    clf1_dist = Counter(v["prediction"] for v in clf1_preds.values())
    print(f"Clf1 distribution: {dict(sorted(clf1_dist.items()))}")

    # Step 2: Load Clf2 argmax predictions
    clf2 = load_json(args.clf2_pred_file)

    # Step 3: Merge into A/B/C
    merged = {}
    for qid, info in clf1_preds.items():
        if info["prediction"] == "A":
            final_label = "A"
        else:
            final_label = clf2[qid]["prediction"]  # B or C
        merged[qid] = {
            "prediction": final_label,
            "dataset_name": info["dataset_name"],
        }

    label_counts = Counter(v["prediction"] for v in merged.values())
    print(f"Merged A/B/C distribution: {dict(sorted(label_counts.items()))}")

    # Step 4: Route to QA predictions
    dataName_to_files = {}
    for dataset in ['musique', 'hotpotqa', '2wikimultihopqa', 'nq', 'trivia', 'squad']:
        dataName_to_files[dataset] = {
            'C': os.path.join("predictions", "test",
                              f'ircot_qa_{m}_{dataset}____prompt_set_1___bm25_retrieval_count__6___distractor_count__1',
                              f'prediction__{dataset}_to_{dataset}__test_subsampled.json'),
            'B': os.path.join("predictions", "test",
                              f'oner_qa_{m}_{dataset}____prompt_set_1___bm25_retrieval_count__15___distractor_count__1',
                              f'prediction__{dataset}_to_{dataset}__test_subsampled.json'),
            'A': os.path.join("predictions", "test",
                              f'nor_qa_{m}_{dataset}____prompt_set_1',
                              f'prediction__{dataset}_to_{dataset}__test_subsampled.json'),
        }

    # Load stepNum
    total_step_num = {}
    for dataset in ['musique', 'hotpotqa', '2wikimultihopqa', 'nq', 'trivia', 'squad']:
        sn_path = os.path.join("predictions", "test",
                               f'ircot_qa_{m}_{dataset}____prompt_set_1___bm25_retrieval_count__6___distractor_count__1',
                               'stepNum.json')
        if os.path.exists(sn_path):
            total_step_num.update(load_json(sn_path))
    consolidated = os.path.join("predictions", "test", f"ircot_qa_{m}", "total", "stepNum.json")
    if os.path.exists(consolidated):
        total_step_num = load_json(consolidated)

    output_path = args.output_path

    for data_name in dataName_to_files:
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

        print(f"  {data_name}: A={a_count}, B={b_count}, C={c_count}, steps={total_steps}")
        save_json(os.path.join(output_path, data_name, f"{data_name}.json"), qid_to_pred)
        save_json(os.path.join(output_path, data_name, f"{data_name}_option.json"), qid_to_pred_option)

    # Save config
    save_json(os.path.join(output_path, "config.json"), {
        "model_name": m,
        "clf1_model_path": args.clf1_model_path,
        "clf2_pred_file": args.clf2_pred_file,
        "threshold": args.threshold,
        "clf1_distribution": dict(sorted(clf1_dist.items())),
        "merged_distribution": dict(sorted(label_counts.items())),
    })

    print(f"\nRouted predictions saved to {output_path}")
    print(f"Run evaluation with: python evaluate_final_acc.py --pred_path {output_path}")


if __name__ == "__main__":
    main()
