"""
Two-stage classifier routing script.

Stage 1 (no_ret_vs_ret): predicts A (no retrieval) vs R (retrieval needed)
Stage 2 (single_vs_multi): for R predictions, predicts B (single-step) vs C (multi-step)

Merges into final A/B/C labels, then routes each test question to the
corresponding QA strategy's pre-computed answer.
"""

import json
import os
import sys
import argparse
from postprocess_utils import load_json, save_json, save_prediction_with_classified_label

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, choices=("flan_t5_xl", "flan_t5_xxl"))
parser.add_argument("--no_ret_vs_ret_file", type=str, required=True,
                    help="Path to no_ret_vs_ret predict/dict_id_pred_results.json")
parser.add_argument("--single_vs_multi_file", type=str, required=True,
                    help="Path to single_vs_multi predict/dict_id_pred_results.json")
parser.add_argument("--output_path", type=str, required=True,
                    help="Output directory for routed predictions")
args = parser.parse_args()

# --- Load both classifier predictions ---
clf1 = load_json(args.no_ret_vs_ret_file)
clf2 = load_json(args.single_vs_multi_file)

# --- Merge into A/B/C ---
merged = {}
for qid in clf1:
    pred1 = clf1[qid]['prediction']
    dataset_name = clf1[qid]['dataset_name']
    if pred1 == 'A':
        final_label = 'A'
    else:  # R
        final_label = clf2[qid]['prediction']  # B or C
    merged[qid] = {
        'prediction': final_label,
        'answer': '',
        'dataset_name': dataset_name,
    }

# Print label distribution
from collections import Counter
label_counts = Counter(v['prediction'] for v in merged.values())
print(f"Merged label distribution: {dict(sorted(label_counts.items()))}")

# --- QA prediction file paths ---
m = args.model_name

dataName_to_multi_one_zero_file = {}
for dataset in ['musique', 'hotpotqa', '2wikimultihopqa', 'nq', 'trivia', 'squad']:
    dataName_to_multi_one_zero_file[dataset] = {
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

# --- stepNum handling ---
# xl has a consolidated total/stepNum.json; xxl has per-dataset stepNum files
if os.path.exists(os.path.join("predictions", "test", f"ircot_qa_{m}", "total", "stepNum.json")):
    total_step_num = load_json(os.path.join("predictions", "test", f"ircot_qa_{m}", "total", "stepNum.json"))
else:
    # merge per-dataset stepNum files
    total_step_num = {}
    for dataset in ['musique', 'hotpotqa', '2wikimultihopqa', 'nq', 'trivia', 'squad']:
        sn_path = os.path.join("predictions", "test",
                               f'ircot_qa_{m}_{dataset}____prompt_set_1___bm25_retrieval_count__6___distractor_count__1',
                               'stepNum.json')
        if os.path.exists(sn_path):
            total_step_num.update(load_json(sn_path))
    print(f"Merged per-dataset stepNum files: {len(total_step_num)} entries")

# --- Route predictions ---
output_path = args.output_path

for data_name in dataName_to_multi_one_zero_file:
    qid_to_pred = {}
    qid_to_pred_option = {}
    total_stepNum = 0

    for qid, info in merged.items():
        if info['dataset_name'] != data_name:
            continue

        predicted_option = info['prediction']

        if predicted_option == 'C':
            stepNum = total_step_num.get(qid, 0)
        elif predicted_option == 'B':
            stepNum = 1
        else:  # A
            stepNum = 0

        pred_file = dataName_to_multi_one_zero_file[data_name][predicted_option]
        all_preds = load_json(pred_file)
        pred = all_preds[qid]

        qid_to_pred[qid] = pred
        qid_to_pred_option[qid] = {
            'prediction': pred,
            'option': predicted_option,
            'stepNum': stepNum,
        }
        total_stepNum += stepNum

    print(f'============== {data_name}')
    save_json(os.path.join(output_path, data_name, f'{data_name}.json'), qid_to_pred)
    save_json(os.path.join(output_path, data_name, f'{data_name}_option.json'), qid_to_pred_option)
    print(f'StepNum {data_name}: {total_stepNum}')
    print(f'  A: {sum(1 for v in qid_to_pred_option.values() if v["option"]=="A")}, '
          f'B: {sum(1 for v in qid_to_pred_option.values() if v["option"]=="B")}, '
          f'C: {sum(1 for v in qid_to_pred_option.values() if v["option"]=="C")}')
