#!/usr/bin/env python3
"""
Iteration 3c: Embedding-based Clf2 (Gate 2, B vs C routing).

Replaces the T5-Large text classifier with an MLP on frozen sentence
embeddings from all-MiniLM-L6-v2 (384-dim, CPU-only).

Produces a drop-in dict_id_pred_results.json compatible with
predict_complexity_agreement.py.

Usage:
    python classifier/postprocess/clf2_embedding_clf.py --model flan_t5_xl
    python classifier/postprocess/clf2_embedding_clf.py --model flan_t5_xxl
    python classifier/postprocess/clf2_embedding_clf.py --model flan_t5_xl --run_evaluation
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print("Install with:  pip install sentence-transformers")
    sys.exit(1)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
ENCODER_MODEL = "all-MiniLM-L6-v2"
PREDICT_JSON = os.path.join(
    REPO_ROOT, "classifier", "data", "musique_hotpot_wiki2_nq_tqa_sqd", "predict.json"
)
AGREEMENT_SCRIPT = os.path.join(
    REPO_ROOT, "classifier", "postprocess", "predict_complexity_agreement.py"
)
EVAL_SCRIPT = os.path.join(REPO_ROOT, "evaluate_final_acc.py")


def detect_train_path(model: str) -> str:
    base = os.path.join(
        REPO_ROOT, "classifier", "data", "musique_hotpot_wiki2_nq_tqa_sqd",
    )
    candidates = [
        os.path.join(base, model, "binary_silver_single_vs_multi", "train.json"),
        os.path.join(base, model, "silver", "single_vs_multi", "train.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    sys.exit(f"ERROR: training data not found for {model}.\nSearched: " + "\n  ".join(candidates))


def detect_t5_clf2_pred(model: str) -> str | None:
    """Find the Iter 2a T5-Large Clf2 prediction file for distribution comparison."""
    pattern = os.path.join(
        REPO_ROOT, "classifier", "outputs", "musique_hotpot_wiki2_nq_tqa_sqd",
        "model", "t5-large", model, "single_vs_multi", "epoch", "*", "*",
        "predict", "dict_id_pred_results.json",
    )
    # Exclude silver_only and feat variants
    files = [f for f in glob.glob(pattern) if "silver_only" not in f and "feat" not in f]
    if files:
        return sorted(files)[-1]  # latest
    # Try 2-level timestamp dirs
    pattern2 = os.path.join(
        REPO_ROOT, "classifier", "outputs", "musique_hotpot_wiki2_nq_tqa_sqd",
        "model", "t5-large", model, "single_vs_multi", "epoch", "*", "*", "*",
        "predict", "dict_id_pred_results.json",
    )
    files2 = [f for f in glob.glob(pattern2) if "silver_only" not in f and "feat" not in f]
    return sorted(files2)[-1] if files2 else None


def load_bc_data(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return [item for item in data if item.get("answer") in ("B", "C")]


def main():
    parser = argparse.ArgumentParser(description="Iter 3c: Embedding-based Clf2")
    parser.add_argument("--model", required=True, choices=("flan_t5_xl", "flan_t5_xxl"))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_evaluation", action="store_true",
                        help="Run agreement gate + evaluation after writing predictions")
    args = parser.parse_args()

    model = args.model

    # --- Output path ---
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            REPO_ROOT, "classifier", "outputs", "embedding_clf2", model, "predict",
        )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dict_id_pred_results.json")

    # =================================================================
    # 1. Load training data
    # =================================================================
    train_path = detect_train_path(model)
    print(f"[data]  {train_path}")
    bc_data = load_bc_data(train_path)
    questions_train = [item["question"] for item in bc_data]
    labels_train = [item["answer"] for item in bc_data]
    n_b = labels_train.count("B")
    n_c = labels_train.count("C")
    print(f"[data]  {len(bc_data)} B/C items  (B={n_b}, C={n_c})")

    # =================================================================
    # 2. Encode training questions
    # =================================================================
    print(f"[enc]   loading {ENCODER_MODEL} ...")
    encoder = SentenceTransformer(ENCODER_MODEL)
    emb_dim = encoder.get_sentence_embedding_dimension()
    print(f"[enc]   dim={emb_dim}")

    est = len(questions_train) / 500
    print(f"[enc]   encoding {len(questions_train)} training questions (est ~{est:.1f}s) ...")
    t0 = time.time()
    X_train_all = encoder.encode(questions_train, batch_size=64, show_progress_bar=True)
    print(f"[enc]   done in {time.time()-t0:.1f}s")

    y_train_all = np.array([1 if l == "C" else 0 for l in labels_train])

    # =================================================================
    # 3. Train MLP + LogReg on validation split, then retrain MLP on full
    # =================================================================
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.20, random_state=42, stratify=y_train_all,
    )

    # LogisticRegression baseline
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    lr.fit(X_tr, y_tr)
    lr_prob = lr.predict_proba(X_val)[:, 1]
    lr_pred = lr.predict(X_val)
    lr_auc = roc_auc_score(y_val, lr_prob)
    lr_acc = accuracy_score(y_val, lr_pred)

    # MLP
    mlp_val = MLPClassifier(
        hidden_layer_sizes=(256, 64), activation="relu", max_iter=500,
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )
    mlp_val.fit(X_tr, y_tr)
    mlp_prob = mlp_val.predict_proba(X_val)[:, 1]
    mlp_pred = mlp_val.predict(X_val)
    mlp_auc = roc_auc_score(y_val, mlp_prob)
    mlp_acc = accuracy_score(y_val, mlp_pred)

    print("\n========== VALIDATION (20% held-out) ==========")
    print(f"  LogReg:  AUC={lr_auc:.4f}  Acc={lr_acc:.4f}")
    print(f"  MLP:     AUC={mlp_auc:.4f}  Acc={mlp_acc:.4f}")
    print()
    print("  MLP classification report (val):")
    print(classification_report(y_val, mlp_pred, target_names=["B (single)", "C (multi)"]))

    # Retrain MLP on full training set for production predictions
    print("[train] retraining MLP on full training set ...")
    mlp_full = MLPClassifier(
        hidden_layer_sizes=(256, 64), activation="relu", max_iter=500,
        early_stopping=True, validation_fraction=0.1, random_state=42,
    )
    mlp_full.fit(X_train_all, y_train_all)
    train_acc = accuracy_score(y_train_all, mlp_full.predict(X_train_all))
    print(f"[train] full-set training accuracy: {train_acc:.4f}")

    # =================================================================
    # 4. Load and encode test questions
    # =================================================================
    print(f"\n[test]  loading {PREDICT_JSON}")
    with open(PREDICT_JSON) as f:
        predict_data = json.load(f)
    print(f"[test]  {len(predict_data)} test questions")
    print(f"[test]  sample entries:")
    for e in predict_data[:2]:
        print(f"        {json.dumps(e)}")

    test_questions = [item["question"] for item in predict_data]
    test_ids = [item["id"] for item in predict_data]
    test_datasets = [item["dataset_name"] for item in predict_data]

    print(f"[enc]   encoding {len(test_questions)} test questions ...")
    t0 = time.time()
    X_test = encoder.encode(test_questions, batch_size=64, show_progress_bar=True)
    print(f"[enc]   done in {time.time()-t0:.1f}s")

    # =================================================================
    # 5. Predict and write drop-in file
    # =================================================================
    test_preds = mlp_full.predict(X_test)
    label_map = {0: "B", 1: "C"}

    result = {}
    for qid, pred, ds in zip(test_ids, test_preds, test_datasets):
        result[qid] = {
            "prediction": label_map[pred],
            "answer": "",
            "dataset_name": ds,
        }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n[save]  {output_file}")

    # Distribution
    b_count = sum(1 for v in result.values() if v["prediction"] == "B")
    c_count = sum(1 for v in result.values() if v["prediction"] == "C")
    total = len(result)
    print(f"[dist]  Emb Clf2:  B={b_count} ({b_count/total:.1%})  C={c_count} ({c_count/total:.1%})  total={total}")

    # Compare to T5-Large Clf2
    t5_file = detect_t5_clf2_pred(model)
    if t5_file:
        with open(t5_file) as f:
            t5_pred = json.load(f)
        t5_b = sum(1 for v in t5_pred.values() if v["prediction"] == "B")
        t5_c = sum(1 for v in t5_pred.values() if v["prediction"] == "C")
        t5_total = len(t5_pred)
        print(f"[dist]  T5 Clf2:   B={t5_b} ({t5_b/t5_total:.1%})  C={t5_c} ({t5_c/t5_total:.1%})  total={t5_total}")
        # Agreement rate between the two classifiers
        agree = sum(1 for qid in result if qid in t5_pred and result[qid]["prediction"] == t5_pred[qid]["prediction"])
        print(f"[dist]  Agreement with T5 Clf2: {agree}/{total} ({agree/total:.1%})")
    else:
        print("[dist]  T5 Clf2 prediction file not found for comparison")

    # =================================================================
    # 6. Run end-to-end evaluation (optional)
    # =================================================================
    if args.run_evaluation:
        pred_output = os.path.join(
            REPO_ROOT, "predictions", "classifier", "t5-large", model,
            "split_agreement", "nor_oner_embclf2",
        )

        print(f"\n[eval]  Running agreement gate → {pred_output}")
        agreement_cmd = [
            sys.executable, AGREEMENT_SCRIPT, model,
            "--clf2_pred_file", output_file,
            "--predict_file", PREDICT_JSON,
            "--output_path", pred_output,
        ]
        print(f"[eval]  {' '.join(agreement_cmd)}")
        subprocess.run(agreement_cmd, cwd=REPO_ROOT, check=True)

        print(f"\n[eval]  Running evaluate_final_acc.py ...")
        eval_cmd = [sys.executable, EVAL_SCRIPT, "--pred_path", pred_output]
        print(f"[eval]  {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
