#!/usr/bin/env python3
"""
SymRAG-inspired structural κ(q) complexity score for Gate 2 (B vs C).

Implements the structural heuristic component of SymRAG's κ(q) complexity
score (Hakim et al., 2025, Definition 1, Appendix A.1.1) as a training-free
alternative to the T5-Large Clf2 classifier.

We omit the attention-based A(q) term (which requires a bert-tiny forward
pass) and use only the structural components L(q) and S_H(q):

    κ(q) = w_L · L(q) · (1 + S_H(q))

where:
    L(q)   = token_len / max_token_len_in_dataset   (normalized query length)
    S_H(q) = w_sh1 · N_ents(q)/|q| + w_sh2 · N_hops(q)/|q|

Published weights: w_L = 1.0, w_sh1 = 0.05, w_sh2 = 0.10.

Usage:
    python classifier/postprocess/predict_complexity_kappa.py flan_t5_xl \\
        --use_agreement_gate \\
        --output_path predictions/classifier/t5-large/flan_t5_xl/kappa_structural/

    python classifier/postprocess/predict_complexity_kappa.py gpt \\
        --use_agreement_gate --tune_threshold \\
        --valid_file classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/gpt/silver/single_vs_multi/valid.json \\
        --output_path predictions/classifier/t5-large/gpt/kappa_tuned/
"""

import argparse
import json
import os
import re
import string
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from postprocess_utils import load_json, save_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASETS = ["musique", "hotpotqa", "2wikimultihopqa", "nq", "trivia", "squad"]

ONER_BM25 = {"flan_t5_xl": 15, "flan_t5_xxl": 15, "gpt": 6}
IRCOT_BM25 = {"flan_t5_xl": 6, "flan_t5_xxl": 6, "gpt": 3}

# SymRAG published weights (minus A(q) attention term)
W_L = 1.0
W_SH1 = 0.05   # entity density weight
W_SH2 = 0.10   # hop indicator density weight

# ---------------------------------------------------------------------------
# Bridging-phrase patterns (copied from clf2_feature_probe.py)
# ---------------------------------------------------------------------------
# These regex patterns target syntactic structures common in multi-hop
# questions that chain two information needs together.  Each pattern is
# case-insensitive and matches anywhere in the question string.
#
#   1. Relative clauses linking two entities
#   2. Possessive chains
#   3. Temporal/causal subordination across entities
#   4. Demonstrative reference to a prior fact
#   5. Explicit comparison bridging two look-ups
#   6. Nested wh-questions
# ---------------------------------------------------------------------------
BRIDGE_PATTERNS = [
    # Relative-clause bridges (who/where/which/that + verb)
    r"\b(?:who|where|which|that)\s+(?:was|were|is|are|did|had|has|does)\b",
    # Double possessive  ("X's … Y's …")
    r"\w+'s\s+\w+(?:\s+\w+){0,5}\s+\w+'s",
    # Temporal subordination before a wh-word
    r"\b(?:before|after|when|while)\b.{3,60}\b(?:who|what|where|which)\b",
    # Demonstrative back-reference ("that country", "this person")
    r"\b(?:that|this|those|these)\s+(?:country|city|person|team|company|film|movie|album|book|organization|university|school)\b",
    # Explicit comparison linking two entities
    r"\b(?:both)\b.{1,40}\band\b",
    r"\bbetween\b.{1,40}\band\b",
    # Nested wh-questions  ("What is the X of the Y that …")
    r"\bof\s+the\s+\w+\s+(?:who|that|which|where)\b",
]
_BRIDGE_RES = [re.compile(p, re.IGNORECASE) for p in BRIDGE_PATTERNS]

# ---------------------------------------------------------------------------
# Answer normalization (same as predict_complexity_agreement.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

def load_strategy_predictions(model_name, split="test"):
    """Load nor_qa and oner_qa predictions for all datasets."""
    nor_preds = {}
    oner_preds = {}
    oner_bm25 = ONER_BM25[model_name]
    base = f"predictions/{split}"
    for ds in DATASETS:
        nor_dir = f"nor_qa_{model_name}_{ds}____prompt_set_1"
        oner_dir = (f"oner_qa_{model_name}_{ds}____prompt_set_1"
                    f"___bm25_retrieval_count__{oner_bm25}___distractor_count__1")
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
        results[qid] = {
            "agree": agree,
            "nor_answer": nor_raw,
            "oner_answer": oner_raw,
        }
    return results


# ---------------------------------------------------------------------------
# Feature extraction & kappa score
# ---------------------------------------------------------------------------

def extract_features(questions, nlp):
    """Extract token_len, entity_count, hop_indicator_count per question."""
    token_lens = []
    entity_counts = []
    hop_counts = []

    for doc in nlp.pipe(questions, batch_size=256):
        text = doc.text
        token_lens.append(len(text.split()))
        entity_counts.append(len(doc.ents))
        hop_counts.append(sum(1 for pat in _BRIDGE_RES if pat.search(text)))

    return token_lens, entity_counts, hop_counts


def compute_kappa(token_lens, entity_counts, hop_counts):
    """Compute κ(q) for each question.

    κ(q) = w_L · L(q) · (1 + S_H(q))
    L(q) = token_len / max_token_len
    S_H(q) = w_sh1 * (entity_count / token_len) + w_sh2 * (hop_count / token_len)
    """
    token_lens = np.array(token_lens, dtype=float)
    entity_counts = np.array(entity_counts, dtype=float)
    hop_counts = np.array(hop_counts, dtype=float)

    max_len = token_lens.max() if token_lens.max() > 0 else 1.0
    L = token_lens / max_len

    # Avoid division by zero for empty questions
    safe_lens = np.where(token_lens > 0, token_lens, 1.0)
    S_H = W_SH1 * (entity_counts / safe_lens) + W_SH2 * (hop_counts / safe_lens)

    kappa = W_L * L * (1.0 + S_H)
    return kappa


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(valid_path, nlp):
    """Tune kappa threshold on validation data.

    Returns (best_acc_threshold, best_acc, best_f1_threshold, best_f1, stats_dict).
    """
    from sklearn.metrics import f1_score

    with open(valid_path) as f:
        data = json.load(f)

    bc_items = [item for item in data if item.get("answer") in ("B", "C")]
    if not bc_items:
        sys.exit(f"ERROR: no B/C items in {valid_path}")

    questions = [item["question"] for item in bc_items]
    labels = np.array([1 if item["answer"] == "C" else 0 for item in bc_items])

    token_lens, entity_counts, hop_counts = extract_features(questions, nlp)
    kappa = compute_kappa(token_lens, entity_counts, hop_counts)

    lo = np.percentile(kappa, 5)
    hi = np.percentile(kappa, 95)
    thresholds = np.linspace(lo, hi, 100)

    best_acc = -1.0
    best_acc_t = lo
    best_f1 = -1.0
    best_f1_t = lo

    for t in thresholds:
        preds = (kappa >= t).astype(int)
        acc = (preds == labels).mean()
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if acc > best_acc:
            best_acc = acc
            best_acc_t = float(t)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_t = float(t)

    b_mask = labels == 0
    c_mask = labels == 1
    best_preds = (kappa >= best_acc_t).astype(int)
    b_acc = (best_preds[b_mask] == labels[b_mask]).mean() if b_mask.any() else 0.0
    c_acc = (best_preds[c_mask] == labels[c_mask]).mean() if c_mask.any() else 0.0

    print(f"Tuned threshold: {best_acc_t:.4f}, validation accuracy: {best_acc:.4f}, "
          f"val B-acc: {b_acc:.4f}, val C-acc: {c_acc:.4f}")
    print(f"Best macro-F1: {best_f1:.4f} at threshold {best_f1_t:.4f}"
          + (" (same as accuracy-optimal)" if abs(best_f1_t - best_acc_t) < 1e-9
             else f" (differs from accuracy-optimal {best_acc_t:.4f})"))

    stats = {
        "best_acc_threshold": best_acc_t,
        "best_accuracy": float(best_acc),
        "val_B_accuracy": float(b_acc),
        "val_C_accuracy": float(c_acc),
        "best_f1_threshold": best_f1_t,
        "best_macro_f1": float(best_f1),
        "n_val_samples": len(bc_items),
        "n_B": int(b_mask.sum()),
        "n_C": int(c_mask.sum()),
    }
    return best_acc_t, best_acc, best_f1_t, best_f1, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SymRAG-inspired structural κ(q) score for Gate 2 (B vs C)."
    )
    parser.add_argument(
        "model_name", type=str,
        choices=("flan_t5_xl", "flan_t5_xxl", "gpt"),
    )
    parser.add_argument(
        "--clf1_pred_file", type=str, default=None,
        help="Path to Gate 1 predictions JSON (dict of qid -> {prediction: A/R}).",
    )
    parser.add_argument(
        "--use_agreement_gate", action="store_true", default=False,
        help="Compute Gate 1 via nor_qa/oner_qa agreement instead of --clf1_pred_file.",
    )
    parser.add_argument(
        "--clf2_pred_file", type=str, default=None,
        help="Path to T5-Large Clf2 predictions — comparison baseline only, not used for routing.",
    )
    parser.add_argument(
        "--predict_file", type=str, default=None,
        help="Path to predict.json. Default: auto-detect from dataset bundle.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
    )
    parser.add_argument(
        "--kappa_threshold", type=float, default=0.5,
    )
    parser.add_argument(
        "--tune_threshold", action="store_true", default=False,
        help="Tune threshold on --valid_file before predicting.",
    )
    parser.add_argument(
        "--valid_file", type=str, default=None,
        help="Clf2 validation JSON with B/C labels (required when --tune_threshold).",
    )
    args = parser.parse_args()

    m = args.model_name

    if not args.use_agreement_gate and args.clf1_pred_file is None:
        parser.error("Provide --clf1_pred_file or --use_agreement_gate.")
    if args.tune_threshold and args.valid_file is None:
        parser.error("--tune_threshold requires --valid_file.")

    # --- Resolve predict.json path ------------------------------------
    predict_path = args.predict_file
    if predict_path is None:
        predict_path = os.path.join(
            "classifier", "data", "musique_hotpot_wiki2_nq_tqa_sqd", "predict.json"
        )
    predict_data = load_json(predict_path)
    qid_to_dataset = {item["id"]: item["dataset_name"] for item in predict_data}
    qid_to_question = {item["id"]: item["question"] for item in predict_data}
    all_qids = list(qid_to_dataset.keys())
    print(f"[data]  {len(all_qids)} questions from {predict_path}")

    # --- Load spaCy ---------------------------------------------------
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        sys.exit(
            "ERROR: spaCy model 'en_core_web_sm' not found.\n"
            "Install with:  python -m spacy download en_core_web_sm"
        )

    # --- Threshold tuning (optional) ----------------------------------
    threshold = args.kappa_threshold
    threshold_tuned = False
    tuning_stats = None

    if args.tune_threshold:
        print("\n[tune]  Tuning threshold on validation data...")
        best_acc_t, best_acc, best_f1_t, best_f1, tuning_stats = tune_threshold(
            args.valid_file, nlp
        )
        threshold = best_acc_t
        threshold_tuned = True
        print(f"[tune]  Using tuned threshold: {threshold:.4f}")
    else:
        print(f"[cfg]   Using default threshold: {threshold:.4f}")

    # --- Gate 1 -------------------------------------------------------
    print("\n[gate1] Loading predictions...")
    nor_preds, oner_preds = load_strategy_predictions(m, split="test")

    if args.use_agreement_gate:
        print("[gate1] Computing nor_qa/oner_qa agreement...")
        agreement = compute_agreement(nor_preds, oner_preds, all_qids)
        gate1 = {}
        for qid in all_qids:
            gate1[qid] = "A" if agreement[qid]["agree"] else "R"
    else:
        print(f"[gate1] Loading Clf1 predictions from {args.clf1_pred_file}")
        clf1_data = load_json(args.clf1_pred_file)
        gate1 = {}
        for qid in all_qids:
            pred = clf1_data[qid]
            if isinstance(pred, dict):
                pred = pred["prediction"]
            gate1[qid] = pred

    a_count = sum(1 for v in gate1.values() if v == "A")
    r_count = len(gate1) - a_count
    print(f"[gate1] A={a_count}, R={r_count}")

    # --- Feature extraction for R-routed questions --------------------
    r_qids = [qid for qid in all_qids if gate1[qid] != "A"]
    r_questions = [qid_to_question[qid] for qid in r_qids]

    print(f"\n[feat]  Extracting features for {len(r_qids)} R-routed questions...")
    token_lens, entity_counts, hop_counts = extract_features(r_questions, nlp)
    kappa_scores = compute_kappa(token_lens, entity_counts, hop_counts)

    qid_to_kappa = dict(zip(r_qids, kappa_scores.tolist()))

    kappa_arr = np.array(list(qid_to_kappa.values()))
    print(f"[feat]  κ stats: mean={kappa_arr.mean():.4f}, std={kappa_arr.std():.4f}, "
          f"min={kappa_arr.min():.4f}, max={kappa_arr.max():.4f}")

    # --- Gate 2: apply κ threshold ------------------------------------
    print(f"\n[gate2] Routing with threshold={threshold:.4f}...")
    merged = {}
    for qid in all_qids:
        ds = qid_to_dataset[qid]
        if gate1[qid] == "A":
            merged[qid] = {"prediction": "A", "dataset_name": ds}
        else:
            k = qid_to_kappa[qid]
            label = "C" if k >= threshold else "B"
            merged[qid] = {"prediction": label, "dataset_name": ds}

    label_counts = Counter(v["prediction"] for v in merged.values())
    print(f"[route] A={label_counts.get('A', 0)}, B={label_counts.get('B', 0)}, "
          f"C={label_counts.get('C', 0)}")

    # --- Comparison with T5-Large Clf2 (if provided) ------------------
    if args.clf2_pred_file:
        clf2_data = load_json(args.clf2_pred_file)
        agree = 0
        for qid in r_qids:
            clf2_pred = clf2_data.get(qid, {})
            if isinstance(clf2_pred, dict):
                clf2_pred = clf2_pred.get("prediction", "")
            kappa_pred = merged[qid]["prediction"]
            if kappa_pred == clf2_pred:
                agree += 1
        print(f"[cmp]   κ vs Clf2 agreement on R-routed: {agree}/{len(r_qids)} "
              f"= {agree / len(r_qids):.3f}" if r_qids else "[cmp]   No R-routed questions")

    # --- Load step numbers for ircot routing --------------------------
    ircot_bm25 = IRCOT_BM25[m]
    total_step_num = {}
    consolidated = os.path.join("predictions", "test", f"ircot_qa_{m}", "total", "stepNum.json")
    if os.path.exists(consolidated):
        total_step_num = load_json(consolidated)
    else:
        for ds in DATASETS:
            sn_path = os.path.join(
                "predictions", "test",
                f"ircot_qa_{m}_{ds}____prompt_set_1"
                f"___bm25_retrieval_count__{ircot_bm25}___distractor_count__1",
                "stepNum.json",
            )
            if os.path.exists(sn_path):
                total_step_num.update(load_json(sn_path))

    # --- Build per-dataset prediction files ---------------------------
    oner_bm25 = ONER_BM25[m]
    dataName_to_files = {}
    for ds in DATASETS:
        dataName_to_files[ds] = {
            "C": os.path.join(
                "predictions", "test",
                f"ircot_qa_{m}_{ds}____prompt_set_1"
                f"___bm25_retrieval_count__{ircot_bm25}___distractor_count__1",
                f"prediction__{ds}_to_{ds}__test_subsampled.json",
            ),
            "B": os.path.join(
                "predictions", "test",
                f"oner_qa_{m}_{ds}____prompt_set_1"
                f"___bm25_retrieval_count__{oner_bm25}___distractor_count__1",
                f"prediction__{ds}_to_{ds}__test_subsampled.json",
            ),
            "A": os.path.join(
                "predictions", "test",
                f"nor_qa_{m}_{ds}____prompt_set_1",
                f"prediction__{ds}_to_{ds}__test_subsampled.json",
            ),
        }

    output_path = args.output_path
    total_steps_all = 0
    per_dataset_stats = {}

    print(f"\n[out]   Writing predictions to {output_path}")
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
            if qid in qid_to_kappa:
                qid_to_pred_option[qid]["kappa"] = round(qid_to_kappa[qid], 6)
            total_steps += step_num

        a_ct = sum(1 for v in qid_to_pred_option.values() if v["option"] == "A")
        b_ct = sum(1 for v in qid_to_pred_option.values() if v["option"] == "B")
        c_ct = sum(1 for v in qid_to_pred_option.values() if v["option"] == "C")

        per_dataset_stats[data_name] = {
            "A": a_ct, "B": b_ct, "C": c_ct, "steps": total_steps,
        }
        total_steps_all += total_steps
        print(f"  {data_name}: A={a_ct}, B={b_ct}, C={c_ct}, steps={total_steps}")

        save_json(os.path.join(output_path, data_name, f"{data_name}.json"), qid_to_pred)
        save_json(
            os.path.join(output_path, data_name, f"{data_name}_option.json"),
            qid_to_pred_option,
        )

    # --- Routing stats JSON -------------------------------------------
    routing_stats = {
        "method": "symrag_kappa_structural",
        "model_name": m,
        "threshold_used": threshold,
        "threshold_tuned": threshold_tuned,
        "kappa_stats": {
            "mean": float(kappa_arr.mean()),
            "std": float(kappa_arr.std()),
            "min": float(kappa_arr.min()),
            "max": float(kappa_arr.max()),
        },
        "routing_counts": {
            "A": label_counts.get("A", 0),
            "B": label_counts.get("B", 0),
            "C": label_counts.get("C", 0),
        },
        "total_questions": len(all_qids),
        "total_steps": total_steps_all,
        "per_dataset": per_dataset_stats,
        "symrag_weights": {"w_L": W_L, "w_sh1": W_SH1, "w_sh2": W_SH2},
        "note": "A(q) attention term omitted; structural heuristic only",
    }
    if tuning_stats:
        routing_stats["tuning_stats"] = tuning_stats
    if args.clf2_pred_file:
        routing_stats["clf2_comparison_file"] = args.clf2_pred_file

    save_json(os.path.join(output_path, "routing_stats.json"), routing_stats)

    print(f"\nRouted predictions saved to {output_path}")
    print(f"Run evaluation with: python evaluate_final_acc.py --pred_path {output_path}")


if __name__ == "__main__":
    main()
