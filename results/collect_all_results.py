#!/usr/bin/env python3
"""Collect all iteration results (IT1–IT7) into a single JSON for thesis writing."""

import json
import glob
import os
import sys
from collections import Counter
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = "musique_hotpot_wiki2_nq_tqa_sqd"
MODELS = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
DATASETS = ["musique", "hotpotqa", "2wikimultihopqa", "nq", "trivia", "squad"]
CLF_BASE = os.path.join(REPO, "classifier/outputs", DATASET, "model/t5-large")
PRED_BASE = os.path.join(REPO, "predictions/classifier/t5-large")
DATA_BASE = os.path.join(REPO, "classifier/data", DATASET)

# ── helpers ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_best_epoch(gate_epoch_dir, valid_file, run_tag=None):
    """Return (best_pred_path, best_valid_path, best_epoch, best_acc)."""
    gold = {item["id"]: item["answer"] for item in load_json(valid_file)}
    KNOWN_TAGS = {"silver_only", "feat"}

    best_acc, best_pred, best_valid, best_epoch = -1, "", "", ""
    for vf in glob.glob(os.path.join(gate_epoch_dir, "**/valid/dict_id_pred_results.json"), recursive=True):
        run_dir = os.path.dirname(os.path.dirname(vf))
        rel = os.path.relpath(run_dir, gate_epoch_dir)
        parts = rel.split(os.sep)
        if run_tag:
            if run_tag not in parts:
                continue
        else:
            if any(p in KNOWN_TAGS for p in parts):
                continue
        pf = os.path.join(run_dir, "predict", "dict_id_pred_results.json")
        if not os.path.exists(pf):
            continue
        preds = load_json(vf)
        correct = sum(1 for qid, info in preds.items()
                      if gold.get(qid) == info.get("prediction"))
        acc = correct / len(preds) if preds else 0
        if acc > best_acc:
            best_acc = acc
            best_pred = pf
            best_valid = vf
            best_epoch = parts[0]
    return best_pred, best_valid, best_epoch, best_acc


def per_class_accuracy(valid_path):
    """Compute overall and per-class accuracy from a valid/dict_id_pred_results.json."""
    preds = load_json(valid_path)
    classes = sorted(set(v["answer"] for v in preds.values()))
    total_correct, total_n = 0, 0
    per_class = {}
    for cls in classes:
        items = {qid: v for qid, v in preds.items() if v["answer"] == cls}
        n = len(items)
        correct = sum(1 for v in items.values() if v["prediction"] == cls)
        per_class[cls] = {"accuracy": correct / n if n else 0, "n": n, "correct": correct}
        total_correct += correct
        total_n += n
    return {
        "overall_accuracy": total_correct / total_n if total_n else 0,
        "n_samples": total_n,
        "per_class": per_class,
    }


def qa_metrics(iter_dir, datasets=DATASETS):
    """Collect per-dataset EM, F1, count from eval_metic_result_acc.json."""
    results = {}
    for ds in datasets:
        metric_file = os.path.join(iter_dir, ds, "eval_metic_result_acc.json")
        if os.path.exists(metric_file):
            m = load_json(metric_file)
            results[ds] = {"em": m.get("em", m.get("acc")), "f1": m["f1"], "count": m.get("count", 500)}
    if results:
        f1s = [results[ds]["f1"] for ds in datasets if ds in results]
        ems = [results[ds]["em"] for ds in datasets if ds in results]
        results["macro_avg_f1"] = sum(f1s) / len(f1s) if f1s else 0
        results["macro_avg_em"] = sum(ems) / len(ems) if ems else 0
    return results


def routing_distribution(iter_dir, datasets=DATASETS):
    """Compute A/B/C routing from *_option.json files."""
    counts = Counter()
    per_ds = {}
    total_steps = 0
    for ds in datasets:
        option_file = os.path.join(iter_dir, ds, f"{ds}_option.json")
        if not os.path.exists(option_file):
            continue
        options = load_json(option_file)
        ds_counts = Counter()
        ds_steps = 0
        for qid, v in options.items():
            label = v["option"]
            ds_counts[label] += 1
            counts[label] += 1
            ds_steps += v.get("stepNum", 0)
            total_steps += v.get("stepNum", 0)
        per_ds[ds] = {
            "A": ds_counts.get("A", 0),
            "B": ds_counts.get("B", 0),
            "C": ds_counts.get("C", 0),
            "total": sum(ds_counts.values()),
            "steps": ds_steps,
        }
    total = sum(counts.values())
    return {
        "A": counts.get("A", 0),
        "B": counts.get("B", 0),
        "C": counts.get("C", 0),
        "total": total,
        "A_pct": counts.get("A", 0) / total * 100 if total else 0,
        "B_pct": counts.get("B", 0) / total * 100 if total else 0,
        "C_pct": counts.get("C", 0) / total * 100 if total else 0,
        "total_steps": total_steps,
        "per_dataset": per_ds,
    }


def compute_deltas(current_qa, baseline_qa, datasets=DATASETS):
    """Compute per-dataset and macro delta F1 vs a baseline."""
    deltas = {}
    for ds in datasets:
        if ds in current_qa and ds in baseline_qa:
            deltas[ds] = {
                "delta_f1": current_qa[ds]["f1"] - baseline_qa[ds]["f1"],
                "delta_em": current_qa[ds]["em"] - baseline_qa[ds]["em"],
            }
    if "macro_avg_f1" in current_qa and "macro_avg_f1" in baseline_qa:
        deltas["macro_avg_delta_f1"] = current_qa["macro_avg_f1"] - baseline_qa["macro_avg_f1"]
        deltas["macro_avg_delta_em"] = current_qa["macro_avg_em"] - baseline_qa["macro_avg_em"]
    return deltas


# ── IT1: Baseline Cascade ────────────────────────────────────────────────

def collect_it1():
    print("Collecting IT1: Baseline Cascade...")
    result = {"iteration": 1, "name": "Baseline Cascade", "models": {}}
    for m in MODELS:
        iter_dir = os.path.join(PRED_BASE, m, "iter1_standard")
        model_result = {}

        # Clf1 (no_ret_vs_ret)
        clf1_dir = os.path.join(CLF_BASE, m, "no_ret_vs_ret/epoch")
        vf1 = os.path.join(DATA_BASE, m, "silver/no_retrieval_vs_retrieval/valid.json")
        clf1_pred, clf1_valid, clf1_epoch, clf1_acc = find_best_epoch(clf1_dir, vf1)
        clf1_metrics = per_class_accuracy(clf1_valid)
        model_result["clf1"] = {
            "best_epoch": int(clf1_epoch),
            **clf1_metrics,
        }

        # Clf2 (single_vs_multi)
        clf2_dir = os.path.join(CLF_BASE, m, "single_vs_multi/epoch")
        vf2 = os.path.join(DATA_BASE, m, "silver/single_vs_multi/valid.json")
        clf2_pred, clf2_valid, clf2_epoch, clf2_acc = find_best_epoch(clf2_dir, vf2)
        clf2_metrics = per_class_accuracy(clf2_valid)
        model_result["clf2"] = {
            "best_epoch": int(clf2_epoch),
            **clf2_metrics,
        }

        # End-to-end QA
        model_result["qa"] = qa_metrics(iter_dir)

        # Routing
        model_result["routing"] = routing_distribution(iter_dir)

        result["models"][m] = model_result
    return result


# ── IT2: Undersampled Clf1 (GPT only) ────────────────────────────────────

def collect_it2():
    print("Collecting IT2: Clf1 Undersampling (GPT only)...")
    result = {"iteration": 2, "name": "Clf1 Undersampling (GPT only)", "models": {}}
    m = "gpt"
    iter_dir = os.path.join(PRED_BASE, m, "iter2_undersampled")

    # Undersampled Clf1
    clf1_dir = os.path.join(CLF_BASE, m, "no_ret_vs_ret_undersampled/epoch")
    vf1 = os.path.join(DATA_BASE, m, "silver/no_retrieval_vs_retrieval/valid.json")
    clf1_pred, clf1_valid, clf1_epoch, clf1_acc = find_best_epoch(clf1_dir, vf1)
    clf1_metrics = per_class_accuracy(clf1_valid)

    # Count undersampled training set size
    train_file = os.path.join(DATA_BASE, m, "silver/no_retrieval_vs_retrieval/train_undersampled.json")
    train_size = len(load_json(train_file)) if os.path.exists(train_file) else None

    model_result = {
        "clf1": {
            "best_epoch": int(clf1_epoch),
            "training_set_size_after_undersampling": train_size,
            **clf1_metrics,
        },
        "qa": qa_metrics(iter_dir),
        "routing": routing_distribution(iter_dir),
    }
    result["models"][m] = model_result
    return result


# ── IT3: Weighted CE ─────────────────────────────────────────────────────

def collect_it3():
    print("Collecting IT3: Clf1 Weighted CE...")
    result = {"iteration": 3, "name": "Clf1 Weighted Cross-Entropy", "models": {}}
    for m in MODELS:
        iter_dir = os.path.join(PRED_BASE, m, "iter3_weighted_ce")
        clf1_dir = os.path.join(CLF_BASE, m, "no_ret_vs_ret_weighted_ce/epoch")
        vf1 = os.path.join(DATA_BASE, m, "silver/no_retrieval_vs_retrieval/valid.json")
        clf1_pred, clf1_valid, clf1_epoch, clf1_acc = find_best_epoch(clf1_dir, vf1)
        clf1_metrics = per_class_accuracy(clf1_valid)
        model_result = {
            "clf1": {
                "best_epoch": int(clf1_epoch),
                **clf1_metrics,
            },
            "qa": qa_metrics(iter_dir),
            "routing": routing_distribution(iter_dir),
        }
        result["models"][m] = model_result
    return result


# ── IT4: Focal Loss ──────────────────────────────────────────────────────

def collect_it4():
    print("Collecting IT4: Clf1 Focal Loss...")
    result = {"iteration": 4, "name": "Clf1 Focal Loss", "models": {}}
    for m in MODELS:
        iter_dir = os.path.join(PRED_BASE, m, "iter4_focal")
        clf1_dir = os.path.join(CLF_BASE, m, "no_ret_vs_ret_focal/epoch")
        vf1 = os.path.join(DATA_BASE, m, "silver/no_retrieval_vs_retrieval/valid.json")
        clf1_pred, clf1_valid, clf1_epoch, clf1_acc = find_best_epoch(clf1_dir, vf1)
        clf1_metrics = per_class_accuracy(clf1_valid)
        model_result = {
            "clf1": {
                "best_epoch": int(clf1_epoch),
                **clf1_metrics,
            },
            "qa": qa_metrics(iter_dir),
            "routing": routing_distribution(iter_dir),
        }
        result["models"][m] = model_result
    return result


# ── IT5: Agreement Gate ──────────────────────────────────────────────────

def collect_it5():
    print("Collecting IT5: Agreement Gate...")
    result = {"iteration": 5, "name": "Agreement Gate (Training-Free Gate 1)", "models": {}}
    for m in MODELS:
        iter_dir = os.path.join(PRED_BASE, m, "iter5_agreement")
        stats_file = os.path.join(iter_dir, "routing_stats.json")
        stats = load_json(stats_file)

        # Agreement rate per dataset
        agreement = {
            "overall_rate": stats["agreement_rate"],
            "total_A": stats["total_A"],
            "total_questions": stats["total_questions"],
            "per_dataset": {},
        }
        for ds, ds_data in stats["per_dataset"].items():
            total = ds_data["A"] + ds_data["B"] + ds_data["C"]
            agreement["per_dataset"][ds] = {
                "rate": ds_data["A"] / total if total else 0,
                "A": ds_data["A"],
                "total": total,
            }

        routing = {
            "A": stats["total_A"],
            "B": stats["total_B"],
            "C": stats["total_C"],
            "total": stats["total_questions"],
            "A_pct": stats["total_A"] / stats["total_questions"] * 100,
            "B_pct": stats["total_B"] / stats["total_questions"] * 100,
            "C_pct": stats["total_C"] / stats["total_questions"] * 100,
            "total_steps": stats["total_steps"],
            "per_dataset": {},
        }
        for ds, ds_data in stats["per_dataset"].items():
            total = ds_data["A"] + ds_data["B"] + ds_data["C"]
            routing["per_dataset"][ds] = {
                "A": ds_data["A"], "B": ds_data["B"], "C": ds_data["C"],
                "total": total, "steps": ds_data["steps"],
            }

        model_result = {
            "agreement": agreement,
            "qa": qa_metrics(iter_dir),
            "routing": routing,
        }
        result["models"][m] = model_result
    return result


# ── IT6: κ(q) Feature Probe ─────────────────────────────────────────────

def collect_it6():
    print("Collecting IT6: κ(q) Feature Probe...")
    probe_file = os.path.join(PRED_BASE, "iter6_kappa_probe/clf2_kappa_probe_results.json")
    probe = load_json(probe_file)

    # Compute per-class recall from CSVs
    import csv

    def class_recall_from_csv(csv_path):
        """Compute B-recall and C-recall from probe data CSV with true_label column."""
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows or "true_label" not in rows[0]:
            return None
        # Use kappa as simple threshold: above median = C, below = B
        # Actually we need the probe predictions, not in the CSV. Use logistic regression.
        return None  # Will compute via sklearn below

    def compute_class_recalls(csv_path):
        """5-fold CV to get per-class recall, matching the probe methodology."""
        import csv
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            return {}
        features = ["token_len_norm", "entity_density", "hop_density", "kappa"]
        X = np.array([[float(r[f]) for f in features] for r in rows])
        y = np.array([1 if r["label"] == "C" else 0 for r in rows])

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        b_recalls, c_recalls = [], []
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y[train_idx])
            y_pred = clf.predict(X_test)
            y_true = y[test_idx]
            # B=0, C=1
            b_mask = y_true == 0
            c_mask = y_true == 1
            b_recall = (y_pred[b_mask] == 0).sum() / b_mask.sum() if b_mask.sum() > 0 else 0
            c_recall = (y_pred[c_mask] == 1).sum() / c_mask.sum() if c_mask.sum() > 0 else 0
            b_recalls.append(b_recall)
            c_recalls.append(c_recall)
        return {
            "B_recall_mean": float(np.mean(b_recalls)),
            "B_recall_std": float(np.std(b_recalls)),
            "C_recall_mean": float(np.mean(c_recalls)),
            "C_recall_std": float(np.std(c_recalls)),
        }

    result = {
        "iteration": 6,
        "name": "κ(q) Feature Probe (Diagnostic)",
        "features": probe["features"],
        "kappa_weights": probe["kappa_weights"],
        "models": {},
    }
    for m in MODELS:
        model_data = probe["models"][m]
        model_result = {}
        for split in ["merged", "silver_only", "ib_only"]:
            d = model_data[split]
            split_result = {
                "n_samples": d["n_samples"],
                "class_counts": d["class_counts"],
                "macro_f1": {"mean": d["mean_macro_f1"], "std": d["std_macro_f1"]},
                "roc_auc": {"mean": d["mean_auc"], "std": d["std_auc"]},
                "accuracy": {"mean": d["mean_accuracy"], "std": d["std_accuracy"]},
                "coefficients": d["mean_coefficients"],
                "intercept": d["intercept"],
                "verdict": d["verdict"],
            }
            # Compute per-class recall
            csv_path = os.path.join(PRED_BASE, f"iter6_kappa_probe/clf2_kappa_probe_data_{m}_{split.replace('_', '-')}.csv")
            if os.path.exists(csv_path):
                recalls = compute_class_recalls(csv_path)
                if recalls:
                    split_result["per_class_recall"] = recalls
            model_result[split] = split_result
        result["models"][m] = model_result
    return result


# ── IT7: UE Kappa (fully training-free) ──────────────────────────────────

def collect_it7():
    print("Collecting IT7: UE Kappa (fully training-free)...")
    result = {"iteration": 7, "name": "UE Kappa (Fully Training-Free)", "models": {}}
    for m in MODELS:
        iter_dir = os.path.join(PRED_BASE, m, "iter7_kappa")
        stats_file = os.path.join(iter_dir, "routing_stats.json")
        stats = load_json(stats_file)

        # κ(q) gate 2 statistics
        kappa_gate2 = {
            "threshold_tuned": stats["threshold_tuned"],
            "threshold_used": stats["threshold_used"],
            "symrag_weights": stats["symrag_weights"],
            "kappa_stats": stats["kappa_stats"],
        }
        if "tuning_stats" in stats:
            ts = stats["tuning_stats"]
            kappa_gate2["tuning"] = {
                "n_val_samples": ts["n_val_samples"],
                "n_B": ts["n_B"],
                "n_C": ts["n_C"],
                "best_accuracy": ts["best_accuracy"],
                "best_acc_threshold": ts["best_acc_threshold"],
                "best_macro_f1": ts["best_macro_f1"],
                "best_f1_threshold": ts["best_f1_threshold"],
                "macro_f1_at_used_threshold": ts.get("macro_f1_at_used_threshold"),
                "val_B_accuracy": ts["val_B_accuracy"],
                "val_C_accuracy": ts["val_C_accuracy"],
            }
            # Normalized top-level aliases for easy thesis lookup
            kappa_gate2["tuned_threshold"] = ts["best_acc_threshold"]
            kappa_gate2["val_accuracy"] = ts["best_accuracy"]
            kappa_gate2["val_macro_f1"] = ts.get("macro_f1_at_used_threshold")

        # Agreement info (shared with IT5)
        agreement_rate = stats.get("agreement_rate")
        if not agreement_rate:
            # Compute from A counts — A means agreement
            total_A = stats["routing_counts"]["A"]
            total_Q = stats["total_questions"]
            agreement_rate = total_A / total_Q

        # Routing
        rc = stats["routing_counts"]
        total = stats["total_questions"]
        routing = {
            "A": rc["A"], "B": rc["B"], "C": rc["C"],
            "total": total,
            "A_pct": rc["A"] / total * 100,
            "B_pct": rc["B"] / total * 100,
            "C_pct": rc["C"] / total * 100,
            "total_steps": stats["total_steps"],
            "per_dataset": {},
        }
        for ds, ds_data in stats["per_dataset"].items():
            t = ds_data["A"] + ds_data["B"] + ds_data["C"]
            routing["per_dataset"][ds] = {
                "A": ds_data["A"], "B": ds_data["B"], "C": ds_data["C"],
                "total": t, "steps": ds_data["steps"],
            }

        model_result = {
            "kappa_gate2": kappa_gate2,
            "agreement_rate": agreement_rate,
            "qa": qa_metrics(iter_dir),
            "routing": routing,
        }
        result["models"][m] = model_result
    return result


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    # Collect all iterations
    all_results["IT1"] = collect_it1()
    all_results["IT2"] = collect_it2()
    all_results["IT3"] = collect_it3()
    all_results["IT4"] = collect_it4()
    all_results["IT5"] = collect_it5()
    all_results["IT6"] = collect_it6()
    all_results["IT7"] = collect_it7()

    # Compute deltas
    print("\nComputing deltas...")
    it1_qa = {m: all_results["IT1"]["models"][m]["qa"] for m in MODELS}

    # IT2 delta vs IT1 (GPT only)
    all_results["IT2"]["models"]["gpt"]["delta_vs_IT1"] = compute_deltas(
        all_results["IT2"]["models"]["gpt"]["qa"], it1_qa["gpt"])

    # IT3, IT4 delta vs IT1
    for itag in ["IT3", "IT4"]:
        for m in MODELS:
            all_results[itag]["models"][m]["delta_vs_IT1"] = compute_deltas(
                all_results[itag]["models"][m]["qa"], it1_qa[m])

    # IT5 delta vs IT1, and vs best of IT2-4
    for m in MODELS:
        all_results["IT5"]["models"][m]["delta_vs_IT1"] = compute_deltas(
            all_results["IT5"]["models"][m]["qa"], it1_qa[m])

        # Best of IT2-4 for this model
        best_it24_f1 = -1
        best_it24_tag = None
        for itag in ["IT2", "IT3", "IT4"]:
            if m in all_results[itag]["models"]:
                f1 = all_results[itag]["models"][m]["qa"].get("macro_avg_f1", 0)
                if f1 > best_it24_f1:
                    best_it24_f1 = f1
                    best_it24_tag = itag
        if best_it24_tag:
            all_results["IT5"]["models"][m]["delta_vs_best_IT2_4"] = {
                "reference": best_it24_tag,
                **compute_deltas(
                    all_results["IT5"]["models"][m]["qa"],
                    all_results[best_it24_tag]["models"][m]["qa"])
            }

    # IT7 delta vs IT1 and IT5
    for m in MODELS:
        all_results["IT7"]["models"][m]["delta_vs_IT1"] = compute_deltas(
            all_results["IT7"]["models"][m]["qa"], it1_qa[m])
        all_results["IT7"]["models"][m]["delta_vs_IT5"] = compute_deltas(
            all_results["IT7"]["models"][m]["qa"],
            all_results["IT5"]["models"][m]["qa"])

    # Save
    out_path = os.path.join(REPO, "results/all_iterations.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
    return all_results


if __name__ == "__main__":
    main()
