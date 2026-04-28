#!/usr/bin/env python3
"""
IT8-aligned feasibility check: can the SymRAG κ(q) structural features
separate B (single-step) from C (multi-step) better than chance?

Uses the same features as predict_complexity_kappa.py (IT8):
  - token_len_norm:  whitespace-split token count / max token count
  - entity_density:  entity_count / token_len
  - hop_density:     hop_indicator_count / token_len
  - kappa:           composite κ(q) = w_L · L(q) · (1 + S_H(q))

Trains a logistic regression under 5-fold stratified cross-validation and
reports mean ROC-AUC ± std, mean accuracy ± std, mean per-feature
coefficients ± std, and a classification report from the last fold.

Usage:
    python classifier/postprocess/clf2_kappa_feature_probe.py
    python classifier/postprocess/clf2_kappa_feature_probe.py --data_path path/to/train.json
    python classifier/postprocess/clf2_kappa_feature_probe.py --model flan_t5_xxl
    python classifier/postprocess/clf2_kappa_feature_probe.py --all_models
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# SymRAG published weights (same as predict_complexity_kappa.py)
# ---------------------------------------------------------------------------
W_L = 1.0
W_SH1 = 0.05   # entity density weight
W_SH2 = 0.10   # hop indicator density weight

# ---------------------------------------------------------------------------
# Bridging-phrase patterns (identical to predict_complexity_kappa.py)
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


def detect_data_paths(model: str) -> list[tuple[str, str]]:
    """Return available Clf2 data files as ``(tag, path)`` pairs."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    base = os.path.join(
        repo_root,
        "classifier", "data", "musique_hotpot_wiki2_nq_tqa_sqd",
    )
    candidates = [
        ("merged", os.path.join(base, model, "binary_silver_single_vs_multi", "train.json")),
        ("silver-only", os.path.join(base, model, "silver", "single_vs_multi", "train.json")),
    ]
    found = [(tag, path) for tag, path in candidates if os.path.isfile(path)]
    if not found:
        sys.exit(
            f"ERROR: could not find any Clf2 training data for model={model}.\n"
            f"Searched:\n  " + "\n  ".join(p for _, p in candidates) + "\n"
            "Pass --data_path explicitly."
        )
    return found


def load_bc_data(path: str) -> list[dict]:
    """Load JSON array and keep only B/C labelled items."""
    with open(path) as f:
        data = json.load(f)
    bc = [item for item in data if item.get("answer") in ("B", "C")]
    if not bc:
        sys.exit(f"ERROR: no B/C items found in {path}")
    return bc


def load_bc_data_ib_only(merged_path: str, silver_path: str) -> list[dict]:
    """Load B/C items that are in merged but NOT in silver (inductive-bias only)."""
    with open(silver_path) as f:
        silver_ids = {item["id"] for item in json.load(f)}
    with open(merged_path) as f:
        merged = json.load(f)
    ib = [item for item in merged
          if item["id"] not in silver_ids and item.get("answer") in ("B", "C")]
    if not ib:
        return []
    return ib


def extract_features(questions: list[str], nlp) -> pd.DataFrame:
    """Extract IT8's κ(q) features for each question.

    Raw features (per question):
      - token_len:       whitespace-split token count
      - entity_count:    spaCy NER entity count
      - hop_count:       number of bridging patterns that fire

    Derived features (matching predict_complexity_kappa.py exactly):
      - token_len_norm:  token_len / max(token_len)
      - entity_density:  entity_count / token_len
      - hop_density:     hop_count / token_len
      - kappa:           W_L * token_len_norm * (1 + W_SH1 * entity_density + W_SH2 * hop_density)
    """
    token_lens = []
    entity_counts = []
    hop_counts = []

    for doc in nlp.pipe(questions, batch_size=256):
        text = doc.text
        token_lens.append(len(text.split()))
        entity_counts.append(len(doc.ents))
        hop_counts.append(sum(1 for pat in _BRIDGE_RES if pat.search(text)))

    token_lens = np.array(token_lens, dtype=float)
    entity_counts = np.array(entity_counts, dtype=float)
    hop_counts = np.array(hop_counts, dtype=float)

    max_len = token_lens.max() if token_lens.max() > 0 else 1.0
    token_len_norm = token_lens / max_len

    safe_lens = np.where(token_lens > 0, token_lens, 1.0)
    entity_density = entity_counts / safe_lens
    hop_density = hop_counts / safe_lens

    kappa = W_L * token_len_norm * (1.0 + W_SH1 * entity_density + W_SH2 * hop_density)

    return pd.DataFrame({
        "token_len": token_lens.astype(int),
        "entity_count": entity_counts.astype(int),
        "hop_count": hop_counts.astype(int),
        "token_len_norm": token_len_norm,
        "entity_density": entity_density,
        "hop_density": hop_density,
        "kappa": kappa,
    })


# Feature columns used in the logistic regression probe
FEATURE_COLS = ["token_len_norm", "entity_density", "hop_density", "kappa"]


def _evaluate_items(
    bc_data: list[dict], tag: str, model: str, output_dir: str, nlp,
) -> dict:
    """Run feature extraction + 5-fold CV on a list of B/C items."""
    if not bc_data:
        return None
    questions = [item["question"] for item in bc_data]
    labels = [item["answer"] for item in bc_data]
    n_b = labels.count("B")
    n_c = labels.count("C")
    print(f"[data]  {len(bc_data)} B/C items  (B={n_b}, C={n_c})")

    print("[feat]  extracting κ(q) features (spaCy NER + regex) ...")
    feat_df = extract_features(questions, nlp)
    feat_df["label"] = labels
    feat_df["question"] = questions
    feat_df["id"] = [item["id"] for item in bc_data]

    print("\n--- Feature means by class ---")
    print(feat_df.groupby("label")[FEATURE_COLS].mean().round(4).to_string())
    print()

    safe_tag = tag.replace(" ", "_").replace("+", "_")
    csv_path = os.path.join(output_dir, f"clf2_kappa_probe_data_{model}_{safe_tag}.csv")
    feat_df.to_csv(csv_path, index=False)
    print(f"[save]  {csv_path}")

    # --- 5-fold stratified CV -----------------------------------------
    X = feat_df[FEATURE_COLS].values
    y = (feat_df["label"] == "C").astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs, fold_accs, fold_f1s, fold_coefs = [], [], [], []
    last_fold_y_test = last_fold_y_pred = None

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        fold_aucs.append(roc_auc_score(y_test, y_prob))
        fold_accs.append(accuracy_score(y_test, y_pred))
        fold_f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        fold_coefs.append(clf.coef_[0].copy())

        last_fold_y_test = y_test
        last_fold_y_pred = y_pred

    fold_aucs = np.array(fold_aucs)
    fold_accs = np.array(fold_accs)
    fold_f1s = np.array(fold_f1s)
    fold_coefs = np.array(fold_coefs)

    mean_auc = fold_aucs.mean()
    std_auc = fold_aucs.std()
    mean_acc = fold_accs.mean()
    std_acc = fold_accs.std()
    mean_f1 = fold_f1s.mean()
    std_f1 = fold_f1s.std()

    print("\n========== RESULTS (5-fold stratified CV) ==========")
    print(f"Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"ROC-AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print()
    print("--- Classification report (last fold) ---")
    print(classification_report(
        last_fold_y_test, last_fold_y_pred,
        target_names=["B (single)", "C (multi)"],
    ))

    mean_coefs = fold_coefs.mean(axis=0)
    std_coefs = fold_coefs.std(axis=0)
    mean_coef_dict = {name: float(mc) for name, mc in zip(FEATURE_COLS, mean_coefs)}
    last_intercept = float(clf.intercept_[0])
    print("--- Feature coefficients (positive → C), mean ± std across folds ---")
    for name, mc, sc in zip(FEATURE_COLS, mean_coefs, std_coefs):
        print(f"  {name:>18s}:  {mc:+.4f} ± {sc:.4f}")
    print(f"  {'intercept':>18s}:  {last_intercept:+.4f}  (last fold)")

    # --- κ distribution plot -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: κ histogram by class
    ax = axes[0]
    b_mask = feat_df["label"] == "B"
    c_mask = feat_df["label"] == "C"
    ax.hist(feat_df.loc[b_mask, "kappa"], bins=40, alpha=0.6, label="B (single)", color="tab:blue")
    ax.hist(feat_df.loc[c_mask, "kappa"], bins=40, alpha=0.6, label="C (multi)", color="tab:red")
    ax.set_xlabel("κ(q)")
    ax.set_ylabel("Count")
    ax.set_title(f"κ(q) distribution ({model}, {tag})")
    ax.legend()

    # Right: entity_density vs hop_density scatter
    ax = axes[1]
    ax.scatter(
        feat_df.loc[b_mask, "entity_density"],
        feat_df.loc[b_mask, "hop_density"],
        c="tab:blue", alpha=0.4, s=18, label="B (single-step)",
    )
    ax.scatter(
        feat_df.loc[c_mask, "entity_density"],
        feat_df.loc[c_mask, "hop_density"],
        c="tab:red", alpha=0.4, s=18, label="C (multi-step)",
    )
    ax.set_xlabel("entity_density")
    ax.set_ylabel("hop_density")
    ax.set_title(f"κ(q) components ({model}, {tag})  —  AUC={mean_auc:.3f}")
    ax.legend()

    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"clf2_kappa_probe_{model}_{safe_tag}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[save]  {plot_path}")

    return {
        "tag": tag,
        "n_samples": len(bc_data),
        "n_b": n_b,
        "n_c": n_c,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_coefficients": mean_coef_dict,
        "intercept": last_intercept,
    }


def _evaluate_dataset(
    data_path: str, tag: str, model: str, output_dir: str, nlp,
) -> dict:
    """Run feature extraction + 5-fold CV on a single data file."""
    print(f"[data]  {data_path}")
    bc_data = load_bc_data(data_path)
    return _evaluate_items(bc_data, tag, model, output_dir, nlp)


DATASET_NAMES = ["musique", "hotpotqa", "2wikimultihopqa", "nq", "trivia", "squad"]


def run_per_dataset_probe(
    model: str, output_dir: str, nlp,
) -> list[dict]:
    """Run probe per source dataset across merged, silver, and IB splits."""
    sources = detect_data_paths(model)
    merged_path = next((p for t, p in sources if t == "merged"), None)
    silver_path = next((p for t, p in sources if t == "silver-only"), None)

    # Load all items once
    merged_items = load_bc_data(merged_path) if merged_path else []
    with open(silver_path) as f:
        silver_ids = {item["id"] for item in json.load(f)} if silver_path else set()
    silver_items = [x for x in load_bc_data(silver_path)] if silver_path else []
    ib_items = [x for x in merged_items if x["id"] not in silver_ids]

    rows = []
    for ds in DATASET_NAMES:
        ds_merged = [x for x in merged_items if x.get("dataset_name") == ds]
        ds_silver = [x for x in silver_items if x.get("dataset_name") == ds]
        ds_ib = [x for x in ib_items if x.get("dataset_name") == ds]

        row = {"dataset": ds, "model": model,
               "merged": None, "silver": None, "ib": None}

        for split_tag, items in [("merged", ds_merged),
                                  ("silver", ds_silver),
                                  ("ib", ds_ib)]:
            bc = [x for x in items if x.get("answer") in ("B", "C")]
            n_b = sum(1 for x in bc if x["answer"] == "B")
            n_c = sum(1 for x in bc if x["answer"] == "C")
            if n_b < 5 or n_c < 5:
                print(f"  [{ds}/{split_tag}] skipped (B={n_b}, C={n_c} — too few)")
                continue
            print(f"\n--- {ds} / {split_tag} (B={n_b}, C={n_c}) ---")
            res = _evaluate_items(
                bc, f"{ds}_{split_tag}", model, output_dir, nlp,
            )
            if res:
                row[split_tag] = res

        rows.append(row)

    return rows


def run_probe(model: str, data_path: str | None, output_dir: str, nlp) -> dict:
    """Run the full feature probe for one model variant.

    Returns a dict with keys: model, merged, silver_only, verdict,
    verdict_f1, verdict_std.
    """
    sub_results: dict[str, dict | None] = {
        "merged": None, "silver-only": None, "ib-only": None,
    }

    if data_path is not None:
        res = _evaluate_dataset(data_path, "override", model, output_dir, nlp)
        verdict_f1 = res["mean_f1"]
        verdict_std = res["std_f1"]
        sub_results["override"] = res
    else:
        sources = detect_data_paths(model)
        has_silver = any(tag == "silver-only" for tag, _ in sources)
        has_merged = any(tag == "merged" for tag, _ in sources)
        if not has_silver:
            print(f"[WARN]  silver-only file not found for {model}; "
                  "proceeding with merged only.")

        for tag, path in sources:
            desc = ("merged (silver + inductive-bias)" if tag == "merged"
                    else "silver-only")
            print(f"\n=== Data: {desc} ===")
            res = _evaluate_dataset(path, tag, model, output_dir, nlp)
            sub_results[tag] = res

        # Inductive-bias-only: merged minus silver
        if has_merged and has_silver:
            merged_path = next(p for t, p in sources if t == "merged")
            silver_path = next(p for t, p in sources if t == "silver-only")
            ib_items = load_bc_data_ib_only(merged_path, silver_path)
            if ib_items:
                import tempfile
                ib_tmp = os.path.join(output_dir, f".ib_only_{model}.json")
                with open(ib_tmp, "w") as f:
                    json.dump(ib_items, f)
                print(f"\n=== Data: inductive-bias-only ===")
                res = _evaluate_dataset(ib_tmp, "ib-only", model, output_dir, nlp)
                sub_results["ib-only"] = res
                os.remove(ib_tmp)

        if sub_results["silver-only"] is not None:
            verdict_f1 = sub_results["silver-only"]["mean_f1"]
            verdict_std = sub_results["silver-only"]["std_f1"]
        else:
            verdict_f1 = sub_results["merged"]["mean_f1"]
            verdict_std = sub_results["merged"]["std_f1"]

    verdict = "GO" if verdict_f1 >= 0.55 else "NO-GO"

    print()
    if verdict == "GO":
        print(f">>> VERDICT:  {verdict}  (macro-F1 {verdict_f1:.4f} >= 0.55)")
    else:
        print(f">>> VERDICT:  {verdict}  (macro-F1 {verdict_f1:.4f} < 0.55)")

    return {
        "model": model,
        "merged": sub_results.get("merged"),
        "silver_only": sub_results.get("silver-only"),
        "ib_only": sub_results.get("ib-only"),
        "override": sub_results.get("override"),
        "verdict": verdict,
        "verdict_f1": verdict_f1,
        "verdict_std": verdict_std,
    }


def main():
    parser = argparse.ArgumentParser(
        description="κ(q) feature probe: can IT8's structural features separate B vs C?"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Override path to Clf2 training JSON.",
    )
    parser.add_argument(
        "--model", type=str, default="flan_t5_xl",
        choices=("flan_t5_xl", "flan_t5_xxl", "gpt"),
        help="LLM variant for auto-detecting the data path (default: flan_t5_xl).",
    )
    parser.add_argument(
        "--all_models", action="store_true", default=False,
        help="Run probe for all model variants and print summary table.",
    )
    parser.add_argument(
        "--per_dataset", action="store_true", default=False,
        help="Break down evaluation by source dataset.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for outputs (plots + CSV). Default: same dir as this script.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[out]   {output_dir}")

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        sys.exit(
            "ERROR: spaCy model 'en_core_web_sm' not found.\n"
            "Install with:  python -m spacy download en_core_web_sm"
        )

    if args.all_models:
        models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    else:
        models = [args.model]

    # --- Per-dataset mode ---------------------------------------------
    if args.per_dataset:
        all_ds_rows = []
        for model in models:
            print(f"\n{'=' * 60}")
            print(f"  MODEL: {model}  (per-dataset)")
            print(f"{'=' * 60}")
            rows = run_per_dataset_probe(model, output_dir, nlp)
            all_ds_rows.extend(rows)

        # Summary table
        print(f"\n\n{'=' * 120}")
        print("  PER-DATASET SUMMARY")
        print(f"{'=' * 120}")
        header = (f"{'Model':<14s}| {'Dataset':<20s}"
                  f"| {'N_m':>4s} {'F1_m':>7s} {'AUC_m':>7s}"
                  f"| {'N_s':>4s} {'F1_s':>7s} {'AUC_s':>7s}"
                  f"| {'N_ib':>4s} {'F1_ib':>7s} {'AUC_ib':>7s}")
        print(header)
        print("-" * len(header))
        for row in all_ds_rows:
            m = row["merged"]
            s = row["silver"]
            ib = row["ib"]
            nm = str(m["n_samples"]) if m else "-"
            f1m = f"{m['mean_f1']:.4f}" if m else "-"
            am = f"{m['mean_auc']:.4f}" if m else "-"
            ns = str(s["n_samples"]) if s else "-"
            f1s = f"{s['mean_f1']:.4f}" if s else "-"
            aus = f"{s['mean_auc']:.4f}" if s else "-"
            nib = str(ib["n_samples"]) if ib else "-"
            f1ib = f"{ib['mean_f1']:.4f}" if ib else "-"
            aib = f"{ib['mean_auc']:.4f}" if ib else "-"
            print(f"{row['model']:<14s}| {row['dataset']:<20s}"
                  f"| {nm:>4s} {f1m:>7s} {am:>7s}"
                  f"| {ns:>4s} {f1s:>7s} {aus:>7s}"
                  f"| {nib:>4s} {f1ib:>7s} {aib:>7s}")
        print()

        # Persist
        json_path = os.path.join(output_dir, "clf2_kappa_probe_per_dataset.json")
        def _row_to_json(sub):
            if sub is None:
                return None
            return {
                "n_samples": sub["n_samples"],
                "class_counts": {"B": sub["n_b"], "C": sub["n_c"]},
                "mean_auc": sub["mean_auc"],
                "mean_macro_f1": sub["mean_f1"],
                "mean_accuracy": sub["mean_acc"],
            }
        json_out = [{"model": r["model"], "dataset": r["dataset"],
                     "merged": _row_to_json(r["merged"]),
                     "silver": _row_to_json(r["silver"]),
                     "ib": _row_to_json(r["ib"])}
                    for r in all_ds_rows]
        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=2)
        print(f"[save]  {json_path}")
        return

    # --- Standard (aggregate) mode ------------------------------------
    results = []
    for model in models:
        if len(models) > 1:
            print(f"\n{'=' * 60}")
            print(f"  MODEL: {model}")
            print(f"{'=' * 60}")
        result = run_probe(
            model=model,
            data_path=args.data_path if not args.all_models else None,
            output_dir=output_dir,
            nlp=nlp,
        )
        results.append(result)

    if len(results) > 1:
        print(f"\n\n{'=' * 90}")
        print("  SUMMARY")
        print(f"{'=' * 90}")
        header = (f"{'Model':<16s}| {'N(merged)':>9s} | {'N(silver)':>9s} | {'N(ib)':>6s} "
                  f"| {'Merged F1':>12s} | {'Silver F1':>12s} | {'IB F1':>12s} "
                  f"| {'Merged AUC':>12s} | {'Silver AUC':>12s} | {'IB AUC':>12s} | Verdict")
        print(header)
        print("-" * len(header))
        for r in results:
            m = r["merged"]
            s = r["silver_only"]
            ib = r["ib_only"]
            n_merged = str(m["n_samples"]) if m else "-"
            n_silver = str(s["n_samples"]) if s else "-"
            n_ib = str(ib["n_samples"]) if ib else "-"
            f1_merged = f"{m['mean_f1']:.4f}" if m else "-"
            f1_silver = f"{s['mean_f1']:.4f}" if s else "-"
            f1_ib = f"{ib['mean_f1']:.4f}" if ib else "-"
            auc_merged = f"{m['mean_auc']:.4f}" if m else "-"
            auc_silver = f"{s['mean_auc']:.4f}" if s else "-"
            auc_ib = f"{ib['mean_auc']:.4f}" if ib else "-"
            print(f"{r['model']:<16s}| {n_merged:>9s} | {n_silver:>9s} | {n_ib:>6s} "
                  f"| {f1_merged:>12s} | {f1_silver:>12s} | {f1_ib:>12s} "
                  f"| {auc_merged:>12s} | {auc_silver:>12s} | {auc_ib:>12s} | {r['verdict']}")
        print()

    # --- Persist results to JSON --------------------------------------
    def _sub_to_json(sub: dict | None) -> dict | None:
        if sub is None:
            return None
        return {
            "n_samples": sub["n_samples"],
            "class_counts": {"B": sub["n_b"], "C": sub["n_c"]},
            "mean_auc": sub["mean_auc"],
            "std_auc": sub["std_auc"],
            "mean_macro_f1": sub["mean_f1"],
            "std_macro_f1": sub["std_f1"],
            "mean_accuracy": sub["mean_acc"],
            "std_accuracy": sub["std_acc"],
            "mean_coefficients": sub["mean_coefficients"],
            "intercept": sub["intercept"],
        }

    json_out: dict = {
        "features": FEATURE_COLS,
        "kappa_weights": {"W_L": W_L, "W_SH1": W_SH1, "W_SH2": W_SH2},
        "models": {},
        "go_no_go_metric": "macro_f1",
        "go_no_go_threshold": 0.55,
        "n_folds": 5,
        "random_state": 42,
    }
    for r in results:
        model_entry: dict = {}
        if r["merged"] is not None:
            merged_json = _sub_to_json(r["merged"])
            merged_json["verdict"] = "GO" if r["merged"]["mean_f1"] >= 0.55 else "NO-GO"
            model_entry["merged"] = merged_json
        if r["silver_only"] is not None:
            silver_json = _sub_to_json(r["silver_only"])
            silver_json["verdict"] = "GO" if r["silver_only"]["mean_f1"] >= 0.55 else "NO-GO"
            model_entry["silver_only"] = silver_json
        if r["ib_only"] is not None:
            ib_json = _sub_to_json(r["ib_only"])
            ib_json["verdict"] = "GO" if r["ib_only"]["mean_f1"] >= 0.55 else "NO-GO"
            model_entry["ib_only"] = ib_json
        if r.get("override") is not None:
            override_json = _sub_to_json(r["override"])
            override_json["verdict"] = r["verdict"]
            model_entry["override"] = override_json
        json_out["models"][r["model"]] = model_entry

    json_path = os.path.join(output_dir, "clf2_kappa_probe_results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"[save]  {json_path}")


if __name__ == "__main__":
    main()
