#!/usr/bin/env python3
"""
Iteration 3b feasibility check: can simple structural query features
separate B (single-step) from C (multi-step) better than chance?

Extracts three features per question:
  - token_len:    whitespace-split token count
  - entity_count: named-entity count (spaCy en_core_web_sm)
  - bridge_flag:  regex match for multi-hop bridging phrases

Trains a logistic regression under 5-fold stratified cross-validation and
reports mean ROC-AUC ± std, mean accuracy ± std, mean per-feature
coefficients ± std, and a classification report from the last fold.

Usage:
    python classifier/postprocess/clf2_feature_probe.py
    python classifier/postprocess/clf2_feature_probe.py --data_path path/to/train.json
    python classifier/postprocess/clf2_feature_probe.py --model flan_t5_xxl
    python classifier/postprocess/clf2_feature_probe.py --all_models
"""

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ---------------------------------------------------------------------------
# Bridging-phrase patterns
# ---------------------------------------------------------------------------
# These regex patterns target syntactic structures common in multi-hop
# questions that chain two information needs together.  Each pattern is
# case-insensitive and matches anywhere in the question string.
#
#   1. Relative clauses linking two entities:
#        "the person who founded …"
#        "the city where … was born"
#   2. Possessive chains:
#        "X's Y's Z"  (two possessives ⇒ likely two hops)
#   3. Temporal/causal subordination across entities:
#        "before/after/when X did Y, what …"
#   4. Demonstrative reference to a prior fact:
#        "that country/person/team" preceded by a wh-clause
#   5. Explicit comparison bridging two look-ups:
#        "both X and Y", "between X and Y"
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
_BRIDGE_RE = re.compile("|".join(BRIDGE_PATTERNS), re.IGNORECASE)


def detect_data_paths(model: str) -> list[tuple[str, str]]:
    """Return available Clf2 data files as ``(tag, path)`` pairs.

    Checks for both the merged (silver + inductive-bias) file and the
    silver-only file.  Returns all that exist, in order:
      ("merged", ...), ("silver-only", ...)
    """
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


def extract_features(questions: list[str], nlp) -> pd.DataFrame:
    """Extract token_len, entity_count, bridge_flag for each question."""
    token_lens = []
    entity_counts = []
    bridge_flags = []

    # Process in batches via spaCy pipe for speed
    for doc in nlp.pipe(questions, batch_size=256):
        token_lens.append(len(doc.text.split()))
        entity_counts.append(len(doc.ents))
        bridge_flags.append(1 if _BRIDGE_RE.search(doc.text) else 0)

    return pd.DataFrame({
        "token_len": token_lens,
        "entity_count": entity_counts,
        "bridge_flag": bridge_flags,
    })


def _evaluate_dataset(
    data_path: str, tag: str, model: str, output_dir: str, nlp,
) -> dict:
    """Run feature extraction + 5-fold CV on a single data file.

    Returns a dict with keys: tag, n_samples, n_b, n_c, mean_auc,
    std_auc, mean_acc, std_acc, feat_df.
    """
    import numpy as np

    print(f"[data]  {data_path}")

    bc_data = load_bc_data(data_path)
    questions = [item["question"] for item in bc_data]
    labels = [item["answer"] for item in bc_data]
    n_b = labels.count("B")
    n_c = labels.count("C")
    print(f"[data]  {len(bc_data)} B/C items  (B={n_b}, C={n_c})")

    print("[feat]  extracting features (spaCy NER + regex) ...")
    feat_df = extract_features(questions, nlp)
    feat_df["label"] = labels
    feat_df["question"] = questions
    feat_df["id"] = [item["id"] for item in bc_data]

    print("\n--- Feature means by class ---")
    print(feat_df.groupby("label")[["token_len", "entity_count", "bridge_flag"]].mean()
          .round(3).to_string())
    print()

    safe_tag = tag.replace(" ", "_").replace("+", "_")
    csv_path = os.path.join(output_dir, f"clf2_feature_probe_data_{model}_{safe_tag}.csv")
    feat_df.to_csv(csv_path, index=False)
    print(f"[save]  {csv_path}")

    # --- 5-fold stratified CV -----------------------------------------
    X = feat_df[["token_len", "entity_count", "bridge_flag"]].values
    y = (feat_df["label"] == "C").astype(int).values
    feature_names = ["token_len", "entity_count", "bridge_flag"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs, fold_accs, fold_coefs = [], [], []
    last_fold_y_test = last_fold_y_pred = None

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        fold_aucs.append(roc_auc_score(y_test, y_prob))
        fold_accs.append(accuracy_score(y_test, y_pred))
        fold_coefs.append(clf.coef_[0].copy())

        last_fold_y_test = y_test
        last_fold_y_pred = y_pred

    fold_aucs = np.array(fold_aucs)
    fold_accs = np.array(fold_accs)
    fold_coefs = np.array(fold_coefs)

    mean_auc = fold_aucs.mean()
    std_auc = fold_aucs.std()
    mean_acc = fold_accs.mean()
    std_acc = fold_accs.std()

    print("\n========== RESULTS (5-fold stratified CV) ==========")
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
    mean_coef_dict = {name: float(mc) for name, mc in zip(feature_names, mean_coefs)}
    last_intercept = float(clf.intercept_[0])
    print("--- Feature coefficients (positive → C), mean ± std across folds ---")
    for name, mc, sc in zip(feature_names, mean_coefs, std_coefs):
        print(f"  {name:>14s}:  {mc:+.4f} ± {sc:.4f}")
    print(f"  {'intercept':>14s}:  {last_intercept:+.4f}  (last fold)")

    # --- Scatter plot --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    b_mask = feat_df["label"] == "B"
    c_mask = feat_df["label"] == "C"
    ax.scatter(
        feat_df.loc[b_mask, "token_len"],
        feat_df.loc[b_mask, "entity_count"],
        c="tab:blue", alpha=0.4, s=18, label="B (single-step)",
    )
    ax.scatter(
        feat_df.loc[c_mask, "token_len"],
        feat_df.loc[c_mask, "entity_count"],
        c="tab:red", alpha=0.4, s=18, label="C (multi-step)",
    )
    ax.set_xlabel("token_len (whitespace-split)")
    ax.set_ylabel("entity_count (spaCy NER)")
    ax.set_title(f"Clf2 Probe ({model}, {tag})  —  B vs C  (AUC={mean_auc:.3f})")
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(output_dir, f"clf2_feature_probe_scatter_{model}_{safe_tag}.png")
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
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_coefficients": mean_coef_dict,
        "intercept": last_intercept,
    }


def run_probe(model: str, data_path: str | None, output_dir: str, nlp) -> dict:
    """Run the full feature probe for one model variant.

    When *data_path* is ``None`` (auto-detect), evaluates on both the merged
    (silver + inductive-bias) and silver-only data files if both exist.
    The go/no-go verdict is based on the silver-only AUC (the stricter
    evaluation); the merged AUC is reported for comparison.

    Returns a dict with keys: model, merged, silver_only, verdict,
    verdict_auc, verdict_std.
    """
    sub_results: dict[str, dict | None] = {"merged": None, "silver-only": None}

    if data_path is not None:
        # Explicit override — single evaluation, no dual-source logic
        res = _evaluate_dataset(data_path, "override", model, output_dir, nlp)
        verdict_auc = res["mean_auc"]
        verdict_std = res["std_auc"]
        sub_results["override"] = res
    else:
        sources = detect_data_paths(model)
        has_silver = any(tag == "silver-only" for tag, _ in sources)
        if not has_silver:
            print(f"[WARN]  silver-only file not found for {model}; "
                  "proceeding with merged only.")

        for tag, path in sources:
            desc = ("merged (silver + inductive-bias)" if tag == "merged"
                    else "silver-only")
            print(f"\n=== Data: {desc} ===")
            res = _evaluate_dataset(path, tag, model, output_dir, nlp)
            sub_results[tag] = res

        # Verdict based on silver-only AUC; fall back to merged
        if sub_results["silver-only"] is not None:
            verdict_auc = sub_results["silver-only"]["mean_auc"]
            verdict_std = sub_results["silver-only"]["std_auc"]
        else:
            verdict_auc = sub_results["merged"]["mean_auc"]
            verdict_std = sub_results["merged"]["std_auc"]

    verdict = "GO" if verdict_auc >= 0.65 else "NO-GO"

    print()
    if verdict == "GO":
        print(f">>> VERDICT:  {verdict}  (AUC {verdict_auc:.4f} >= 0.65)")
    else:
        print(f">>> VERDICT:  {verdict}  (AUC {verdict_auc:.4f} < 0.65)")

    return {
        "model": model,
        "merged": sub_results.get("merged"),
        "silver_only": sub_results.get("silver-only"),
        "override": sub_results.get("override"),
        "verdict": verdict,
        "verdict_auc": verdict_auc,
        "verdict_std": verdict_std,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Clf2 feature probe: can structural features separate B vs C?"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Override path to Clf2 training JSON (must contain 'question' and 'answer' fields).",
    )
    parser.add_argument(
        "--model", type=str, default="flan_t5_xl",
        choices=("flan_t5_xl", "flan_t5_xxl", "gpt"),
        help="LLM variant for auto-detecting the data path (default: flan_t5_xl).",
    )
    parser.add_argument(
        "--all_models", action="store_true", default=False,
        help="Run probe for all model variants (flan_t5_xl, flan_t5_xxl, gpt) and print summary table.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for outputs (scatter plot + CSV). Default: same dir as this script.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[out]   {output_dir}")

    # --- Load spaCy ----------------------------------------------------
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        sys.exit(
            "ERROR: spaCy model 'en_core_web_sm' not found.\n"
            "Install with:  python -m spacy download en_core_web_sm"
        )

    # --- Determine models to probe ------------------------------------
    if args.all_models:
        models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    else:
        models = [args.model]

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

    # --- Summary table (when multiple models) -------------------------
    if len(results) > 1:
        print(f"\n\n{'=' * 90}")
        print("  SUMMARY")
        print(f"{'=' * 90}")
        header = (f"{'Model':<16s}| {'N(merged)':>9s} | {'N(silver)':>9s} "
                  f"| {'Merged AUC':>12s} | {'Silver AUC':>12s} | Verdict")
        print(header)
        print("-" * len(header))
        for r in results:
            m = r["merged"]
            s = r["silver_only"]
            n_merged = str(m["n_samples"]) if m else "-"
            n_silver = str(s["n_samples"]) if s else "-"
            auc_merged = f"{m['mean_auc']:.4f}" if m else "-"
            auc_silver = f"{s['mean_auc']:.4f}" if s else "-"
            print(f"{r['model']:<16s}| {n_merged:>9s} | {n_silver:>9s} "
                  f"| {auc_merged:>12s} | {auc_silver:>12s} | {r['verdict']}")
        print()

    # --- Persist all results to JSON ----------------------------------
    def _sub_to_json(sub: dict | None) -> dict | None:
        if sub is None:
            return None
        return {
            "n_samples": sub["n_samples"],
            "class_counts": {"B": sub["n_b"], "C": sub["n_c"]},
            "mean_auc": sub["mean_auc"],
            "std_auc": sub["std_auc"],
            "mean_accuracy": sub["mean_acc"],
            "std_accuracy": sub["std_acc"],
            "mean_coefficients": sub["mean_coefficients"],
            "intercept": sub["intercept"],
        }

    json_out: dict = {
        "models": {},
        "go_no_go_threshold": 0.65,
        "n_folds": 5,
        "random_state": 42,
    }
    for r in results:
        model_entry: dict = {}
        if r["merged"] is not None:
            merged_json = _sub_to_json(r["merged"])
            merged_json["verdict"] = "GO" if r["merged"]["mean_auc"] >= 0.65 else "NO-GO"
            model_entry["merged"] = merged_json
        if r["silver_only"] is not None:
            silver_json = _sub_to_json(r["silver_only"])
            silver_json["verdict"] = "GO" if r["silver_only"]["mean_auc"] >= 0.65 else "NO-GO"
            model_entry["silver_only"] = silver_json
        if r.get("override") is not None:
            override_json = _sub_to_json(r["override"])
            override_json["verdict"] = r["verdict"]
            model_entry["override"] = override_json
        json_out["models"][r["model"]] = model_entry

    json_path = os.path.join(output_dir, "clf2_feature_probe_results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"[save]  {json_path}")


if __name__ == "__main__":
    main()
