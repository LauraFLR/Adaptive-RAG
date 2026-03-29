#!/usr/bin/env python3
"""
Iteration 3b feasibility check: can simple structural query features
separate B (single-step) from C (multi-step) better than chance?

Extracts three features per question:
  - token_len:    whitespace-split token count
  - entity_count: named-entity count (spaCy en_core_web_sm)
  - bridge_flag:  regex match for multi-hop bridging phrases

Trains a logistic regression on 80/20 stratified split and reports
ROC-AUC, accuracy, classification report, and per-feature coefficients.

Usage:
    python classifier/postprocess/clf2_feature_probe.py
    python classifier/postprocess/clf2_feature_probe.py --data_path path/to/train.json
    python classifier/postprocess/clf2_feature_probe.py --model flan_t5_xxl
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
from sklearn.model_selection import train_test_split

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


def detect_data_path(model: str) -> str:
    """Auto-detect the Clf2 training JSON (binary_silver merged preferred)."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    base = os.path.join(
        repo_root,
        "classifier", "data", "musique_hotpot_wiki2_nq_tqa_sqd",
    )
    # Prefer the larger merged file; fall back to silver-only
    candidates = [
        os.path.join(base, model, "binary_silver_single_vs_multi", "train.json"),
        os.path.join(base, model, "silver", "single_vs_multi", "train.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    sys.exit(
        f"ERROR: could not find Clf2 training data for model={model}.\n"
        f"Searched:\n  " + "\n  ".join(candidates) + "\n"
        "Pass --data_path explicitly."
    )


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
        "--output_dir", type=str, default=None,
        help="Directory for outputs (scatter plot + CSV). Default: same dir as this script.",
    )
    args = parser.parse_args()

    # --- Resolve paths -------------------------------------------------
    data_path = args.data_path or detect_data_path(args.model)
    print(f"[data]  {data_path}")

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

    # --- Load & filter data -------------------------------------------
    bc_data = load_bc_data(data_path)
    questions = [item["question"] for item in bc_data]
    labels = [item["answer"] for item in bc_data]
    print(f"[data]  {len(bc_data)} B/C items  "
          f"(B={labels.count('B')}, C={labels.count('C')})")

    # --- Extract features ---------------------------------------------
    print("[feat]  extracting features (spaCy NER + regex) ...")
    feat_df = extract_features(questions, nlp)
    feat_df["label"] = labels
    feat_df["question"] = questions
    feat_df["id"] = [item["id"] for item in bc_data]

    # Summary stats
    print("\n--- Feature means by class ---")
    print(feat_df.groupby("label")[["token_len", "entity_count", "bridge_flag"]].mean()
          .round(3).to_string())
    print()

    # --- Save CSV ------------------------------------------------------
    csv_path = os.path.join(output_dir, "clf2_feature_probe_data.csv")
    feat_df.to_csv(csv_path, index=False)
    print(f"[save]  {csv_path}")

    # --- Train / evaluate ---------------------------------------------
    X = feat_df[["token_len", "entity_count", "bridge_flag"]].values
    y = (feat_df["label"] == "C").astype(int).values  # 1 = C, 0 = B

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    print("\n========== RESULTS ==========")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print()
    print(classification_report(
        y_test, y_pred, target_names=["B (single)", "C (multi)"],
    ))

    # Per-feature coefficients
    feature_names = ["token_len", "entity_count", "bridge_flag"]
    coefs = clf.coef_[0]
    print("--- Feature coefficients (positive → C) ---")
    for name, coef in zip(feature_names, coefs):
        print(f"  {name:>14s}:  {coef:+.4f}")
    print(f"  {'intercept':>14s}:  {clf.intercept_[0]:+.4f}")

    # --- Verdict -------------------------------------------------------
    print()
    if auc >= 0.65:
        print(f">>> VERDICT:  GO  (AUC {auc:.4f} >= 0.65)")
    else:
        print(f">>> VERDICT:  NO-GO  (AUC {auc:.4f} < 0.65)")

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
    ax.set_title(f"Clf2 Feature Probe  —  B vs C  (AUC={auc:.3f})")
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "clf2_feature_probe_scatter.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[save]  {plot_path}")


if __name__ == "__main__":
    main()
