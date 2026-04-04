#!/usr/bin/env python3
"""
Clf2 (Gate 2) feasibility probe: can frozen sentence embeddings separate
B (single-step) from C (multi-step) better than surface features?

Encodes questions with all-MiniLM-L6-v2 (384-dim, CPU-only), trains logistic
regression on 80/20 stratified split, reports ROC-AUC + classification report,
produces a PCA scatter plot, and compares against the Iter 3b surface-feature
probe (AUC=0.676).

Usage:
    python classifier/postprocess/clf2_embedding_probe.py
    python classifier/postprocess/clf2_embedding_probe.py --model flan_t5_xxl
"""

import argparse
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print("Install with:  pip install sentence-transformers")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRIMARY_MODEL = "all-MiniLM-L6-v2"
FALLBACK_MODEL = "paraphrase-MiniLM-L3-v2"
SURFACE_FEATURE_AUC = 0.676  # Iter 3b result
GO_THRESHOLD = 0.75
ENCODE_SPEED_EST = 500  # questions/sec on CPU (rough MiniLM estimate)


def detect_data_path(model: str) -> str:
    """Auto-detect the Clf2 training JSON."""
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    base = os.path.join(
        repo_root,
        "classifier", "data", "musique_hotpot_wiki2_nq_tqa_sqd",
    )
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


def load_encoder(model_name: str) -> tuple:
    """Try to load sentence-transformers model; fall back if needed."""
    try:
        encoder = SentenceTransformer(model_name)
        return encoder, model_name, False
    except Exception:
        if model_name == PRIMARY_MODEL:
            print(f"[warn]  {PRIMARY_MODEL} not available, falling back to {FALLBACK_MODEL}")
            try:
                encoder = SentenceTransformer(FALLBACK_MODEL)
                return encoder, FALLBACK_MODEL, True
            except Exception as e2:
                sys.exit(f"ERROR: could not load fallback model {FALLBACK_MODEL}: {e2}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Clf2 embedding probe: sentence embeddings for B vs C separability"
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
        "--output_dir", type=str, default=None,
        help="Directory for outputs. Default: same dir as this script.",
    )
    args = parser.parse_args()

    # --- Resolve paths -------------------------------------------------
    data_path = args.data_path or detect_data_path(args.model)
    print(f"[data]  {data_path}")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[out]   {output_dir}")

    # --- Load data -----------------------------------------------------
    bc_data = load_bc_data(data_path)
    questions = [item["question"] for item in bc_data]
    labels = [item["answer"] for item in bc_data]
    n_b = labels.count("B")
    n_c = labels.count("C")
    print(f"[data]  {len(bc_data)} B/C items  (B={n_b}, C={n_c})")

    # --- Load encoder --------------------------------------------------
    est_time = len(questions) / ENCODE_SPEED_EST
    print(f"[enc]   estimated encoding time: ~{est_time:.1f}s "
          f"({len(questions)} questions @ ~{ENCODE_SPEED_EST}/s on CPU)")

    encoder, encoder_name, used_fallback = load_encoder(PRIMARY_MODEL)
    emb_dim = encoder.get_sentence_embedding_dimension()
    print(f"[enc]   model: {encoder_name}  (dim={emb_dim})"
          + ("  [FALLBACK]" if used_fallback else ""))

    # --- Encode --------------------------------------------------------
    print("[enc]   encoding questions ...")
    t0 = time.time()
    embeddings = encoder.encode(
        questions, batch_size=64, show_progress_bar=True, normalize_embeddings=False,
    )
    t1 = time.time()
    actual_speed = len(questions) / (t1 - t0)
    print(f"[enc]   done in {t1 - t0:.1f}s  ({actual_speed:.0f} questions/s)")

    # --- Embedding norms -----------------------------------------------
    norms = np.linalg.norm(embeddings, axis=1)

    # --- Train / evaluate ----------------------------------------------
    y = np.array([1 if lab == "C" else 0 for lab in labels])  # 1 = C, 0 = B

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.20, random_state=42, stratify=y,
    )

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)

    print("\n========== RESULTS ==========")
    print(f"ROC-AUC:  {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Samples:  train={len(y_train)}, test={len(y_test)}")
    print()
    print(classification_report(
        y_test, y_pred, target_names=["B (single)", "C (multi)"],
    ))

    # --- PCA scatter plot ----------------------------------------------
    print("[pca]   fitting PCA (2 components) ...")
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(embeddings)
    pca_x = pca_coords[:, 0]
    pca_y = pca_coords[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    b_mask = np.array(labels) == "B"
    c_mask = np.array(labels) == "C"
    ax.scatter(pca_x[b_mask], pca_y[b_mask],
               c="tab:blue", alpha=0.35, s=14, label=f"B (single, n={n_b})")
    ax.scatter(pca_x[c_mask], pca_y[c_mask],
               c="tab:red", alpha=0.35, s=14, label=f"C (multi, n={n_c})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"Clf2 Embedding Probe — B vs C  (AUC={auc:.3f}, {encoder_name})")
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(output_dir, "clf2_embedding_probe_pca.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[save]  {plot_path}")

    # --- Save CSV ------------------------------------------------------
    # Build full predictions for CSV (re-predict on all data)
    all_pred = clf.predict(embeddings)

    csv_df = pd.DataFrame({
        "question": questions,
        "label": labels,
        "pca_x": pca_x,
        "pca_y": pca_y,
        "predicted_label": ["C" if p == 1 else "B" for p in all_pred],
        "correct": [lab == ("C" if p == 1 else "B") for lab, p in zip(labels, all_pred)],
        "embedding_norm": norms,
    })
    csv_path = os.path.join(output_dir, "clf2_embedding_probe_data.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"[save]  {csv_path}")

    # --- Comparison table ----------------------------------------------
    print()
    print("=" * 55)
    print("  Probe Comparison: Gate 2 B/C Separability")
    print("=" * 55)
    print(f"  {'Method':<35s} {'AUC':>6s}")
    print(f"  {'-'*35} {'-'*6}")
    print(f"  {'Surface features (Iter 3b)':<35s} {SURFACE_FEATURE_AUC:>6.3f}")
    print(f"  {'Sentence embeddings (this)':<35s} {auc:>6.3f}")
    print(f"  {'-'*35} {'-'*6}")
    print(f"  {'Go/no-go threshold':<35s} {GO_THRESHOLD:>6.3f}")
    print()
    if auc >= GO_THRESHOLD:
        print(f"  >>> VERDICT:  GO  (AUC {auc:.4f} >= {GO_THRESHOLD})")
    else:
        print(f"  >>> VERDICT:  NO-GO  (AUC {auc:.4f} < {GO_THRESHOLD})")
    print("=" * 55)

    if used_fallback:
        print(f"\n[note]  Results used fallback model '{encoder_name}' — "
              f"re-run with '{PRIMARY_MODEL}' for best accuracy.")


if __name__ == "__main__":
    main()
