#!/usr/bin/env python3
"""
Iteration 3b: prepend structural features to Clf2 question strings.

Reads the original Clf2 JSON splits (train, valid, predict) and writes
new copies with the question field prefixed:
    [LEN:X] [ENT:Y] [BRIDGE:Z] <original question>

Feature extraction reuses the spaCy + regex logic from
classifier/postprocess/clf2_feature_probe.py.

Usage:
    python classifier/data_utils/add_feature_prefix.py

Outputs (for each model in flan_t5_xl, flan_t5_xxl, gpt):
    .../binary_silver_feat_single_vs_multi/train.json
    .../silver_feat_single_vs_multi/valid.json
    .../feat_predict.json
"""

import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Bridging-phrase patterns (copied from clf2_feature_probe.py)
# ---------------------------------------------------------------------------
BRIDGE_PATTERNS = [
    r"\b(?:who|where|which|that)\s+(?:was|were|is|are|did|had|has|does)\b",
    r"\w+'s\s+\w+(?:\s+\w+){0,5}\s+\w+'s",
    r"\b(?:before|after|when|while)\b.{3,60}\b(?:who|what|where|which)\b",
    r"\b(?:that|this|those|these)\s+(?:country|city|person|team|company|film|movie|album|book|organization|university|school)\b",
    r"\b(?:both)\b.{1,40}\band\b",
    r"\bbetween\b.{1,40}\band\b",
    r"\bof\s+the\s+\w+\s+(?:who|that|which|where)\b",
]
_BRIDGE_RE = re.compile("|".join(BRIDGE_PATTERNS), re.IGNORECASE)


def compute_prefix(question: str, nlp) -> str:
    """Return '[LEN:X] [ENT:Y] [BRIDGE:Z] ' for a question string."""
    token_len = len(question.split())
    doc = nlp(question)
    entity_count = len(doc.ents)
    bridge_flag = 1 if _BRIDGE_RE.search(question) else 0
    return f"[LEN:{token_len}] [ENT:{entity_count}] [BRIDGE:{bridge_flag}] "


def augment_file(input_path: str, output_path: str, nlp) -> int:
    """Read JSON array, prepend feature prefix to each 'question', write out.
    Returns the number of items processed."""
    with open(input_path) as f:
        data = json.load(f)

    # Batch NER for speed
    questions = [item["question"] for item in data]
    docs = list(nlp.pipe(questions, batch_size=512))

    for item, doc in zip(data, docs):
        q = item["question"]
        token_len = len(q.split())
        entity_count = len(doc.ents)
        bridge_flag = 1 if _BRIDGE_RE.search(q) else 0
        item["question"] = (
            f"[LEN:{token_len}] [ENT:{entity_count}] [BRIDGE:{bridge_flag}] {q}"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return len(data)


def main():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        sys.exit(
            "ERROR: spaCy model 'en_core_web_sm' not found.\n"
            "Install with:  python -m spacy download en_core_web_sm"
        )

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    base = os.path.join(
        repo_root, "classifier", "data", "musique_hotpot_wiki2_nq_tqa_sqd"
    )

    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]

    for model in models:
        print(f"\n=== {model} ===")

        # 1. Training data: binary_silver_single_vs_multi → binary_silver_feat_single_vs_multi
        train_in = os.path.join(base, model, "binary_silver_single_vs_multi", "train.json")
        train_out = os.path.join(base, model, "binary_silver_feat_single_vs_multi", "train.json")
        n = augment_file(train_in, train_out, nlp)
        print(f"  train: {n:>5d} items  →  {train_out}")

        # 2. Validation data: silver/single_vs_multi/valid.json → silver_feat_single_vs_multi/valid.json
        valid_in = os.path.join(base, model, "silver", "single_vs_multi", "valid.json")
        valid_out = os.path.join(base, model, "silver_feat_single_vs_multi", "valid.json")
        n = augment_file(valid_in, valid_out, nlp)
        print(f"  valid: {n:>5d} items  →  {valid_out}")

    # 3. Predict data (shared across models): predict.json → feat_predict.json
    predict_in = os.path.join(base, "predict.json")
    predict_out = os.path.join(base, "feat_predict.json")
    n = augment_file(predict_in, predict_out, nlp)
    print(f"\n  predict: {n:>5d} items  →  {predict_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
