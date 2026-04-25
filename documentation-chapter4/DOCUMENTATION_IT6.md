## 1. Files Involved

| File | Role |
|---|---|
| classifier/postprocess/clf2_feature_probe.py | **The entire Iteration 6.** Standalone diagnostic script: extracts features, trains logistic regression, evaluates, plots, saves CSV. |
| classifier/utils.py | Not imported — the probe is self-contained |
| classifier/run_classifier.py | Not imported — the probe does not use T5 at all |

**External dependencies** (Python libraries, not project files):
- `spacy` with `en_core_web_sm` ([line 170](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L170))
- `sklearn` — `LogisticRegression`, `train_test_split`, `roc_auc_score`, `accuracy_score`, `classification_report` ([lines 30–35](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L30-L35))
- `pandas`, `matplotlib` ([lines 28–29](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L28-L29))

**Data input** (resolved at clf2_feature_probe.py):
- `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/binary_silver_single_vs_multi/train.json` (preferred)
- Falls back to `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/silver/single_vs_multi/train.json`

The files from later **Iteration 7** — classifier/data_utils/add_feature_prefix.py and classifier/run/run_large_train_feat_single_vs_multi.sh — are **not** part of Iteration 6. They consume the same feature logic but inject it into the T5 pipeline, which is a separate experiment.

---

## 2. Diagnostic-Only Scope — Confirmed

The probe is **completely isolated** from the main pipeline:

- It does **not** import run_classifier.py, utils.py, or any `postprocess_utils` module.
- It does **not** load or modify any T5 model weights.
- It does **not** write to any directory under `classifier/outputs/` or `predictions/`.
- It reads the Clf2 training JSON as **input only** and writes two output artifacts to its own directory ([lines 192, 251–252](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L192-L252)).
- The shell scripts for Gate 1 (`*_no_ret_vs_ret.sh`) and Gate 2 (`*_single_vs_multi.sh`) are entirely untouched.
- Gate 2 in the live cascade pipeline remains the standard T5-Large classifier from Iteration 1.

---

## 3. Structural Features

Three features are extracted per question ([lines 6–9](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L6-L9) and clf2_feature_probe.py):

| Feature | Extraction | Library |
|---|---|---|
| `token_len` | `len(doc.text.split())` — whitespace-split token count | Pure Python |
| `entity_count` | `len(doc.ents)` — named entity count | **spaCy** `en_core_web_sm` (NER pipeline only; parser and lemmatizer disabled at clf2_feature_probe.py) |
| `bridge_flag` | `1 if _BRIDGE_RE.search(doc.text) else 0` — binary match against 7 regex patterns | Python `re` module |

**Bridge patterns** ([lines 45–75](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L45-L75)) target multi-hop syntactic structures:
1. Relative-clause bridges: `who/where/which/that + was/were/is/...`
2. Double possessive: `X's ... Y's`
3. Temporal subordination before a wh-word: `before/after/when ... who/what/where`
4. Demonstrative back-reference: `that country/person/team/...`
5. Explicit comparison: `both ... and`, `between ... and`
6. Nested wh-questions: `of the X who/that/which/where`

**Computation timing:** Features are computed **at runtime** during probe execution, not precomputed or cached. spaCy processes questions in batches of 256 via `nlp.pipe()` ([line 121](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L121)). The resulting feature table is saved to CSV for later inspection, but there is no persistent cache mechanism.

---

## 4. Diagnostic Classifier

**Model:** `LogisticRegression(max_iter=1000, random_state=42)` from scikit-learn ([line 206](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L206)).

**Hyperparameters** (all scikit-learn defaults except those listed):
| Parameter | Value |
|---|---|
| `max_iter` | 1000 |
| `random_state` | 42 |
| Solver | `lbfgs` (sklearn default) |
| Penalty | `l2` (sklearn default) |
| `C` (regularization) | `1.0` (sklearn default) |

**Label encoding:** Binary — `y = (label == "C").astype(int)`, so 1=C (multi-step), 0=B (single-step) ([line 203](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L203)).

**Training data:** The same `binary_silver_single_vs_multi/train.json` used by Gate 2 in Iteration 1 — the merged silver + binary inductive-bias labels, filtered to B and C only ([lines 100–106](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L100-L106)). The `--model` argument defaults to `flan_t5_xl` ([line 146](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L146)), so by default it uses the xl variant.

---

## 5. No Text Features — Confirmed

The feature matrix `X` is constructed exclusively from the three numeric columns ([line 201](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L201)):

```python
X = feat_df[["token_len", "entity_count", "bridge_flag"]].values
```

No question text, no token embeddings, no TF-IDF, no bag-of-words, and no T5 representations are included in `X`. The diagnostic classifier operates on a 3-dimensional numeric feature vector only, cleanly isolating the discriminative signal of structure alone.

---

## 6. Evaluation of the Diagnostic Classifier

**Split:** 80/20 stratified train/test split, `random_state=42` ([lines 198–200](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L198-L200)):
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y,
)
```

This is an internal train/test split of the Clf2 training data — **not** the pipeline's validation or test split.

**Metrics reported** ([lines 211–224](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L211-L224)):

| Metric | Implementation |
|---|---|
| ROC-AUC | `roc_auc_score(y_test, y_prob)` where `y_prob = clf.predict_proba(X_test)[:, 1]` |
| Accuracy | `accuracy_score(y_test, y_pred)` |
| Classification report | `classification_report(y_test, y_pred, target_names=["B (single)", "C (multi)"])` — includes per-class precision, recall, F1 |

**Feature importances** ([lines 227–231](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L227-L231)):
- Logistic regression coefficients (`clf.coef_[0]`) for each of the 3 features
- Intercept (`clf.intercept_[0]`)
- Positive coefficients indicate signal toward class C (multi-step)

**Go/no-go verdict** ([lines 234–237](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L234-L237)):
```python
if auc >= 0.65:
    print(f">>> VERDICT:  GO  (AUC {auc:.4f} >= 0.65)")
else:
    print(f">>> VERDICT:  NO-GO  (AUC {auc:.4f} < 0.65)")
```

---

## 7. Gate 1

The probe script does **not involve Gate 1 at all**. It only evaluates the diagnostic classifier's ability to separate B vs C in isolation. There is no cascade routing and no A/R decision.

If results from this probe were later combined with a cascade for end-to-end QA evaluation (Iteration 7), the Gate 1 used would be specified by whichever routing script is invoked — but that is outside the scope of Iteration 6 itself.

---

## 8. End-to-End Evaluation

**No end-to-end QA F1 evaluation is performed in Iteration 6.** The script does not call evaluate_final_acc.py, does not load QA predictions from `predictions/test/`, and does not invoke any routing logic. The empty "Full Pipeline Results" section in OFFICIAL_EXP_RESULTS.md confirms this was intentionally left blank — the diagnostic probe produces classifier-level metrics only.

---

## 9. Output Artifacts

Two files are written to `--output_dir` (defaults to the same directory as the script, i.e., `classifier/postprocess/`):

| Artifact | Path | Content |
|---|---|---|
| CSV | classifier/postprocess/clf2_feature_probe_data.csv | Full feature table: `token_len`, `entity_count`, `bridge_flag`, `label`, `question`, `id` for every B/C sample |
| Scatter plot | classifier/postprocess/clf2_feature_probe_scatter.png | `token_len` vs `entity_count` coloured by B/C, with AUC in the title |

These are in a **completely separate location** from all other iteration outputs (which go to `classifier/outputs/...` or `predictions/classifier/...`). No overlap with Iterations 1–5.

All numeric results (AUC, accuracy, classification report, coefficients, verdict) are printed to stdout only — they are not persisted to a JSON file.

---

## 10. Suspicious Items / Flags

1. **Evaluation on a subset of training data, not a held-out split.** The probe takes the Clf2 *training* file (`binary_silver_single_vs_multi/train.json`) and does an 80/20 internal split ([line 198](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L198-L200)). It does **not** evaluate on the pipeline's silver validation set (`silver/single_vs_multi/valid.json`). This is acceptable for a feasibility probe (the question is "can these features discriminate at all?"), but the reported accuracy/AUC may not transfer to the actual validation distribution, especially since the training file contains ~2,400 binary inductive-bias labels that are deterministically assigned by dataset identity (single-hop datasets → B, multi-hop → C). Those labels make the task artificially easier — structural features likely correlate strongly with dataset origin.

2. **No leakage into the main Gate 2 pipeline.** The probe writes only to `classifier/postprocess/` (CSV + PNG). It does not modify any training data, model checkpoint, or routing script. Iteration 7's files (add_feature_prefix.py, run_large_train_feat_single_vs_multi.sh) are entirely separate scripts that would need to be explicitly run. No premature contamination.

3. **Binary inductive-bias labels inflate probe metrics.** The 2,400 binary labels in the merged training file encode a near-perfect dataset→label mapping. Since structural features (especially `bridge_flag` and `entity_count`) also correlate with dataset identity, the reported AUC may overestimate the probe's ability to discriminate genuinely ambiguous B vs C cases. The silver-only subset (~868 samples for xl) would be a stricter evaluation — the script supports this via `--data_path` pointing to `silver/single_vs_multi/train.json`, but the default path favours the inflated merged file.

4. **Feature summary stats are computed on all data, but classification is on the 80% split.** The "Feature means by class" table ([line 189](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L189)) uses the full dataset, not just the training split. This is fine for descriptive statistics but could mislead if someone compares those means to the classifier's decision boundary.

5. **Results not persisted to JSON.** All numeric outputs go to stdout. If the terminal output is lost, there's no recoverable record other than the CSV (which has raw data but not the model's metrics). A minor robustness gap.

6. **`--model` defaults to `flan_t5_xl`.** Running the probe without arguments evaluates only the xl variant. To assess gpt or xxl, the user must explicitly pass `--model`. This isn't a bug, but it means a single invocation doesn't cover all LLM variants.