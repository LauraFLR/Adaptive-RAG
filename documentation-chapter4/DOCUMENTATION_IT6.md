# Iteration 6 — Diagnostic Logistic Regression Probe for Gate 2 Features

> **Design Science Research Artifact:** A standalone diagnostic script that
> tests whether three simple structural query features (token length, named-
> entity count, bridging-phrase flag) carry discriminative signal for the
> B-vs-C (single-step vs multi-step retrieval) decision.  No T5 model is
> loaded, no checkpoint is written, no downstream QA evaluation is performed.
> This is a **feasibility check only** — a go/no-go gate that decides whether
> it is worth injecting these features into a full Gate 2 classifier.

---

## 1. Files Involved

| File | Role |
|---|---|
| `classifier/postprocess/clf2_feature_probe.py` (443 lines) | **The ENTIRE iteration.** One standalone script: feature extraction, logistic-regression 5-fold CV, scatter plot, CSV, JSON results. |

**This is the only file in IT6.** There is no shell wrapper, no config file, no supporting training script.

### 1.1 External dependencies

| Library | Import line | Purpose |
|---|---|---|
| `spacy` (+ `en_core_web_sm` model) | [L351] | Named-entity recognition for `entity_count` feature |
| `sklearn` (`LogisticRegression`, `StratifiedKFold`, `roc_auc_score`, `accuracy_score`, `classification_report`, `cross_val_predict`) | [L32–38] | Diagnostic classifier + evaluation metrics |
| `pandas` | [L31] | DataFrame for feature matrix, CSV export |
| `matplotlib` | [L28–30] | Scatter plot (Agg backend, no display) |
| `numpy` | [L142] (imported inside `_evaluate_dataset`) | Array operations for fold aggregation |

### 1.2 NOT imported

| Library | Relevance |
|---|---|
| `torch` | Not imported. No tensor operations. |
| `transformers` | Not imported. No T5 model, no tokenizer, no `generate()`. |
| `run_classifier.py` | Not imported. No training loop, no `FocalLossTrainer`. |
| `utils.py` | Not imported. No `load_model()`, no `preprocess_features_function()`. |
| `accelerate` | Not imported. No distributed training. |

---

## 2. Diagnostic-Only Scope — Confirmed

### 2.1 No T5 model loaded

The script loads **only** a spaCy `en_core_web_sm` NLP pipeline [L351–352]:

```python
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
```

This is a small (12 MB) NER-only model used exclusively for entity counting. No T5 model, no HuggingFace model, no GPU allocation.

### 2.2 No writes to classifier/outputs/ or predictions/

The script's write targets are controlled by `--output_dir`, which defaults to `os.path.dirname(os.path.abspath(__file__))` [L349] — i.e. `classifier/postprocess/` itself. It writes:

| Artifact | Path pattern |
|---|---|
| CSV | `{output_dir}/clf2_feature_probe_data_{model}_{tag}.csv` [L168] |
| Scatter plot | `{output_dir}/clf2_feature_probe_scatter_{model}_{tag}.png` [L241] |
| JSON results | `{output_dir}/clf2_feature_probe_results.json` [L437] |

None of these targets are inside `classifier/outputs/` (the checkpoint tree) or `predictions/` (the QA predictions tree).

### 2.3 No modification of training data or checkpoints

The script **reads** Clf2 training JSON files via `load_bc_data()` [L105–112] but does not write to them. There is no `save_json()`, no `shutil.copy()`, no `open(..., "w")` targeting any data or checkpoint path.

### 2.4 Iteration 7's files are separate

The probe is purely diagnostic. If the go/no-go verdict is "GO," a **separate** script in a future iteration would incorporate the features into an actual Gate 2 classifier. This script does not feed into any downstream training pipeline.

---

## 3. Structural Features

### 3.1 Feature table

| Feature | Extraction method | Library | Line(s) |
|---|---|---|---|
| `token_len` | `len(doc.text.split())` — whitespace-split token count | Built-in `str.split()` (spaCy doc only used for its `.text`) | [L123] |
| `entity_count` | `len(doc.ents)` — number of named entities | spaCy `en_core_web_sm` NER | [L124] |
| `bridge_flag` | `1 if _BRIDGE_RE.search(doc.text) else 0` — binary regex match | `re` stdlib | [L125] |

All three features are extracted in `extract_features()` [L115–131], which processes questions in batches via `nlp.pipe(questions, batch_size=256)` [L122].

### 3.2 Bridge-flag regex patterns

Seven regex patterns are defined in `BRIDGE_PATTERNS` [L59–73] and compiled into a single alternation at [L74]:

```python
_BRIDGE_RE = re.compile("|".join(BRIDGE_PATTERNS), re.IGNORECASE)
```

| # | Pattern | Description | Example match |
|---|---|---|---|
| 1 | `\b(?:who\|where\|which\|that)\s+(?:was\|were\|is\|are\|did\|had\|has\|does)\b` | Relative-clause bridges linking two entities | "the person **who was** born in…" |
| 2 | `\w+'s\s+\w+(?:\s+\w+){0,5}\s+\w+'s` | Double possessive — two possessives suggest two hops | "**Obama's** mother **'s** birthplace" |
| 3 | `\b(?:before\|after\|when\|while)\b.{3,60}\b(?:who\|what\|where\|which)\b` | Temporal/causal subordination before a wh-word | "**after** X was elected, **what** happened…" |
| 4 | `\b(?:that\|this\|those\|these)\s+(?:country\|city\|person\|team\|company\|film\|movie\|album\|book\|organization\|university\|school)\b` | Demonstrative back-reference to a prior fact | "**that country**'s capital" |
| 5 | `\b(?:both)\b.{1,40}\band\b` | Explicit comparison linking two entities ("both X and Y") | "**both** France **and** Germany" |
| 6 | `\bbetween\b.{1,40}\band\b` | Explicit comparison ("between X and Y") | "**between** Paris **and** Berlin" |
| 7 | `\bof\s+the\s+\w+\s+(?:who\|that\|which\|where)\b` | Nested wh-question ("X of the Y that…") | "the capital **of the country that** won…" |

All patterns are case-insensitive [L74: `re.IGNORECASE`]. The `bridge_flag` is 1 if **any** of the seven patterns matches anywhere in the question string, 0 otherwise.

### 3.3 Computation timing

Features are computed **at runtime** from the raw question text. There is no caching, no pre-computed feature file. The spaCy `nlp.pipe()` call processes all questions in a single batch [L122], but NER inference still takes a few seconds for thousands of questions. The regex match is negligible.

---

## 4. Diagnostic Classifier

### 4.1 Model: Logistic Regression

A fresh `LogisticRegression` instance is created **per fold** at [L182]:

```python
clf = LogisticRegression(max_iter=1000, random_state=42)
```

### 4.2 Hyperparameters

| Parameter | Value | Source | Note |
|---|---|---|---|
| `max_iter` | `1000` | [L182] | Explicit |
| `random_state` | `42` | [L182] | Explicit |
| `penalty` | `"l2"` | sklearn default | Not set; L2 regularization |
| `C` | `1.0` | sklearn default | Inverse regularization strength |
| `solver` | `"lbfgs"` | sklearn default | Not set |
| `class_weight` | `None` | sklearn default | No class balancing |
| `fit_intercept` | `True` | sklearn default | Intercept is reported |
| `tol` | `1e-4` | sklearn default | Convergence tolerance |
| `multi_class` | `"auto"` | sklearn default | Binary in this case |

### 4.3 Label encoding

At [L171]:

```python
y = (feat_df["label"] == "C").astype(int).values
```

| Encoded value | Original label | Meaning |
|---|---|---|
| 0 | B | Single-step retrieval |
| 1 | C | Multi-step retrieval |

Positive class = C. A positive LR coefficient means "this feature pushes toward C (multi-step)."

### 4.4 Training data

The data source depends on `--model` and `--data_path`:

**Auto-detection** (when `--data_path` is not specified): `detect_data_paths()` [L77–102] checks for two files in priority order:

| Tag | Path pattern | Contents |
|---|---|---|
| `"merged"` | `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/binary_silver_single_vs_multi/train.json` | Silver labels + inductive-bias labels combined |
| `"silver-only"` | `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/silver/single_vs_multi/train.json` | Silver labels only |

Both files are evaluated if they exist. The **go/no-go verdict** is based on the silver-only AUC (the stricter evaluation) [L296–301]; the merged AUC is reported for comparison.

**Filtering:** `load_bc_data()` [L105–112] loads the full JSON array and keeps only items where `item["answer"]` is `"B"` or `"C"` [L109]:

```python
bc = [item for item in data if item.get("answer") in ("B", "C")]
```

Items with label `"A"` (if any exist in the file) are discarded.

### 4.5 Default model

`--model` defaults to `flan_t5_xl` [L332]:

```python
parser.add_argument("--model", type=str, default="flan_t5_xl", ...)
```

The `--all_models` flag [L337] runs the probe for all three variants (`flan_t5_xl`, `flan_t5_xxl`, `gpt`) sequentially and prints a summary table.

### 4.6 Cross-validation setup

5-fold stratified CV at [L174]:

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

| CV parameter | Value |
|---|---|
| `n_splits` | 5 |
| `shuffle` | `True` |
| `random_state` | 42 |

Each fold: train on 80 %, evaluate on 20 %. A new `LogisticRegression` is fitted from scratch per fold.

---

## 5. No Text Features — Confirmed

The feature matrix is constructed at [L170]:

```python
X = feat_df[["token_len", "entity_count", "bridge_flag"]].values
```

This selects **exactly three numeric columns**. No other columns are included:

| NOT in X | Why |
|---|---|
| `question` (text) | Stored in `feat_df` for CSV export [L157], but not in `X` |
| `label` | Used only for `y` [L171] |
| `id` | Stored for CSV export [L158], not in `X` |
| TF-IDF / BoW | Not computed anywhere in the script |
| Embeddings | No embedding model is loaded |
| T5 hidden states | No T5 model is loaded |

The diagnostic classifier sees only structural metadata about the question, not its semantic content.

---

## 6. Evaluation of the Diagnostic Classifier

### 6.1 Split: 5-fold stratified CV

Each of the 5 folds uses an 80/20 stratified train/test split (inherent to `StratifiedKFold(n_splits=5)` [L174]). Stratification ensures each fold preserves the B/C class ratio.

### 6.2 Metrics

| Metric | Function | Scope | Line(s) |
|---|---|---|---|
| ROC-AUC | `sklearn.metrics.roc_auc_score(y_test, y_prob)` | Per-fold, then mean ± std | [L187, L195–196] |
| Accuracy | `sklearn.metrics.accuracy_score(y_test, y_pred)` | Per-fold, then mean ± std | [L188, L197–198] |
| Classification report | `sklearn.metrics.classification_report(last_fold_y_test, last_fold_y_pred, target_names=["B (single)", "C (multi)"])` | **Last fold only** | [L203–206] |

The AUC uses `predict_proba()[:, 1]` [L186] (probability of class C). Accuracy uses `predict()` [L185] (hard 0.5 threshold).

### 6.3 Feature importances

LR coefficients and intercept are reported at [L211–216]:

```python
mean_coefs = fold_coefs.mean(axis=0)
std_coefs  = fold_coefs.std(axis=0)
```

- **Coefficients:** Mean ± std across 5 folds, for each of the 3 features. Positive coefficient → pushes toward C (multi-step).
- **Intercept:** Reported from the **last fold only** [L214]: `float(clf.intercept_[0])`.

### 6.4 Go/no-go verdict

At [L303]:

```python
verdict = "GO" if verdict_auc >= 0.65 else "NO-GO"
```

| Condition | Verdict | Meaning |
|---|---|---|
| Mean ROC-AUC ≥ 0.65 | **GO** | Structural features carry enough signal to proceed with a feature-augmented Gate 2 classifier |
| Mean ROC-AUC < 0.65 | **NO-GO** | Features are insufficiently discriminative; do not proceed |

The threshold `0.65` is also persisted in the JSON output [L417]:

```python
json_out: dict = { ..., "go_no_go_threshold": 0.65, ... }
```

When both merged and silver-only data exist, the verdict is based on the **silver-only** AUC [L296–301] (the stricter evaluation, since inductive-bias labels may inflate performance).

---

## 7. Gate 1

**The probe does NOT involve Gate 1 at all.**

- No A/R classification is performed.
- No Clf1 checkpoint or prediction file is loaded.
- The input data is filtered to B/C items only [L109].
- The script is entirely about the Gate 2 (B vs C) decision boundary.

---

## 8. End-to-End Evaluation

**NO end-to-end QA F1 evaluation is performed.**

- `evaluate_final_acc.py` is not imported or invoked.
- No QA prediction files are routed.
- No per-dataset EM/F1 scores are computed.
- The only metrics are the diagnostic classifier's ROC-AUC, accuracy, and classification report (sklearn metrics on B/C labels, not QA answers).

---

## 9. Output Artifacts

### 9.1 File table

| Artifact | Path pattern | Content |
|---|---|---|
| CSV | `{output_dir}/clf2_feature_probe_data_{model}_{tag}.csv` [L168] | Per-question row: `token_len`, `entity_count`, `bridge_flag`, `label`, `question`, `id`. One file per (model, data-source) combination. |
| Scatter plot | `{output_dir}/clf2_feature_probe_scatter_{model}_{tag}.png` [L241] | `token_len` (x) vs `entity_count` (y), colored by B (blue) / C (red). Title includes model name, data tag, and AUC. DPI=150. |
| JSON results | `{output_dir}/clf2_feature_probe_results.json` [L437] | Structured results for all evaluated models: sample counts, class counts, mean AUC ± std, mean accuracy ± std, mean coefficients, intercept, per-source verdict. |

### 9.2 Default output directory

When `--output_dir` is not specified, defaults to `classifier/postprocess/` [L349]:

```python
output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
```

### 9.3 JSON results structure

The JSON output [L414–439] persists the full numeric results:

```json
{
  "models": {
    "flan_t5_xl": {
      "merged": {
        "n_samples": ..., "class_counts": {"B": ..., "C": ...},
        "mean_auc": ..., "std_auc": ...,
        "mean_accuracy": ..., "std_accuracy": ...,
        "mean_coefficients": {"token_len": ..., "entity_count": ..., "bridge_flag": ...},
        "intercept": ..., "verdict": "GO"
      },
      "silver_only": { ... }
    }
  },
  "go_no_go_threshold": 0.65,
  "n_folds": 5,
  "random_state": 42
}
```

### 9.4 Stdout output

In addition to the persisted files, the script prints to stdout:
- Feature means by class [L159–161]
- 5-fold CV results (AUC ± std, accuracy ± std) [L200–201]
- Classification report from last fold [L203–206]
- Per-feature LR coefficients (mean ± std) and intercept [L213–216]
- Go/no-go verdict [L305–309]
- Multi-model summary table (when `--all_models`) [L381–394]

---

## 10. Suspicious Items / Flags

### 10.1 Evaluation on training data subset — not held-out validation

| Issue | Detail |
|---|---|
| **What** | The probe evaluates on the Clf2 **training** file (`train.json`). The 5-fold CV provides internal train/test splits, but all data comes from the training set. The Clf2 validation set (`valid.json`) and prediction set (`predict.json`) are never used. |
| **Risk** | The reported AUC measures how well structural features separate B from C **in the training distribution**, which may differ from the validation/test distribution. The go/no-go verdict may not generalize. |
| **Mitigation** | 5-fold CV reduces overfitting risk compared to a single train/test split, but does not address distribution shift between train and validation sets. |

### 10.2 Binary inductive-bias label inflation risk

| Issue | Detail |
|---|---|
| **What** | When using the merged data file (`binary_silver_single_vs_multi/train.json`), the training set includes inductive-bias labels. These labels are derived from the dataset's known complexity structure (e.g., MuSiQue questions are always multi-hop → label C). If the structural features (especially `bridge_flag`) correlate with dataset-of-origin rather than genuine question complexity, the AUC may be inflated. |
| **Mitigation** | The script evaluates on **both** merged and silver-only data [L279–301], and bases the verdict on the silver-only AUC when available [L296–301]. The merged AUC is reported for comparison only. |

### 10.3 Feature statistics computed on full data, classifier on 80% folds

| Issue | Detail |
|---|---|
| **What** | The "Feature means by class" table [L159–161] is computed on the **full** dataset before the CV loop. The LR classifier in each fold sees only 80 % of the data. |
| **Impact** | The printed feature means reflect the global distribution, not the per-fold training distribution. This is a reporting inconsistency, not a correctness issue — the classifier itself only sees fold-appropriate data. |

### 10.4 `--model` defaults to `flan_t5_xl` only

| Issue | Detail |
|---|---|
| **What** | Running the script without arguments probes only the `flan_t5_xl` data [L332]. The other two model variants (`flan_t5_xxl`, `gpt`) are evaluated only if `--all_models` is passed. |
| **Risk** | A user might run the default and conclude "structural features work" or "don't work" based on one model variant. The GPT data has a different class distribution (more A labels filtered out, different B/C ratio) and may yield a different verdict. |

### 10.5 Classification report is from last fold only

| Issue | Detail |
|---|---|
| **What** | `classification_report()` is called with `last_fold_y_test` and `last_fold_y_pred` [L203–206], which are from the **fifth and final** fold. The per-class precision/recall/F1 are not averaged across folds. |
| **Impact** | The reported precision/recall may not be representative of all folds. Only AUC and accuracy are properly averaged. |

### 10.6 Intercept reported from last fold only

| Issue | Detail |
|---|---|
| **What** | The intercept is `float(clf.intercept_[0])` [L214], where `clf` is the **last fold's** fitted model. Unlike coefficients (which are averaged across folds via `fold_coefs.mean(axis=0)`), the intercept has no fold-averaging. |
| **Impact** | Minor reporting inconsistency. The intercept may differ across folds. |

### 10.7 No feature scaling

| Issue | Detail |
|---|---|
| **What** | `LogisticRegression` is applied to raw feature values without standardization (no `StandardScaler`, no `MinMaxScaler`). The three features have very different scales: `token_len` (typically 5–50), `entity_count` (typically 0–10), `bridge_flag` (binary 0/1). |
| **Impact** | L2 regularization (`penalty="l2"`, default) penalizes coefficients proportionally to feature magnitude. A large-scale feature like `token_len` will have a smaller coefficient not because it's less important, but because its values are larger. The coefficient magnitudes are not directly comparable across features. |
| **AUC impact** | Logistic regression's predictions are invariant to feature scaling (the decision boundary adjusts), so the **AUC is unaffected**. Only the coefficient interpretation and regularization balance are affected. |

### 10.8 `cross_val_predict` imported but unused

| Issue | Detail |
|---|---|
| **What** | `from sklearn.model_selection import StratifiedKFold, cross_val_predict` [L38] imports `cross_val_predict`, but the script implements its own manual fold loop [L176–190] and never calls `cross_val_predict`. |
| **Impact** | Dead import. No functional issue. |

### 10.9 spaCy components selectively disabled

| Issue | Detail |
|---|---|
| **What** | The spaCy pipeline is loaded with `disable=["parser", "lemmatizer"]` [L352]. Only the NER component runs. The tokenizer always runs (cannot be disabled). |
| **Benefit** | Faster processing — no dependency parsing overhead. |
| **Note** | `token_len` is computed via `doc.text.split()` (whitespace split), **not** via spaCy's tokenizer (`len(doc)`). This means the token count is a raw whitespace count, not a linguistically-informed token count. |

### 10.10 Scatter plot shows only 2 of 3 features

| Issue | Detail |
|---|---|
| **What** | The scatter plot [L222–240] displays `token_len` (x-axis) vs `entity_count` (y-axis). The `bridge_flag` (binary) is not visualized — it would require a different plot type (e.g., separate panels or marker shapes). |
| **Impact** | The visual may miss separation that exists along the `bridge_flag` dimension. |
