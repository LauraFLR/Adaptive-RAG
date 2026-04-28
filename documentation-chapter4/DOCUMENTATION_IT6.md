# Iteration 6 — Diagnostic Logistic Regression Probe for Gate 2 κ(q) Features

> **Design Science Research Artifact:** A standalone diagnostic script that
> tests whether the SymRAG κ(q) structural query features (normalised token
> length, entity density, hop-indicator density, and the composite κ score)
> carry discriminative signal for the B-vs-C (single-step vs multi-step
> retrieval) decision.  No T5 model is loaded, no checkpoint is written, no
> downstream QA evaluation is performed.  This is a **feasibility check
> only** — a go/no-go gate that decides whether it is worth injecting these
> features into a full Gate 2 classifier.

---

## 1. Files Involved

| File | Role |
|---|---|
| `classifier/postprocess/clf2_kappa_feature_probe.py` (637 lines) | **The ENTIRE iteration.** One standalone script: κ(q) feature extraction, logistic-regression 5-fold CV, histogram + scatter plot, CSV, JSON results. |

**This is the only file in IT6.** There is no shell wrapper, no config file, no supporting training script.

### 1.1 External dependencies

| Library | Import line | Purpose |
|---|---|---|
| `spacy` (+ `en_core_web_sm` model) | [L468–469] | Named-entity recognition for `entity_count` raw feature |
| `sklearn` (`LogisticRegression`, `StratifiedKFold`, `roc_auc_score`, `accuracy_score`, `classification_report`, `f1_score`) | [L36–43] | Diagnostic classifier + evaluation metrics |
| `pandas` | [L34] | DataFrame for feature matrix, CSV export |
| `matplotlib` | [L30–32] | Histogram + scatter plot (Agg backend, no display) |
| `numpy` | [L29] (module-level import) | Array operations throughout feature extraction and fold aggregation |

### 1.2 NOT imported

| Library | Relevance |
|---|---|
| `torch` | Not imported. No tensor operations. |
| `transformers` | Not imported. No T5 model, no tokenizer, no `generate()`. |
| `run_classifier.py` | Not imported. No training loop, no `FocalLossTrainer`. |
| `utils.py` | Not imported. No `load_model()`, no `preprocess_features_function()`. |
| `accelerate` | Not imported. No distributed training. |
| `cross_val_predict` | Not imported (unlike the older `clf2_feature_probe.py`). |

---

## 2. Diagnostic-Only Scope — Confirmed

### 2.1 No T5 model loaded

The script loads **only** a spaCy `en_core_web_sm` NLP pipeline [L469]:

```python
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
```

This is a small (12 MB) NER-only model used exclusively for entity counting. No T5 model, no HuggingFace model, no GPU allocation.

### 2.2 No writes to classifier/outputs/ or predictions/

The script's write targets are controlled by `--output_dir`, which defaults to `os.path.dirname(os.path.abspath(__file__))` [L463] — i.e. `classifier/postprocess/` itself. When invoked by `run-all-iterations.sh`, the output directory is overridden to `predictions/classifier/t5-large/iter6_kappa_probe`. It writes:

| Artifact | Path pattern |
|---|---|
| CSV | `{output_dir}/clf2_kappa_probe_data_{model}_{tag}.csv` [L192] |
| Plot (histogram + scatter) | `{output_dir}/clf2_kappa_probe_{model}_{tag}.png` [L286] |
| JSON results | `{output_dir}/clf2_kappa_probe_results.json` [L630] |

None of these targets are inside `classifier/outputs/` (the checkpoint tree).

### 2.3 No modification of training data or checkpoints

The script **reads** Clf2 training JSON files via `load_bc_data()` [L94–100] but does not write to them. There is no `save_json()`, no `shutil.copy()`, no `open(..., "w")` targeting any data or checkpoint path.

### 2.4 Iteration 7's files are separate

The probe is purely diagnostic. If the go/no-go verdict is "GO," a **separate** script in a future iteration would incorporate the features into an actual Gate 2 classifier. This script does not feed into any downstream training pipeline.

---

## 3. Structural Features

### 3.1 Raw and derived features

The script extracts three **raw** features per question and derives four **model-input** features from them. The logistic regression uses only the derived features.

#### Raw features (per question)

| Feature | Extraction method | Library | Line(s) |
|---|---|---|---|
| `token_len` | `len(text.split())` — whitespace-split token count | Built-in `str.split()` | [L137] |
| `entity_count` | `len(doc.ents)` — number of named entities | spaCy `en_core_web_sm` NER | [L138] |
| `hop_count` | `sum(1 for pat in _BRIDGE_RES if pat.search(text))` — number of bridging patterns that fire | `re` stdlib | [L139] |

#### Derived features (matching `predict_complexity_kappa.py`)

| Feature | Formula | Line(s) |
|---|---|---|
| `token_len_norm` | `token_len / max(token_len)` | [L148] |
| `entity_density` | `entity_count / token_len` | [L151] |
| `hop_density` | `hop_count / token_len` | [L152] |
| `kappa` | $W_L \cdot \text{token\_len\_norm} \cdot (1 + W_{SH1} \cdot \text{entity\_density} + W_{SH2} \cdot \text{hop\_density})$ | [L154] |

The SymRAG published weights are defined at [L46–48]:

| Weight | Value | Purpose |
|---|---|---|
| `W_L` | `1.0` | Token-length scaling |
| `W_SH1` | `0.05` | Entity-density contribution |
| `W_SH2` | `0.10` | Hop-indicator-density contribution |

The four derived features are stored in `FEATURE_COLS` at [L166]:

```python
FEATURE_COLS = ["token_len_norm", "entity_density", "hop_density", "kappa"]
```

All four features are extracted in `extract_features()` [L117–164], which processes questions in batches via `nlp.pipe(questions, batch_size=256)` [L135].

**Key difference from `clf2_feature_probe.py`:** The older script used three raw features (`token_len`, `entity_count`, `bridge_flag`). This script uses four derived/normalised features that match the κ(q) formulation used in IT7's `predict_complexity_kappa.py`. Notably, `bridge_flag` (binary: any pattern matched?) is replaced by `hop_count` (integer: how many patterns matched?), and all features are normalised to density or unit-range form.

### 3.2 Bridge-pattern compilation

Unlike the older `clf2_feature_probe.py` which compiled all patterns into a single alternation regex, this script compiles each pattern **individually** [L68]:

```python
_BRIDGE_RES = [re.compile(p, re.IGNORECASE) for p in BRIDGE_PATTERNS]
```

This enables **counting** matching patterns (via `sum(1 for pat in _BRIDGE_RES if pat.search(text))`) rather than producing a binary match/no-match flag.

Seven regex patterns are defined in `BRIDGE_PATTERNS` [L53–67]:

| # | Pattern | Description | Example match |
|---|---|---|---|
| 1 | `\b(?:who\|where\|which\|that)\s+(?:was\|were\|is\|are\|did\|had\|has\|does)\b` | Relative-clause bridges linking two entities | "the person **who was** born in…" |
| 2 | `\w+'s\s+\w+(?:\s+\w+){0,5}\s+\w+'s` | Double possessive — two possessives suggest two hops | "**Obama's** mother **'s** birthplace" |
| 3 | `\b(?:before\|after\|when\|while)\b.{3,60}\b(?:who\|what\|where\|which)\b` | Temporal/causal subordination before a wh-word | "**after** X was elected, **what** happened…" |
| 4 | `\b(?:that\|this\|those\|these)\s+(?:country\|city\|person\|team\|company\|film\|movie\|album\|book\|organization\|university\|school)\b` | Demonstrative back-reference to a prior fact | "**that country**'s capital" |
| 5 | `\b(?:both)\b.{1,40}\band\b` | Explicit comparison linking two entities ("both X and Y") | "**both** France **and** Germany" |
| 6 | `\bbetween\b.{1,40}\band\b` | Explicit comparison ("between X and Y") | "**between** Paris **and** Berlin" |
| 7 | `\bof\s+the\s+\w+\s+(?:who\|that\|which\|where)\b` | Nested wh-question ("X of the Y that…") | "the capital **of the country that** won…" |

All patterns are case-insensitive [L68: `re.IGNORECASE`].

### 3.3 Computation timing

Features are computed **at runtime** from the raw question text. There is no caching, no pre-computed feature file. The spaCy `nlp.pipe()` call processes all questions in a single batch [L135], but NER inference still takes a few seconds for thousands of questions. The regex match is negligible.

---

## 4. Diagnostic Classifier

### 4.1 Model: Logistic Regression

A fresh `LogisticRegression` instance is created **per fold** at [L208]:

```python
clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
```

### 4.2 Hyperparameters

| Parameter | Value | Source | Note |
|---|---|---|---|
| `max_iter` | `1000` | [L208] | Explicit |
| `random_state` | `42` | [L208] | Explicit |
| **`class_weight`** | **`'balanced'`** | [L208] | **Explicit — automatic inverse-frequency class weighting** |
| `penalty` | `"l2"` | sklearn default | Not set; L2 regularization |
| `C` | `1.0` | sklearn default | Inverse regularization strength |
| `solver` | `"lbfgs"` | sklearn default | Not set |
| `fit_intercept` | `True` | sklearn default | Intercept is reported |
| `tol` | `1e-4` | sklearn default | Convergence tolerance |
| `multi_class` | `"auto"` | sklearn default | Binary in this case |

**Key difference from `clf2_feature_probe.py`:** The older script used `class_weight=None` (sklearn default). This script uses `class_weight='balanced'`, which automatically adjusts weights inversely proportional to class frequencies. This compensates for any B/C class imbalance in the training data.

### 4.3 Label encoding

At [L198]:

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

**Auto-detection** (when `--data_path` is not specified): `detect_data_paths()` [L71–91] checks for two files in priority order:

| Tag | Path pattern | Contents |
|---|---|---|
| `"merged"` | `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/binary_silver_single_vs_multi/train.json` | Silver labels + inductive-bias labels combined |
| `"silver-only"` | `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/silver/single_vs_multi/train.json` | Silver labels only |

Both files are evaluated if they exist. Additionally, if both exist, an **inductive-bias-only** split (items in merged but not in silver) is also evaluated [L395–407]. The **go/no-go verdict** is based on the silver-only macro-F1 (the stricter evaluation) [L410–414]; the merged and IB macro-F1 are reported for comparison.

**Filtering:** `load_bc_data()` [L94–100] loads the full JSON array and keeps only items where `item["answer"]` is `"B"` or `"C"` [L98]:

```python
bc = [item for item in data if item.get("answer") in ("B", "C")]
```

Items with label `"A"` (if any exist in the file) are discarded.

### 4.5 Default model

`--model` defaults to `flan_t5_xl` [L445]:

```python
parser.add_argument("--model", type=str, default="flan_t5_xl", ...)
```

The `--all_models` flag [L450] runs the probe for all three variants (`flan_t5_xl`, `flan_t5_xxl`, `gpt`) sequentially and prints a summary table.

### 4.6 Cross-validation setup

5-fold stratified CV at [L200]:

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

The feature matrix is constructed at [L197]:

```python
X = feat_df[FEATURE_COLS].values
```

Where `FEATURE_COLS = ["token_len_norm", "entity_density", "hop_density", "kappa"]` [L166]. This selects **exactly four numeric columns**. No other columns are included:

| NOT in X | Why |
|---|---|
| `question` (text) | Stored in `feat_df` for CSV export [L185], but not in `X` |
| `label` | Used only for `y` [L198] |
| `id` | Stored for CSV export [L186], not in `X` |
| Raw `token_len`, `entity_count`, `hop_count` | Stored in `feat_df` but not in `FEATURE_COLS`; only the normalised/derived forms are used |
| TF-IDF / BoW | Not computed anywhere in the script |
| Embeddings | No embedding model is loaded |
| T5 hidden states | No T5 model is loaded |

The diagnostic classifier sees only structural metadata about the question, not its semantic content.

---

## 6. Evaluation of the Diagnostic Classifier

### 6.1 Split: 5-fold stratified CV

Each of the 5 folds uses an 80/20 stratified train/test split (inherent to `StratifiedKFold(n_splits=5)` [L200]). Stratification ensures each fold preserves the B/C class ratio.

### 6.2 Metrics

| Metric | Function | Scope | Line(s) |
|---|---|---|---|
| **Macro-F1** | `sklearn.metrics.f1_score(y_test, y_pred, average='macro')` | Per-fold, then mean ± std | [L216, L234–235] |
| ROC-AUC | `sklearn.metrics.roc_auc_score(y_test, y_prob)` | Per-fold, then mean ± std | [L214, L230–231] |
| Accuracy | `sklearn.metrics.accuracy_score(y_test, y_pred)` | Per-fold, then mean ± std | [L215, L232–233] |
| Classification report | `sklearn.metrics.classification_report(last_fold_y_test, last_fold_y_pred, target_names=["B (single)", "C (multi)"])` | **Last fold only** | [L241–244] |

The AUC uses `predict_proba()[:, 1]` [L212] (probability of class C). Accuracy and F1 use `predict()` [L213] (hard 0.5 threshold). **Macro-F1** is the primary metric used for the go/no-go verdict (see §6.4).

### 6.3 Feature importances

LR coefficients and intercept are reported at [L245–251]:

```python
mean_coefs = fold_coefs.mean(axis=0)
std_coefs  = fold_coefs.std(axis=0)
```

- **Coefficients:** Mean ± std across 5 folds, for each of the 4 features. Positive coefficient → pushes toward C (multi-step).
- **Intercept:** Reported from the **last fold only** [L248]: `float(clf.intercept_[0])`.

### 6.4 Go/no-go verdict

At [L416]:

```python
verdict = "GO" if verdict_f1 >= 0.55 else "NO-GO"
```

| Condition | Verdict | Meaning |
|---|---|---|
| Mean macro-F1 ≥ 0.55 | **GO** | κ(q) features carry enough signal to proceed with a feature-augmented Gate 2 classifier |
| Mean macro-F1 < 0.55 | **NO-GO** | Features are insufficiently discriminative; do not proceed |

The threshold `0.55` is also persisted in the JSON output [L618]:

```python
json_out: dict = { ..., "go_no_go_threshold": 0.55, ... }
```

**Key difference from `clf2_feature_probe.py`:** The older script used ROC-AUC ≥ 0.65 as the verdict criterion. This script uses macro-F1 ≥ 0.55, a stricter and more practical threshold given that F1 penalises both false positives and false negatives.

When both merged and silver-only data exist, the verdict is based on the **silver-only** macro-F1 [L410–414] (the stricter evaluation, since inductive-bias labels may inflate performance).

---

## 7. Gate 1

**The probe does NOT involve Gate 1 at all.**

- No A/R classification is performed.
- No Clf1 checkpoint or prediction file is loaded.
- The input data is filtered to B/C items only [L98].
- The script is entirely about the Gate 2 (B vs C) decision boundary.

---

## 8. End-to-End Evaluation

**NO end-to-end QA F1 evaluation is performed.**

- `evaluate_final_acc.py` is not imported or invoked.
- No QA prediction files are routed.
- No per-dataset EM/F1 scores are computed.
- The only metrics are the diagnostic classifier's macro-F1, ROC-AUC, accuracy, and classification report (sklearn metrics on B/C labels, not QA answers).

---

## 9. Output Artifacts

### 9.1 File table

| Artifact | Path pattern | Content |
|---|---|---|
| CSV | `{output_dir}/clf2_kappa_probe_data_{model}_{tag}.csv` [L192] | Per-question row: `token_len`, `entity_count`, `hop_count`, `token_len_norm`, `entity_density`, `hop_density`, `kappa`, `label`, `question`, `id`. One file per (model, data-source) combination. |
| Plot (κ histogram + scatter) | `{output_dir}/clf2_kappa_probe_{model}_{tag}.png` [L286] | **Left panel:** κ(q) histogram by class (B=blue, C=red). **Right panel:** `entity_density` (x) vs `hop_density` (y), coloured by B (blue) / C (red). Title includes model name, data tag, and AUC. DPI=150. |
| JSON results | `{output_dir}/clf2_kappa_probe_results.json` [L630] | Structured results for all evaluated models: sample counts, class counts, mean macro-F1 ± std, mean AUC ± std, mean accuracy ± std, mean coefficients, intercept, per-source verdict. |

### 9.2 Default output directory

When `--output_dir` is not specified, defaults to `classifier/postprocess/` [L463]:

```python
output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
```

When invoked by `run-all-iterations.sh`, the output directory is explicitly set to `predictions/classifier/t5-large/iter6_kappa_probe`.

### 9.3 JSON results structure

The JSON output [L605–632] persists the full numeric results:

```json
{
  "features": ["token_len_norm", "entity_density", "hop_density", "kappa"],
  "kappa_weights": {"W_L": 1.0, "W_SH1": 0.05, "W_SH2": 0.10},
  "models": {
    "flan_t5_xl": {
      "merged": {
        "n_samples": ..., "class_counts": {"B": ..., "C": ...},
        "mean_auc": ..., "std_auc": ...,
        "mean_macro_f1": ..., "std_macro_f1": ...,
        "mean_accuracy": ..., "std_accuracy": ...,
        "mean_coefficients": {"token_len_norm": ..., "entity_density": ..., "hop_density": ..., "kappa": ...},
        "intercept": ..., "verdict": "GO"
      },
      "silver_only": { ... },
      "ib_only": { ... }
    }
  },
  "go_no_go_metric": "macro_f1",
  "go_no_go_threshold": 0.55,
  "n_folds": 5,
  "random_state": 42
}
```

### 9.4 Stdout output

In addition to the persisted files, the script prints to stdout:
- Feature means by class [L187–189]
- 5-fold CV results (macro-F1 ± std, ROC-AUC ± std, accuracy ± std) [L236–239]
- Classification report from last fold [L241–244]
- Per-feature LR coefficients (mean ± std) and intercept [L249–251]
- Go/no-go verdict [L418–422]
- Multi-model summary table (when `--all_models`) [L540–570]

---

## 10. Suspicious Items / Flags

### 10.1 Evaluation on training data subset — not held-out validation

| Issue | Detail |
|---|---|
| **What** | The probe evaluates on the Clf2 **training** file (`train.json`). The 5-fold CV provides internal train/test splits, but all data comes from the training set. The Clf2 validation set (`valid.json`) and prediction set (`predict.json`) are never used. |
| **Risk** | The reported metrics measure how well κ(q) features separate B from C **in the training distribution**, which may differ from the validation/test distribution. The go/no-go verdict may not generalize. |
| **Mitigation** | 5-fold CV reduces overfitting risk compared to a single train/test split, but does not address distribution shift between train and validation sets. |

### 10.2 Inductive-bias label inflation risk

| Issue | Detail |
|---|---|
| **What** | When using the merged data file (`binary_silver_single_vs_multi/train.json`), the training set includes inductive-bias labels. These labels are derived from the dataset's known complexity structure (e.g., MuSiQue questions are always multi-hop → label C). If the structural features (especially `hop_density`) correlate with dataset-of-origin rather than genuine question complexity, the metrics may be inflated. |
| **Mitigation** | The script evaluates on **three** splits — merged, silver-only, and inductive-bias-only [L369–407] — and bases the verdict on the silver-only macro-F1 when available [L410–414]. The merged and IB metrics are reported for comparison only. |

### 10.3 Feature statistics computed on full data, classifier on 80% folds

| Issue | Detail |
|---|---|
| **What** | The "Feature means by class" table [L187–189] is computed on the **full** dataset before the CV loop. The LR classifier in each fold sees only 80 % of the data. |
| **Impact** | The printed feature means reflect the global distribution, not the per-fold training distribution. This is a reporting inconsistency, not a correctness issue — the classifier itself only sees fold-appropriate data. |

### 10.4 `--model` defaults to `flan_t5_xl` only

| Issue | Detail |
|---|---|
| **What** | Running the script without arguments probes only the `flan_t5_xl` data [L445]. The other two model variants (`flan_t5_xxl`, `gpt`) are evaluated only if `--all_models` is passed. |
| **Risk** | A user might run the default and conclude "κ(q) features work" or "don't work" based on one model variant. The GPT data has a different class distribution (more A labels filtered out, different B/C ratio) and may yield a different verdict. |
| **Note** | When invoked by `run-all-iterations.sh`, the `--all_models` flag is always passed, so all three models are evaluated. |

### 10.5 Classification report is from last fold only

| Issue | Detail |
|---|---|
| **What** | `classification_report()` is called with `last_fold_y_test` and `last_fold_y_pred` [L241–244], which are from the **fifth and final** fold. The per-class precision/recall/F1 are not averaged across folds. |
| **Impact** | The reported precision/recall may not be representative of all folds. Only macro-F1, AUC, and accuracy are properly averaged. |

### 10.6 Intercept reported from last fold only

| Issue | Detail |
|---|---|
| **What** | The intercept is `float(clf.intercept_[0])` [L248], where `clf` is the **last fold's** fitted model. Unlike coefficients (which are averaged across folds via `fold_coefs.mean(axis=0)`), the intercept has no fold-averaging. |
| **Impact** | Minor reporting inconsistency. The intercept may differ across folds. |

### 10.7 Partial feature normalisation

| Issue | Detail |
|---|---|
| **What** | The κ(q) features are partially normalised by design: `entity_density` and `hop_density` are divided by token length, and `token_len_norm` is divided by the global max. However, no explicit `StandardScaler` or `MinMaxScaler` is applied. The four features still have different scales and value ranges. |
| **Impact** | L2 regularization (`penalty="l2"`, default) penalizes coefficients proportionally to feature magnitude. The coefficient magnitudes are not directly comparable across features. Using `class_weight='balanced'` addresses class imbalance but does not affect feature scaling. |
| **AUC/F1 impact** | Logistic regression's predictions are invariant to feature scaling (the decision boundary adjusts), so the **AUC and F1 are unaffected**. Only the coefficient interpretation and regularization balance are affected. |

### 10.8 spaCy components selectively disabled

| Issue | Detail |
|---|---|
| **What** | The spaCy pipeline is loaded with `disable=["parser", "lemmatizer"]` [L469]. Only the NER component runs. The tokenizer always runs (cannot be disabled). |
| **Benefit** | Faster processing — no dependency parsing overhead. |
| **Note** | `token_len` is computed via `text.split()` (whitespace split), **not** via spaCy's tokenizer (`len(doc)`). This means the token count is a raw whitespace count, not a linguistically-informed token count. |

### 10.9 κ(q) max-normalisation is global, not per-fold

| Issue | Detail |
|---|---|
| **What** | `token_len_norm = token_lens / max_len` [L148] divides by the global maximum token length across all questions (not per-fold). This means the normalisation leaks information from the test fold into the training fold (all questions' max token length is used). |
| **Impact** | Negligible in practice — the max token length is a single scalar and the CV is diagnostic only. But strictly speaking, normalisation should be fit on training data per fold. |

### 10.10 Plot shows κ histogram + 2-of-4 features

| Issue | Detail |
|---|---|
| **What** | The plot [L256–287] has two panels: (left) κ(q) histogram by class, (right) `entity_density` (x) vs `hop_density` (y) scatter. `token_len_norm` and the full `kappa` value are not visualised in the scatter. |
| **Impact** | The visual may miss separation patterns that involve `token_len_norm` directly. The κ histogram compensates partially since κ is a composite of all sub-features. |
