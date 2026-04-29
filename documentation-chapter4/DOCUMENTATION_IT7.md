# Iteration 7 — Fully Training-Free SymRAG κ(q) Cascade (Agreement Gate + Structural Heuristic)

> **Design Science Research Artifact:** Replace **both** trained classifiers
> with training-free heuristics: Gate 1 uses the IT5 cross-strategy answer
> agreement gate (nor_qa vs oner_qa), and Gate 2 uses a SymRAG-inspired
> structural complexity score κ(q) with threshold tuning on validation data.
> No model is trained, no checkpoint is loaded, no GPU is required for
> routing decisions. The only external dependency beyond stdlib is spaCy
> `en_core_web_sm` (12 MB NER model) for entity counting.

---

## 1. Files Involved

| File | Role |
|---|---|
| `classifier/postprocess/predict_complexity_kappa.py` (586 lines) | **The ONLY new script.** Implements the full training-free cascade: agreement-based Gate 1, κ(q)-based Gate 2, threshold tuning, prediction routing, and output writing. |
| `classifier/postprocess/postprocess_utils.py` | Shared — provides `load_json()` and `save_json()` helpers. **Identical to IT1.** |
| `evaluate_final_acc.py` (341 lines) | QA evaluation — **identical to IT1.** |
| `run-all-iterations.sh` | Top-level orchestrator. Invokes IT7 via the `route_kappa()` helper, which calls `predict_complexity_kappa.py` with `--use_agreement_gate --tune_threshold`, then runs `evaluate_final_acc.py`. |
| `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/binary_silver_single_vs_multi/train.json` | IB+silver merged B/C labels (~2 800–3 300 items per model, ~1.1–1.4:1 B:C ratio) — used for threshold tuning only. Not used for training. |
| `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json` | Unlabelled test set (3 000 questions, 500 per dataset). |

### 1.1 Relationship to prior iterations

| Component | IT5 (Gate 1 only) | IT6 (diagnostic) | **IT7 (this)** |
|---|---|---|---|
| Gate 1 | Agreement gate | Not involved | Agreement gate (reused from IT5) |
| Gate 2 | Trained Clf2 consumed as-is | Logistic regression probe (feasibility only) | **κ(q) structural heuristic (new)** |
| Training required? | No (Gate 1), Yes (Clf2) | No | **No** |
| QA routing? | Yes | No | **Yes** |
| End-to-end evaluation? | Yes | No | **Yes** |

IT7 combines IT5's Gate 1 replacement with a new Gate 2 replacement derived from IT6's go/no-go feasibility check. It is the first fully training-free iteration.

### 1.2 External dependencies

| Library | Import line | Purpose |
|---|---|---|
| `spacy` (+ `en_core_web_sm` model) | [L362–365] | Named-entity recognition for `entity_count` raw feature |
| `numpy` | [L39] | Array operations in `compute_kappa()` and threshold tuning |
| `sklearn.metrics.f1_score` | [L221] (lazy import inside `tune_threshold()`) | Macro-F1 during threshold search |

### 1.3 NOT imported

| Library | Relevance |
|---|---|
| `torch` | Not imported. No tensor operations. |
| `transformers` | Not imported. No T5 model, no tokenizer, no `generate()`. |
| `run_classifier.py` | Not imported. No training loop, no `FocalLossTrainer`. |
| `utils.py` | Not imported. No `load_model()`, no `preprocess_features_function()`. |
| `accelerate` | Not imported. No distributed training. |
| `pandas` | Not imported. No DataFrame operations. |
| `matplotlib` | Not imported. No plots generated. |

---

## 2. Gate 1 — Agreement Gate (Reused from IT5)

### 2.1 What is replaced

In IT1–IT4, Gate 1 is a trained T5-Large binary classifier (Clf1) that reads a question and predicts A or R. The classifier requires a fine-tuned model checkpoint (~770 M parameters) and GPU inference.

### 2.2 What replaces it

The same agreement gate from IT5. The logic is embedded directly in `predict_complexity_kappa.py` — the `normalize_answer()`, `answer_extractor()`, `compute_agreement()`, and `load_strategy_predictions()` functions are copied from `predict_complexity_agreement.py`.

### 2.3 Agreement computation

The `compute_agreement()` function [L144–171] implements the full comparison pipeline:

| Step | Code | Description |
|---|---|---|
| 1 | `nor_raw = nor_preds[qid]` | Load raw no-retrieval answer |
| 2 | `oner_raw = oner_preds[qid]` | Load raw single-step retrieval answer |
| 3 | Handle list answers | `if isinstance(nor_raw, list): nor_raw = nor_raw[0]` [L152–155] |
| 4 | Cast to string | `nor_raw = str(nor_raw)` [L156–157] |
| 5 | Extract answer from CoT | `answer_extractor(nor_raw)` — regex `".* answer is:? (.*)\\.?"` [L107–121] |
| 6 | Normalize | `normalize_answer()` — lower → remove punctuation → remove articles → collapse whitespace [L93–104] |
| 7 | Compare | `agree = bool(nor_norm and oner_norm and nor_norm == oner_norm)` [L163] — exact string match |

### 2.4 Invocation path

When `--use_agreement_gate` is passed (always the case in `run-all-iterations.sh`), the script at [L399–402]:

```python
agreement = compute_agreement(nor_preds, oner_preds, all_qids)
gate1 = {}
for qid in all_qids:
    gate1[qid] = "A" if agreement[qid]["agree"] else "R"
```

When `--clf1_pred_file` is passed instead (alternative path), the script loads a Clf1 prediction JSON and reads each question's A/R label directly [L403–411].

### 2.5 No trained model is loaded for Gate 1

Same as IT5: no `torch`, `transformers`, or GPU allocation. The entire Gate 1 decision is string comparison on pre-computed QA outputs.

---

## 3. Gate 2 — SymRAG Structural κ(q) Score

### 3.1 Theoretical basis

Gate 2 uses the structural heuristic component of SymRAG's κ(q) complexity score (Hakim et al., 2025, Definition 1, Appendix A.1.1):

$$\kappa(q) = (w_A \cdot A(q) + w_L \cdot L(q)) \cdot (1 + S_H(q))$$

where:
- $A(q)$ = mean attention score from `prajjwal1/bert-tiny` (4.4 M parameter model)
- $L(q) = \text{token\_len} / \max(\text{token\_len})$ — normalised query length
- $S_H(q) = w_{sh1} \cdot \frac{N_{ents}(q)}{|q|} + w_{sh2} \cdot \frac{N_{hops}(q)}{|q|}$ — structural heuristic

**IT7 omits $A(q)$** and uses only the structural components:

$$\kappa(q) = w_L \cdot L(q) \cdot (1 + S_H(q))$$

### 3.2 Justification for omitting A(q)

Two reasons:

1. **No new model dependency:** Including $A(q)$ would require loading `prajjwal1/bert-tiny` and running a forward pass for each question. While cheap (4.4 M parameters), it introduces a model dependency that breaks the "no model required" property.
2. **Clean comparison:** Using only structural features provides a direct comparison against the T5-Large Clf2 — both use the same question text, but IT7 uses hand-crafted features while Clf2 uses learned representations.

### 3.3 Published SymRAG weights

Defined as module-level constants [L54–56]:

| Weight | Variable | Value | Purpose |
|---|---|---|---|
| $w_L$ | `W_L` | `1.0` | Token-length scaling |
| $w_{sh1}$ | `W_SH1` | `0.05` | Entity-density contribution |
| $w_{sh2}$ | `W_SH2` | `0.10` | Hop-indicator-density contribution |

Source: SymRAG (Hakim et al., 2025), Appendix A.1.1.

### 3.4 SymRAG initial thresholds (reference only)

SymRAG publishes $T_{low,\kappa} = 0.4$ and $T_{high,\kappa} = 0.8$ (Table 7, page 23) for their 3-way routing (symbolic / neural / hybrid). IT7 uses a **binary** threshold (B vs C) that is tuned on validation data, so these reference values are not directly used.

---

## 4. Feature Extraction

### 4.1 Raw features

The `extract_features()` function [L175–186] extracts three raw features per question:

| Feature | Extraction method | Library | Line |
|---|---|---|---|
| `token_len` | `len(text.split())` — whitespace-split token count | Built-in `str.split()` | [L183] |
| `entity_count` | `len(doc.ents)` — number of named entities | spaCy `en_core_web_sm` NER | [L184] |
| `hop_count` | `sum(1 for pat in _BRIDGE_RES if pat.search(text))` — count of matching bridging patterns | `re` stdlib | [L185] |

Processing: all questions are passed through `nlp.pipe(questions, batch_size=256)` [L180].

### 4.2 spaCy configuration

Loaded at [L363–365]:

```python
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
```

Only the NER component runs. The tokenizer always runs (cannot be disabled). The parser and lemmatizer are disabled for speed. This is a ~12 MB model — no GPU required.

**Note:** `token_len` is computed via `text.split()` (whitespace split), **not** via spaCy's tokenizer. The spaCy pipeline is used exclusively for entity counting.

### 4.3 Bridge-pattern compilation

Seven regex patterns are compiled individually [L87]:

```python
_BRIDGE_RES = [re.compile(p, re.IGNORECASE) for p in BRIDGE_PATTERNS]
```

This enables **counting** matching patterns (via `sum(...)`) rather than producing a binary match/no-match flag. All patterns are case-insensitive.

### 4.4 Bridge patterns

Seven patterns are defined in `BRIDGE_PATTERNS` [L72–86]:

| # | Pattern | Description | Example match |
|---|---|---|---|
| 1 | `\b(?:who\|where\|which\|that)\s+(?:was\|were\|is\|are\|did\|had\|has\|does)\b` | Relative-clause bridges linking two entities | "the person **who was** born in…" |
| 2 | `\w+'s\s+\w+(?:\s+\w+){0,5}\s+\w+'s` | Double possessive — two possessives suggest two hops | "**Obama's** mother**'s** birthplace" |
| 3 | `\b(?:before\|after\|when\|while)\b.{3,60}\b(?:who\|what\|where\|which)\b` | Temporal/causal subordination before a wh-word | "**after** X was elected, **what** happened…" |
| 4 | `\b(?:that\|this\|those\|these)\s+(?:country\|city\|person\|team\|company\|film\|movie\|album\|book\|organization\|university\|school)\b` | Demonstrative back-reference to a prior fact | "**that country**'s capital" |
| 5 | `\b(?:both)\b.{1,40}\band\b` | Explicit comparison linking two entities | "**both** France **and** Germany" |
| 6 | `\bbetween\b.{1,40}\band\b` | Explicit comparison | "**between** Paris **and** Berlin" |
| 7 | `\bof\s+the\s+\w+\s+(?:who\|that\|which\|where)\b` | Nested wh-question | "the capital **of the country that** won…" |

These patterns are **identical** to those in `clf2_kappa_feature_probe.py` (IT6).

### 4.5 Mapping to SymRAG's N_hops(q)

SymRAG's $N_{hops}(q)$ "counts multi-hop keywords" in the query. The IT7 implementation maps this to the count of bridging patterns that fire (`hop_count`). This is an integer count (0–7), not a binary flag. The older `clf2_feature_probe.py` (pre-IT6) used a binary `bridge_flag`; the IT6/IT7 scripts both use the multi-valued count.

---

## 5. Complexity Score κ(q)

### 5.1 `compute_kappa()` implementation

Defined at [L190–209]:

```python
def compute_kappa(token_lens, entity_counts, hop_counts):
    token_lens = np.array(token_lens, dtype=float)
    entity_counts = np.array(entity_counts, dtype=float)
    hop_counts = np.array(hop_counts, dtype=float)

    max_len = token_lens.max() if token_lens.max() > 0 else 1.0
    L = token_lens / max_len

    safe_lens = np.where(token_lens > 0, token_lens, 1.0)
    S_H = W_SH1 * (entity_counts / safe_lens) + W_SH2 * (hop_counts / safe_lens)

    kappa = W_L * L * (1.0 + S_H)
    return kappa
```

### 5.2 Step-by-step computation

For each question $q$ with whitespace-split token count $|q|$:

| Step | Formula | Variable | Line |
|---|---|---|---|
| 1. Normalise query length | $L(q) = |q| / \max_{q' \in Q}|q'|$ | `L` | [L202] |
| 2. Avoid division by zero | $|q|_{safe} = \max(|q|, 1)$ | `safe_lens` | [L205] |
| 3. Entity density | $w_{sh1} \cdot N_{ents}(q) / |q|_{safe}$ | first term of `S_H` | [L206] |
| 4. Hop density | $w_{sh2} \cdot N_{hops}(q) / |q|_{safe}$ | second term of `S_H` | [L206] |
| 5. Structural heuristic | $S_H(q) = 0.05 \cdot \frac{N_{ents}}{|q|} + 0.10 \cdot \frac{N_{hops}}{|q|}$ | `S_H` | [L206] |
| 6. Final score | $\kappa(q) = 1.0 \cdot L(q) \cdot (1 + S_H(q))$ | `kappa` | [L208] |

### 5.3 Score range

- $L(q) \in [0, 1]$ (normalised by max token length)
- $S_H(q) \geq 0$ (all terms non-negative)
- Therefore $\kappa(q) \in [0, 1 + S_H^{max}]$ — bounded above by $\approx 1 + \epsilon$ since $w_{sh1}$ and $w_{sh2}$ are small (0.05 and 0.10)
- In practice, $\kappa(q) \approx L(q)$ with small perturbations from entity and hop features

### 5.4 Max-normalisation scope

`max_len = token_lens.max()` [L201] is computed over **all R-routed questions** (questions that Gate 1 classified as "needs retrieval"), not over the full predict set. A-routed questions are excluded from feature extraction [L418–419]:

```python
r_qids = [qid for qid in all_qids if gate1[qid] != "A"]
r_questions = [qid_to_question[qid] for qid in r_qids]
```

This means the normalisation base depends on Gate 1's decisions. Different Gate 1 configurations (agreement gate vs trained Clf1) will produce different R-question pools, leading to different `max_len` values and therefore slightly different κ(q) scores for the same question.

---

## 6. Threshold Tuning

### 6.1 When tuning is activated

Tuning is activated when `--tune_threshold` is passed. This requires `--valid_file` (a Clf2 validation JSON with B/C labels). When invoked via `run-all-iterations.sh`, the `route_kappa()` helper always passes both flags:

```bash
python classifier/postprocess/predict_complexity_kappa.py "${model}" \
    --use_agreement_gate \
    --tune_threshold \
    --valid_file "classifier/data/${DATASET}/${model}/binary_silver_single_vs_multi/train.json" \
    --output_path "${out}"
```

The `--valid_file` points to the **IB+silver merged** dataset, not the silver-only validation split. This merged set combines empirical silver labels with inductive-bias labels (multi-hop datasets → C, single-hop → B), producing a much less skewed B:C ratio (~1.1–1.4:1 vs ~3:1 in silver-only). Since κ(q) has no learned parameters, the full merged set can be used for threshold selection without risk of overfitting.

### 6.2 `tune_threshold()` implementation

Defined at [L216–290]. Step-by-step:

| Step | Code | Line(s) | Description |
|---|---|---|---|
| 1 | `data = json.load(f)` | [L224] | Load validation JSON |
| 2 | `bc_items = [item for item in data if item.get("answer") in ("B", "C")]` | [L226] | Filter to B/C items only |
| 3 | `labels = np.array([1 if item["answer"] == "C" else 0 for item in bc_items])` | [L231] | Encode: C=1, B=0 |
| 4 | `extract_features(questions, nlp)` | [L233] | Extract raw features for validation questions |
| 5 | `compute_kappa(token_lens, entity_counts, hop_counts)` | [L234] | Compute κ(q) for each validation question |
| 6 | `lo = np.percentile(kappa, 5)` / `hi = np.percentile(kappa, 95)` | [L236–237] | Search range: 5th to 95th percentile |
| 7 | `thresholds = np.linspace(lo, hi, 100)` | [L238] | 100 candidate thresholds |
| 8 | For each threshold: `preds = (kappa >= t).astype(int)` | [L246] | κ ≥ t → C, else → B |
| 9 | `acc = (preds == labels).mean()` | [L247] | Accuracy at this threshold |
| 10 | `f1 = f1_score(labels, preds, average="macro", zero_division=0)` | [L248] | Macro-F1 at this threshold |
| 11 | Track best accuracy threshold and best F1 threshold separately | [L249–254] | Two optima may differ |
| 12 | Report per-class accuracy at **F1-optimal** threshold | [L256–267] | B-accuracy, C-accuracy, accuracy at F1 threshold |

### 6.3 Threshold selection criterion

The **macro-F1–optimal** threshold is used for prediction [L383]:

```python
threshold = best_f1_t
```

Macro-F1 was chosen over accuracy because the B/C label distribution can be imbalanced. Accuracy-optimal thresholds tend to over-predict the majority class (B), suppressing C-recall. Macro-F1 weights both classes equally, producing thresholds that route more genuinely complex questions to multi-step retrieval.

The accuracy-optimal threshold is still computed and reported for reference. If the two differ, a diagnostic message is printed [L272–274]:

```
Best accuracy: {best_acc:.4f} at threshold {best_acc_t:.4f}
  (differs from F1-optimal {best_f1_t:.4f})
```

### 6.4 Default threshold (when tuning is off)

If `--tune_threshold` is not passed, the default `--kappa_threshold` of `0.5` is used [L325]:

```python
parser.add_argument("--kappa_threshold", type=float, default=0.5)
```

### 6.5 Tuning data used for threshold selection

| Model | Tuning file | Items | B | C | B:C ratio |
|---|---|---|---|---|---|
| `flan_t5_xl` | `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/binary_silver_single_vs_multi/train.json` | 3 268 | 1 871 | 1 397 | 1.34 |
| `flan_t5_xxl` | `...flan_t5_xxl/binary_silver_single_vs_multi/train.json` | 3 298 | 1 903 | 1 395 | 1.36 |
| `gpt` | `...gpt/binary_silver_single_vs_multi/train.json` | 2 804 | 1 475 | 1 329 | 1.11 |

These are the **IB+silver merged** training sets — the same files used for Clf2 training in IT1. They concatenate:
- **Silver labels** — empirical: based on which retrieval strategy actually got the answer right for each LLM (model-specific, ~400–900 B/C items)
- **Inductive-bias (IB) labels** — heuristic: multi-hop datasets (MuSiQue, HotpotQA, 2WikiMultiHopQA) → C, single-hop datasets (NQ, TriviaQA, SQuAD) → B (model-independent, 2 400 items with 1:1 B:C ratio)

The merged set is ~3× larger and much less skewed than the silver-only validation split (which has ~3:1 B:C ratio). Since κ(q) tuning involves no learned parameters (just a threshold sweep), using the full merged set does not risk overfitting.

**Comparison with prior silver-only approach:** The silver-only valid.json files (e.g. XL: 911 items, 691 B / 220 C) produced accuracy-skewed thresholds around τ ≈ 0.29–0.46 that under-routed to C. The IB+silver merged set produces F1-optimal thresholds around τ ≈ 0.165, substantially increasing C-routing.

### 6.6 Tuning return values

`tune_threshold()` returns a 5-tuple [L276–290]:

```python
return best_f1_t, best_f1, best_acc_t, best_acc, stats
```

The primary return value is the **F1-optimal threshold** (`best_f1_t`). The accuracy-optimal threshold is returned as a secondary reference.

The `stats` dict contains:

```python
{
    "tuning_criterion": "macro_f1",
    "best_f1_threshold": float,
    "best_macro_f1": float,
    "accuracy_at_used_threshold": float,
    "val_B_accuracy": float,
    "val_C_accuracy": float,
    "best_acc_threshold": float,
    "best_accuracy": float,
    "macro_f1_at_acc_threshold": float,
    "n_val_samples": int,
    "n_B": int,
    "n_C": int,
}
```

---

## 7. Routing Logic

### 7.1 Full cascade decision tree

For each question in predict.json:

```
Question q
    │
    ▼
Gate 1: Agreement gate
    │
    ├── Agree (nor_qa == oner_qa after normalization) → Route A → use nor_qa answer
    │
    └── Disagree → Gate 2: κ(q) threshold
                     │
                     ├── κ(q) ≥ threshold → Route C → use ircot_qa answer
                     │
                     └── κ(q) < threshold → Route B → use oner_qa answer
```

### 7.2 Implementation

The routing decision is at [L434–441]:

```python
for qid in all_qids:
    ds = qid_to_dataset[qid]
    if gate1[qid] == "A":
        merged[qid] = {"prediction": "A", "dataset_name": ds}
    else:
        k = qid_to_kappa[qid]
        label = "C" if k >= threshold else "B"
        merged[qid] = {"prediction": label, "dataset_name": ds}
```

### 7.3 Feature extraction scope

Features are computed **only for R-routed questions** (questions where Gate 1 = "R"). A-routed questions bypass feature extraction entirely [L418–419]:

```python
r_qids = [qid for qid in all_qids if gate1[qid] != "A"]
r_questions = [qid_to_question[qid] for qid in r_qids]
```

### 7.4 Optional Clf2 comparison

When `--clf2_pred_file` is provided, the script computes agreement between κ(q) routing and Clf2 routing on R-routed questions [L448–458]:

```python
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
```

This is for diagnostic comparison only — Clf2 predictions are never used for routing in IT7.

---

## 8. Data Pipeline

### 8.1 How queries are fed

Questions are **not** fed through a trained model. The script:

1. Loads `predict.json` for the qid → dataset_name and qid → question mappings [L355–359]
2. Loads pre-computed QA prediction files (nor_qa, oner_qa) for agreement computation [L396]
3. Extracts structural features from question text using spaCy NER + regex [L422]
4. Computes κ(q) from features [L423]
5. Routes each question to the appropriate pre-computed QA answer [L513–535]

### 8.2 Prediction file loading

`load_strategy_predictions()` [L126–142] loads nor_qa and oner_qa predictions for all datasets, following the same file path pattern as `predict_complexity_agreement.py`:

| Strategy | Pattern |
|---|---|
| nor_qa | `predictions/test/nor_qa_{model}_{ds}____prompt_set_1/prediction__{ds}_to_{ds}__test_subsampled.json` |
| oner_qa | `predictions/test/oner_qa_{model}_{ds}____prompt_set_1___bm25_retrieval_count__{N}___distractor_count__1/prediction__{ds}_to_{ds}__test_subsampled.json` |

### 8.3 BM25 retrieval counts

| Model | `ONER_BM25` | `IRCOT_BM25` | Source |
|---|---|---|---|
| `flan_t5_xl` | 15 | 6 | [L50–51] |
| `flan_t5_xxl` | 15 | 6 | [L50–51] |
| `gpt` | 6 | 3 | [L50–51] |

These are **identical** to all prior iterations.

### 8.4 predict.json format

JSON list of 3 000 objects (500 per dataset). The script extracts `id`, `dataset_name`, and `question` [L355–359]:

```python
qid_to_dataset = {item["id"]: item["dataset_name"] for item in predict_data}
qid_to_question = {item["id"]: item["question"] for item in predict_data}
```

### 8.5 Per-dataset QA prediction file routing

Built at [L480–500]:

```python
dataName_to_files[ds] = {
    "C": "predictions/test/ircot_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__{ircot_bm25}___distractor_count__1/prediction__{ds}_to_{ds}__test_subsampled.json",
    "B": "predictions/test/oner_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__{oner_bm25}___distractor_count__1/prediction__{ds}_to_{ds}__test_subsampled.json",
    "A": "predictions/test/nor_qa_{m}_{ds}____prompt_set_1/prediction__{ds}_to_{ds}__test_subsampled.json",
}
```

### 8.6 stepNum loading

The script loads step numbers for ircot-routed questions [L462–476] with a fallback path:

1. **Primary:** `predictions/test/ircot_qa_{m}/total/stepNum.json` (consolidated file)
2. **Fallback:** per-dataset `predictions/test/ircot_qa_{m}_{ds}____prompt_set_1___bm25_retrieval_count__{N}___distractor_count__1/stepNum.json`
3. **Default:** `total_step_num.get(qid, 0)` — if neither exists, stepNum defaults to 0

Step number values: A → 0, B → 1, C → variable (loaded from stepNum file).

---

## 9. No Training Artifacts — Confirmed

### 9.1 Complete import list

The script's imports [L30–43]:

```python
import argparse
import json
import os
import re
import string
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ...)
from postprocess_utils import load_json, save_json
```

Plus a lazy import of `sklearn.metrics.f1_score` inside `tune_threshold()` [L221].

### 9.2 What is absent

| Library | Present? | Implication |
|---|---|---|
| `torch` | **No** | No tensor operations, no GPU usage |
| `transformers` | **No** | No model loading, no tokenizer, no generate() |
| `accelerate` | **No** | No distributed training/inference |
| `datasets` | **No** | No HuggingFace dataset loading |
| `pandas` | **No** | No DataFrame operations |
| `matplotlib` | **No** | No plots (unlike IT6's probe) |

### 9.3 No checkpoint references

The script takes no `--model_name_or_path` argument, no `--checkpoint` argument, no path to a trained model. The only model loaded is spaCy `en_core_web_sm` (12 MB NER model, CPU-only).

### 9.4 No writes to classifier/outputs/

All output is written to `--output_path` (typically `predictions/classifier/t5-large/{model}/iter7_kappa/`). No files are created under `classifier/outputs/` (the checkpoint tree).

---

## 10. Evaluation Setup

### 10.1 End-to-end QA evaluation

**Identical to Iteration 1.** After routing, the output directory structure is compatible with `evaluate_final_acc.py`:

```
python evaluate_final_acc.py --pred_path predictions/classifier/t5-large/{model}/iter7_kappa/
```

| Step | Procedure | Difference from IT1? |
|---|---|---|
| Routed prediction format | `{dataset}/{dataset}.json` + `{dataset}_option.json` | No |
| Evaluation script | `evaluate_final_acc.py --pred_path ...` | No |
| Single-hop metrics (nq, trivia, squad) | `SquadAnswerEmF1Metric` | No |
| Multi-hop metrics (musique, hotpotqa, 2wikimultihopqa) | Official evaluator scripts via `subprocess` | No |
| Per-dataset output | Printed to stdout | No |

### 10.2 Invocation via `run-all-iterations.sh`

The `route_kappa()` helper [L147–159] runs both routing and evaluation:

```bash
route_kappa() {
    local tag="$1" model="$2"
    local out="predictions/classifier/t5-large/${model}/${tag}"
    echo "  [${tag}/${model}] Routing (agreement gate + kappa)..."
    python classifier/postprocess/predict_complexity_kappa.py "${model}" \
        --use_agreement_gate \
        --tune_threshold \
        --valid_file "classifier/data/${DATASET}/${model}/binary_silver_single_vs_multi/train.json" \
        --output_path "${out}"
    echo "  [${tag}/${model}] Evaluating..."
    python evaluate_final_acc.py --pred_path "${out}"
}
```

Called for all three models [L332–334]:

```bash
for m in "${MODELS[@]}"; do
    route_kappa "iter7_kappa" "$m"
done
```

### 10.3 No Phase 0 dependency

IT7 does **not** depend on Phase 0 (standard classifier training). The `needs_std` flag at [run-all-iterations.sh L167–168] includes only iterations 1–5:

```bash
for i in 1 2 3 4 5; do should_run "$i" && needs_std=true; done
```

IT7 can run independently with `bash run-all-iterations.sh 7` — no prior training is required.

---

## 11. Output Artifacts

### 11.1 Directory tree

When invoked via `run-all-iterations.sh`:

```
predictions/classifier/t5-large/{model}/iter7_kappa/
  routing_stats.json
  musique/
    musique.json
    musique_option.json
  hotpotqa/
    hotpotqa.json
    hotpotqa_option.json
  2wikimultihopqa/
    2wikimultihopqa.json
    2wikimultihopqa_option.json
  nq/
    nq.json
    nq_option.json
  trivia/
    trivia.json
    trivia_option.json
  squad/
    squad.json
    squad_option.json
```

### 11.2 Per-dataset files

**`{dataset}.json`** — maps qid → final answer string:
```json
{"single_nq_dev_9": "Gal Gadot", ...}
```

**`{dataset}_option.json`** — maps qid → routing metadata, **including κ(q) score** for R-routed questions [L526–532]:

```json
{
    "2hop__511176_22458": {
        "prediction": "answer text",
        "option": "C",
        "stepNum": 4,
        "kappa": 0.847231
    },
    "single_nq_dev_9": {
        "prediction": "answer text",
        "option": "A",
        "stepNum": 0
    }
}
```

The `kappa` field is present only for R-routed questions (those that passed through Gate 2). A-routed questions have no `kappa` field.

### 11.3 `routing_stats.json`

Saved at [L552–579]:

```json
{
    "method": "symrag_kappa_structural",
    "model_name": "flan_t5_xl",
    "threshold_used": 0.1654,
    "threshold_tuned": true,
    "kappa_stats": {
        "mean": 0.2474,
        "std": 0.1100,
        "min": 0.0575,
        "max": 1.0066
    },
    "routing_counts": {"A": 581, "B": 527, "C": 1892},
    "total_questions": 3000,
    "total_steps": 9233,
    "per_dataset": {
        "musique": {"A": 48, "B": 16, "C": 436, "steps": 1577},
        ...
    },
    "symrag_weights": {"w_L": 1.0, "w_sh1": 0.05, "w_sh2": 0.10},
    "note": "A(q) attention term omitted; structural heuristic only",
    "tuning_stats": {
        "tuning_criterion": "macro_f1",
        "best_f1_threshold": 0.1654,
        "best_macro_f1": 0.6331,
        "accuracy_at_used_threshold": 0.6398,
        "val_B_accuracy": 0.6772,
        "val_C_accuracy": 0.5898,
        "best_acc_threshold": 0.1654,
        "best_accuracy": 0.6398,
        "macro_f1_at_acc_threshold": 0.6331,
        "n_val_samples": 3268,
        "n_B": 1871,
        "n_C": 1397
    }
}
```

### 11.4 Stdout output

The script prints structured progress to stdout:

```
[data]  3000 questions from classifier/data/.../predict.json

[tune]  Tuning threshold on validation data (criterion: macro-F1)...
Tuned threshold (macro-F1): 0.1654, best macro-F1: 0.6331, val B-acc: 0.6772, val C-acc: 0.5898, accuracy at this threshold: 0.6398
Best accuracy: 0.6398 at threshold 0.1654 (same as F1-optimal)
[tune]  Using tuned threshold: 0.1654 (macro-F1=0.6331)

[gate1] Loading predictions...
[gate1] Computing nor_qa/oner_qa agreement...
[gate1] A=581, R=2419

[feat]  Extracting features for 2419 R-routed questions...
[feat]  κ stats: mean=0.2474, std=0.1100, min=0.0575, max=1.0066

[gate2] Routing with threshold=0.1654...
[route] A=581, B=527, C=1892

[out]   Writing predictions to predictions/classifier/t5-large/flan_t5_xl/iter7_kappa/
  musique: A=48, B=16, C=436, steps=1577
  hotpotqa: A=120, B=14, C=366, steps=1972
  ...

Routed predictions saved to predictions/classifier/t5-large/flan_t5_xl/iter7_kappa/
Run evaluation with: python evaluate_final_acc.py --pred_path predictions/classifier/t5-large/flan_t5_xl/iter7_kappa/
```

---

## 12. Suspicious / Noteworthy Items

### 12.1 Max-normalisation computed on R-routed questions only

| Issue | Detail |
|---|---|
| **What** | `compute_kappa()` normalises token lengths by `max_len = token_lens.max()` [L201], where `token_lens` is extracted only from R-routed questions [L418–419]. A-routed questions are excluded. |
| **Risk** | The normalisation base depends on Gate 1's output. If the longest question happens to be A-routed, the max-normalisation denominator is smaller, inflating all κ(q) values. Different Gate 1 configurations produce different R-question pools, making κ(q) scores non-comparable across iterations. |
| **Severity** | Low — the threshold is tuned on validation data using the same `compute_kappa()` function (which normalises by its own max), so the tuning and prediction use consistent normalisation. The issue only arises when comparing raw κ(q) values across different Gate 1 configurations. |

### 12.2 Validation-set max-normalisation differs from test-set normalisation

| Issue | Detail |
|---|---|
| **What** | During threshold tuning, `compute_kappa()` is called on the **validation** questions [L233–234]. During prediction, `compute_kappa()` is called on the **R-routed test** questions [L422–423]. Each call computes its own `max_len`. If the validation set has a different max token length than the test set, the same question would receive a different κ(q) score in tuning vs prediction. |
| **Risk** | The tuned threshold is calibrated to validation-set κ(q) values, but applied to test-set κ(q) values with a potentially different normalisation base. This is a **distribution shift in feature space** introduced by the normalisation. |
| **Severity** | Medium — if max token lengths are similar between validation and test sets, the effect is negligible. If they differ substantially, the tuned threshold may be suboptimal. |

### 12.3 Threshold tuning uses macro-F1 on IB+silver merged data (resolved)

| Issue | Detail |
|---|---|
| **What** | The tuned threshold now maximises **macro-F1** on the IB+silver merged data [L248, L383], not accuracy. The accuracy-optimal threshold is computed and reported as a secondary reference. |
| **Status** | **Resolved.** The earlier version used accuracy on the silver-only validation split (e.g., XL: 691 B vs 220 C, 3:1 ratio), which produced high thresholds (~0.46) that under-routed to C. Switching to macro-F1 on the IB+silver merged set (~1.3:1 ratio) lowers τ to ~0.165 and matches or exceeds IT5's trained Clf2 on end-to-end QA F1. |

### 12.4 Prediction files are loaded redundantly per question

| Issue | Detail |
|---|---|
| **What** | In the per-dataset loop [L513–535], `load_json(dataName_to_files[data_name][option])` is called once per question inside the inner loop. This re-reads the same JSON file from disk for every question. |
| **Performance** | For 500 questions per dataset, this means up to 500 redundant file reads per strategy per dataset. |
| **Impact** | Correctness unaffected. Runtime slower than necessary but acceptable for 3 000 questions. This is the same pattern present in IT1's `predict_complexity_split_classifiers.py` and IT5's `predict_complexity_agreement.py`. |

### 12.5 `answer_extractor()` and `normalize_answer()` are duplicated

| Issue | Detail |
|---|---|
| **What** | Both functions are copy-pasted from `predict_complexity_agreement.py` (which itself copied from `evaluate_final_acc.py`). They are not imported from a shared module. |
| **Risk** | If one copy is updated and the others are not, the normalization applied during routing and evaluation could diverge. |
| **Current state** | All three copies (predict_complexity_kappa.py, predict_complexity_agreement.py, evaluate_final_acc.py) are identical. |

### 12.6 κ(q) is dominated by L(q) due to small SymRAG weights

| Issue | Detail |
|---|---|
| **What** | With $w_{sh1} = 0.05$ and $w_{sh2} = 0.10$, the $S_H(q)$ term contributes very little to $\kappa(q)$. For a typical question with 2 entities in 15 tokens and 1 hop pattern: $S_H = 0.05 \cdot (2/15) + 0.10 \cdot (1/15) = 0.0067 + 0.0067 = 0.013$. So $\kappa(q) \approx L(q) \cdot 1.013$. |
| **Implication** | The threshold-based B/C decision is effectively a **query-length threshold** with minor perturbations from entity and hop features. This means IT7's Gate 2 is approximately: "long questions → multi-step (C), short questions → single-step (B)." |
| **Relevance** | This is consistent with IT6's probe findings: `token_len_norm` had the largest logistic regression coefficient among the four features. The structural features add marginal discriminative power. |

### 12.7 spaCy entity count may not correspond to SymRAG's N_ents(q)

| Issue | Detail |
|---|---|
| **What** | SymRAG defines $N_{ents}(q)$ as the count of named entities in the query but does not specify which NER model. IT7 uses spaCy `en_core_web_sm`, which has a known NER F1 of ~85% on OntoNotes. The entity count may differ from what SymRAG's pipeline produces. |
| **Impact** | Minor — given $w_{sh1} = 0.05$, entity density's contribution to κ(q) is very small. Even a 15% NER error rate translates to negligible κ(q) differences. |

### 12.8 Hop patterns are a superset of SymRAG's multi-hop keywords

| Issue | Detail |
|---|---|
| **What** | SymRAG mentions "multi-hop keyword indicators" for $N_{hops}(q)$ but does not publish the exact patterns. IT7's 7 bridging patterns were designed for the Adaptive-RAG dataset mix (MuSiQue, HotpotQA, 2WikiMultiHopQA, NQ, TriviaQA, SQuAD) and may not match SymRAG's keyword list. |
| **Impact** | The κ(q) scores are not directly comparable to SymRAG's published results. This is a known approximation. |

### 12.9 No early termination on empty R-set

| Issue | Detail |
|---|---|
| **What** | If Gate 1 routes all 3 000 questions to A (complete agreement), `r_qids` would be empty, and `compute_kappa()` would receive empty arrays. `np.array([]).max()` raises `ValueError: zero-size array reduction`. |
| **Risk** | Extremely unlikely in practice — complete agreement across 3 000 questions with different retrieval strategies would require identical answers for every question. |
| **Mitigation** | None present. A defensive check like `if not r_qids: ...` would prevent the crash. |

### 12.10 stepNum default of 0 for missing C-routed questions

| Issue | Detail |
|---|---|
| **What** | `step_num = total_step_num.get(qid, 0)` [L518] defaults to 0 if the question ID is not found in the stepNum file. For C-routed questions, the actual step count should be ≥ 1. |
| **Impact** | Affects cost accounting only (total retrieval steps reported), not answer selection. Same pattern as IT5's `predict_complexity_agreement.py`. |

### 12.11 `--tune_threshold` without `--valid_file` is caught

| Issue | Detail |
|---|---|
| **What** | `parser.error("--tune_threshold requires --valid_file.")` [L347] is called if `--tune_threshold` is passed without `--valid_file`. This is a proper argument validation. |
| **Impact** | None — correct behaviour. |

### 12.12 No `--clf1_pred_file` and no `--use_agreement_gate` is caught

| Issue | Detail |
|---|---|
| **What** | `parser.error("Provide --clf1_pred_file or --use_agreement_gate.")` [L345] is called if neither Gate 1 option is specified. |
| **Impact** | None — correct behaviour. |
