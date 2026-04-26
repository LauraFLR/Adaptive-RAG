# Iteration 7 — Feature-Augmented Gate 2 Classifier

> **Design Science Research Artifact:** Retrain the Gate 2 (Clf2: B vs C)
> classifier with three structural features — token length, named-entity
> count, and bridging-phrase flag — prepended as a plain-text prefix to the
> question string before T5 tokenization.  The T5-Large model architecture is
> completely unmodified; the features are injected purely through the input
> text.  Gate 1 is not retrained.

---

## 1. Files Involved

| File | Role |
|---|---|
| `classifier/run/run_large_train_feat_single_vs_multi.sh` (99 lines) | **New.** Single shell script that trains, validates, and predicts for all three model variants. Parameterized by positional arg (`flan_t5_xl`, `flan_t5_xxl`, `gpt`). |
| `classifier/data_utils/add_feature_prefix.py` (118 lines) | **New.** Offline preprocessing script that reads original Clf2 JSON files, computes features via spaCy + regex, prepends `[LEN:X] [ENT:Y] [BRIDGE:Z]` to each question, and writes new files. Run once before training. |
| `classifier/postprocess/clf2_feature_probe.py` (443 lines) | **Reference only (IT6).** Defines the same three features and seven bridge-flag regex patterns. Not imported by the training pipeline. |
| `classifier/run_classifier.py` (937 lines) | Shared — the manual Accelerate training loop (non-focal path) is the active code path. |
| `classifier/utils.py` (254 lines) | Shared — `preprocess_features_function()` tokenizes the (now-prefixed) question string. |
| `classifier/postprocess/predict_complexity_split_classifiers.py` | Routing — merges Clf1 + Clf2 predictions into A/B/C, routes to QA strategy answers. **Identical to IT1.** |
| `classifier/postprocess/predict_complexity_agreement.py` (251 lines) | Routing — IT5 agreement gate. Intended Gate 1 pairing for IT7. |
| `evaluate_final_acc.py` (341 lines) | QA evaluation — **identical to IT1.** |

**`run_classifier.py` and `utils.py` are NOT modified.** The feature injection happens entirely in the data files — the training script reads the pre-augmented JSON files and processes them through the same tokenization pipeline as IT1.

---

## 2. Model Architecture

### 2.1 T5-Large — completely unmodified

The model is the same T5-Large (~770 M parameters) loaded via `AutoModelForSeq2SeqLM` as in IT1. No new layers, no additional input heads, no embedding modifications. The model's `config.json`, vocabulary, and architecture are identical.

### 2.2 How features are prepended

Features are injected as a **plain-text prefix** before the original question string. The preprocessing script `add_feature_prefix.py` modifies the `"question"` field in each JSON item at [L65–68]:

```python
item["question"] = (
    f"[LEN:{token_len}] [ENT:{entity_count}] [BRIDGE:{bridge_flag}] {q}"
)
```

### 2.3 Concrete example

Original question:
```
What country was the director of the film born in?
```

After feature augmentation:
```
[LEN:14] [ENT:2] [BRIDGE:1] What country was the director of the film born in?
```

Actual sample from the XL training file:
```
[LEN:9] [ENT:1] [BRIDGE:0] When was the institute that owned The Collegian founded?
```

### 2.4 How T5 processes the prefix

The feature prefix is tokenized by T5's SentencePiece tokenizer as ordinary text. At training time, `preprocess_features_function()` in `utils.py` [L79–105] tokenizes the full prefixed string:

```python
model_inputs = tokenizer(
    examples[question_column],  # now includes "[LEN:X] [ENT:Y] ..." prefix
    truncation=True,
    max_length=max_seq_length,
    ...
)
```

The tokenizer decomposes each bracket-tag into subword tokens. The T5-Large tokenizer splits the prefix `[LEN:9] [ENT:1] [BRIDGE:0] ` into **18 subword tokens**:

```
▁[  LE  N  :  9  ]  ▁[  ENT  :  1  ]  ▁[  BR  ID  GE  :  0  ]
```

This 18-token overhead is **constant** regardless of the feature values (single-digit vs double-digit numbers tokenize to the same count, as T5 treats numbers as single tokens up to 2–3 digits).

The model learns to attend to these prefix tokens during fine-tuning. No explicit attention mask modification is needed — the prefix is simply part of the input sequence.

---

## 3. Structural Features

### 3.1 Feature table

| Feature | Tag format | Extraction logic | Library | Source file |
|---|---|---|---|---|
| `token_len` | `[LEN:X]` | `len(question.split())` — whitespace-split word count | Built-in `str.split()` | `add_feature_prefix.py` [L65] |
| `entity_count` | `[ENT:Y]` | `len(doc.ents)` — spaCy named-entity count | spaCy `en_core_web_sm` | `add_feature_prefix.py` [L66] |
| `bridge_flag` | `[BRIDGE:Z]` | `1 if _BRIDGE_RE.search(question) else 0` — binary regex match | `re` stdlib | `add_feature_prefix.py` [L67] |

### 3.2 Feature set is identical to Iteration 6

The three features and their extraction logic are the same as in `clf2_feature_probe.py` (IT6). The `BRIDGE_PATTERNS` list and `_BRIDGE_RE` compiled regex are **literally copied** between the two files:

| Pattern | `add_feature_prefix.py` lines | `clf2_feature_probe.py` lines |
|---|---|---|
| 7 regex patterns | [L30–37] | [L60–73] |
| Compiled alternation | [L38] | [L74] |

Both files import `re`, define the same 7 patterns in the same order, and compile them with `re.IGNORECASE`.

### 3.3 The seven bridge-flag patterns

| # | Pattern | Description |
|---|---|---|
| 1 | `\b(?:who\|where\|which\|that)\s+(?:was\|were\|is\|are\|did\|had\|has\|does)\b` | Relative-clause bridges |
| 2 | `\w+'s\s+\w+(?:\s+\w+){0,5}\s+\w+'s` | Double possessive |
| 3 | `\b(?:before\|after\|when\|while)\b.{3,60}\b(?:who\|what\|where\|which)\b` | Temporal subordination |
| 4 | `\b(?:that\|this\|those\|these)\s+(?:country\|city\|person\|...)\b` | Demonstrative back-reference |
| 5 | `\b(?:both)\b.{1,40}\band\b` | Explicit comparison (both…and) |
| 6 | `\bbetween\b.{1,40}\band\b` | Explicit comparison (between…and) |
| 7 | `\bof\s+the\s+\w+\s+(?:who\|that\|which\|where)\b` | Nested wh-question |

### 3.4 bridge_flag distribution by dataset (XL training data)

| Dataset | N | bridge=1 | % |
|---|---|---|---|
| 2wikimultihopqa | 513 | 190 | 37.0 % |
| musique | 522 | 175 | 33.5 % |
| hotpotqa | 561 | 124 | 22.1 % |
| nq | 546 | 120 | 22.0 % |
| trivia | 580 | 81 | 14.0 % |
| squad | 546 | 46 | 8.4 % |

The flag varies substantially across datasets — from 8.4 % (squad) to 37.0 % (2wikimultihopqa). It is not near-constant globally but may be near-constant within individual datasets that are predominantly single-hop (e.g., squad).

---

## 4. Training Parameters

### 4.1 Comparison table: IT7 feature-augmented Clf2 vs IT1 standard Clf2

| Parameter | IT7 (feat Clf2) | IT1 (standard Clf2) | Match? |
|---|---|---|---|
| Base model | `t5-large` | `t5-large` | ✓ |
| Learning rate | `3e-5` | `3e-5` | ✓ |
| Train batch size | `32` | `32` | ✓ |
| Eval batch size | `100` | `100` | ✓ |
| Max seq length | `384` | `384` | ✓ |
| Doc stride | `128` | `128` | ✓ |
| Weight decay | `0.0` (default) | `0.0` | ✓ |
| Grad accum steps | `1` (default) | `1` | ✓ |
| Seed | `42` | `42` | ✓ |
| Labels | `B C` | `B C` | ✓ |
| Epochs (XL) | `15, 20, 25, 30, 35` | `15, 20, 25, 30, 35` | ✓ |
| Epochs (XXL) | `15, 20, 25, 30, 35` | `15, 20, 25, 30, 35` | ✓ |
| Epochs (GPT) | `35, 40` | `35, 40` | ✓ |
| Loss | Standard CE (T5 built-in) | Standard CE | ✓ |
| `--use_focal_loss` | **Not set** | Not set | ✓ |
| `--auto_class_weights` | **Not set** | Not set | ✓ |
| Optimizer | `torch.optim.AdamW` [run_classifier.py L640] | `torch.optim.AdamW` | ✓ |
| LR scheduler | `get_scheduler("linear")` | `get_scheduler("linear")` | ✓ |
| Code path | Manual Accelerate loop [run_classifier.py L730–840] | Manual Accelerate loop | ✓ |
| **Training file** | **`binary_silver_feat_single_vs_multi/train.json`** | `binary_silver_single_vs_multi/train.json` | **Different (feature-prefixed)** |
| **Validation file** | **`silver_feat_single_vs_multi/valid.json`** | `silver/single_vs_multi/valid.json` | **Different (feature-prefixed)** |
| **Predict file** | **`feat_predict.json`** | `predict.json` | **Different (feature-prefixed)** |
| **Output dir tag** | **`feat_single_vs_multi/`** | `single_vs_multi/` | **Different** |
| GPU | `GPU=0` | `GPU=0` | ✓ |
| Shell script | Single unified script (all 3 models) | 3 separate scripts (one per model) | Different structure |

### 4.2 Epoch ranges match standard Clf2

The script conditionally sets epoch ranges [run_large_train_feat_single_vs_multi.sh L30–34]:

```bash
if [ "$LLM_NAME" = "gpt" ]; then
    EPOCHS="35 40"
else
    EPOCHS="15 20 25 30 35"
fi
```

These are **identical** to the standard Clf2 scripts (`run_large_train_{xl,xxl}_single_vs_multi.sh`: `15 20 25 30 35`; `run_large_train_gpt_single_vs_multi.sh`: `35 40`).

### 4.3 Code path: manual Accelerate training loop

Since `--use_focal_loss` is not set, the training enters the manual Accelerate loop at [run_classifier.py L730–840]:
- `torch.optim.AdamW` optimizer [L640]
- `accelerator.prepare(model, optimizer)` [L650–651]
- Standard `outputs = model(**batch)` → `loss = outputs.loss` [L785–786]
- T5's built-in full-vocabulary cross-entropy (not the 2-class softmax used in IT3/IT4)

---

## 5. Feature Preprocessing

### 5.1 Preprocessing script: `add_feature_prefix.py`

The script is run **once** before training to create the feature-prefixed data files. It is not part of the training loop.

### 5.2 No binning, no normalization — raw integers

Features are injected as raw integer values with no transformation:

| Feature | Type | Range (typical) | Representation |
|---|---|---|---|
| `token_len` | int | 5–50+ | Raw integer: `[LEN:14]` |
| `entity_count` | int | 0–10+ | Raw integer: `[ENT:2]` |
| `bridge_flag` | int (binary) | 0 or 1 | Raw integer: `[BRIDGE:0]` or `[BRIDGE:1]` |

No binning (e.g., "short"/"medium"/"long"), no min-max normalization, no z-score standardization. The T5 model must learn the relationship between raw numeric values and the B/C decision from the training data alone.

### 5.3 The f-string format

At [add_feature_prefix.py L65–68]:

```python
item["question"] = (
    f"[LEN:{token_len}] [ENT:{entity_count}] [BRIDGE:{bridge_flag}] {q}"
)
```

Each tag is enclosed in square brackets with a colon separator. Tags are separated by single spaces. The prefix ends with a space before the original question text.

### 5.4 How T5 tokenizer processes the prefix

T5-Large uses SentencePiece tokenization. The bracket/colon/number tokens are decomposed as follows:

| Token group | Subword tokens |
|---|---|
| `[LEN:9]` | `▁[`, `LE`, `N`, `:`, `9`, `]` (6 tokens) |
| `[ENT:1]` | `▁[`, `ENT`, `:`, `1`, `]` (5 tokens) |
| `[BRIDGE:0]` | `▁[`, `BR`, `ID`, `GE`, `:`, `0`, `]` (7 tokens) |

Total prefix overhead: **18 subword tokens** (constant across all feature values, since single- and double-digit integers are each encoded as one token by T5).

### 5.5 Token overhead vs max_seq_length

With `max_seq_length=384` and a prefix overhead of 18 tokens, the effective capacity for the question text is 384 − 18 − 1 (EOS) = **365 tokens**. Clf2 questions are typically short (< 50 tokens), so the overhead is negligible in practice.

### 5.6 Batch NER processing

The `augment_file()` function [add_feature_prefix.py L52–74] processes all questions in a single spaCy batch for efficiency [L58]:

```python
docs = list(nlp.pipe(questions, batch_size=512))
```

The spaCy model is loaded once with `disable=["parser", "lemmatizer"]` [L80], keeping only the NER pipeline component.

---

## 6. No Residual Changes from Iterations 2–4

### 6.1 Confirmed: no `--use_focal_loss`

The shell script [run_large_train_feat_single_vs_multi.sh] does not pass `--use_focal_loss` to `run_classifier.py`. Verified: zero occurrences of "focal", "undersampl", "class_weight", or "auto_class" in the script.

### 6.2 Confirmed: no undersampling

The training data files (`binary_silver_feat_single_vs_multi/train.json`) are feature-prefixed copies of the **full** `binary_silver_single_vs_multi/train.json` files. Sample counts are identical:

| Model | Original Clf2 train | Feature-prefixed train | Labels match? |
|---|---|---|---|
| `flan_t5_xl` | 3 268 (B=1 871, C=1 397) | 3 268 (B=1 871, C=1 397) | ✓ |
| `flan_t5_xxl` | 3 298 (B=1 903, C=1 395) | 3 298 (B=1 903, C=1 395) | ✓ |
| `gpt` | 2 804 (B=1 475, C=1 329) | 2 804 (B=1 475, C=1 329) | ✓ |

### 6.3 Confirmed: no class weighting

No `--auto_class_weights`, no `--focal_alpha`. Standard unweighted cross-entropy on the full training data.

### 6.4 Summary: IT7 changes ONLY the input text

The sole difference from IT1's Clf2 is the `[LEN:X] [ENT:Y] [BRIDGE:Z]` prefix in the question field. Everything else — model, optimizer, loss, scheduler, hyperparameters, code path — is identical.

---

## 7. Gate 1

### 7.1 No Gate 1 training

The shell script trains only Clf2 (B vs C). There is no Clf1 (A vs R) training in IT7. The `--labels B C` argument confirms this [run_large_train_feat_single_vs_multi.sh L49].

### 7.2 Intended Gate 1 pairing

At evaluation time, the IT7 feature-augmented Clf2 is intended to be paired with the **IT5 agreement gate** (`predict_complexity_agreement.py`). The agreement gate produces A/R decisions (agree → A, disagree → Clf2's B/C). The IT7 Clf2 checkpoint is passed to the agreement script via `--clf2_pred_file`.

This pairing can also be done with any IT1–IT4 Clf1 checkpoint via `predict_complexity_split_classifiers.py`, but the agreement gate is the primary intended pairing for IT7.

### 7.3 Routing flow

```
Question → Agreement Gate (IT5)
              ├── Agree → A (no retrieval)
              └── Disagree → IT7 Clf2 prediction
                                ├── B → single-step retrieval
                                └── C → multi-step retrieval
```

---

## 8. Evaluation Setup

**Identical to all prior iterations.**

| Step | Procedure | Difference from IT1? |
|---|---|---|
| Per-epoch validation | `run_classifier.py --do_eval` on `silver_feat_single_vs_multi/valid.json` | Different input file (feature-prefixed), same evaluation code |
| Per-epoch prediction | `run_classifier.py --do_eval` on `feat_predict.json` | Different input file, same evaluation code |
| Cascade routing | `predict_complexity_agreement.py` or `predict_complexity_split_classifiers.py` | No change |
| QA evaluation | `evaluate_final_acc.py --pred_path ...` | No change |

The validation and prediction files are feature-prefixed versions of the same data. The evaluation code in `run_classifier.py` (`calculate_accuracy`, `calculate_accuracy_perClass`) is unchanged — it compares predicted labels to ground-truth labels regardless of the input format.

---

## 9. Output Artifacts

### 9.1 Directory tree

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/
  {model}/                                          # flan_t5_xl, flan_t5_xxl, or gpt
    feat_single_vs_multi/                           ← SEPARATE from IT1's single_vs_multi/
      epoch/
        {epoch}/                                    # 15..35 for xl/xxl; 35,40 for gpt
          feat/                                     ← DATE=feat (static, not timestamped)
            config.json
            generation_config.json
            model.safetensors                       # or pytorch_model.bin
            spiece.model
            special_tokens_map.json
            tokenizer.json
            tokenizer_config.json
            valid/
              dict_id_pred_results.json
              final_eval_results.json
              final_eval_results_perClass.json
              logs.log
            predict/
              dict_id_pred_results.json
              final_eval_results.json
              final_eval_results_perClass.json
              logs.log
```

### 9.2 `DATE=feat` is static, not timestamped

The shell script sets `DATE=feat` [run_large_train_feat_single_vs_multi.sh L24] — a fixed string, not `$(date ...)`. This means the innermost directory is always named `feat/`.

Contrast with IT1's standard Clf2 scripts, which use `DATE=$(date +"%Y_%m_%d")` and `TIME=$(date +"%H_%M_%S")`, creating unique `{YYYY_MM_DD}/{HH_MM_SS}/` subdirectories per run.

### 9.3 Separation from other iterations

| Iteration | Clf2 output path |
|---|---|
| IT1 (standard) | `.../single_vs_multi/epoch/{N}/{DATE}/{TIME}/` |
| **IT7 (features)** | **`.../feat_single_vs_multi/epoch/{N}/feat/`** |

The directory names (`feat_single_vs_multi` vs `single_vs_multi`) are distinct. No risk of overwriting IT1 outputs.

### 9.4 Data file artifacts

| File | Path | Size |
|---|---|---|
| Feature-prefixed training data | `classifier/data/.../{model}/binary_silver_feat_single_vs_multi/train.json` | Same row count as original |
| Feature-prefixed validation data | `classifier/data/.../{model}/silver_feat_single_vs_multi/valid.json` | Same row count as original |
| Feature-prefixed predict data | `classifier/data/.../feat_predict.json` | 3 000 items (shared across models) |

---

## 10. Suspicious Items / Flags

### 10.1 `DATE=feat` — static, overwrites on re-run

| Issue | Detail |
|---|---|
| **What** | `DATE=feat` [run_large_train_feat_single_vs_multi.sh L24] is a fixed string. If the script is run twice for the same model, the second run **overwrites** the first run's outputs without warning. |
| **Contrast** | IT1's standard Clf2 scripts use `DATE=$(date ...)` + `TIME=$(date ...)`, creating unique per-run subdirectories. |
| **Impact** | No run isolation. Previous results are silently destroyed on re-run. |

### 10.2 Feature prefix token overhead vs max_seq_length

| Issue | Detail |
|---|---|
| **What** | The 18-token prefix reduces effective input capacity from 384 to ~365 tokens. |
| **Impact** | Negligible for Clf2 questions (typically < 50 tokens). However, if the prefix grew (e.g., more features), it could start truncating actual question content. |

### 10.3 No data leakage in features

| Issue | Detail |
|---|---|
| **What** | All three features — token length, entity count, bridge flag — are computed solely from the question text itself. They do not use the answer, the label (B/C), the dataset name, or any external information. |
| **Status** | **No leakage.** The features are legitimate question-intrinsic properties. |

### 10.4 bridge_flag near-constant for some datasets

| Issue | Detail |
|---|---|
| **What** | In the XL training data: squad has only 8.4 % bridge=1, trivia has 14.0 %. Within these datasets, the bridge_flag provides almost no discriminative signal. |
| **Impact** | The model may learn to ignore the bridge_flag for questions from these datasets. However, across the full mixed dataset, the flag varies from 8.4 % to 37.0 %, providing some signal. |

### 10.5 Raw integers not binned

| Issue | Detail |
|---|---|
| **What** | `token_len` and `entity_count` are raw integers (e.g., `[LEN:14]`, `[ENT:2]`). T5 treats each number as a separate token. The model must learn that `[LEN:14]` and `[LEN:15]` are similar, while `[LEN:5]` and `[LEN:50]` are very different — purely from training data. |
| **Alternative** | Binning (e.g., `[LEN:short]`, `[LEN:medium]`, `[LEN:long]`) would reduce the vocabulary burden and make nearby values equivalent. Not implemented. |
| **Impact** | With ~3 000 training examples, the model may not see enough instances of each specific number to learn robust associations. |

### 10.6 `feat_predict.json` shared across models

| Issue | Detail |
|---|---|
| **What** | The predict file `feat_predict.json` is generated once (not per-model) at [add_feature_prefix.py L107–109]. All three model variants use the same file [run_large_train_feat_single_vs_multi.sh L84]. |
| **Implication** | This is correct — predict.json contains the same 3 000 questions for all models. The features are question-intrinsic, so they don't depend on the model variant. However, training files are per-model because different models produce different silver labels (different B/C splits). |
| **Risk** | None — this is intentional and correct. |

### 10.7 Duplicated feature logic between two files

| Issue | Detail |
|---|---|
| **What** | `add_feature_prefix.py` [L30–38] and `clf2_feature_probe.py` [L59–74] define identical `BRIDGE_PATTERNS` lists and `_BRIDGE_RE` compiled regex. The entity-count and token-length logic is also duplicated. Neither file imports from the other. |
| **Risk** | If one file's patterns are updated and the other is not, the features used in the diagnostic probe (IT6) and the training data (IT7) would diverge. |
| **Status** | Currently identical. |

### 10.8 No feature validation at inference time

| Issue | Detail |
|---|---|
| **What** | The training script reads pre-augmented JSON files. At inference time (validation, prediction), the files must also be pre-augmented. If someone accidentally passes a non-prefixed file, the model would receive questions without the feature prefix it was trained on. |
| **Detection** | There is no runtime check that the question starts with `[LEN:`. The model would silently produce degraded predictions. |
| **Mitigation** | The shell script explicitly uses `silver_feat_single_vs_multi/valid.json` and `feat_predict.json`, which are the correct feature-prefixed files. |

### 10.9 spaCy model version sensitivity

| Issue | Detail |
|---|---|
| **What** | NER results from `en_core_web_sm` may differ across spaCy versions. If `add_feature_prefix.py` is run with one version and `clf2_feature_probe.py` is run with another, the entity counts could differ. |
| **Impact** | The training data features and the diagnostic probe features could be inconsistent. |
| **Mitigation** | Both scripts are expected to run in the same virtual environment. |

### 10.10 Fresh-from-scratch training per epoch value

| Issue | Detail |
|---|---|
| **What** | Same as IT1: each epoch value (15, 20, 25, 30, 35 for XL/XXL; 35, 40 for GPT) trains from the base `t5-large` model, not from a previous checkpoint. |
| **Cost** | XL: 15+20+25+30+35 = 125 epochs total. XXL: same. GPT: 35+40 = 75 epochs. |
| **Note** | This is identical to the standard Clf2 training pattern and is not a new issue in IT7. |
