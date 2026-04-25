## 1. Files Involved

| File | Role |
|---|---|
| classifier/run/run_large_train_feat_single_vs_multi.sh | Gate 2 training launcher (feature-augmented); takes model name as `$1` |
| classifier/data_utils/add_feature_prefix.py | Offline data-generation utility: reads original Clf2 JSONs, prepends feature tags, writes new files |
| classifier/postprocess/clf2_feature_probe.py | Iteration 6 diagnostic: logistic-regression probe that established the feature set. Not called by Iteration 7, but defines the same feature logic |
| classifier/run_classifier.py | Core training/eval entry point (shared by all experiments) |
| classifier/utils.py | Model loading, tokenisation (`preprocess_features_function`), scheduler, accuracy metrics |
| classifier/postprocess/predict_complexity_split_classifiers.py | Routing: merges Clf1 + Clf2 predictions → routed QA answers |
| classifier/postprocess/predict_complexity_agreement.py | Alternative routing using agreement gate as Clf1 + any Clf2 |
| evaluate_final_acc.py | Final EM/F1 evaluation on routed predictions |
| `classifier/data/.../predict.json` | Original unlabelled test set (input to add_feature_prefix.py) |
| `classifier/data/.../{model}/binary_silver_single_vs_multi/train.json` | Original Clf2 training data (input) |
| `classifier/data/.../{model}/silver/single_vs_multi/valid.json` | Original Clf2 validation data (input) |
| `classifier/data/.../{model}/binary_silver_feat_single_vs_multi/train.json` | **Output**: feature-prefixed training data |
| `classifier/data/.../{model}/silver_feat_single_vs_multi/valid.json` | **Output**: feature-prefixed validation data |
| `classifier/data/.../feat_predict.json` | **Output**: feature-prefixed test set (shared across models) |
| spaCy `en_core_web_sm` | External dependency for NER feature extraction |

---

## 2. Model Architecture

**T5-Large** (`AutoModelForSeq2SeqLM`), identical to all prior iterations. Set at run_large_train_feat_single_vs_multi.sh: `MODEL=t5-large`. The model itself is **completely unmodified** — no architectural changes, no additional embedding layers, no auxiliary heads.

**How features are prepended:** The features are injected as a **plain-text prefix** to the question string, before tokenisation. In add_feature_prefix.py:

```python
item["question"] = (
    f"[LEN:{token_len}] [ENT:{entity_count}] [BRIDGE:{bridge_flag}] {q}"
)
```

A concrete example: `[LEN:14] [ENT:2] [BRIDGE:1] What country was the director of film The Milky Way born in?`

This modified string is written to JSON at rest. When run_classifier.py loads and tokenises it, T5's standard tokeniser converts the prefix into subword tokens that precede the actual question tokens — no special embedding or hidden-state concatenation is used.

---

## 3. Structural Features

Three features, all defined identically in both clf2_feature_probe.py and add_feature_prefix.py:

| Feature | Tag | Extraction logic | Source |
|---|---|---|---|
| **Token length** | `[LEN:X]` | `len(question.split())` — whitespace-split word count ([add_feature_prefix.py line 67](Adaptive-RAG/classifier/data_utils/add_feature_prefix.py#L67)) | Same as Iteration 6 probe ([clf2_feature_probe.py line 114](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L114)) |
| **Entity count** | `[ENT:Y]` | `len(doc.ents)` — spaCy `en_core_web_sm` NER count ([add_feature_prefix.py line 68](Adaptive-RAG/classifier/data_utils/add_feature_prefix.py#L68)) | Same as Iteration 6 probe ([clf2_feature_probe.py line 115](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L115)) |
| **Bridge flag** | `[BRIDGE:Z]` | Regex match against 7 multi-hop bridging-phrase patterns → 0 or 1 ([add_feature_prefix.py lines 31–39](Adaptive-RAG/classifier/data_utils/add_feature_prefix.py#L31-L39)) | Same patterns as Iteration 6 ([clf2_feature_probe.py lines 55–67](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L55-L67)) |

The `BRIDGE_PATTERNS` list is literally copied between the two files — 7 identical regex patterns targeting relative clauses, double possessives, temporal subordination, demonstrative back-references, explicit comparisons, and nested wh-questions.

**The feature set is identical to Iteration 6.** The add_feature_prefix.py docstring says explicitly: "Feature extraction reuses the spaCy + regex logic from classifier/postprocess/clf2_feature_probe.py" ([line 9](Adaptive-RAG/classifier/data_utils/add_feature_prefix.py#L9)).

---

## 4. Training Parameters

All parameters from run_large_train_feat_single_vs_multi.sh:

| Parameter | Value | Standard Clf2 (Iteration 1) | Difference? |
|---|---|---|---|
| `--model_name_or_path` | `t5-large` | `t5-large` | None |
| `--learning_rate` | `3e-5` | `3e-5` | None |
| `--per_device_train_batch_size` | `32` | `32` | None |
| `--max_seq_length` | `384` | `384` | None |
| `--doc_stride` | `128` | `128` | None |
| `--labels` | `B C` | `B C` | None |
| `--num_train_epochs` | `15 20 25 30 35` | xl/xxl: `15 20 25 30 35`; **gpt: `35 40`** | **YES — see below** |
| Optimizer | AdamW ([run_classifier.py L636](Adaptive-RAG/classifier/run_classifier.py#L636)) | Same | None |
| `--weight_decay` | `0.0` (default) | Same | None |
| Scheduler | `linear`, 0 warmup steps (defaults) | Same | None |
| Loss | Standard cross-entropy (`outputs.loss`) | Same | None |
| Early stopping | None | None | None |
| `--seed` | Not passed (defaults to `None`) | Same | None |

**Epoch range difference for GPT:** The standard GPT Clf2 ([run_large_train_gpt_single_vs_multi.sh line 11](Adaptive-RAG/classifier/run/run_large_train_gpt_single_vs_multi.sh#L11)) uses `for EPOCH in 35 40`, but the feature-augmented script uses `for EPOCH in 15 20 25 30 35` for **all** models including GPT. This is a **notable deviation**: GPT will be evaluated at epochs 15–35 instead of 35–40. This means shorter training schedules are tested that weren't tested for the standard GPT Clf2, while the 40-epoch checkpoint that was the standard GPT best is never produced.

---

## 5. Feature Preprocessing

Features are converted to **discretized text tokens** — not normalized, not continuous, not embedded. The formatting is defined in add_feature_prefix.py and the batch version at add_feature_prefix.py:

```python
f"[LEN:{token_len}] [ENT:{entity_count}] [BRIDGE:{bridge_flag}] "
```

- `LEN` is a raw integer (e.g., `7`, `14`, `23`) — no binning, no normalisation
- `ENT` is a raw integer (e.g., `0`, `1`, `4`)
- `BRIDGE` is binary (`0` or `1`)

The prefix is prepended directly to the question string before it's written to JSON. At training/inference time, the standard `preprocess_features_function` in utils.py strips leading whitespace and tokenises the whole string:

```python
examples[question_column] = ['{}'.format(q.strip()) for q in examples[question_column]]
```

The T5 tokeniser then processes `[LEN:14] [ENT:2] [BRIDGE:1] What country was...` as a standard input sequence. The square-bracket tags are tokenised as subword tokens like `▁[`, `LEN`, `:`, `14`, `]` etc.

---

## 6. No Residual Changes from Iterations 2–4

Confirmed. The shell script does **not** pass `--use_focal_loss`, `--focal_gamma`, or `--focal_alpha` ([lines 39–52](Adaptive-RAG/classifier/run/run_large_train_feat_single_vs_multi.sh#L39-L52)). Since `--use_focal_loss` is an `action="store_true"` flag ([run_classifier.py line 401](Adaptive-RAG/classifier/run_classifier.py#L401)), it defaults to `False`. The training path hits the standard `if args.do_train and not args.use_focal_loss:` branch at run_classifier.py, using the model's built-in cross-entropy loss (`outputs.loss` at run_classifier.py).

No undersampling is applied — the training data is the full `binary_silver_feat_single_vs_multi/train.json` (the same merged binary+silver set as standard Clf2, just feature-prefixed). No class weighting of any kind.

---

## 7. Gate 1

The training script **does not train or load any Gate 1 checkpoint** — it only trains Gate 2. At evaluation time, you choose which Clf1 to pair it with by passing the appropriate file to `predict_complexity_split_classifiers.py --no_ret_vs_ret_file` or predict_complexity_agreement.py.

The README documents the intended usage as pairing with the **agreement gate** from Iteration 2a (see README.md and DOCUMENTATION_IT6.md). The agreement gate is a training-free heuristic (nor_qa/oner_qa answer match) implemented in predict_complexity_agreement.py — it loads no checkpoint, so there's no checkpoint path to verify. It can also be paired with any trained Clf1 checkpoint via predict_complexity_split_classifiers.py.

---

## 8. Evaluation Setup

Evaluation uses the same pipeline as all prior iterations:

- **Routing scripts:** predict_complexity_split_classifiers.py or predict_complexity_agreement.py — both accept any Clf2's `dict_id_pred_results.json` via the `--single_vs_multi_file` / `--clf2_pred_file` arg
- **Final evaluation:** `evaluate_final_acc.py --pred_path <routed_predictions_dir>`
- **Datasets:** All 6 — musique, hotpotqa, 2wikimultihopqa, nq, trivia, squad (hardcoded in routing scripts at predict_complexity_split_classifiers.py)
- **Backbone models:** The script takes model name as `$1` and supports all 3: `flan_t5_xl`, `flan_t5_xxl`, `gpt` ([line 12–14](Adaptive-RAG/classifier/run/run_large_train_feat_single_vs_multi.sh#L12-L14))
- **QA predictions:** Same pre-computed `predictions/test/` files (BM25 counts identical per model family)
- **F1 computation:** `DropAnswerEmAndF1` in evaluate_final_acc.py, identical to all iterations

---

## 9. Output Artifacts

Checkpoints, validation results, and predictions are written to a **separate directory tree** from all prior iterations:

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/{model}/
├── single_vs_multi/epoch/{N}/{DATE}/       ← Standard Clf2 (Iterations 1+)
├── feat_single_vs_multi/epoch/{N}/feat/    ← Iteration 7 ✓ SEPARATE
│   ├── model.safetensors, config.json      (trained checkpoint)
│   ├── valid/
│   │   ├── dict_id_pred_results.json
│   │   ├── final_eval_results.json
│   │   └── final_eval_results_perClass.json
│   └── predict/
│       ├── dict_id_pred_results.json
│       └── final_eval_results.json
```

The distinguishing path components are `feat_single_vs_multi/` (vs `single_vs_multi/`) and the static date tag `DATE=feat` ([line 23](Adaptive-RAG/classifier/run/run_large_train_feat_single_vs_multi.sh#L23)) instead of a timestamp. No risk of overwriting prior iteration outputs.

Feature-prefixed data files are similarly separated: `binary_silver_feat_single_vs_multi/` and `silver_feat_single_vs_multi/` directories sit alongside (not inside) the original data directories.

---

## 10. Suspicious Items

1. **`DATE=feat` is static, not timestamped** ([line 23](Adaptive-RAG/classifier/run/run_large_train_feat_single_vs_multi.sh#L23)). Every other training script uses `DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)`. With a static tag, re-running the script **silently overwrites previous results** for the same model and epoch. All other iteration scripts are protected by unique timestamps.

2. **GPT epoch range mismatch.** The feature script uses `for EPOCH in 15 20 25 30 35` for all models ([line 30](Adaptive-RAG/classifier/run/run_large_train_feat_single_vs_multi.sh#L30)), but the standard GPT Clf2 uses `35 40`. This means:
   - GPT trains at epochs 15, 20, 25 that were never tested in the standard GPT Clf2
   - GPT epoch 40 (the standard best) is **never trained**
   - This could either be intentional (to explore earlier convergence with feature enrichment) or an oversight from copying the xl/xxl template

3. **Feature prefix token overhead vs. `max_seq_length=384`.** A typical prefix like `[LEN:14] [ENT:2] [BRIDGE:1] ` adds approximately **15–20 subword tokens** after T5 tokenisation (the bracket/colon/number tokens). With `max_seq_length=384` ([line 46](Adaptive-RAG/classifier/run/run_large_train_feat_single_vs_multi.sh#L46)), this is unlikely to cause truncation for typical questions (most are well under 100 tokens). However, the longest questions in multi-hop datasets can approach 50+ words, and with the prefix overhead, questions near the boundary could lose their final tokens. This is a minor risk — 384 tokens provides ample headroom for question-length inputs.

4. **No data leakage in features.** All three features (token length, entity count, bridge flag) are computed solely from the **question text** using `question.split()`, spaCy NER, and regex. They do not use any information from the QA answers, retrieval results, or ground-truth labels. They are available at routing time. This is clean.

5. **Bridge flag could produce near-constant values for some subsets.** For single-hop datasets (NQ, TriviaQA, SQuAD), most questions are simple factoid queries that won't match the bridging patterns, so `BRIDGE` will be `0` for nearly all of them. Conversely, many multi-hop questions will match. This is actually the desired signal, not a bug — but it means the bridge flag provides redundant information with the dataset-level heuristic already encoded in the binary labels (single-hop → B, multi-hop → C).

6. **Feature values are not normalised or binned.** `LEN` ranges from ~3 to ~50+ as a raw integer. The T5 tokeniser will encode `[LEN:7]` and `[LEN:42]` as different subword sequences (`7` vs `4`, `2`). Since T5 processes these as text tokens (not numeric values), the model must learn the semantic meaning of each number independently. Binning (e.g., `[LEN:short]`, `[LEN:medium]`, `[LEN:long]`) could improve generalisation with limited training data by reducing the vocabulary of prefix tokens the model must learn.

7. **`feat_predict.json` is shared across all models** ([add_feature_prefix.py lines 112–114](Adaptive-RAG/classifier/data_utils/add_feature_prefix.py#L112-L114)), since predict.json is model-independent. This is correct — the features depend only on the question text, not on any model-specific labels.

8. **Duplicated feature logic between two files.** The `BRIDGE_PATTERNS` list and feature extraction logic are copy-pasted between clf2_feature_probe.py ([lines 55–67](Adaptive-RAG/classifier/postprocess/clf2_feature_probe.py#L55-L67)) and add_feature_prefix.py ([lines 31–39](Adaptive-RAG/classifier/data_utils/add_feature_prefix.py#L31-L39)). If one is updated and the other isn't, the probe results and the actual training features would diverge silently.