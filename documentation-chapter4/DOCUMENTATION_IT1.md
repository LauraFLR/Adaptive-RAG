# Iteration 1 — Baseline Cascaded Binary Classifier

> **Design Science Research Artifact:** Replace Adaptive-RAG's single 3-class
> complexity classifier (A / B / C) with two sequential binary classifiers
> (Clf1: A vs R, then Clf2: B vs C).

---

## 1. Files Involved

| File | Role |
|---|---|
| `classifier/run_classifier.py` | Core training and evaluation entry point. Parses args, loads data, tokenises, trains (standard CE or focal-loss path), runs greedy-decode inference, computes accuracy, and writes results JSON files. |
| `classifier/utils.py` | Helper library: `load_model()` (config + tokenizer + `AutoModelForSeq2SeqLM`), `preprocess_dataset()` / `preprocess_features_function()` (tokenisation), `post_processing_function()` (decode predictions), `prepare_scheduler()` (LR scheduler + step math), `calculate_accuracy()`, `calculate_accuracy_perClass()`. |
| `classifier/run/run_large_train_xl_no_ret_vs_ret.sh` | Shell launcher — Clf1 (A vs R) with `flan_t5_xl` silver labels. |
| `classifier/run/run_large_train_xxl_no_ret_vs_ret.sh` | Shell launcher — Clf1 (A vs R) with `flan_t5_xxl` silver labels. |
| `classifier/run/run_large_train_gpt_no_ret_vs_ret.sh` | Shell launcher — Clf1 (A vs R) with `gpt` silver labels. |
| `classifier/run/run_large_train_xl_single_vs_multi.sh` | Shell launcher — Clf2 (B vs C) with `flan_t5_xl` silver labels. |
| `classifier/run/run_large_train_xxl_single_vs_multi.sh` | Shell launcher — Clf2 (B vs C) with `flan_t5_xxl` silver labels. |
| `classifier/run/run_large_train_gpt_single_vs_multi.sh` | Shell launcher — Clf2 (B vs C) with `gpt` silver labels. |
| `classifier/postprocess/predict_complexity_split_classifiers.py` | Two-stage cascade routing script: merges Clf1 + Clf2 predictions → A/B/C labels → routes each test question to the correct QA strategy's pre-computed answer. |
| `classifier/postprocess/predict_complexity_on_classification_results.py` | Original single-classifier routing script (used for the 3-class baseline; not used in IT1 split pipeline). |
| `classifier/postprocess/postprocess_utils.py` | Shared helpers: `load_json()`, `save_json()`, `save_prediction_with_classified_label()`. |
| `evaluate_final_acc.py` | End-to-end QA evaluation: computes EM, F1, accuracy per dataset using official evaluation scripts for multi-hop datasets and `SquadAnswerEmF1Metric` for single-hop. |
| `classifier/data/.../silver/no_retrieval_vs_retrieval/{train,valid}.json` | Clf1 silver-labelled data (A / R). |
| `classifier/data/.../binary_silver_single_vs_multi/train.json` | Clf2 training data (B / C) — silver + inductive-bias binary labels merged. |
| `classifier/data/.../silver/single_vs_multi/valid.json` | Clf2 validation data (B / C) — silver labels only. |
| `classifier/data/.../predict.json` | Unlabelled test set (3 000 questions, 500 per dataset). |

---

## 2. Model Architecture

### Base model

| Property | Value |
|---|---|
| Model name | `t5-large` |
| Architecture | Encoder–decoder (seq2seq) transformer |
| Parameter count | ~770 M |
| HuggingFace class | `AutoModelForSeq2SeqLM` [run_classifier.py L47, utils.py L48] |
| Loaded via | `utils.load_model()` → `AutoConfig.from_pretrained()` + `AutoTokenizer.from_pretrained()` + `AutoModelForSeq2SeqLM.from_pretrained()` [utils.py L20–52] |

### Classification mechanism

Classification is implemented as **generative decoding** constrained to label tokens, not as a traditional classification head:

1. **Training:** The model is fine-tuned as a seq2seq task where the input is the question text and the target is a single label token (e.g. `"A"`, `"R"`, `"B"`, `"C"`). Standard teacher-forced cross-entropy loss on the decoder output is used (unless `--use_focal_loss` is passed, which is **not** the case in Iteration 1). [run_classifier.py L739–808]

2. **Inference:** `model.generate()` is called with `return_dict_in_generate=True, output_scores=True`. The raw logits at the first decoder position (`scores[0]`) are extracted. For each label in `args.labels`, the score at that label's token ID is gathered. Softmax is applied across those label-specific columns, and `argmax` selects the predicted class. [run_classifier.py L862–880]

   ```
   scores = model.generate(...).scores[0]          # (batch, vocab_size)
   probs = softmax(stack([scores[:, tok_id(label)] for label in labels]), dim=0)
   pred = argmax(probs, dim=0)
   ```

### Label sets

| Classifier | Labels | Meaning |
|---|---|---|
| Clf1 (Gate 1) | `A R` | A = no retrieval needed; R = retrieval needed (merges original B + C) |
| Clf2 (Gate 2) | `B C` | B = single-step retrieval; C = multi-step retrieval |

Labels are passed via `--labels A R` or `--labels B C` in each shell script. The `label_to_option` mapping is built dynamically at [run_classifier.py L490]: `label_to_option = {i: label for i, label in enumerate(args.labels)}`.

---

## 3. Training Parameters

### 3.1 Parameters table

| Parameter | Value | Source | Default or explicit? |
|---|---|---|---|
| **Learning rate** | `3e-5` | Shell scripts `--learning_rate 3e-5` | Explicit (argparse default is `5e-5` [run_classifier.py L369]) |
| **Per-device train batch size** | `32` | Shell scripts `--per_device_train_batch_size 32` | Explicit (default `8` [run_classifier.py L358]) |
| **Per-device eval batch size** | `100` | Shell scripts `--per_device_eval_batch_size 100` | Explicit (default `8` [run_classifier.py L363]) |
| **Max sequence length** | `384` | Shell scripts `--max_seq_length 384` | Explicit (matches default [run_classifier.py L201]) |
| **Doc stride** | `128` | Shell scripts `--doc_stride 128` | Explicit (matches default [run_classifier.py L463]) |
| **Weight decay** | `0.0` | argparse default [run_classifier.py L371] | Default — never overridden by any IT1 shell script |
| **Gradient accumulation steps** | `1` | argparse default [run_classifier.py L378] | Default — never overridden |
| **Num warmup steps** | `0` | argparse default [run_classifier.py L390] | Default — never overridden |
| **Optimizer** | `AdamW` | Hardcoded [run_classifier.py L643] | Hardcoded — `torch.optim.AdamW` |
| **LR scheduler** | `linear` | argparse default [run_classifier.py L384] | Default — `SchedulerType("linear")` via `get_scheduler()` [utils.py L213] |
| **Seed** | `42` | Shell scripts `--seed 42` | Explicit (default `None` [run_classifier.py L392]) |
| **Max train steps** | `None` | argparse default [run_classifier.py L374] | Default — computed from `num_train_epochs × steps_per_epoch` in `prepare_scheduler()` [utils.py L207–210] |
| **Epochs (XL / XXL)** | `15, 20, 25, 30, 35` | Shell scripts `for EPOCH in 15 20 25 30 35` | Explicit |
| **Epochs (GPT)** | `35, 40` | Shell scripts `for EPOCH in 35 40` | Explicit |
| **Early stopping** | **None** | Not implemented anywhere | N/A |
| **Checkpointing** | End-of-training only | `--checkpointing_steps` never passed; model saved after all epochs via `unwrapped_model.save_pretrained()` [run_classifier.py L832–840] | Default (no intermediate checkpoints) |
| **Max answer length** | `30` | argparse default [run_classifier.py L261] | Default — never overridden |
| **Pad to max length** | `False` | argparse default (store_true, not passed) [run_classifier.py L306] | Default — dynamic padding used |
| **Ignore pad token for loss** | `True` | argparse default [run_classifier.py L193] | Default |
| **Mixed precision** | None (fp32) | Accelerator default | Default — no `--mixed_precision` passed |

### 3.2 Loss function

**Standard path (Iteration 1):** The IT1 shell scripts do **not** pass `--use_focal_loss`. Training uses the default seq2seq cross-entropy computed by the HuggingFace `AutoModelForSeq2SeqLM` forward pass:

```python
outputs = model(**batch)
loss = outputs.loss        # [run_classifier.py L785]
```

This is the standard `CrossEntropyLoss` on the decoder logits, applied by T5's internal `lm_head` [run_classifier.py L785–788].

**Focal-loss path (NOT active in IT1):** Enabled only when `--use_focal_loss` is passed. Uses `FocalLossTrainer` (subclass of HF `Trainer`) with a `FocalLoss` module [run_classifier.py L64–142]. In IT1, this code path is dead.

### 3.3 Epoch sweep behaviour

Each epoch value in the `for EPOCH in ...` loop launches a **completely fresh training run from the pre-trained `t5-large` weights**. The script does not continue from a previous epoch's checkpoint. For example, epoch 25 trains from scratch for 25 epochs, not 5 additional epochs on top of epoch 20. This means:

- 5 independent models are trained per script (XL/XXL) or 2 per script (GPT).
- Total GPU time scales as `sum(epochs)`, not `max(epochs)`.

---

## 4. Data Pipeline

### 4.1 Clf1 — No-retrieval (A) vs Retrieval (R)

#### flan_t5_xl

| Property | Training | Validation |
|---|---|---|
| **File** | `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/no_retrieval_vs_retrieval/train.json` | `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/silver/no_retrieval_vs_retrieval/valid.json` |
| **Size** | 1 292 samples | 1 350 samples |
| **A count** | 424 (32.8 %) | 439 (32.5 %) |
| **R count** | 868 (67.2 %) | 911 (67.5 %) |
| **Inductive-bias binary labels?** | No — silver labels only | No — silver labels only |

#### flan_t5_xxl

| Property | Training | Validation |
|---|---|---|
| **File** | `...flan_t5_xxl/silver/no_retrieval_vs_retrieval/train.json` | `...flan_t5_xxl/silver/no_retrieval_vs_retrieval/valid.json` |
| **Size** | 1 409 samples | 1 415 samples |
| **A count** | 511 (36.3 %) | 519 (36.7 %) |
| **R count** | 898 (63.7 %) | 896 (63.3 %) |
| **Inductive-bias binary labels?** | No — silver labels only | No — silver labels only |

#### gpt

| Property | Training | Validation |
|---|---|---|
| **File** | `...gpt/silver/no_retrieval_vs_retrieval/train.json` | `...gpt/silver/no_retrieval_vs_retrieval/valid.json` |
| **Size** | 1 417 samples | 1 431 samples |
| **A count** | 1 013 (71.5 %) | 1 038 (72.5 %) |
| **R count** | 404 (28.5 %) | 393 (27.5 %) |
| **Inductive-bias binary labels?** | No — silver labels only | No — silver labels only |

**Note:** GPT Clf1 has an **inverted** class imbalance compared to XL/XXL: A is the majority class (~71 %) rather than R.

### 4.2 Clf2 — Single-step (B) vs Multi-step (C)

#### flan_t5_xl

| Property | Training | Validation |
|---|---|---|
| **File** | `...flan_t5_xl/binary_silver_single_vs_multi/train.json` | `...flan_t5_xl/silver/single_vs_multi/valid.json` |
| **Size** | 3 268 samples | 911 samples |
| **B count** | 1 871 (57.3 %) | 691 (75.8 %) |
| **C count** | 1 397 (42.7 %) | 220 (24.2 %) |
| **Inductive-bias binary labels?** | **Yes** — `binary_silver` prefix means silver + inductive-bias labels merged | No — silver only |

#### flan_t5_xxl

| Property | Training | Validation |
|---|---|---|
| **File** | `...flan_t5_xxl/binary_silver_single_vs_multi/train.json` | `...flan_t5_xxl/silver/single_vs_multi/valid.json` |
| **Size** | 3 298 samples | 896 samples |
| **B count** | 1 903 (57.7 %) | 701 (78.2 %) |
| **C count** | 1 395 (42.3 %) | 195 (21.8 %) |
| **Inductive-bias binary labels?** | **Yes** | No |

#### gpt

| Property | Training | Validation |
|---|---|---|
| **File** | `...gpt/binary_silver_single_vs_multi/train.json` | `...gpt/silver/single_vs_multi/valid.json` |
| **Size** | 2 804 samples | 393 samples |
| **B count** | 1 475 (52.6 %) | 272 (69.2 %) |
| **C count** | 1 329 (47.4 %) | 121 (30.8 %) |
| **Inductive-bias binary labels?** | **Yes** | No |

### 4.3 Predict set (unlabelled test)

| Property | Value |
|---|---|
| **File** | `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json` |
| **Size** | 3 000 samples |
| **Datasets** | 500 each: musique, hotpotqa, 2wikimultihopqa, nq, trivia, squad |
| **Answer field** | Empty string (`""`) — labels absent |
| **Columns** | `answer`, `dataset_name`, `id`, `question`, `total_answer` |

### 4.4 Tokenisation / preprocessing pipeline

Step-by-step, for both training and evaluation:

1. **File loading** — `load_dataset("json", data_files={split: path})` creates an HF `DatasetDict`. The shell script passes `--train_file` and/or `--validation_file`; `run_classifier.py` maps them into `data_files` dict keyed by `"train"` / `"validation"`. [run_classifier.py L544–555]

2. **Column identification** — `preprocess_dataset()` reads `question_column` and `answer_column` from args, validates they exist in the dataset columns. [utils.py L57–76]

3. **Question whitespace strip** — Leading/trailing whitespace is stripped from every question string. [utils.py L89]

4. **Input tokenisation** — `tokenizer(examples[question_column], truncation=True, max_length=min(384, model_max_length), stride=128, return_overflowing_tokens=True, return_offsets_mapping=True, padding=False)`. Dynamic padding is used (no `--pad_to_max_length`). Overflow stride is configured but rarely triggers on short questions. [utils.py L95–103]

5. **Target tokenisation** — `tokenizer(text_target=targets, max_length=30, padding=False, truncation=True)`. For classification, targets are single label tokens (`"A"`, `"R"`, `"B"`, `"C"`), so max_length 30 is never hit. [utils.py L107]

6. **Pad-token masking** — If `pad_to_max_length` and `ignore_pad_token_for_loss`, pad tokens in labels are replaced with `-100`. In practice, this branch is not entered in IT1 (dynamic padding is used). [utils.py L111–114]

7. **Overflow sample mapping** — `overflow_to_sample_mapping` maps tokenised features back to original examples. `example_id` and `labels` are aligned to the mapped features. [utils.py L118–127]

8. **Dataset `.map()` call** — `train_dataset.map(preprocess_features_function, fn_kwargs={...}, batched=True, remove_columns=...)`. [run_classifier.py L573–581]

9. **DataLoader creation** — `DataLoader(dataset, shuffle=True, collate_fn=DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=-100), batch_size=32)`. The collator pads each batch to the longest sequence in that batch. [run_classifier.py L614–619]

---

## 5. Imbalance Handling

**Iteration 1 uses NO imbalance handling.** Specifically:

| Technique | Used in IT1? | Evidence |
|---|---|---|
| Focal loss (`--use_focal_loss`) | **No** | Not passed in any of the 6 IT1 shell scripts |
| Class-weighted cross-entropy | **No** | `--auto_class_weights` not passed; `--focal_alpha` not passed |
| Oversampling | **No** | No oversampling code exists in `run_classifier.py` |
| Undersampling | **No** | `make_no_ret_vs_ret_undersampled.py` exists but is not called by IT1 scripts |
| Cost-sensitive loss | **No** | Only available via focal-loss path |
| Stratified batching | **No** | `DataLoader(shuffle=True)` does uniform random shuffling [run_classifier.py L617] |

**Class imbalance present in IT1 data:**

| Classifier | Model | Majority class | Ratio |
|---|---|---|---|
| Clf1 | flan_t5_xl | R | 67:33 |
| Clf1 | flan_t5_xxl | R | 64:36 |
| Clf1 | gpt | A | 72:28 |
| Clf2 | flan_t5_xl | B | 57:43 (train), 76:24 (valid) |
| Clf2 | flan_t5_xxl | B | 58:42 (train), 78:22 (valid) |
| Clf2 | gpt | B | 53:47 (train), 69:31 (valid) |

The Clf2 validation sets show notably more skewed B/C ratios than the training sets because the training sets include inductive-bias binary labels that add more balanced C samples, while validation uses silver-only labels.

---

## 6. Evaluation Setup

### 6.1 Per-epoch validation procedure

Within each epoch iteration of the shell script `for EPOCH in ...`:

1. **Train** → writes the model to `${TRAIN_OUTPUT_DIR}` (the output directory). [shell scripts, train block]
2. **Validate** → loads the just-trained model from `${TRAIN_OUTPUT_DIR}`, runs inference on the labelled validation file, writes results to `${TRAIN_OUTPUT_DIR}/valid/`. [shell scripts, valid block]
3. **Predict** → loads the same model, runs inference on `predict.json` (unlabelled test), writes results to `${TRAIN_OUTPUT_DIR}/predict/`. [shell scripts, predict block]

Each step runs `python run_classifier.py` as a separate process. Validation and prediction both use `--do_eval`.

### 6.2 What is saved per step

| Step | Files written | Directory |
|---|---|---|
| Train | Model weights (`pytorch_model.bin`, `config.json`, `tokenizer*`, `spiece.model`), `logs.log` | `${TRAIN_OUTPUT_DIR}/` |
| Valid | `dict_id_pred_results.json` (per-question predictions), `final_eval_results.json` (overall accuracy), `final_eval_results_perClass.json` (per-class accuracy), `logs.log` | `${TRAIN_OUTPUT_DIR}/valid/` |
| Predict | Same three JSON files + `logs.log` | `${TRAIN_OUTPUT_DIR}/predict/` |

### 6.3 Accuracy computation

- **Overall accuracy** — `calculate_accuracy(gold_answers, predictions)` [utils.py L237–242]: simple exact-match ratio `(# correct) / (# total) × 100`.
- **Per-class accuracy** — `calculate_accuracy_perClass(gold_answers, predictions, labels)` [utils.py L244–254]: for each label, computes `correct[l] / gold_num[l] × 100` and reports `pred_num` and `gold_num` counts.
- Both are called at [run_classifier.py L907–912, L915–922].

### 6.4 End-to-end QA evaluation

**Step 1: Cascade routing** — `predict_complexity_split_classifiers.py` [classifier/postprocess/]:

1. Loads Clf1 predictions from `no_ret_vs_ret/.../predict/dict_id_pred_results.json`.
2. Loads Clf2 predictions from `single_vs_multi/.../predict/dict_id_pred_results.json`.
3. Merges: if Clf1 predicted `A` → final label `A`; if Clf1 predicted `R` → final label = Clf2's prediction (`B` or `C`). [predict_complexity_split_classifiers.py L41–50]
4. For each question, loads the pre-computed QA answer from the strategy corresponding to its final label:
   - `A` → `nor_qa` (no retrieval) answer
   - `B` → `oner_qa` (one-step retrieval) answer
   - `C` → `ircot_qa` (iterative retrieval chain-of-thought) answer
5. Writes per-dataset prediction files to the output directory.

**Step 2: QA metric evaluation** — `evaluate_final_acc.py`:

- For **single-hop** datasets (nq, trivia, squad): uses `SquadAnswerEmF1Metric` + custom `calculate_acc` (normalised substring match). [evaluate_final_acc.py L90–108]
- For **multi-hop** datasets (musique, hotpotqa, 2wikimultihopqa): calls the official evaluation scripts via subprocess:
  - `hotpot_evaluate_v1.py` [evaluate_final_acc.py L128–179]
  - `2wikimultihop_evaluate_v1.1.py` [evaluate_final_acc.py L181–232]
  - `evaluate_v1.0.py` (MuSiQue) [evaluate_final_acc.py L234–279]
- Reports: **EM** (exact match), **F1**, **accuracy**, **precision**, **recall**, **count** per dataset.
- Results saved to `eval_metic_result_acc.json` per dataset subdirectory.

### 6.5 BM25 retrieval counts per model

| Model | `oner_qa` BM25 count | `ircot_qa` BM25 count | Source |
|---|---|---|---|
| `flan_t5_xl` | 15 | 6 | [predict_complexity_split_classifiers.py L17–18] |
| `flan_t5_xxl` | 15 | 6 | [predict_complexity_split_classifiers.py L17–18] |
| `gpt` | 6 | 3 | [predict_complexity_split_classifiers.py L17–18] |

### 6.6 Datasets evaluated

| Dataset | Type | Evaluation method |
|---|---|---|
| musique | Multi-hop | Official MuSiQue evaluator |
| hotpotqa | Multi-hop | Official HotpotQA evaluator |
| 2wikimultihopqa | Multi-hop | Official 2WikiMultiHop evaluator |
| nq (Natural Questions) | Single-hop | SquadAnswerEmF1Metric |
| trivia (TriviaQA) | Single-hop | SquadAnswerEmF1Metric |
| squad (SQuAD) | Single-hop | SquadAnswerEmF1Metric |

---

## 7. Output Artifacts

### 7.1 Directory tree (Clf1 example)

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/
  {model}/                                          # flan_t5_xl, flan_t5_xxl, or gpt
    no_ret_vs_ret/
      epoch/
        {epoch}/                                    # 15, 20, 25, 30, 35 (or 35, 40 for gpt)
          {YYYY_MM_DD}/{HH_MM_SS}/                  # DATE timestamp
            config.json
            generation_config.json
            pytorch_model.bin                       # (or model.safetensors)
            spiece.model
            special_tokens_map.json
            tokenizer.json
            tokenizer_config.json
            logs.log
            valid/
              dict_id_pred_results.json
              final_eval_results.json
              final_eval_results_perClass.json
              logs.log
            predict/
              dict_id_pred_results.json
              final_eval_results.json               # accuracy is meaningless (labels are empty)
              final_eval_results_perClass.json
              logs.log
```

### 7.2 Directory tree (Clf2 example)

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/
  {model}/
    single_vs_multi/
      epoch/
        {epoch}/
          {YYYY_MM_DD}/{HH_MM_SS}/
            [same structure as Clf1]
```

### 7.3 Routed predictions (after cascade)

```
predictions/classifier/t5-large/
  {model}/
    split/                                          # or whatever --output_path is
      {routing_run_name}/
        musique/
          musique.json                              # qid → answer string
          musique_option.json                       # qid → {prediction, option, stepNum}
        hotpotqa/
          hotpotqa.json
          hotpotqa_option.json
        2wikimultihopqa/
          ...
        nq/
          ...
        trivia/
          ...
        squad/
          ...
```

### 7.4 Evaluation results (after `evaluate_final_acc.py`)

Each dataset subdirectory within the routed predictions gets:

```
{dataset}/
  eval_metic_result_acc.json                        # {"f1": ..., "em": ..., "acc": ..., "count": ...}
```

---

## 8. Anything Suspicious / Premature Termination Risks

### 8.1 Model overwriting within epoch loop

**File:** `run_classifier.py` L832–840  
**Issue:** At the end of every training epoch, `unwrapped_model.save_pretrained(args.output_dir)` overwrites the model in the output directory. This means only the **final epoch's weights** survive — there is no best-epoch selection or epoch-indexed saving. If training degrades in later epochs (overfitting), the saved model may be suboptimal.  
**Impact:** Medium. No early stopping exists to protect against this.

```python
if args.output_dir is not None:
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.output_dir, ...
    )
```

### 8.2 Silent training cap via `max_train_steps`

**File:** `run_classifier.py` L805–806, `utils.py` L207–222  
**Issue:** `prepare_scheduler()` computes `max_train_steps = num_train_epochs × num_update_steps_per_epoch`. The training loop then breaks at `if completed_steps >= args.max_train_steps: break` [run_classifier.py L805–806]. Due to integer rounding in `math.ceil(len(dataloader) / gradient_accumulation_steps)`, the actual number of steps may differ from the expected count by ±1 step per epoch. Not a practical problem given batch sizes involved, but worth noting.  
**Impact:** Negligible.

### 8.3 Each epoch value trains from scratch

**File:** All 6 IT1 shell scripts  
**Issue:** The `for EPOCH in 15 20 25 30 35` loop calls `--model_name_or_path t5-large` for every iteration. This means each epoch budget is a **fresh training run** from pre-trained T5-Large, not incremental fine-tuning. Training epoch 35 does not start from the epoch-30 checkpoint — it trains from scratch for 35 full epochs.  
**Impact:** Very high GPU cost. For XL/XXL, total training is `15+20+25+30+35 = 125` epochs (×2 classifiers = 250 epochs). For GPT: `35+40 = 75` epochs (×2 = 150 epochs).

### 8.4 No early stopping

**File:** `run_classifier.py` — no early stopping logic exists anywhere  
**Issue:** Training runs for the full `num_train_epochs` with no validation-based early stopping. Combined with the model-overwriting issue (§8.1), the final saved model is always the last-epoch model regardless of performance.  
**Impact:** Medium. Overfitting is likely at higher epoch counts (e.g. 35 epochs on 1 292 samples), but the epoch sweep partially compensates by trying multiple budgets.

### 8.5 Missing seed in original 3-class scripts (not IT1, but contextual)

**File:** `run_large_train_xl.sh`, `run_large_train_xxl.sh`, `run_large_train_gpt.sh`  
**Issue:** The original 3-class scripts do not pass `--seed`. The IT1 split-classifier scripts **do** pass `--seed 42`.  
**Impact on IT1:** None — IT1 scripts are reproducible.

### 8.6 predict.json accuracy scores are meaningless

**File:** `run_classifier.py` L905–922  
**Issue:** When running `--do_eval` on `predict.json`, the code computes `calculate_accuracy(gold_answers, predictions)` where `gold_answers` are all empty strings (`""`). Since no prediction will be `""`, accuracy will always be 0 %. The code writes this to `final_eval_results.json` in the predict directory, which could be mistaken for actual evaluation results.  
**Impact:** Confusion risk. The predict-directory accuracy files should be ignored.

### 8.7 `TRANSFORMERS_CACHE` set relative to CWD

**File:** `run_classifier.py` L27  
**Issue:** `os.environ['TRANSFORMERS_CACHE'] = os.path.dirname(os.getcwd()) + '/cache'`. This resolves to `dirname(classifier/) + '/cache'` = `Adaptive-RAG/cache` when run from the `classifier/` directory as instructed. If run from a different directory, the cache location changes.  
**Impact:** Low. Correct when following the documented working-directory convention.

### 8.8 GPU hardcoding

**File:** All 6 IT1 shell scripts  
**Issue:** GPU is hardcoded to `GPU=0` via `CUDA_VISIBLE_DEVICES=${GPU}`. Unlike the focal-loss scripts that use `GPU=${GPU:-0}` (allowing environment override), the IT1 standard scripts set `GPU=0` as a plain assignment — which cannot be overridden with `GPU=1 bash run/...`.  
**Impact:** Low operational inconvenience. Must edit the script to use a different GPU.

### 8.9 Clf2 training/validation distribution mismatch

**File:** Clf2 shell scripts + data files  
**Issue:** Clf2 training uses `binary_silver_single_vs_multi/train.json` (silver + inductive-bias labels, ~3 268 samples for XL) but validation uses `silver/single_vs_multi/valid.json` (silver-only, ~911 samples for XL). The class distribution differs substantially:
- XL train: B=57 %, C=43 %
- XL valid: B=76 %, C=24 %

The inductive-bias binary labels in the training set add approximately 2 400 extra samples (mostly balanced B/C), which shifts the training distribution relative to the validation distribution.  
**Impact:** Medium. Training-set statistics may not represent the evaluation-time distribution, and accuracy numbers on validation may not track training performance faithfully.

### 8.10 `DataLoader` shuffle without worker seeding

**File:** `run_classifier.py` L617  
**Issue:** `DataLoader(train_dataset_for_model, shuffle=True, ...)` — while `set_seed(42)` is called globally at [run_classifier.py L538], PyTorch DataLoader worker processes may not be seeded unless `worker_init_fn` is provided. In this case, `num_workers` defaults to 0 (main-process loading), so this is not a practical issue.  
**Impact:** Negligible.

### 8.11 Routing script loads QA prediction file once per question

**File:** `postprocess_utils.py` L32 (`save_prediction_with_classified_label`)  
**Issue:** For every question where `predicted_option == 'C'`, the function calls `load_json(stepNum_result_file)` inside the loop — re-reading the file on every iteration. Similarly, `load_json(dataName_to_multi_one_zero_file[dataset_name][predicted_option])` is called per question.  
**Impact:** Performance only — `predict_complexity_split_classifiers.py` avoids this by loading files upfront. `predict_complexity_on_classification_results.py` (the older single-classifier script) still has this issue but is not used in the split pipeline.

### 8.12 `predict_complexity_on_classification_results.py` has a hardcoded classification result path

**File:** `predict_complexity_on_classification_results.py` L11  
**Issue:** `classification_result_file = './classifier/outputs/.../epoch/25/2024_04_19/01_53_50/predict/dict_id_pred_results.json'` — a fully hardcoded path to one specific run. This script is the **old** single-classifier router and is not used in the IT1 split pipeline (which uses `predict_complexity_split_classifiers.py` with CLI arguments instead).  
**Impact on IT1:** None — this file is unused in the split pipeline.

### 8.13 No validation-based model selection across epoch sweep

**File:** All 6 IT1 shell scripts  
**Issue:** The scripts train 5 (or 2) independent models at different epoch budgets but perform no automated comparison to select the best epoch. The user must manually inspect `final_eval_results.json` files across epoch directories and pick the best one for routing.  
**Impact:** Operational — requires manual intervention to determine which epoch's predictions to pass to the routing script.
