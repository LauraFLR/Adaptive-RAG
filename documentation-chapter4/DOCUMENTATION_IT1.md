## 1. Files Involved

**Training entry point:**
- classifier/run_classifier.py — all training and evaluation logic

**Utility module:**
- classifier/utils.py — model loading, tokenization, scheduler setup, accuracy computation

**Shell scripts (Clf1 — A vs R):**
- classifier/run/run_large_train_xl_no_ret_vs_ret.sh
- classifier/run/run_large_train_xxl_no_ret_vs_ret.sh
- classifier/run/run_large_train_gpt_no_ret_vs_ret.sh

**Shell scripts (Clf2 — B vs C):**
- classifier/run/run_large_train_xl_single_vs_multi.sh
- classifier/run/run_large_train_xxl_single_vs_multi.sh
- classifier/run/run_large_train_gpt_single_vs_multi.sh

**Cascade routing / evaluation:**
- classifier/postprocess/predict_complexity_on_classification_results.py — routes QA answers based on classifier predictions
- evaluate_final_acc.py — computes EM/F1 on routed predictions

**Data files (per LLM variant):**
- `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/silver/no_retrieval_vs_retrieval/train.json` — Clf1 train
- `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/silver/no_retrieval_vs_retrieval/valid.json` — Clf1 valid
- `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/binary_silver_single_vs_multi/train.json` — Clf2 train
- `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/silver/single_vs_multi/valid.json` — Clf2 valid
- `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json` — unlabeled test set (3,000 questions)

---

## 2. Model Architecture

Both Gate 1 and Gate 2 use **the same architecture**: `AutoModelForSeq2SeqLM` loaded from HuggingFace with `model_name_or_path = "t5-large"` (utils.py).

- **Base model:** T5-Large (770M parameters), a seq2seq encoder-decoder
- **Head type:** No added classification head. Classification is done via **generative decoding**: the model generates a single token, and inference selects the highest-probability token among the allowed label set
- At inference (run_classifier.py), `model.generate()` is called with `output_scores=True`, and the score at decoder position 0 is narrowed to only the label token IDs. `argmax` selects the prediction
- **Clf1 labels:** `A R` (passed via `--labels A R`)
- **Clf2 labels:** `B C` (passed via `--labels B C`)

---

## 3. Training Parameters

All values below are identical across all 6 scripts. Sourced from the shell scripts and the argparse defaults in run_classifier.py:

| Parameter | Value | Source |
|---|---|---|
| Learning rate | `3e-5` | Shell scripts `--learning_rate 3e-5` |
| Batch size (train) | `32` | Shell scripts `--per_device_train_batch_size 32` |
| Batch size (eval) | `100` | Shell scripts `--per_device_eval_batch_size 100` |
| Max seq length | `384` | Shell scripts `--max_seq_length 384` |
| Doc stride | `128` | Shell scripts `--doc_stride 128` |
| Weight decay | `0.0` | Default at run_classifier.py |
| Gradient accumulation | `1` | Default at run_classifier.py |
| Warmup steps | `0` | Default at run_classifier.py |
| Optimizer | AdamW | Hardcoded at run_classifier.py |
| LR scheduler | `linear` (linear decay) | Default at run_classifier.py |
| Seed | `None` (not set) | Default at run_classifier.py — no `--seed` in any shell script |
| `max_train_steps` | `None` | Default at run_classifier.py — computed dynamically from epochs |
| Epochs (XL/XXL) | `15, 20, 25, 30, 35` | Shell scripts `for EPOCH in 15 20 25 30 35` |
| Epochs (GPT) | `35, 40` | Shell scripts `for EPOCH in 35 40` |
| Early stopping | **None** | Not implemented anywhere in the code |
| Checkpointing | **None** | `--checkpointing_steps` not passed; no intermediate checkpoints saved |

**Loss function:** Standard **cross-entropy** — the model's native seq2seq loss (`outputs.loss`) at run_classifier.py. The `--use_focal_loss` flag is **not** passed in any Iteration 1 script, so the `FocalLoss`/`FocalLossTrainer` classes (lines 68–146) are never activated.

**Scheduler detail** (utils.py): `get_scheduler("linear", ...)` with 0 warmup steps. Total training steps = `num_epochs * ceil(len(train_dataloader) / gradient_accumulation_steps)`.

---

## 4. Data Pipeline

### Clf1 (A vs R)

| | flan_t5_xl | flan_t5_xxl | gpt |
|---|---|---|---|
| Train file | `silver/no_retrieval_vs_retrieval/train.json` | same pattern | same pattern |
| Train size | 1,292 (A=424, R=868) | 1,409 (A=511, R=898) | 1,417 (A=1013, R=404) |
| Valid file | `silver/no_retrieval_vs_retrieval/valid.json` | same pattern | same pattern |
| Valid size | 1,350 (A=439, R=911) | 1,415 (A=519, R=896) | 1,431 (A=1038, R=393) |

Training data is **silver labels only** — no binary inductive-bias data is added for Clf1.

### Clf2 (B vs C)

| | flan_t5_xl | flan_t5_xxl | gpt |
|---|---|---|---|
| Train file | `binary_silver_single_vs_multi/train.json` | same pattern | same pattern |
| Train size | ~3,268 | ~3,298 | ~2,804 |
| Valid file | `silver/single_vs_multi/valid.json` | same pattern | same pattern |
| Valid size | 911 (B=691, C=220) | 896 (B=701, C=195) | 393 (B=272, C=121) |

Training data is **silver + binary inductive-bias labels merged**. Binary labels contribute B (single-hop datasets) and C (multi-hop datasets) only.

### Loading and tokenization

Data is loaded via HuggingFace `load_dataset("json", data_files=...)` at run_classifier.py.

Preprocessing is in `preprocess_features_function()` (utils.py):
1. Questions are stripped of leading whitespace (line 100)
2. Input tokenization: `tokenizer(examples[question_column], truncation=True, max_length=384, stride=128, return_overflowing_tokens=True, ...)` — uses T5's fast tokenizer
3. Target tokenization: `tokenizer(text_target=targets, max_length=30, ...)` — the answer is a single label token (`"A"`, `"R"`, `"B"`, or `"C"`)
4. Padding is **dynamic** (not `pad_to_max_length`), with `DataCollatorForSeq2Seq` handling it at collation time
5. Pad tokens in labels are replaced with `-100` to be ignored in loss

---

## 5. Imbalance Handling

**Confirmed: none.** Iteration 1 uses vanilla cross-entropy with no class weighting, no focal loss, no sampling strategy.

- `--use_focal_loss` is **not passed** in any of the 6 shell scripts
- `--focal_gamma` and `--focal_alpha` are **not passed**
- The standard training loop at run_classifier.py simply uses `outputs.loss` (the model's built-in cross-entropy)
- The `DataLoader` uses `shuffle=True` (run_classifier.py) but no weighted sampler
- There is **no oversampling, undersampling, or class-weight tensor** anywhere in the Iteration 1 path

Note the class imbalance is significant:
- **Clf1 XL/XXL:** R outnumbers A roughly 2:1
- **Clf1 GPT:** A outnumbers R roughly 2.5:1 (reversed!)
- **Clf2 all variants:** B outnumbers C roughly 2-3:1

---

## 6. Evaluation Setup

### Validation (per-epoch)
Each epoch sweep runs three phases per epoch:
1. **Train** → saves model to `TRAIN_OUTPUT_DIR`
2. **Validate** → runs `--do_eval` on silver valid labels → writes to `TRAIN_OUTPUT_DIR/valid/`
3. **Predict** → runs `--do_eval` on unlabeled `predict.json` → writes to `TRAIN_OUTPUT_DIR/predict/`

Validation computes **accuracy** (exact string match between predicted label and gold label) via `calculate_accuracy()` (utils.py):
```python
acc_score = int(gold_answer == prediction)
final_acc_score = (total_acc_score / len(gold_answers)) * 100
```

Per-class accuracy is also computed via `calculate_accuracy_perClass()` (utils.py), which reports per-label accuracy, predicted count, and gold count.

Results are saved as:
- `final_eval_results.json` — `{"final_acc_score": X}`
- `final_eval_results_perClass.json` — per-class breakdown
- `dict_id_pred_results.json` — per-question predictions with gold answers

### End-to-end QA evaluation
evaluate_final_acc.py uses `SquadAnswerEmF1Metric` to compute token-level EM and F1 on the routed QA predictions across all 6 datasets × 3 LLM variants.

---

## 7. Output Artifacts

### Per-epoch outputs
```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/{model}/
├── no_ret_vs_ret/epoch/{15,20,25,30,35}/{DATE}/
│   ├── model.safetensors          # Full model weights
│   ├── config.json                # Model config
│   ├── generation_config.json
│   ├── tokenizer.json + tokenizer_config.json
│   ├── logs.log                   # Training log
│   ├── valid/
│   │   ├── final_eval_results.json
│   │   ├── final_eval_results_perClass.json
│   │   └── dict_id_pred_results.json
│   └── predict/
│       ├── final_eval_results.json
│       └── dict_id_pred_results.json
└── single_vs_multi/epoch/{15,20,25,30,35}/{DATE}/
    └── (same structure)
```

### Routed QA predictions (after cascade assembly)
```
predictions/classifier/t5-large/{model}/split/{combo}/
├── {dataset}/{dataset}.json
├── {dataset}/{dataset}_option.json
└── {dataset}/eval_metic_result_acc.json
```

---

## 8. Anything Suspicious / Premature Termination Risks

1. **`max_train_steps` can silently cap training.** At run_classifier.py, default is `None` so epochs control training. But if `max_train_steps` were passed, it would **override** `num_train_epochs` — the inner loop breaks at run_classifier.py: `if completed_steps >= args.max_train_steps: break`. In Iteration 1 this is not passed, so **no risk**.

2. **Model is saved every epoch, not just at the end.** The save block at run_classifier.py runs unconditionally after every epoch, overwriting the same `output_dir`. This means the saved model is always the **last epoch's** weights, not the best. There is no best-model selection during training — only manual selection after all epochs finish by comparing `valid/final_eval_results.json`.

3. **No seed is set.** `--seed` defaults to `None` and is never passed. At run_classifier.py, `set_seed()` is only called if seed is not None. Combined with `DataLoader(shuffle=True)`, this means training is **not reproducible** across runs.

4. **GPT Clf2 epoch 40 crash risk.** As you observed, the GPT epoch 40 run died at internal training epoch 32/40 with no validation produced. The log shows it was still training normally (loss ~0.057) when it stopped. This is likely an OOM or process kill — there's nothing in the code that would cause it. The GPT Clf2 dataset is smaller (2,804 samples), so each epoch is faster but the 40-epoch run is still ~14 minutes of continuous GPU usage.

5. **No early stopping exists.** There is no patience mechanism, no validation-during-training callback, and no best-model checkpointing. Every epoch trains to completion from scratch (the model is initialized fresh from `t5-large` for each epoch value in the sweep, not continued).

6. **Each epoch value in the sweep is a fresh training run.** The loop `for EPOCH in 15 20 25 30 35` trains **five independent models from scratch**, each for the specified number of epochs. Epoch 35 is NOT epoch 15 continued for 20 more epochs — it's a completely separate 35-epoch training run starting from base `t5-large`.