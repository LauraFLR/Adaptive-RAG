## 1. Files Involved

| File | Role |
|---|---|
| classifier/run/run_large_train_xl_no_ret_vs_ret_weighted_ce.sh | Gate 1 weighted-CE launcher for flan_t5_xl |
| classifier/run/run_large_train_xxl_no_ret_vs_ret_weighted_ce.sh | Gate 1 weighted-CE launcher for flan_t5_xxl |
| classifier/run/run_large_train_gpt_no_ret_vs_ret_weighted_ce.sh | Gate 1 weighted-CE launcher for gpt |
| classifier/run_classifier.py | Core training/eval entry point — `FocalLoss` class (L69–113), `FocalLossTrainer` class (L116–155), focal training path in `main()` (L666–708) |
| classifier/utils.py | Model loading, tokenisation, accuracy metrics (shared with all iterations) |
| classifier/run/run_large_train_{xl,xxl,gpt}_single_vs_multi.sh | Gate 2 training (unchanged from Iteration 1) |
| classifier/postprocess/predict_complexity_split_classifiers.py | End-to-end routing: merges Clf1 + Clf2 predictions → QA answers |
| evaluate_final_acc.py | Final EM/F1 computation |
| `classifier/data/.../{model}/silver/no_retrieval_vs_retrieval/train.json` | Training data (full, unmodified — same as Iteration 1) |
| `classifier/data/.../{model}/silver/no_retrieval_vs_retrieval/valid.json` | Validation data (unchanged) |
| `classifier/data/.../predict.json` | Unlabelled 3,000-question test set (unchanged) |

No undersampling utility is called. No make_no_ret_vs_ret_undersampled.py invocation exists in any weighted-CE script.

---

## 2. Model Architecture

Both Gate 1 and Gate 2 use **T5-Large** (`AutoModelForSeq2SeqLM`), identical to Iterations 1 and 2. All three shell scripts set `MODEL=t5-large` (line 4 in each). Gate 1 predicts `A` vs `R`; Gate 2 predicts `B` vs `C`.

---

## 3. Training Parameters

Parameters from the xl weighted-CE script run_large_train_xl_no_ret_vs_ret_weighted_ce.sh, compared to Iterations 1 and 2:

| Parameter | Iteration 3 (weighted CE) | Iteration 1 (standard) | Iteration 2 (undersampled) | Difference? |
|---|---|---|---|---|
| `--learning_rate` | `3e-5` | `3e-5` | `3e-5` | None |
| `--per_device_train_batch_size` | `32` | `32` | `32` | None |
| `--max_seq_length` | `384` | `384` | `384` | None |
| `--doc_stride` | `128` | `128` | `128` | None |
| `--labels` | `A R` | `A R` | `A R` | None |
| Epochs (xl) | **20, 25, 30, 35** | 15, 20, 25, 30, 35 | N/A (GPT-only) | **xl skips epoch 15** |
| Epochs (xxl) | 15, 20, 25, 30, 35 | 15, 20, 25, 30, 35 | N/A | None |
| Epochs (gpt) | 35, 40 | 35, 40 | 35, 40 | None |
| `--use_focal_loss` | **Yes** | No | No | **Different code path** |
| `--focal_gamma` | **0.0** | N/A | N/A | gamma=0 disables focal weighting |
| `--focal_alpha` | **0.6** | N/A | N/A | class weights [0.4, 0.6] |
| Optimizer | **HF Trainer's internal AdamW** | `torch.optim.AdamW` (manual) | `torch.optim.AdamW` (manual) | **Different construction** (see below) |
| Scheduler | **HF Trainer's default linear** | `get_scheduler("linear")` (manual) | `get_scheduler("linear")` (manual) | **Functionally equivalent** |
| `--weight_decay` | `0.0` (default) | `0.0` (default) | `0.0` (default) | None |
| `--seed` | **Effective 42** (see below) | `None` (not seeded) | `None` (not seeded) | **Weighted CE IS seeded** |
| Early stopping | None | None | None | None |

### Key differences flagged:

**Optimizer construction:** The `--use_focal_loss` path activates `FocalLossTrainer` (run_classifier.py), which is a `Trainer` subclass. The HF Trainer creates its own optimizer internally (default: `AdamW` with the same LR and weight decay from `TrainingArguments`). The manually-constructed `optimizer` at run_classifier.py is **never used** — it's created unconditionally but the `accelerator.prepare(model, optimizer)` call is gated behind `if not args.use_focal_loss` at run_classifier.py, and the manual training loop at run_classifier.py is gated behind `if args.do_train and not args.use_focal_loss`. So the HF Trainer manages its own AdamW. Functionally equivalent, but a different code path.

**Seed behaviour:** The `FocalLossTrainer` path constructs `TrainingArguments(seed=args.seed if args.seed is not None else 42)` at run_classifier.py. Since no `--seed` is passed, `args.seed` is `None`, so the seed defaults to **42**. In contrast, the standard Iteration 1 path skips `set_seed()` entirely when `args.seed is None` (run_classifier.py). So Iteration 3 is reproducibly seeded; Iterations 1 and 2 are not.

**Epoch range (xl only):** The xl weighted-CE script starts at epoch 20, skipping epoch 15. The xxl and gpt scripts are unchanged.

---

## 4. Class-Weighted Loss Implementation

### How it works — gamma=0 reduces FocalLoss to weighted CE

The weighted-CE scripts pass `--use_focal_loss --focal_gamma 0.0 --focal_alpha 0.6`. The `FocalLoss.forward()` method at run_classifier.py:

```python
focal_weight = (1.0 - p_t) ** self.gamma   # gamma=0 → focal_weight = 1.0
loss = -focal_weight * torch.log(p_t)       # = -log(p_t) = standard CE
if self.alpha is not None:
    alpha_t = self.alpha.to(logits.device)[targets]
    loss = alpha_t * loss                    # weighted CE
```

With `gamma=0.0`, the focal modulation term `(1 - p_t)^gamma` evaluates to **1.0** for all samples, so the loss reduces to plain cross-entropy. The alpha tensor then applies per-class weights.

### How alpha is constructed

1. Shell script: `FOCAL_ALPHA=${FOCAL_ALPHA:-0.6}` → passed as `--focal_alpha 0.6`
2. In `main()` at run_classifier.py:
   ```python
   _alpha_tensor = torch.tensor([1.0 - args.focal_alpha, args.focal_alpha], dtype=torch.float)
   # → tensor([0.4, 0.6])
   ```
3. Passed to `FocalLoss(gamma=0.0, alpha=_alpha_tensor)` at run_classifier.py
4. In `FocalLoss.__init__` at run_classifier.py, the tensor is stored as-is via `alpha.float().clone().detach()`

### How weights are applied

In `FocalLossTrainer.compute_loss()` at run_classifier.py:
1. Extract decoder logits at position 0: `logits[:, 0, :]`
2. Narrow to label-token columns: `label_logits = logits[:, tid]` → shape `(batch, 2)` for [A, R]
3. Convert ground-truth token IDs to 0-based class indices
4. Call `self.focal_loss_fn(label_logits, class_indices)` — which applies the weighted CE

### Resulting per-class weights (all three models)

With `--labels A R` and `--focal_alpha 0.6`:
- **Index 0 = A → weight 0.4**
- **Index 1 = R → weight 0.6**

These weights are **hardcoded as a constant** (via the default `FOCAL_ALPHA=0.6`), **not computed from data**. No call to `sklearn.utils.class_weight.compute_class_weight` or any data-driven calculation exists. The same alpha=0.6 is used for all three models.

---

## 5. No Residual Undersampling

**Confirmed fully removed.** None of the three weighted-CE scripts call make_no_ret_vs_ret_undersampled.py or reference `train_undersampled.json`. All three point `--train_file` at the original full file:

```
./data/${DATASET_NAME}/${LLM_NAME}/silver/no_retrieval_vs_retrieval/train.json
```

(xl script run_large_train_xl_no_ret_vs_ret_weighted_ce.sh, xxl run_large_train_xxl_no_ret_vs_ret_weighted_ce.sh, gpt run_large_train_gpt_no_ret_vs_ret_weighted_ce.sh)

Full training set sizes: xl=1,292 (A=424, R=868); xxl=1,409 (A=511, R=898); gpt=1,417 (A=1,013, R=404).

---

## 6. Gate 2 — Completely Unchanged

Gate 2 is the standard Iteration 1 Clf2. The weighted-CE experiment only touches Gate 1. Gate 2 is trained by the same `run_large_train_{xl,xxl,gpt}_single_vs_multi.sh` scripts with:
- Training data: `binary_silver_single_vs_multi/train.json`
- Validation data: `silver/single_vs_multi/valid.json`
- Labels: `B C`
- All hyperparameters identical to Iteration 1

At routing time, predict_complexity_split_classifiers.py takes separate `--no_ret_vs_ret_file` and `--single_vs_multi_file` arguments, so the weighted-CE Clf1 predictions are paired with any existing Clf2 checkpoint.

---

## 7. Evaluation Setup

**Identical to all previous iterations.** The same evaluation code path is used:
- Validation/prediction runs are separate `run_classifier.py --do_eval` invocations in the shell scripts
- Same 6 datasets, same validation file (`silver/no_retrieval_vs_retrieval/valid.json`), same `predict.json`
- Same `calculate_accuracy` / `calculate_accuracy_perClass` in utils.py
- Routing via predict_complexity_split_classifiers.py, final EM/F1 via evaluate_final_acc.py
- Same BM25 retrieval counts per model, same backbone QA predictions from `predictions/test/`

---

## 8. Output Artifacts

Checkpoints are written to a **separate directory** from Iterations 1 and 2:

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/{model}/
├── no_ret_vs_ret/epoch/...                      ← Iteration 1
├── no_ret_vs_ret_undersampled/epoch/...         ← Iteration 2 (GPT only)
├── no_ret_vs_ret_weighted_ce/epoch/{N}/{DATE}/  ← Iteration 3 ✓ SEPARATE
│   ├── config.json, model.safetensors, ...      (saved by focal_trainer.save_model())
│   ├── valid/
│   │   ├── dict_id_pred_results.json
│   │   ├── final_eval_results.json
│   │   └── final_eval_results_perClass.json
│   └── predict/
│       ├── dict_id_pred_results.json
│       └── final_eval_results.json
└── single_vs_multi/epoch/...                    ← Gate 2 (shared)
```

The output subdirectory `no_ret_vs_ret_weighted_ce` (set at line 15 in each script) is distinct from `no_ret_vs_ret` (Iteration 1) and `no_ret_vs_ret_undersampled` (Iteration 2). Date-stamped paths provide additional isolation. **No risk of overwriting.**

---

## 9. Suspicious / Noteworthy Items

### (a) Alpha=0.6 is backwards for xl and xxl

This is the most significant finding. With `--labels A R` and `--focal_alpha 0.6`:
- Index 0 (A) gets weight **0.4**
- Index 1 (R) gets weight **0.6**

The actual class distributions are:

| Model | A (index 0) | R (index 1) | Minority class | What alpha=0.6 does |
|---|---|---|---|---|
| **flan_t5_xl** | 424 (32.8%) | 868 (67.2%) | A | **Downweights** minority A (0.4), **upweights** majority R (0.6) |
| **flan_t5_xxl** | 511 (36.3%) | 898 (63.7%) | A | **Downweights** minority A (0.4), **upweights** majority R (0.6) |
| **gpt** | 1,013 (71.5%) | 404 (28.5%) | R | **Upweights** minority R (0.6) — **correct** |

For xl and xxl, the weighting is **inverted** — it amplifies the majority class and suppresses the minority class. This is the opposite of what class-weighted CE is intended to do. Compare with the focal-loss scripts, which set model-specific alphas: xl=0.33, xxl=0.36, gpt=0.71 — those correctly upweight each model's minority class.

The `--focal_alpha` docstring itself says "weight for index 1 (minority class)" (run_classifier.py), which assumes index 1 is always the minority. This holds for GPT (R is minority at index 1) but is false for xl/xxl (A is minority at index 0).

### (b) Different training code path (HF Trainer vs manual loop)

The `--use_focal_loss` flag routes training through `FocalLossTrainer` (run_classifier.py), which is a HuggingFace `Trainer` subclass. This is a fundamentally different code path from Iterations 1 and 2 (which use a manual training loop with `accelerator` at run_classifier.py). Differences include:

- **Optimizer:** HF Trainer creates its own `AdamW` internally; the manually-constructed `optimizer` at run_classifier.py is **created but never used** (wasted computation; the `accelerator.prepare` guard at run_classifier.py skips it).
- **Scheduler:** HF Trainer uses its own default linear scheduler. The manual path uses `prepare_scheduler()` from utils.py.
- **Seed:** HF Trainer applies seed=42 via `TrainingArguments`; the manual path leaves training unseeded.

The loss **is** actually applied — `FocalLossTrainer.compute_loss()` overrides the Trainer's default `compute_loss`, so the HF Trainer does **not** silently revert to the model's built-in cross-entropy. This is confirmed by the method at run_classifier.py.

### (c) `save_strategy="no"` and checkpoint resolution

The `TrainingArguments` at run_classifier.py sets `save_strategy="no"`, meaning no intermediate checkpoints are saved by the Trainer. The model is saved only by the explicit `focal_trainer.save_model(args.output_dir)` call at run_classifier.py, which writes directly to `TRAIN_OUTPUT_DIR`.

The shell scripts then try to find `checkpoint-*` directories ([line 41](Adaptive-RAG/classifier/run/run_large_train_xl_no_ret_vs_ret_weighted_ce.sh#L41)). Since none exist, `CKPT_PATH` correctly falls back to `TRAIN_OUTPUT_DIR` ([line 43](Adaptive-RAG/classifier/run/run_large_train_xl_no_ret_vs_ret_weighted_ce.sh#L43)). The cleanup loop is a no-op. **No premature termination risk here** — but the checkpoint-finding logic is vestigial dead code for this path.

### (d) Tolerant error handling

All three scripts use `|| echo "[WARN] ..."` after validation and prediction commands ([xl line 72](Adaptive-RAG/classifier/run/run_large_train_xl_no_ret_vs_ret_weighted_ce.sh#L72), run_large_train_xl_no_ret_vs_ret_weighted_ce.sh). Unlike Iteration 2 (which uses `set -euo pipefail` and would abort on failure), these scripts silently continue if validation or prediction fails, logging only a warning. This means a broken checkpoint could produce an incomplete run without any error signal.

### (e) xl skips epoch 15

The xl weighted-CE script sweeps epochs `20 25 30 35` ([line 13](Adaptive-RAG/classifier/run/run_large_train_xl_no_ret_vs_ret_weighted_ce.sh#L13)), while the standard xl sweeps `15 20 25 30 35`. Epoch 15 is omitted. The xxl script retains the full range `15 20 25 30 35`. This is a minor inconsistency but means xl has one fewer checkpoint to compare.


---

# CHANGES

Based on your documentation, the fix is straightforward. You need to change the alpha values in the three shell scripts so that each model's minority class receives the higher weight. Everything else stays identical.

### What to Change

The alpha parameter in your implementation means: `alpha = weight for index 1 (R)`, and `1 - alpha = weight for index 0 (A)`. So `--focal_alpha X` produces weights `[1-X, X]` for `[A, R]`.

The SOTA approach is inverse class frequency, normalized to sum to 1. Using `w_c = N / (K × n_c)`:

| Model | N | n_A | n_R | w_A raw | w_R raw | w_A norm | w_R norm | `--focal_alpha` (= w_R norm) |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **XL** | 1292 | 424 | 868 | 1.524 | 0.744 | **0.672** | **0.328** | **0.33** |
| **XXL** | 1409 | 511 | 898 | 1.379 | 0.784 | **0.638** | **0.362** | **0.36** |
| **GPT** | 1417 | 1013 | 404 | 0.699 | 1.754 | **0.285** | **0.715** | **0.71** |

### Concrete Changes Per File

**`run_large_train_xl_no_ret_vs_ret_weighted_ce.sh`:**
Change `FOCAL_ALPHA=${FOCAL_ALPHA:-0.6}` → `FOCAL_ALPHA=${FOCAL_ALPHA:-0.33}`

**`run_large_train_xxl_no_ret_vs_ret_weighted_ce.sh`:**
Change `FOCAL_ALPHA=${FOCAL_ALPHA:-0.6}` → `FOCAL_ALPHA=${FOCAL_ALPHA:-0.36}`

**`run_large_train_gpt_no_ret_vs_ret_weighted_ce.sh`:**
Change `FOCAL_ALPHA=${FOCAL_ALPHA:-0.6}` → `FOCAL_ALPHA=${FOCAL_ALPHA:-0.71}`

This produces the correct per-class weights:

| Model | A weight (minority?) | R weight (minority?) | Effect |
|:--|:--|:--|:--|
| XL | **0.67** (minority ✓) | 0.33 | Upweights minority A |
| XXL | **0.64** (minority ✓) | 0.36 | Upweights minority A |
| GPT | 0.29 | **0.71** (minority ✓) | Upweights minority R |

### What NOT to Change

Everything else stays identical: `--focal_gamma 0.0`, all hyperparameters, the full training data (no undersampling), same epoch sweeps, same Gate 2. The only variable that changes relative to the old Iter 3 is the alpha value — which is exactly the controlled-experiment principle.

### Additional Recommendation: Add Epoch 15 Back for XL

While you're editing `run_large_train_xl_no_ret_vs_ret_weighted_ce.sh`, also change the epoch sweep from `20 25 30 35` to `15 20 25 30 35` to match Iterations 1 and the other models. This eliminates the minor inconsistency flagged earlier.

### Justification for the Thesis

You can now cite the inverse class frequency formula `w_c = N / (K × n_c)` as the standard approach (used by scikit-learn's `compute_class_weight('balanced', ...)`; see also King & Zeng, 2001 for the theoretical grounding of inverse frequency weighting in imbalanced binary classification). The per-model computation is justified because class distributions differ substantially across backbone models (32.8% vs. 36.3% vs. 71.5% for class A), making a single fixed alpha inappropriate.