## 1. Files Involved

| File | Role |
|---|---|
| classifier/run/run_large_train_xl_no_ret_vs_ret_focal.sh | Gate 1 focal-loss launcher for flan_t5_xl |
| classifier/run/run_large_train_xxl_no_ret_vs_ret_focal.sh | Gate 1 focal-loss launcher for flan_t5_xxl |
| classifier/run/run_large_train_gpt_no_ret_vs_ret_focal.sh | Gate 1 focal-loss launcher for gpt |
| classifier/run_classifier.py | Core training/eval entry point — contains `FocalLoss` class (L68–113), `FocalLossTrainer` class (L116–153), and the focal-loss training branch (L667–710) |
| classifier/utils.py | Model loading, tokenisation, scheduler, accuracy metrics (shared) |
| classifier/run/run_large_train_xl_single_vs_multi.sh | Gate 2 launcher for xl (Iteration 1, reused) |
| classifier/run/run_large_train_xxl_single_vs_multi.sh | Gate 2 launcher for xxl (reused) |
| classifier/run/run_large_train_gpt_single_vs_multi.sh | Gate 2 launcher for gpt (reused) |
| classifier/postprocess/predict_complexity_split_classifiers.py | Two-stage Clf1+Clf2 routing to QA answers |
| evaluate_final_acc.py | Final EM/F1 computation |
| `classifier/data/.../silver/no_retrieval_vs_retrieval/train.json` | Gate 1 training data (per-model) |
| `classifier/data/.../silver/no_retrieval_vs_retrieval/valid.json` | Gate 1 validation data (per-model) |
| `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json` | 3,000-question unlabelled test set |

---

## 2. Model Architecture

Both Gate 1 and Gate 2 use **T5-Large** (`AutoModelForSeq2SeqLM`), identical to Iterations 1–3. Each shell script sets `MODEL=t5-large` (line 5 in all three scripts). Gate 1 predicts `A` vs `R`; Gate 2 predicts `B` vs `C`.

---

## 3. Training Parameters

| Parameter | Focal scripts | Iteration 1 (standard CE) | Iteration 2 (undersampled) | Iteration 3 (weighted CE) | Difference? |
|---|---|---|---|---|---|
| `--learning_rate` | `3e-5` | `3e-5` | `3e-5` | `3e-5` | None |
| `--per_device_train_batch_size` | `32` | `32` | `32` | `32` | None |
| `--max_seq_length` | `384` | `384` | `384` | `384` | None |
| `--doc_stride` | `128` | `128` | `128` | `128` | None |
| `--labels` | `A R` | `A R` | `A R` | `A R` | None |
| Epochs (xl/xxl) | `15 20 25 30 35` | `15 20 25 30 35` | N/A (GPT-only) | `20 25 30 35` | **Weighted CE starts at 20** |
| Epochs (gpt) | `35 40` | `35 40` | `35 40` | `35 40` | None |
| `--use_focal_loss` | **Yes** | No | No | **Yes** (gamma=0) | **Key difference** |
| Optimizer | HF `Trainer` default AdamW | Manual `torch.optim.AdamW` + Accelerate | Manual AdamW + Accelerate | HF `Trainer` default AdamW | **Different optimizer path** (see §10) |
| Scheduler | HF `Trainer` default (linear warmup) | Manual `get_scheduler("linear", warmup=0)` | Manual linear, warmup=0 | HF `Trainer` default | **Different scheduler path** |
| `--weight_decay` | `0.0` (default) | `0.0` | `0.0` | `0.0` | None |
| `--seed` | Not passed → defaults to `42` inside `TrainingArguments` (run_classifier.py) | Not passed → `None` (no `set_seed`) | Not passed → `None` | Not passed → defaults to `42` | **Focal/weighted-CE get seed=42; Iterations 1–2 do not** |
| Early stopping | None | None | None | None | None |
| `save_strategy` | `"no"` (run_classifier.py) | N/A (manual save after all epochs) | N/A (manual save) | `"no"` | Same within focal path |

**Key difference from Iteration 1:** The focal-loss path uses `FocalLossTrainer` (HF `Trainer` subclass, run_classifier.py), which creates its own AdamW optimizer internally via `TrainingArguments`, rather than the manual `torch.optim.AdamW` + Accelerate loop used by the non-focal path. The hyperparameters (`lr=3e-5`, `weight_decay=0.0`, `warmup_steps=0`) are forwarded into `TrainingArguments` at run_classifier.py, so the effective training configuration should be equivalent, but the optimizer/scheduler internals are HF Trainer's defaults (AdamW with linear warmup schedule).

---

## 4. Focal Loss Implementation

### Custom class, not a third-party library

`FocalLoss` is defined at run_classifier.py as a `torch.nn.Module`. No external library (e.g., `focal_loss`, `kornia`) is used.

### Mathematical formula (run_classifier.py)

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Implementation:
```python
probs = torch.softmax(logits, dim=-1).clamp(min=1e-8)
p_t = probs[batch_idx, targets]
focal_weight = (1.0 - p_t) ** self.gamma
loss = -focal_weight * torch.log(p_t)
if self.alpha is not None:
    alpha_t = self.alpha.to(logits.device)[targets]
    loss = alpha_t * loss
return loss.mean()
```

### Gamma and alpha values per model

| Model | `FOCAL_GAMMA` | `FOCAL_ALPHA` | Resulting weight vector `[1−α, α]` | Interpretation |
|---|---|---|---|---|
| flan_t5_xl | `2.0` (run_large_train_xl_no_ret_vs_ret_focal.sh) | `0.33` (run_large_train_xl_no_ret_vs_ret_focal.sh) | `[0.67, 0.33]` | A (idx 0) upweighted — A is minority for xl (424 vs 868) |
| flan_t5_xxl | `2.0` (run_large_train_xxl_no_ret_vs_ret_focal.sh) | `0.36` (run_large_train_xxl_no_ret_vs_ret_focal.sh) | `[0.64, 0.36]` | A upweighted (511 vs 898) |
| gpt | `2.0` (run_large_train_gpt_no_ret_vs_ret_focal.sh) | `0.71` (run_large_train_gpt_no_ret_vs_ret_focal.sh) | `[0.29, 0.71]` | R (idx 1) upweighted — R is minority for gpt (404 vs 1013) |

All values are **environment-overridable** (`${FOCAL_GAMMA:-2.0}`, `${FOCAL_ALPHA:-0.33}`), not hardcoded constants.

### How alpha maps to label indices

`--labels A R` means `args.labels = ["A", "R"]`. At run_classifier.py, when `focal_alpha` is a float:
```python
_alpha_tensor = torch.tensor([1.0 - args.focal_alpha, args.focal_alpha])
```
Index 0 = A, Index 1 = R. So `alpha=0.33` → `A_weight=0.67, R_weight=0.33`, confirming the minority class (A for xl/xxl) gets the larger weight.

### Where the loss is applied

`FocalLossTrainer.compute_loss()` at run_classifier.py **overrides** the HF `Trainer.compute_loss()` method. It:
1. Extracts logits at decoder position 0: `outputs.logits[:, 0, :]`
2. Narrows to label token columns: `logits[:, tid]` where `tid` = tokenizer IDs for `["A", "R"]`
3. Converts ground-truth label token IDs to 0-based class indices
4. Calls `self.focal_loss_fn(label_logits, class_indices)`

The `FocalLossTrainer` is instantiated at run_classifier.py and `.train()` is called at run_classifier.py. The model is then saved at run_classifier.py.

### Is the focal loss actually used?

**Yes.** The guard condition is `if args.do_train and args.use_focal_loss:` at run_classifier.py. The shell scripts pass `--use_focal_loss` (e.g., xl script run_large_train_xl_no_ret_vs_ret_focal.sh), which sets `args.use_focal_loss = True`. The subsequent non-focal path at run_classifier.py (`if args.do_train and not args.use_focal_loss`) is **skipped**. The HF Trainer never sees the model's built-in `outputs.loss` because `compute_loss` is fully overridden.

---

## 5. No Residual Undersampling or Class-Weighted CE

**Random undersampling (Iteration 2):** Fully absent. The focal scripts do **not** call make_no_ret_vs_ret_undersampled.py anywhere. The training file is the full, unmodified `silver/no_retrieval_vs_retrieval/train.json` (e.g., xl script run_large_train_xl_no_ret_vs_ret_focal.sh), not `train_undersampled.json`.

**Weighted CE (Iteration 3):** Weighted CE used `gamma=0.0` with the same `FocalLoss` class. The focal scripts use `gamma=2.0`, not `gamma=0.0`. The `alpha` values also differ:

| Model | Focal alpha | Weighted CE alpha |
|---|---|---|
| xl | `0.33` | `0.6` |
| xxl | `0.36` | `0.6` |
| gpt | `0.71` | `0.6` |

These are entirely different settings — no residual from Iteration 3.

---

## 6. Data Pipeline

**Identical to Iteration 1.** All three focal scripts train on the same file as their Iteration 1 counterparts:

| Model | Training file | Validation file | Predict file |
|---|---|---|---|
| xl | `silver/no_retrieval_vs_retrieval/train.json` (1,292 samples) | `silver/no_retrieval_vs_retrieval/valid.json` | `predict.json` (3,000) |
| xxl | `silver/no_retrieval_vs_retrieval/train.json` (1,409 samples) | `silver/no_retrieval_vs_retrieval/valid.json` | `predict.json` |
| gpt | `silver/no_retrieval_vs_retrieval/train.json` (1,417 samples) | `silver/no_retrieval_vs_retrieval/valid.json` | `predict.json` |

No preprocessing step, no undersampling, no data modification.

---

## 7. Gate 2 — Completely Unchanged

Gate 2 is the standard Iteration 1 Clf2. The focal scripts only train Gate 1. Gate 2 is trained separately by the `*_single_vs_multi.sh` scripts with `--labels B C`, `binary_silver_single_vs_multi/train.json`, and standard cross-entropy (no `--use_focal_loss`). At routing time, predict_complexity_split_classifiers.py takes both Clf1 and Clf2 prediction files as arguments — you point Clf1 at the focal run's output and Clf2 at any Iteration 1 checkpoint.

---

## 8. Evaluation Setup

**Identical to Iteration 1.** The same routing script (predict_complexity_split_classifiers.py) and evaluator (evaluate_final_acc.py) are used. Same 6 datasets, same backbone QA predictions from `predictions/test/`, same BM25 counts, same EM/F1 computation via `DropAnswerEmAndF1`. Validation accuracy uses `calculate_accuracy` / `calculate_accuracy_perClass` in utils.py.

---

## 9. Output Artifacts

Outputs are written under a **separate `no_ret_vs_ret_focal` subdirectory**, fully isolated from Iterations 1–3:

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/{model}/
├── no_ret_vs_ret/epoch/...              ← Iteration 1
├── no_ret_vs_ret_undersampled/epoch/... ← Iteration 2 (GPT only)
├── no_ret_vs_ret_weighted_ce/epoch/...  ← Iteration 3
├── no_ret_vs_ret_focal/epoch/{N}/{DATE}/  ← Iteration 4 ✓ SEPARATE
│   ├── model.safetensors, config.json   (or checkpoint-*/...)
│   ├── valid/
│   │   ├── dict_id_pred_results.json
│   │   ├── final_eval_results.json
│   │   └── final_eval_results_perClass.json
│   └── predict/
│       ├── dict_id_pred_results.json
│       └── final_eval_results.json
└── single_vs_multi/epoch/...            ← Gate 2 (shared)
```

No risk of overwriting any prior iteration's results. Date-stamped `${DATE}` paths add further protection.

**Checkpoint detail:** The `FocalLossTrainer` uses `save_strategy="no"` (run_classifier.py), so no `checkpoint-*` directories are created during training. After training completes, `focal_trainer.save_model(args.output_dir)` saves the final model directly into `TRAIN_OUTPUT_DIR`. The shell script's fallback logic (`CKPT_PATH=$(ls -d ... checkpoint-* ...)`) will find nothing and fall back to `CKPT_PATH=${TRAIN_OUTPUT_DIR}` — which is the correct path to the saved model.

---

## 10. Suspicious / Noteworthy Items

1. **`save_strategy="no"` plus checkpoint-scanning shell logic is redundant but harmless.** The shell scripts ([xl L47–53](Adaptive-RAG/classifier/run/run_large_train_xl_no_ret_vs_ret_focal.sh#L47-L53)) look for `checkpoint-*` dirs and clean up all but the latest. Since `save_strategy="no"`, no checkpoint dirs are created, so the `CKPT_PATH` fallback at run_large_train_xl_no_ret_vs_ret_focal.sh always fires. This is correct but indicates the checkpoint-cleanup code is dead weight.

2. **Different optimizer path from Iteration 1.** Iteration 1 uses a manually constructed `torch.optim.AdamW` with explicit parameter groups ([run_classifier.py L619–636](Adaptive-RAG/classifier/run_classifier.py#L619-L636)) and a manually built `get_scheduler("linear", warmup=0)` via `prepare_scheduler()`. The focal path uses HF `Trainer`'s internal optimizer, which by default also creates AdamW but with slightly different internals (HF uses `torch.optim.AdamW` with default betas `(0.9, 0.999)` and epsilon `1e-8`, same as the manual path). The learning rate, weight decay, and warmup steps are forwarded identically, so the effective configuration should match, but there could be **subtle numerical differences** due to HF Trainer's own linear-warmup-to-zero scheduler computation vs the manually constructed one.

3. **Gamma = 2.0 is genuinely focal.** Gamma=0 would reduce focal loss to alpha-weighted cross-entropy (which is what the Iteration 3 weighted-CE scripts do). Gamma=2.0 is the standard focal-loss value from the original paper (Lin et al., 2017), so this is a real focal loss, not a disguised CE.

4. **Alpha interpretation for xl/xxl may seem counterintuitive.** For xl, `alpha=0.33` means index-1 (R) gets weight 0.33 and index-0 (A) gets weight 0.67. Since A is the minority class in xl (424 A vs 868 R), giving A the *higher* weight is correct. For gpt, `alpha=0.71` gives R weight 0.71 and A weight 0.29 — since R is the minority (404 R vs 1013 A), this is also correct. The alpha values are **manually tuned per-model** to roughly reflect the inverse class frequency.

5. **GPT script downgrades validation/prediction failures to warnings** (lines run_large_train_gpt_no_ret_vs_ret_focal.sh and run_large_train_gpt_no_ret_vs_ret_focal.sh: `|| echo "[WARN] ..."`) while xl/xxl scripts use `set -euo pipefail` without such guards. This means **xl/xxl will abort the entire run if any epoch's validation fails**, while **gpt will continue to the next epoch**. This is a behavioral inconsistency — if eval fails for an early epoch in xl, you lose all subsequent epochs.

6. **No `--seed` passed to the shell scripts**, but the `TrainingArguments` in the focal path default it to `42` (run_classifier.py: `seed=args.seed if args.seed is not None else 42`). This means focal-loss runs are reproducible (seeded at 42), unlike Iteration 1's manual training loop which receives `seed=None` and skips `set_seed()`.

7. **No early stopping and high epoch counts.** 35 epochs on ~1,292 samples (xl) is ~1,400+ gradient steps. With focal loss actively down-weighting easy examples after they become well-classified, the model may oscillate in later epochs. No early stopping or best-checkpoint selection during training is implemented — selection happens post-hoc by comparing validation accuracy across the epoch sweep.

8. **Alpha values are not derived from class counts.** The `alpha` values (0.33, 0.36, 0.71) are not computed from the actual A/R ratios:
   - xl: A/(A+R) = 424/1292 = 0.328 → `alpha=0.33` (almost exactly 1−minority_frac, i.e., alpha ≈ minority_frac itself)
   - xxl: 511/1409 = 0.363 → `alpha=0.36` (same pattern)
   - gpt: 404/1417 = 0.285 → `alpha=0.71` (close to 1 − 0.285 = 0.715)
   
   These alphas closely approximate the minority class fraction, which is a standard heuristic for focal-loss alpha. They appear intentionally set rather than arbitrary.