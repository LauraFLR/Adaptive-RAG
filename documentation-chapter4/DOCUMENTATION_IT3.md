# Iteration 3 — Class-Weighted Cross-Entropy Loss for Gate 1

> **Design Science Research Artifact:** Replace the standard (unweighted)
> cross-entropy loss in Clf1 (A vs R) with inverse-frequency class-weighted
> cross-entropy to address class imbalance, implemented via `FocalLoss` with
> `gamma = 0` and auto-computed class weights.  Applied to all three model
> variants (flan_t5_xl, flan_t5_xxl, gpt).  Gate 2 remains unchanged.

---

## 1. Files Involved

| File | Role |
|---|---|
| `classifier/run/run_large_train_xl_no_ret_vs_ret_weighted_ce.sh` | **New in IT3.** Shell launcher — Clf1 (A vs R) with weighted CE for XL silver labels. Sweeps epochs `20 25 30 35`. |
| `classifier/run/run_large_train_xxl_no_ret_vs_ret_weighted_ce.sh` | **New in IT3.** Shell launcher — Clf1 (A vs R) with weighted CE for XXL silver labels. Sweeps epochs `15 20 25 30 35`. |
| `classifier/run/run_large_train_gpt_no_ret_vs_ret_weighted_ce.sh` | **New in IT3.** Shell launcher — Clf1 (A vs R) with weighted CE for GPT silver labels. Sweeps epochs `35 40`. |
| `classifier/run_classifier.py` | Shared — **same file as IT1/IT2**, but a different code path is now active: `FocalLoss`, `FocalLossTrainer`, and the focal training branch in `main()`. |
| `classifier/utils.py` | Shared — **identical** to IT1/IT2. |
| `classifier/run/run_large_train_{xl,xxl,gpt}_single_vs_multi.sh` | Gate 2 — **identical** to IT1. |
| `classifier/postprocess/predict_complexity_split_classifiers.py` | Cascade routing — **identical** to IT1. |
| `evaluate_final_acc.py` | QA evaluation — **identical** to IT1. |

**No undersampling utility is called.** The training data files are the original, full-size silver-label files used in IT1. No `make_no_ret_vs_ret_undersampled.py` invocation appears in any IT3 script.

---

## 2. Model Architecture

**Identical to Iteration 1.** T5-Large (770 M parameters), `AutoModelForSeq2SeqLM`, generative decoding with constrained softmax over label token IDs. Clf1 labels: `A R`. Clf2 labels: `B C`. See DOCUMENTATION_IT1.md §2 for full details.

---

## 3. Training Parameters

| Parameter | IT3 value | IT1 value | IT2 value (GPT only) | Difference from IT1? |
|---|---|---|---|---|
| Base model | `t5-large` | `t5-large` | `t5-large` | No |
| Learning rate | `3e-5` | `3e-5` | `3e-5` | No |
| Per-device train batch size | `32` | `32` | `32` | No |
| Per-device eval batch size | `100` | `100` | `100` | No |
| Max sequence length | `384` | `384` | `384` | No |
| Doc stride | `128` | `128` | `128` | No |
| Weight decay | `0.0` | `0.0` | `0.0` | No |
| Gradient accumulation steps | `1` | `1` | `1` | No |
| Warmup steps | `0` | `0` | `0` | No |
| **Optimizer** | **HF Trainer's AdamW** | `torch.optim.AdamW` | `torch.optim.AdamW` | **YES** — different implementation |
| **LR scheduler** | **HF Trainer default (`linear`)** | `get_scheduler("linear")` via Accelerate | `get_scheduler("linear")` | Same type, different orchestration |
| Seed | `42` | `42` | `42` | No |
| Epochs (XL) | **`20, 25, 30, 35`** | `15, 20, 25, 30, 35` | N/A | **YES** — XL skips epoch 15 |
| Epochs (XXL) | `15, 20, 25, 30, 35` | `15, 20, 25, 30, 35` | N/A | No |
| Epochs (GPT) | `35, 40` | `35, 40` | `35, 40` | No |
| Labels | `A R` | `A R` | `A R` | No |
| **`--use_focal_loss`** | **Yes** | No | No | **YES** — activates FocalLossTrainer path |
| **`--focal_gamma`** | **`0.0`** | N/A | N/A | **YES** — reduces focal loss to weighted CE |
| **`--auto_class_weights`** | **Yes** | N/A | N/A | **YES** — computes weights from data |
| `--focal_alpha` | Not passed | N/A | N/A | N/A (overridden by `--auto_class_weights`) |
| **Training file** | `train.json` (original) | `train.json` | `train_undersampled.json` | Same as IT1, differs from IT2 |
| Validation file | `valid.json` | `valid.json` | `valid.json` | No |
| **Training code path** | **`FocalLossTrainer`** (HF `Trainer` subclass) | Manual Accelerate loop | Manual Accelerate loop | **YES** — entirely different training loop |
| **`save_strategy`** | **`"no"`** | N/A (manual per-epoch save) | N/A | **YES** — no intermediate checkpoints during training |
| **Model save** | **`focal_trainer.save_model()`** once at end | `save_pretrained()` per epoch (overwrites) | `save_pretrained()` per epoch | **YES** — single save after all epochs |
| GPU handling | `GPU=${GPU:-0}` (overridable) | `GPU=0` (hardcoded) | `GPU=${GPU:-0}` | Minor shell difference |
| Error handling | `|| echo "[WARN]"` on valid/predict | None | `set -euo pipefail` (strict) | **YES** — tolerant failure mode |

### Key differences summarised

1. **`--use_focal_loss` activates** the `FocalLossTrainer` code path instead of the manual Accelerate training loop.
2. **`--focal_gamma 0.0`** reduces the focal loss formula to class-weighted cross-entropy (see §4).
3. **`--auto_class_weights`** computes per-class weights from the training data distribution using inverse-frequency weighting (see §4).
4. **HF Trainer's AdamW** is used instead of `torch.optim.AdamW`. HF's `TrainingArguments` constructs the optimizer internally.
5. **XL skips epoch 15** (starts at 20), while XXL and GPT keep their IT1 epoch ranges.
6. **`save_strategy="no"`** means no `checkpoint-*` directories are created during training. The model is saved once at the end via `focal_trainer.save_model()`.

---

## 4. Class-Weighted Loss Implementation

### 4.1 How `gamma = 0` reduces focal loss to weighted cross-entropy

The focal loss formula [run_classifier.py L69]:

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

When $\gamma = 0$:

$$(1 - p_t)^0 = 1$$

So the formula reduces to:

$$\text{FL}(p_t) = -\alpha_t \cdot \log(p_t)$$

This is exactly **class-weighted cross-entropy**: the standard cross-entropy loss $-\log(p_t)$ multiplied by a per-class weight $\alpha_t$.

### 4.2 How class weights are constructed

The `--auto_class_weights` flag triggers the inverse-frequency weight computation [run_classifier.py L678–686]:

```python
label_counts = Counter(raw_datasets[args.train_column]["answer"])
labels_list = args.labels          # ["A", "R"]
n_total = sum(label_counts[l] for l in labels_list)
n_classes = len(labels_list)       # 2
weights = [n_total / (n_classes * label_counts[l]) for l in labels_list]
_alpha_tensor = torch.tensor(weights, dtype=torch.float)   # [w_A, w_R]
```

Formula per class:

$$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$

where $N_{\text{total}}$ is total samples, $K = 2$ (number of classes), and $N_c$ is the count for class $c$. This is the same formula used by scikit-learn's `class_weight="balanced"`.

### 4.3 How `_alpha_tensor` enters `FocalLoss`

`_alpha_tensor` is a `torch.Tensor` with shape `(2,)` containing `[w_A, w_R]`. It is passed to `FocalLoss(gamma=0.0, alpha=_alpha_tensor)` [run_classifier.py L697].

In `FocalLoss.__init__`, this hits the `isinstance(alpha, (list, torch.Tensor))` branch [run_classifier.py L89–93], which stores the tensor directly as `self.alpha` — **no** `[1-alpha, alpha]` transformation is applied (that transformation only occurs when `alpha` is a scalar `float`).

### 4.4 How weights map to label indices

| Index | Label | Mapping source |
|---|---|---|
| 0 | A | `args.labels[0]` = `"A"` |
| 1 | R | `args.labels[1]` = `"R"` |

In `FocalLoss.forward()` [run_classifier.py L107–112]:

```python
alpha_t = self.alpha.to(logits.device)[targets]   # targets are 0-based class indices
loss = alpha_t * loss
```

So `self.alpha[0]` (= $w_A$) is applied to all A-labelled samples, and `self.alpha[1]` (= $w_R$) is applied to all R-labelled samples.

The `class_indices` are computed in `FocalLossTrainer.compute_loss()` [run_classifier.py L144–146]:

```python
tid = torch.tensor(self.label_token_ids, dtype=torch.long, device=logits.device)
token_ids = labels[:, 0]
class_indices = (token_ids.unsqueeze(1) == tid.unsqueeze(0)).long().argmax(dim=1)
```

`label_token_ids` is built as `[tokenizer("A").input_ids[0], tokenizer("R").input_ids[0]]` [run_classifier.py L696], matching the `args.labels` order. A sample with ground-truth token "A" gets class index 0; "R" gets class index 1.

### 4.5 Per-model weights

| Model | A count | R count | Total | $w_A$ | $w_R$ | Minority | Minority gets higher weight? |
|---|---|---|---|---|---|---|---|
| `flan_t5_xl` | 424 | 868 | 1 292 | **1.5236** | 0.7442 | A | **Yes** ✓ |
| `flan_t5_xxl` | 511 | 898 | 1 409 | **1.3787** | 0.7845 | A | **Yes** ✓ |
| `gpt` | 1 013 | 404 | 1 417 | 0.6994 | **1.7537** | R | **Yes** ✓ |

With `--auto_class_weights`, the inverse-frequency formula correctly assigns higher weight to the minority class in all three models.

**Historical note:** The shell scripts contain a comment `# FOCAL_ALPHA removed: --auto_class_weights computes weights from training data`, indicating that a previous version used a scalar `--focal_alpha` value. Had a scalar `--focal_alpha` been used instead, it would have been transformed via `[1 - alpha, alpha]`, meaning `alpha` would directly set the R weight (index 1). For XL and XXL where A is the minority, this would require `alpha < 0.5` to upweight A — an easy mistake to get backwards. The switch to `--auto_class_weights` eliminates this ambiguity.

### 4.6 Complete loss computation flow

For a single training sample with ground-truth class $c$ (0 = A or 1 = R):

1. **Decoder logits** extracted at position 0: `outputs.logits[:, 0, :]` — shape `(batch, vocab_size)` [run_classifier.py L138]
2. **Narrow** to label columns: `label_logits = logits[:, tid]` — shape `(batch, 2)` [run_classifier.py L141]
3. **Softmax**: `probs = softmax(label_logits, dim=-1)` [run_classifier.py L105]
4. **Select** $p_t = \text{probs}[c]$ [run_classifier.py L107]
5. **Focal weight**: $(1 - p_t)^{0.0} = 1.0$ [run_classifier.py L108]
6. **Base loss**: $-1.0 \cdot \log(p_t) = -\log(p_t)$ [run_classifier.py L109]
7. **Class weight**: $\alpha_t = w_c$ from the weight tensor [run_classifier.py L111]
8. **Final**: $\text{loss} = w_c \cdot (-\log(p_t))$ [run_classifier.py L112]
9. **Batch mean**: `loss.mean()` [run_classifier.py L113]

---

## 5. No Residual Undersampling

**No undersampling is applied in IT3.** The training files are the original, full-size silver-label files — the same files used in IT1.

| Model | Training file | Sample count | A | R |
|---|---|---|---|---|
| `flan_t5_xl` | `.../flan_t5_xl/silver/no_retrieval_vs_retrieval/train.json` | 1 292 | 424 (32.8 %) | 868 (67.2 %) |
| `flan_t5_xxl` | `.../flan_t5_xxl/silver/no_retrieval_vs_retrieval/train.json` | 1 409 | 511 (36.3 %) | 898 (63.7 %) |
| `gpt` | `.../gpt/silver/no_retrieval_vs_retrieval/train.json` | 1 417 | 1 013 (71.5 %) | 404 (28.5 %) |

None of the three IT3 shell scripts call `make_no_ret_vs_ret_undersampled.py` or reference `train_undersampled.json`. No data is discarded. The class imbalance is handled entirely through the loss function weights.

---

## 6. Gate 2 — Completely Unchanged

Gate 2 (Clf2: B vs C) is **identical to Iteration 1** for all three model variants. The weighted-CE experiment changes only Gate 1.

| Property | XL | XXL | GPT |
|---|---|---|---|
| Script | `run_large_train_xl_single_vs_multi.sh` | `run_large_train_xxl_single_vs_multi.sh` | `run_large_train_gpt_single_vs_multi.sh` |
| Labels | `B C` | `B C` | `B C` |
| Training file | `binary_silver_single_vs_multi/train.json` | same pattern | same pattern |
| Train size | 3 268 (B=1 871, C=1 397) | 3 298 (B=1 903, C=1 395) | 2 804 (B=1 475, C=1 329) |
| Validation file | `silver/single_vs_multi/valid.json` | same pattern | same pattern |
| Epochs | `15 20 25 30 35` | `15 20 25 30 35` | `35 40` |
| Seed | `42` | `42` | `42` |
| Loss | Standard CE (no focal loss) | Standard CE | Standard CE |
| Training code path | Manual Accelerate loop | Manual Accelerate loop | Manual Accelerate loop |

At routing time, the IT3 Clf1 (weighted-CE) predictions and the standard Clf2 predictions are combined via `predict_complexity_split_classifiers.py`:
- Clf1 predicts A → final label A
- Clf1 predicts R → use Clf2's prediction (B or C)

The routing script and its arguments are identical to IT1/IT2.

---

## 7. Evaluation Setup

**Identical to Iteration 1.** Specifically:

| Step | Procedure | Difference from IT1? |
|---|---|---|
| Per-epoch validation | `run_classifier.py --do_eval` on `valid.json` → accuracy + per-class accuracy | No |
| Per-epoch prediction | `run_classifier.py --do_eval` on `predict.json` → classification labels | No |
| Checkpoint selection | Shell script finds `checkpoint-*`, falls back to `TRAIN_OUTPUT_DIR` | In practice always falls back (see §9.3) |
| Cascade routing | `predict_complexity_split_classifiers.py` | No |
| QA evaluation | `evaluate_final_acc.py --pred_path ...` | No |

See DOCUMENTATION_IT1.md §6 for full evaluation details.

---

## 8. Output Artifacts

### 8.1 Directory tree

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/
  {model}/                                          # flan_t5_xl, flan_t5_xxl, or gpt
    no_ret_vs_ret_weighted_ce/                      ← SEPARATE from IT1's no_ret_vs_ret/
      epoch/
        {epoch}/                                    # 20,25,30,35 for xl; 15..35 for xxl; 35,40 for gpt
          {YYYY_MM_DD}/{HH_MM_SS}/
            config.json
            generation_config.json
            model.safetensors                       # or pytorch_model.bin
            spiece.model
            special_tokens_map.json
            tokenizer.json
            tokenizer_config.json
            training_args.bin                       # HF Trainer metadata (new in IT3)
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

### 8.2 Separation from IT1 and IT2

| Classifier | IT1 path | IT2 path (GPT) | IT3 path | Risk of overwrite? |
|---|---|---|---|---|
| Clf1 | `.../no_ret_vs_ret/epoch/...` | `.../no_ret_vs_ret_undersampled/epoch/...` | `.../no_ret_vs_ret_weighted_ce/epoch/...` | **No** — all three use different directory names |
| Clf2 | `.../single_vs_multi/epoch/...` | same | same | **Shared** — but timestamped `{DATE}` subdirectories prevent overwrite |

### 8.3 Notable output differences from IT1

- `training_args.bin` is written by HF Trainer's `save_model()` (absent in IT1's manual save).
- No `logs.log` in the training output root (HF Trainer uses its own logging; the manual `logging.basicConfig(filename=...)` call at [run_classifier.py L522] is only reached but the log file may not contain training-step details from the Trainer).
- No `checkpoint-*` subdirectories (because `save_strategy="no"`).

---

## 9. Suspicious / Noteworthy Items

### 9.1 Different training code path (HF Trainer vs manual Accelerate loop)

| Issue | Detail |
|---|---|
| **What** | IT1/IT2 use a manual training loop with `Accelerator`, `torch.optim.AdamW`, and `get_scheduler()`. IT3 uses `FocalLossTrainer` (subclass of HF `Trainer`) with `TrainingArguments`. |
| **Risk** | Subtle behavioural differences between the two paths: (a) HF Trainer uses its own AdamW which may differ in epsilon, beta, or weight-decay implementation; (b) Trainer handles gradient accumulation, mixed precision, and learning-rate warmup internally; (c) The loss computation operates on the first decoder position only (IT1's CE is computed over the full sequence by T5's internal `lm_head`). |
| **Files** | IT3 path: [run_classifier.py L698–728]. IT1 path: [run_classifier.py L730–840]. |
| **Impact** | Results are not attributable solely to the loss weighting — the training loop itself changed. |

### 9.2 Loss computed on position 0 only vs full-sequence CE

| Issue | Detail |
|---|---|
| **What** | `FocalLossTrainer.compute_loss()` extracts logits at `[:, 0, :]` (first decoder position) and computes loss only on that position [run_classifier.py L138]. IT1's standard path uses `outputs.loss`, which is the T5 model's built-in cross-entropy over **all** decoder positions (though for single-token labels the subsequent positions are padding with `-100`). |
| **Risk** | For single-token targets (A, R), the practical difference is minimal — subsequent positions contribute zero loss due to `-100` masking. However, the softmax normalisation is different: IT3 applies softmax over only the 2 label token IDs, while IT1 applies softmax over the full ~32 000-token vocabulary. This means IT3 effectively ignores probability mass on non-label tokens, which could change the gradient signal. |
| **Files** | [run_classifier.py L138–141] vs T5's internal CE. |

### 9.3 `save_strategy="no"` and checkpoint-finding shell logic

| Issue | Detail |
|---|---|
| **What** | `TrainingArguments(save_strategy="no")` [run_classifier.py L710] means no `checkpoint-*` directories are created during training. Yet the shell scripts contain a `CKPT_PATH=$(ls -d ${TRAIN_OUTPUT_DIR}/checkpoint-* ...)` block that searches for them. |
| **Behaviour** | `ls -d ...checkpoint-*` will find no matches and produce an empty string (stderr suppressed by `2>/dev/null`). The `if [[ -z "${CKPT_PATH}" ]]` fallback sets `CKPT_PATH=${TRAIN_OUTPUT_DIR}`, which is correct — `focal_trainer.save_model(args.output_dir)` writes the model to `${TRAIN_OUTPUT_DIR}`. The cleanup loop `for ckpt in ...checkpoint-*` also matches nothing and is harmless. |
| **Impact** | No functional problem. The checkpoint-finding code is dead but benign. |
| **Files** | Shell scripts L42–49 (all three variants). |

### 9.4 Tolerant error handling (`|| echo "[WARN]"`)

| Issue | Detail |
|---|---|
| **What** | All three IT3 scripts append `|| echo "[WARN] Validation failed for epoch ${EPOCH}"` and `|| echo "[WARN] Prediction failed for epoch ${EPOCH}"` to the validation and prediction commands. |
| **Risk** | If validation or prediction crashes (e.g., OOM, corrupt model file, missing data), the script prints a warning and **continues to the next epoch**. In IT1 (no error handling) or IT2 (`set -euo pipefail`), such a crash would abort the entire script. |
| **Note** | The IT3 scripts do **not** use `set -euo pipefail` — they have no shell-level strictness at all (the `#!/usr/bin/env bash` shebang does not imply `set -e`). Only the training step runs without a `||` fallback; a training failure would exit the script. |
| **Files** | [run_large_train_xl_no_ret_vs_ret_weighted_ce.sh L67, L77] and equivalents. |

### 9.5 XL epoch 15 skip

| Issue | Detail |
|---|---|
| **What** | The XL weighted-CE script sweeps `for EPOCH in 20 25 30 35`, omitting epoch 15. The XXL script includes `15 20 25 30 35`. |
| **Risk** | The XL IT1 baseline uses `15 20 25 30 35`. Comparisons at epoch 15 between IT1 and IT3 for XL are impossible. |
| **File** | [run_large_train_xl_no_ret_vs_ret_weighted_ce.sh L13] |

### 9.6 No `set -euo pipefail` in XL and XXL scripts

| Issue | Detail |
|---|---|
| **What** | Unlike the undersampled script (IT2) which uses `set -euo pipefail`, none of the three IT3 weighted-CE scripts set any shell strictness options. |
| **Risk** | Undefined variables expand to empty strings silently. Pipe failures are ignored. Only the `|| echo "[WARN]"` on validation/prediction provides any error awareness. |
| **Files** | All three weighted-CE scripts lack `set -e` or `set -euo pipefail`. |

### 9.7 `eval_strategy="no"` — no trainer-side validation

| Issue | Detail |
|---|---|
| **What** | `TrainingArguments(eval_strategy="no")` [run_classifier.py L711] means the HF Trainer never runs validation during training. Validation is done by a **separate** `run_classifier.py --do_eval` invocation after training completes. |
| **Risk** | No training-time metrics are available for early stopping or best-model selection. The model saved at the end of all epochs is the only model. Combined with the per-epoch-value loop (each epoch budget is a fresh run from `t5-large`), this means each epoch value produces exactly one model with no internal validation signal. |
| **File** | [run_classifier.py L711] |

### 9.8 Fresh-from-scratch training per epoch value

| Issue | Detail |
|---|---|
| **What** | Same as IT1 — each epoch value in the shell `for` loop starts from the pre-trained `t5-large` base model. Epoch 35 trains from scratch for 35 epochs, not 10 more on top of epoch 25. |
| **Impact** | High GPU cost. XL: `20+25+30+35 = 110` epochs. XXL: `15+20+25+30+35 = 125` epochs. GPT: `35+40 = 75` epochs. |
| **Files** | All three weighted-CE shell scripts. |

### 9.9 Validation set imbalance unchanged

| Issue | Detail |
|---|---|
| **What** | Validation files remain unchanged from IT1. The model is trained with loss weights that compensate for the training imbalance, but validation accuracy is computed on the original (imbalanced) distribution. |
| **Implication** | Per-class accuracy (`final_eval_results_perClass.json`) is essential for evaluation. Overall accuracy will be dominated by the majority class in the validation set. |
| **Validation distributions** | XL: 439 A / 911 R (32.5 % / 67.5 %). XXL: 519 A / 896 R (36.7 % / 63.3 %). GPT: 1 038 A / 393 R (72.5 % / 27.5 %). |
