# Iteration 4 — Focal Loss (γ = 2.0) for Gate 1

> **Design Science Research Artifact:** Replace the standard cross-entropy loss
> in Clf1 (A vs R) with focal loss using a focusing parameter γ = 2.0 and
> per-model α weights that approximate inverse class frequency.  Applied to all
> three model variants (flan_t5_xl, flan_t5_xxl, gpt).  Gate 2 remains
> unchanged.

---

## 1. Files Involved

| File | Role |
|---|---|
| `classifier/run/run_large_train_xl_no_ret_vs_ret_focal.sh` | **New in IT4.** Shell launcher — Clf1 (A vs R) with focal loss for XL silver labels. γ = 2.0, α = 0.33. Sweeps epochs `15 20 25 30 35`. |
| `classifier/run/run_large_train_xxl_no_ret_vs_ret_focal.sh` | **New in IT4.** Shell launcher — Clf1 (A vs R) with focal loss for XXL silver labels. γ = 2.0, α = 0.36. Sweeps epochs `15 20 25 30 35`. |
| `classifier/run/run_large_train_gpt_no_ret_vs_ret_focal.sh` | **New in IT4.** Shell launcher — Clf1 (A vs R) with focal loss for GPT silver labels. γ = 2.0, α = 0.71. Sweeps epochs `35 40`. |
| `classifier/run_classifier.py` | Shared — **same file as IT1–IT3**. The `FocalLoss` class, `FocalLossTrainer` class, and the focal training branch in `main()` are the active code paths (same paths as IT3, but with different hyperparameters). |
| `classifier/utils.py` | Shared — **identical** to IT1–IT3. |
| `classifier/run/run_large_train_{xl,xxl,gpt}_single_vs_multi.sh` | Gate 2 — **identical** to IT1. |
| `classifier/postprocess/predict_complexity_split_classifiers.py` | Cascade routing — **identical** to IT1. |
| `evaluate_final_acc.py` | QA evaluation — **identical** to IT1. |

---

## 2. Model Architecture

**Identical to Iteration 1.** T5-Large (770 M parameters), `AutoModelForSeq2SeqLM`, generative decoding with constrained softmax over label token IDs. Clf1 labels: `A R`. Clf2 labels: `B C`. See DOCUMENTATION_IT1.md §2 for full details.

---

## 3. Training Parameters

| Parameter | IT4 value | IT3 value | IT2 (GPT) | IT1 |
|---|---|---|---|---|
| Base model | `t5-large` | `t5-large` | `t5-large` | `t5-large` |
| Learning rate | `3e-5` | `3e-5` | `3e-5` | `3e-5` |
| Train batch size | `32` | `32` | `32` | `32` |
| Eval batch size | `100` | `100` | `100` | `100` |
| Max seq length | `384` | `384` | `384` | `384` |
| Doc stride | `128` | `128` | `128` | `128` |
| Weight decay | `0.0` | `0.0` | `0.0` | `0.0` |
| Grad accum steps | `1` | `1` | `1` | `1` |
| Warmup steps | `0` | `0` | `0` | `0` |
| Optimizer | HF Trainer's AdamW | HF Trainer's AdamW | `torch.optim.AdamW` | `torch.optim.AdamW` |
| LR scheduler | HF Trainer default (`linear`) | HF Trainer default | `get_scheduler("linear")` | `get_scheduler("linear")` |
| Seed | `42` | `42` | `42` | `42` |
| Epochs (XL) | `15, 20, 25, 30, 35` | `20, 25, 30, 35` | N/A | `15, 20, 25, 30, 35` |
| Epochs (XXL) | `15, 20, 25, 30, 35` | `15, 20, 25, 30, 35` | N/A | `15, 20, 25, 30, 35` |
| Epochs (GPT) | `35, 40` | `35, 40` | `35, 40` | `35, 40` |
| Labels | `A R` | `A R` | `A R` | `A R` |
| **`--use_focal_loss`** | **Yes** | Yes | No | No |
| **`--focal_gamma`** | **`2.0`** | `0.0` | N/A | N/A |
| **`--focal_alpha`** | **Per-model (see below)** | N/A | N/A | N/A |
| **`--auto_class_weights`** | **No** | Yes | N/A | N/A |
| Training file | `train.json` (original) | `train.json` | `train_undersampled.json` | `train.json` |
| Training code path | `FocalLossTrainer` | `FocalLossTrainer` | Manual Accelerate loop | Manual Accelerate loop |
| `save_strategy` | `"no"` | `"no"` | N/A | N/A |
| Shell strictness | `set -euo pipefail` (XL, XXL); tolerant (GPT) | None | `set -euo pipefail` | None |
| Error handling | XL/XXL: strict; GPT: `\|\| echo "[WARN]"` | `\|\| echo "[WARN]"` | Strict | None |
| GPU | `GPU=${GPU:-0}` | `GPU=${GPU:-0}` | `GPU=${GPU:-0}` | `GPU=0` |

### Per-model `--focal_alpha` values

| Model | `--focal_alpha` | Source |
|---|---|---|
| `flan_t5_xl` | `0.33` | [run_large_train_xl_no_ret_vs_ret_focal.sh L12: `FOCAL_ALPHA=${FOCAL_ALPHA:-0.33}`] |
| `flan_t5_xxl` | `0.36` | [run_large_train_xxl_no_ret_vs_ret_focal.sh L12: `FOCAL_ALPHA=${FOCAL_ALPHA:-0.36}`] |
| `gpt` | `0.71` | [run_large_train_gpt_no_ret_vs_ret_focal.sh L13: `FOCAL_ALPHA=${FOCAL_ALPHA:-0.71}`] |

### Key differences from IT3

| Aspect | IT4 (Focal) | IT3 (Weighted CE) |
|---|---|---|
| γ | `2.0` (genuine focal loss) | `0.0` (reduces to weighted CE) |
| Alpha source | Manually set `--focal_alpha` per model | Auto-computed `--auto_class_weights` from data |
| Alpha type | Scalar float → `[1-α, α]` tensor | Direct inverse-frequency tensor `[w_A, w_R]` |
| Epoch range (XL) | `15 20 25 30 35` (includes 15) | `20 25 30 35` (skips 15) |
| Shell strictness | XL/XXL: `set -euo pipefail`; GPT: tolerant | All three: tolerant |

---

## 4. Focal Loss Implementation

### 4.1 The focal loss formula

`FocalLoss` is a **custom `torch.nn.Module`** defined at [run_classifier.py L69–113], not imported from any external library.

The formula [run_classifier.py L69]:

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

where:
- $p_t$ is the model's predicted probability for the ground-truth class
- $\gamma = 2.0$ is the focusing parameter — down-weights easy (well-classified) examples
- $\alpha_t$ is the per-class weight for the ground-truth class

When $\gamma = 2.0$:
- If the model is confident on the correct class ($p_t \to 1$): $(1 - p_t)^2 \to 0$, loss $\to 0$
- If the model is uncertain ($p_t \approx 0.5$): $(1 - 0.5)^2 = 0.25$, loss is scaled to 25 % of CE
- If the model is wrong ($p_t \to 0$): $(1 - p_t)^2 \to 1$, loss approaches standard CE

This is **genuinely focal** (unlike IT3 where $\gamma = 0$ eliminated the focusing effect).

### 4.2 `FocalLoss.forward()` — step by step

| Step | Code | Line |
|---|---|---|
| 1. Softmax + clamp | `probs = torch.softmax(logits, dim=-1).clamp(min=1e-8)` | [L105] |
| 2. Select $p_t$ for ground-truth class | `p_t = probs[batch_idx, targets]` | [L107] |
| 3. Focal weight | `focal_weight = (1.0 - p_t) ** self.gamma` | [L108] |
| 4. Base CE | `loss = -focal_weight * torch.log(p_t)` | [L109] |
| 5. Class weight (if alpha set) | `alpha_t = self.alpha.to(logits.device)[targets]` | [L111] |
| 6. Apply class weight | `loss = alpha_t * loss` | [L112] |
| 7. Batch mean | `return loss.mean()` | [L113] |

### 4.3 How scalar `--focal_alpha` is converted to per-class weights

The `--focal_alpha` argument is a scalar float. In `main()` at [run_classifier.py L689–691]:

```python
_alpha_tensor = torch.tensor(
    [1.0 - args.focal_alpha, args.focal_alpha], dtype=torch.float
)
```

This tensor is passed to `FocalLoss(gamma=..., alpha=_alpha_tensor)` at [L697]. Since it's already a tensor, it enters `FocalLoss.__init__` via the `isinstance(alpha, (list, torch.Tensor))` branch at [L89–93], which stores it as-is.

### 4.4 How weights map to label indices

| Index | Label | Weight |
|---|---|---|
| 0 | A | `1 - focal_alpha` |
| 1 | R | `focal_alpha` |

This mapping is established by:
1. `args.labels = ["A", "R"]` (from `--labels A R`)
2. `label_token_ids = [tokenizer("A").input_ids[0], tokenizer("R").input_ids[0]]` [L696]
3. `class_indices` in `FocalLossTrainer.compute_loss()` maps token IDs to 0-based indices matching this order [L144–146]
4. `self.alpha[targets]` indexes into `[1-α, α]` using these class indices [L111]

### 4.5 Per-model weight table

| Model | A count | R count | Minority | α | A weight (1−α) | R weight (α) | Minority gets higher weight? |
|---|---|---|---|---|---|---|---|
| `flan_t5_xl` | 424 | 868 | A | 0.33 | **0.67** | 0.33 | **Yes** ✓ (A=0.67 > R=0.33) |
| `flan_t5_xxl` | 511 | 898 | A | 0.36 | **0.64** | 0.36 | **Yes** ✓ (A=0.64 > R=0.36) |
| `gpt` | 1 013 | 404 | R | 0.71 | 0.29 | **0.71** | **Yes** ✓ (R=0.71 > A=0.29) |

All three models correctly assign a higher weight to the minority class.

### 4.6 Alpha values approximate inverse class frequency

The `--focal_alpha` values are **manually set** in each shell script, not auto-computed. However, they closely approximate the normalized inverse-frequency weights:

| Model | α (set) | Ideal norm. R weight | Ratio |
|---|---|---|---|
| `flan_t5_xl` | 0.33 | 0.3282 | 1.006 |
| `flan_t5_xxl` | 0.36 | 0.3627 | 0.993 |
| `gpt` | 0.71 | 0.7149 | 0.993 |

The ideal normalized R weight is computed as:

$$\text{norm}\_R = \frac{N / (2 \cdot N_R)}{N / (2 \cdot N_A) + N / (2 \cdot N_R)} = \frac{N_A}{N_A + N_R}$$

which simplifies to the proportion of A samples (i.e. the minority proportion for XL/XXL, majority proportion for GPT). The manually set alpha values match this to within 0.7 %.

### 4.7 `FocalLossTrainer.compute_loss()` — where the loss is applied

`FocalLossTrainer` [run_classifier.py L118–148] is a subclass of HuggingFace's `Trainer` that overrides `compute_loss()`:

1. Extract decoder logits at position 0: `logits = outputs.logits[:, 0, :]` [L138]
2. Narrow to label columns: `label_logits = logits[:, tid]` where `tid` = label token IDs [L141]
3. Convert ground-truth vocab token IDs to 0-based class indices [L144–146]
4. Call `self.focal_loss_fn(label_logits, class_indices)` [L148]

This is the same code path as IT3. The only difference is the `FocalLoss` instance: IT4 has `gamma=2.0` (genuine focal) vs IT3's `gamma=0.0` (weighted CE).

### 4.8 Confirmation: FocalLoss is a custom `nn.Module`

`FocalLoss` is defined at [run_classifier.py L69–113] as a subclass of `torch.nn.Module`. It is not imported from any external library (no `focal_loss`, `torchvision`, or third-party dependency). The implementation is self-contained.

---

## 5. No Residual Undersampling or Class-Weighted CE

**No undersampling is applied in IT4.** None of the three focal-loss shell scripts call `make_no_ret_vs_ret_undersampled.py` or reference `train_undersampled.json`.

**IT4 is not class-weighted CE.** Unlike IT3 where `gamma=0.0` reduced focal loss to weighted CE, IT4 uses `gamma=2.0` — a genuinely focal loss that additionally down-weights easy examples. The α values are also set differently: IT4 uses per-model scalar `--focal_alpha` values, while IT3 uses auto-computed inverse-frequency tensors via `--auto_class_weights`.

| Property | IT4 (Focal) | IT3 (Weighted CE) | IT2 (Undersample) |
|---|---|---|---|
| Data modification | None | None | Remove 609 A samples |
| Loss modification | Focal (γ=2.0) + class weights | Weighted CE (γ=0.0) + auto weights | Standard CE |
| `--focal_gamma` | `2.0` | `0.0` | N/A |
| Class weight source | Manual `--focal_alpha` | Auto `--auto_class_weights` | N/A |

---

## 6. Data Pipeline

**Identical to Iteration 1.** The training files are the original, full-size silver-label files.

| Model | Training file | Samples | A | R |
|---|---|---|---|---|
| `flan_t5_xl` | `.../flan_t5_xl/silver/no_retrieval_vs_retrieval/train.json` | 1 292 | 424 (32.8 %) | 868 (67.2 %) |
| `flan_t5_xxl` | `.../flan_t5_xxl/silver/no_retrieval_vs_retrieval/train.json` | 1 409 | 511 (36.3 %) | 898 (63.7 %) |
| `gpt` | `.../gpt/silver/no_retrieval_vs_retrieval/train.json` | 1 417 | 1 013 (71.5 %) | 404 (28.5 %) |

Validation files and predict.json are unchanged from IT1. See DOCUMENTATION_IT1.md §4 for full details.

---

## 7. Gate 2 — Completely Unchanged

Gate 2 (Clf2: B vs C) is **identical to Iteration 1** for all three model variants. The focal-loss experiment changes only Gate 1.

| Property | XL | XXL | GPT |
|---|---|---|---|
| Script | `run_large_train_xl_single_vs_multi.sh` | `run_large_train_xxl_single_vs_multi.sh` | `run_large_train_gpt_single_vs_multi.sh` |
| Labels | `B C` | `B C` | `B C` |
| Training file | `binary_silver_single_vs_multi/train.json` | same pattern | same pattern |
| Train size | 3 268 (B=1 871, C=1 397) | 3 298 (B=1 903, C=1 395) | 2 804 (B=1 475, C=1 329) |
| Epochs | `15 20 25 30 35` | `15 20 25 30 35` | `35 40` |
| Seed | `42` | `42` | `42` |
| Loss | Standard CE (no focal loss) | Standard CE | Standard CE |

At routing time, IT4 Clf1 (focal) predictions and the standard Clf2 predictions are combined via `predict_complexity_split_classifiers.py` identically to IT1.

---

## 8. Evaluation Setup

**Identical to Iteration 1.** Specifically:

| Step | Procedure | Difference from IT1? |
|---|---|---|
| Per-epoch validation | `run_classifier.py --do_eval` on `valid.json` → accuracy + per-class accuracy | No |
| Per-epoch prediction | `run_classifier.py --do_eval` on `predict.json` → classification labels | No |
| Checkpoint selection | Shell script finds `checkpoint-*`, falls back to `TRAIN_OUTPUT_DIR` | In practice always falls back (see §10.1) |
| Cascade routing | `predict_complexity_split_classifiers.py` | No |
| QA evaluation | `evaluate_final_acc.py --pred_path ...` | No |

See DOCUMENTATION_IT1.md §6 for full evaluation details.

---

## 9. Output Artifacts

### 9.1 Directory tree

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/
  {model}/                                          # flan_t5_xl, flan_t5_xxl, or gpt
    no_ret_vs_ret_focal/                            ← SEPARATE from IT1/IT2/IT3
      epoch/
        {epoch}/                                    # 15..35 for xl/xxl; 35,40 for gpt
          {YYYY_MM_DD}/{HH_MM_SS}/
            config.json
            generation_config.json
            model.safetensors                       # or pytorch_model.bin
            spiece.model
            special_tokens_map.json
            tokenizer.json
            tokenizer_config.json
            training_args.bin                       # HF Trainer metadata
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

### 9.2 Separation from other iterations

| Iteration | Clf1 output path | Risk of overwrite? |
|---|---|---|
| IT1 | `.../no_ret_vs_ret/epoch/...` | No |
| IT2 | `.../no_ret_vs_ret_undersampled/epoch/...` | No |
| IT3 | `.../no_ret_vs_ret_weighted_ce/epoch/...` | No |
| **IT4** | **`.../no_ret_vs_ret_focal/epoch/...`** | **No** |

All four iterations use distinct directory names. Clf2 outputs share `.../single_vs_multi/epoch/...` but are separated by `{DATE}` timestamps.

---

## 10. Suspicious / Noteworthy Items

### 10.1 `save_strategy="no"` + vestigial checkpoint-scanning shell logic

| Issue | Detail |
|---|---|
| **What** | `TrainingArguments(save_strategy="no")` [run_classifier.py L710] prevents the HF Trainer from creating `checkpoint-*` directories. Yet all three IT4 shell scripts contain an elaborate block that scans for `checkpoint-*`, selects the latest, and deletes older ones. |
| **Shell comment** | The XL/XXL scripts include a misleading comment: `# FocalLossTrainer saves in checkpoint-* dirs; evaluate the latest checkpoint.` — This is **incorrect** with `save_strategy="no"`. |
| **Behaviour** | `ls -d checkpoint-*` matches nothing (stderr suppressed), `CKPT_PATH` falls back to `${TRAIN_OUTPUT_DIR}`, and the cleanup loop is a no-op. Validation/prediction correctly use `${TRAIN_OUTPUT_DIR}` where `focal_trainer.save_model()` wrote the model. |
| **Impact** | No functional problem. Dead code. |
| **Files** | [run_large_train_xl_no_ret_vs_ret_focal.sh L48–56], same in XXL and GPT. |

### 10.2 Different training code path from IT1

| Issue | Detail |
|---|---|
| **What** | IT4 uses `FocalLossTrainer` (HF `Trainer` subclass) while IT1 uses a manual Accelerate training loop. Same difference as IT3. |
| **Differences** | (a) HF Trainer's AdamW vs `torch.optim.AdamW`; (b) Loss computed on first decoder position only (2-class softmax) vs T5's built-in full-vocabulary CE; (c) Trainer handles gradient accumulation, scheduling internally. |
| **Impact** | Results are not attributable solely to the focal loss — the training loop itself differs from IT1. However, IT4 is directly comparable to IT3 since both use the same `FocalLossTrainer` path (only γ and α differ). |
| **Files** | IT4 path: [run_classifier.py L676–728]. IT1 path: [run_classifier.py L730–840]. |

### 10.3 Manually set alpha values vs auto-computed (IT3)

| Issue | Detail |
|---|---|
| **What** | IT4 uses `--focal_alpha` with manually chosen per-model values (0.33, 0.36, 0.71). IT3 uses `--auto_class_weights` which computes weights automatically from training data. |
| **Risk** | Manual values are fragile — if the training data changes (e.g. new silver labels), the alpha values become stale. The IT4 values are currently accurate to within 0.7 % of the ideal inverse-frequency weights, but this is not guaranteed to hold for future data. |
| **Overridable** | All three scripts use `FOCAL_ALPHA=${FOCAL_ALPHA:-default}`, allowing environment override (e.g. `FOCAL_ALPHA=0.40 bash run/...`). |

### 10.4 GPT error handling inconsistency vs XL/XXL

| Issue | Detail |
|---|---|
| **What** | The GPT focal script appends `\|\| echo "[WARN] ..."` to validation and prediction commands [run_large_train_gpt_no_ret_vs_ret_focal.sh L73, L85], tolerating failures. The XL and XXL scripts do **not** — they have no `\|\|` fallback. |
| **Combined with** | All three scripts use `set -euo pipefail` [L2]. For XL/XXL, a validation or prediction failure aborts the entire run. For GPT, it prints a warning and continues to the next epoch. |
| **Impact** | Inconsistent failure semantics across model variants. |
| **Files** | GPT: [L73, L85]. XL: [L68, L78] — no fallback. XXL: same as XL. |

### 10.5 No early stopping

| Issue | Detail |
|---|---|
| **What** | Same as IT1–IT3: no early stopping, no validation-based model selection. `eval_strategy="no"` [run_classifier.py L711] means the Trainer never evaluates during training. |
| **Impact** | The saved model is always the final-epoch model. With γ=2.0 focal loss, the training dynamics differ from standard CE — the loss landscape may have different convergence properties, making the lack of early stopping potentially more impactful. |

### 10.6 Fresh-from-scratch training per epoch value

| Issue | Detail |
|---|---|
| **What** | Same as all previous iterations — each epoch value trains from the base `t5-large` model. |
| **Cost** | XL: `15+20+25+30+35 = 125` epochs. XXL: same. GPT: `35+40 = 75` epochs. |

### 10.7 Gamma and alpha interact non-trivially

| Issue | Detail |
|---|---|
| **What** | Focal loss with both γ > 0 and α ≠ None applies **two** forms of reweighting simultaneously: (a) α gives static per-class weights (minority upweighting); (b) (1−p_t)^γ gives dynamic per-sample weights (hard-example upweighting). |
| **Implication** | The effective weight on a minority-class hard example is amplified by both factors. Conversely, a majority-class easy example is doubly down-weighted. This interaction makes the loss harder to interpret and tune compared to either mechanism alone (IT2 undersampling or IT3 weighted CE). |

### 10.8 Gamma and alpha are both overridable via environment

| Issue | Detail |
|---|---|
| **What** | All three scripts use `FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}` and `FOCAL_ALPHA=${FOCAL_ALPHA:-default}`, allowing both to be overridden from the environment. |
| **Risk** | A user could unknowingly run with non-default values if `FOCAL_GAMMA` or `FOCAL_ALPHA` are set in their shell environment. This is a convenience feature but could cause confusing results if forgotten. |

### 10.9 Validation set imbalance unchanged

| Issue | Detail |
|---|---|
| **What** | Same as IT1–IT3: validation files are not reweighted or resampled. The model is trained with focal loss that compensates for training imbalance, but validation accuracy is computed on the original distribution. |
| **Validation distributions** | XL: 439 A / 911 R (32.5 % / 67.5 %). XXL: 519 A / 896 R (36.7 % / 63.3 %). GPT: 1 038 A / 393 R (72.5 % / 27.5 %). |
