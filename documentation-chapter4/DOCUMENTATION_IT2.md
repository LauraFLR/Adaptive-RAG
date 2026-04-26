# Iteration 2 — Random Undersampling for Gate 1 (GPT only)

> **Design Science Research Artifact:** Apply random undersampling to the
> majority class in Clf1 (A vs R) training data to address the severe class
> imbalance in the GPT silver labels (A ≈ 71 %, R ≈ 29 %).  Gate 2 (Clf2)
> remains identical to Iteration 1.

---

## 1. Files Involved

| File | Role |
|---|---|
| `classifier/data_utils/make_no_ret_vs_ret_undersampled.py` | **New in IT2.** Offline script that reads the original Clf1 training JSON, undersamples the majority class to match the minority class size, and writes `train_undersampled.json`. |
| `classifier/run/run_large_train_gpt_no_ret_vs_ret_undersampled.sh` | **New in IT2.** Shell launcher — Clf1 (A vs R) for GPT, using `train_undersampled.json` as training data. Calls the undersampling script before entering the training loop. |
| `classifier/run_classifier.py` | Shared — **identical** to IT1. No code changes. |
| `classifier/utils.py` | Shared — **identical** to IT1. |
| `classifier/run/run_large_train_gpt_single_vs_multi.sh` | Gate 2 — **identical** to IT1 (B vs C, GPT, binary_silver training data). |
| `classifier/postprocess/predict_complexity_split_classifiers.py` | Cascade routing — **identical** to IT1. |
| `evaluate_final_acc.py` | QA evaluation — **identical** to IT1. |
| `classifier/data/.../gpt/silver/no_retrieval_vs_retrieval/train.json` | Input to undersampling script (original 1 417-sample Clf1 training set). |
| `classifier/data/.../gpt/silver/no_retrieval_vs_retrieval/train_undersampled.json` | Output of undersampling script (balanced 808-sample Clf1 training set). |
| `classifier/data/.../gpt/silver/no_retrieval_vs_retrieval/valid.json` | Clf1 validation — **unchanged** from IT1. |
| `classifier/data/.../gpt/binary_silver_single_vs_multi/train.json` | Clf2 training — **unchanged** from IT1. |
| `classifier/data/.../gpt/silver/single_vs_multi/valid.json` | Clf2 validation — **unchanged** from IT1. |
| `classifier/data/.../predict.json` | Test set — **unchanged** from IT1. |

**Scope limitation:** No undersampling scripts or shell launchers exist for `flan_t5_xl` or `flan_t5_xxl`. The `make_no_ret_vs_ret_undersampled.py` script accepts `--model flan_t5_xl` and `--model flan_t5_xxl` as valid choices, but no corresponding `run_large_train_xl_no_ret_vs_ret_undersampled.sh` or `run_large_train_xxl_no_ret_vs_ret_undersampled.sh` scripts exist. Iteration 2 applies to GPT only.

---

## 2. Model Architecture

**Identical to Iteration 1.** T5-Large (770 M parameters), `AutoModelForSeq2SeqLM`, generative decoding with constrained softmax over label token IDs. Clf1 labels: `A R`. Clf2 labels: `B C`. See DOCUMENTATION_IT1.md §2 for full details.

---

## 3. Training Parameters

| Parameter | Iteration 2 value | Iteration 1 value | Difference? |
|---|---|---|---|
| Base model | `t5-large` | `t5-large` | No |
| Learning rate | `3e-5` | `3e-5` | No |
| Per-device train batch size | `32` | `32` | No |
| Per-device eval batch size | `100` | `100` | No |
| Max sequence length | `384` | `384` | No |
| Doc stride | `128` | `128` | No |
| Weight decay | `0.0` (default) | `0.0` (default) | No |
| Gradient accumulation steps | `1` (default) | `1` (default) | No |
| Warmup steps | `0` (default) | `0` (default) | No |
| Optimizer | AdamW (hardcoded) | AdamW (hardcoded) | No |
| LR scheduler | `linear` (default) | `linear` (default) | No |
| Seed | `42` | `42` | No |
| Epochs | `35, 40` | `35, 40` | No |
| Labels | `A R` | `A R` | No |
| Loss function | Standard CE | Standard CE | No |
| `--use_focal_loss` | Not passed | Not passed | No |
| `--auto_class_weights` | Not passed | Not passed | No |
| **Training file** | **`train_undersampled.json`** | **`train.json`** | **YES** |
| Validation file | `valid.json` | `valid.json` | No |
| Predict file | `predict.json` | `predict.json` | No |
| GPU handling | `GPU=${GPU:-0}` (overridable) | `GPU=0` (hardcoded) | Minor shell difference |
| Error handling | `set -euo pipefail` | None | Minor shell difference |
| Output subdir | `no_ret_vs_ret_undersampled/` | `no_ret_vs_ret/` | Directory name only |

**The only substantive difference from Iteration 1 is the training data file.** All hyperparameters, model architecture, loss function, evaluation procedure, and code paths are identical.

---

## 4. Random Undersampling Implementation

### 4.1 Overview

Undersampling is performed **offline, before training starts**, as a static data preprocessing step. The shell script `run_large_train_gpt_no_ret_vs_ret_undersampled.sh` calls the undersampling script at line 11 before entering the `for EPOCH in ...` loop:

```bash
python ./data_utils/make_no_ret_vs_ret_undersampled.py --model ${LLM_NAME}
```

This writes `train_undersampled.json` to disk; the training loop then reads that file via `--train_file`.

### 4.2 Algorithm — step by step

| Step | Code | Location |
|---|---|---|
| 1. Parse `--model gpt --seed 42` | `parse_args()` | [make_no_ret_vs_ret_undersampled.py L8–28] |
| 2. Resolve input path to `.../gpt/silver/no_retrieval_vs_retrieval/train.json` and output to `train_undersampled.json` | `default_paths(model)` | [make_no_ret_vs_ret_undersampled.py L31–34] |
| 3. Load full training data as a Python list of dicts | `json.load(f)` | [make_no_ret_vs_ret_undersampled.py L42–43] |
| 4. Partition into `by_label["A"]` and `by_label["R"]` buckets | for-loop over items | [make_no_ret_vs_ret_undersampled.py L45–49] |
| 5. Compute `minority_size = min(len(A), len(R))` | `min()` | [make_no_ret_vs_ret_undersampled.py L52] |
| 6. Seed an independent RNG: `rng = random.Random(42)` | `random.Random(args.seed)` | [make_no_ret_vs_ret_undersampled.py L53] |
| 7. Start balanced set with **all R samples** (full copy) | `balanced = list(by_label["R"])` | [make_no_ret_vs_ret_undersampled.py L55] |
| 8. Sample `minority_size` A samples **without replacement** | `rng.sample(by_label["A"], minority_size)` | [make_no_ret_vs_ret_undersampled.py L56] |
| 9. Shuffle the combined list | `rng.shuffle(balanced)` | [make_no_ret_vs_ret_undersampled.py L57] |
| 10. Write to `train_undersampled.json` | `json.dump(balanced, f, indent=4)` | [make_no_ret_vs_ret_undersampled.py L59–60] |

### 4.3 Target ratio

**1:1 (perfect balance).** `minority_size` A samples + all R samples → equal counts.

### 4.4 Concrete numbers (GPT model)

| Metric | Value |
|---|---|
| Input file | `train.json` |
| Input total | 1 417 samples |
| Input A count | 1 013 (71.5 %) |
| Input R count | 404 (28.5 %) |
| Minority class | R (404) |
| Minority size | 404 |
| Output file | `train_undersampled.json` |
| Output total | 808 samples |
| Output A count | 404 (50.0 %) |
| Output R count | 404 (50.0 %) |
| **Samples discarded** | **609 A samples (60.1 % of the A class; 43.0 % of total data)** |

### 4.5 Reproducibility

| Seed aspect | Value | Notes |
|---|---|---|
| Undersampling seed | `42` | Default value of `--seed` [make_no_ret_vs_ret_undersampled.py L14]. Passed to `random.Random(42)` — an **independent** RNG instance, not the global `random` module state. |
| Training seed | `42` | Passed as `--seed 42` in the shell script. Applied via `accelerate.utils.set_seed(42)` inside `run_classifier.py`. |
| Relationship | **Independent** | The undersampling RNG is created and consumed in a separate Python process that exits before training begins. The two seeds happen to share the same value `42` but are applied to different RNGs in different processes. |

### 4.6 Implicit assumption in the code

The algorithm at line 55 starts with `balanced = list(by_label["R"])` (all R samples), then samples from A. This **assumes R is the minority class**. For GPT data this is correct (R = 404 < A = 1 013). However, for `flan_t5_xl` and `flan_t5_xxl` where R is the **majority** class, the code would still work correctly: `minority_size = min(A, R)` would be A's count, `balanced` would start with all R, then sample `minority_size` from A — but then R would still be unsampled at full size while A is fully included. The resulting set would be `(all R) + (all A)` = original data, since both A and R contribute their full counts when A ≤ R. Actually: `balanced = list(by_label["R"])` takes all 868 R, then `rng.sample(by_label["A"], 424)` takes all 424 A → total 1 292 = original size. So for XL the script would produce the full dataset unmodified, because the minority class (A = 424) gets fully sampled and the majority class (R = 868) is taken in full. This is a latent defect: R should also be downsampled to `minority_size`. In practice it doesn't matter because no XL/XXL undersampling shell script exists.

---

## 5. Data Pipeline

### 5.1 Clf1 training data comparison (GPT)

| Aspect | Iteration 1 | Iteration 2 |
|---|---|---|
| **Training file** | `.../gpt/silver/no_retrieval_vs_retrieval/train.json` | `.../gpt/silver/no_retrieval_vs_retrieval/train_undersampled.json` |
| **Total samples** | 1 417 | 808 |
| **A count (%)** | 1 013 (71.5 %) | 404 (50.0 %) |
| **R count (%)** | 404 (28.5 %) | 404 (50.0 %) |
| **A:R ratio** | 2.51:1 | 1:1 |
| **Inductive-bias labels?** | No (silver only) | No (silver only) |
| **Data reduction** | — | 609 samples removed (43.0 %) |
| Validation file | `.../gpt/silver/no_retrieval_vs_retrieval/valid.json` (1 431 samples) | **Same** |
| Validation A:R | 1 038 A / 393 R (72.5 % / 27.5 %) | **Same** |
| Predict file | `.../predict.json` (3 000 samples) | **Same** |
| Tokenisation | Identical pipeline (§4.4 of DOCUMENTATION_IT1.md) | **Same** |

### 5.2 Training steps per epoch

| Metric | Iteration 1 | Iteration 2 |
|---|---|---|
| Training samples | 1 417 | 808 |
| Batch size | 32 | 32 |
| Steps per epoch | ⌈1 417 / 32⌉ = 45 | ⌈808 / 32⌉ = 26 |
| Total steps (epoch 35) | 1 575 | 910 |
| Total steps (epoch 40) | 1 800 | 1 040 |

---

## 6. Gate 2 — Completely Unchanged

Gate 2 (Clf2: B vs C) is **identical to Iteration 1**. The undersampling experiment changes only Gate 1.

| Property | Value |
|---|---|
| Script | `classifier/run/run_large_train_gpt_single_vs_multi.sh` |
| Labels | `B C` |
| Training file | `.../gpt/binary_silver_single_vs_multi/train.json` (2 804 samples: 1 475 B / 1 329 C) |
| Validation file | `.../gpt/silver/single_vs_multi/valid.json` (393 samples: 272 B / 121 C) |
| Epochs | `35, 40` |
| All hyperparameters | Identical to IT1 GPT Clf2 (see DOCUMENTATION_IT1.md §3) |
| Seed | `42` |
| Loss | Standard CE (no focal loss) |
| Output subdir | `gpt/single_vs_multi/epoch/{35,40}/{date}/` |

At routing time, the IT2 Clf1 (undersampled) predictions and the standard Clf2 predictions are combined via `predict_complexity_split_classifiers.py`:
- Clf1 predicts A → final label A
- Clf1 predicts R → use Clf2's prediction (B or C)

The routing script and evaluation pipeline are identical to IT1. The only operational difference is that the Clf1 `dict_id_pred_results.json` comes from the `no_ret_vs_ret_undersampled/` output directory instead of `no_ret_vs_ret/`.

---

## 7. Evaluation Setup

**Identical to Iteration 1.** Specifically:

| Step | Procedure | Difference from IT1? |
|---|---|---|
| Per-epoch validation | `run_classifier.py --do_eval` on `valid.json` → accuracy + per-class accuracy | No |
| Per-epoch prediction | `run_classifier.py --do_eval` on `predict.json` → classification labels | No |
| Cascade routing | `predict_complexity_split_classifiers.py` merging Clf1 + Clf2 predictions | No |
| QA evaluation | `evaluate_final_acc.py --pred_path ...` computing EM/F1/acc per dataset | No |
| Accuracy function | `calculate_accuracy()` [utils.py L237] | No |
| Per-class accuracy | `calculate_accuracy_perClass()` [utils.py L244] | No |
| Official evaluators | HotpotQA, 2WikiMultiHop, MuSiQue | No |
| SquadAnswerEmF1 | nq, trivia, squad | No |

See DOCUMENTATION_IT1.md §6 for full evaluation details.

---

## 8. Output Artifacts

### 8.1 Clf1 output directory tree (undersampled)

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/
  gpt/
    no_ret_vs_ret_undersampled/               ← SEPARATE from IT1's no_ret_vs_ret/
      epoch/
        35/
          {YYYY_MM_DD}/{HH_MM_SS}/
            config.json
            generation_config.json
            pytorch_model.bin
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
              final_eval_results.json
              final_eval_results_perClass.json
              logs.log
        40/
          {YYYY_MM_DD}/{HH_MM_SS}/
            [same structure as 35]
```

### 8.2 Separation from IT1

| Classifier | IT1 output path | IT2 output path | Risk of overwrite? |
|---|---|---|---|
| Clf1 (GPT) | `.../gpt/no_ret_vs_ret/epoch/...` | `.../gpt/no_ret_vs_ret_undersampled/epoch/...` | **No** — different directory |
| Clf2 (GPT) | `.../gpt/single_vs_multi/epoch/...` | `.../gpt/single_vs_multi/epoch/...` | **Shared** — but timestamped `{DATE}` subdirectories prevent overwrite unless run at the exact same second |

### 8.3 Generated data file

| File | Location | Overwrites on re-run? |
|---|---|---|
| `train_undersampled.json` | `classifier/data/.../gpt/silver/no_retrieval_vs_retrieval/train_undersampled.json` | **Yes** — the shell script calls the undersampling script unconditionally before every training run [run_large_train_gpt_no_ret_vs_ret_undersampled.sh L11]. The file is silently overwritten. |

---

## 9. Suspicious / Noteworthy Items

### 9.1 Aggressive data discard

| Issue | Detail |
|---|---|
| **What** | 609 of 1 417 samples (43.0 %) are discarded from Clf1 training. |
| **Risk** | The model sees only 808 training samples. With batch size 32, that is just 26 steps per epoch. Information from 60 % of the A-class samples is permanently lost. |
| **File** | [make_no_ret_vs_ret_undersampled.py L52–57] |
| **Mitigation** | None — no oversampling, SMOTE, or weighted loss is combined with the undersampling. |

### 9.2 High epoch count on small dataset

| Issue | Detail |
|---|---|
| **What** | 35–40 epochs on 808 samples (26 steps/epoch). The model sees every training sample ~35–40 times. |
| **Risk** | Severe overfitting risk. With no early stopping, no validation-based selection, and the model overwrite-per-epoch behaviour (§8.1 of DOCUMENTATION_IT1.md), the saved model is always the epoch-40 model regardless of when peak performance occurred. |
| **File** | [run_large_train_gpt_no_ret_vs_ret_undersampled.sh L13: `for EPOCH in 35 40`] |

### 9.3 Undersampling re-run behaviour

| Issue | Detail |
|---|---|
| **What** | The undersampling script is called **every time** the shell script runs [L11], unconditionally. It overwrites `train_undersampled.json` on disk. |
| **Risk** | Since the undersampling seed is fixed at `42` and the input file is deterministic, the output is reproducible across re-runs. However, if someone manually edits `train.json` or the undersampling script between runs, the generated file changes silently. There is no checksum or staleness check. |
| **File** | [run_large_train_gpt_no_ret_vs_ret_undersampled.sh L11] |

### 9.4 GPT-only scope

| Issue | Detail |
|---|---|
| **What** | No `run_large_train_xl_no_ret_vs_ret_undersampled.sh` or `run_large_train_xxl_no_ret_vs_ret_undersampled.sh` exists. |
| **Risk** | Results cannot be compared across model variants. This is likely intentional — XL and XXL have R as the majority class, so the same A-discarding approach would make less sense (it would discard R samples instead). |
| **Note** | The Python script `make_no_ret_vs_ret_undersampled.py` accepts `flan_t5_xl` and `flan_t5_xxl` as valid `--model` values, but the shell launcher to invoke them does not exist. |

### 9.5 Latent code defect: majority class always taken in full

| Issue | Detail |
|---|---|
| **What** | `balanced = list(by_label["R"])` always takes **all** R samples, then `rng.sample(by_label["A"], minority_size)` samples from A. The code assumes R is always the smaller bucket. |
| **Risk** | For XL data (A=424, R=868): `minority_size = 424`, `balanced` = all 868 R + 424 sampled A = 1 292. This equals the original dataset — no undersampling occurs. For XXL (A=511, R=898): same issue — result would be 898 R + 511 A = 1 409 = original. The script silently produces the full dataset when R > A. |
| **Correct fix** | Should be: undersample whichever class is larger to `minority_size`, keeping the smaller class intact. |
| **Impact on IT2** | **None** — the script is only invoked for GPT where R (404) < A (1 013), so the defect is not triggered. |
| **File** | [make_no_ret_vs_ret_undersampled.py L55–56] |

### 9.6 Shell strictness difference from IT1

| Issue | Detail |
|---|---|
| **What** | The IT2 script uses `set -euo pipefail` [L2] and `GPU=${GPU:-0}` [L8]. The IT1 GPT Clf1 script has neither. |
| **Risk** | `set -euo pipefail` means any non-zero exit code (including from the undersampling script or `mkdir -p`) will abort the entire run. IT1 would silently continue past failures. This is **safer** behaviour in IT2 but means the two scripts have different failure semantics. |
| **File** | [run_large_train_gpt_no_ret_vs_ret_undersampled.sh L2] vs [run_large_train_gpt_no_ret_vs_ret.sh — no equivalent line] |

### 9.7 Validation set remains imbalanced

| Issue | Detail |
|---|---|
| **What** | The validation file `valid.json` is unchanged: 1 038 A / 393 R (72.5 % / 27.5 %). The model is trained on 50/50 data but validated on 72/28 data. |
| **Risk** | Overall validation accuracy will be dominated by A-class performance. A model that learns to predict more R (the intended goal of undersampling) may show lower overall accuracy even if recall on R improves. Per-class accuracy in `final_eval_results_perClass.json` should be used instead of overall accuracy for evaluation. |
| **File** | [run_large_train_gpt_no_ret_vs_ret_undersampled.sh L42: `--validation_file .../valid.json`] |

### 9.8 No Clf2-side imbalance handling

| Issue | Detail |
|---|---|
| **What** | IT2 addresses Clf1 imbalance only. Clf2 (GPT) training data is 53 % B / 47 % C (relatively balanced), so this is less concerning. However, the Clf2 validation set is 69 % B / 31 % C, creating a similar train/validation distribution mismatch to the one documented in DOCUMENTATION_IT1.md §8.9. |
| **Impact** | Low for IT2 specifically — Clf2 is out of scope for this iteration. |
