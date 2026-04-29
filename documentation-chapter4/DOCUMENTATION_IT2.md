# Iteration 2 — Random Undersampling for Gate 1

> **Design Science Research Artifact:** Apply random undersampling to the
> majority class in Clf1 (A vs R) training data to address class imbalance
> in the silver labels.  The undersampling script is run for all three model
> variants (flan_t5_xl, flan_t5_xxl, GPT).  Gate 2 (Clf2) remains identical
> to Iteration 1.

---

## 1. Files Involved

| File | Role |
|---|---|
| `classifier/data_utils/make_no_ret_vs_ret_undersampled.py` | **New in IT2.** Offline script that reads the original Clf1 training JSON, undersamples the majority class to match the minority class size, and writes `train_undersampled.json`. Accepts `--model {flan_t5_xl, flan_t5_xxl, gpt}`. |
| `classifier/run/run_large_train_gpt_no_ret_vs_ret_undersampled.sh` | **New in IT2.** Shell launcher — Clf1 (A vs R) for GPT, using `train_undersampled.json`. Epochs: `35, 40`. |
| `classifier/run/run_large_train_xl_no_ret_vs_ret_undersampled.sh` | **New in IT2.** Shell launcher — Clf1 (A vs R) for flan_t5_xl, using `train_undersampled.json`. Epochs: `15, 20, 25, 30, 35`. |
| `classifier/run/run_large_train_xxl_no_ret_vs_ret_undersampled.sh` | **New in IT2.** Shell launcher — Clf1 (A vs R) for flan_t5_xxl, using `train_undersampled.json`. Epochs: `15, 20, 25, 30, 35`. |
| `classifier/run_classifier.py` | Shared — **identical** to IT1. No code changes. |
| `classifier/utils.py` | Shared — **identical** to IT1. |
| `classifier/run/run_large_train_{xl,xxl,gpt}_single_vs_multi.sh` | Gate 2 — **identical** to IT1 (B vs C, binary_silver training data). |
| `classifier/postprocess/predict_complexity_split_classifiers.py` | Cascade routing — **identical** to IT1. |
| `evaluate_final_acc.py` | QA evaluation — **identical** to IT1. |
| `run-all-iterations.sh` | Top-level orchestrator — loops over all three models, trains undersampled Clf1 (IT2 section), selects best epoch via `find_best_epoch()`, routes predictions via `route_split()`, and evaluates via `evaluate_final_acc.py`. |
| `classifier/postprocess/postprocess_utils.py` | Shared helpers: `load_json()`, `save_json()`, `save_prediction_with_classified_label()`. |
| `classifier/data/.../{model}/silver/no_retrieval_vs_retrieval/train.json` | Input to undersampling script (original Clf1 training set, per model). |
| `classifier/data/.../{model}/silver/no_retrieval_vs_retrieval/train_undersampled.json` | Output of undersampling script (per model). |
| `classifier/data/.../{model}/silver/no_retrieval_vs_retrieval/valid.json` | Clf1 validation — **unchanged** from IT1. |
| `classifier/data/.../{model}/binary_silver_single_vs_multi/train.json` | Clf2 training — **unchanged** from IT1. |
| `classifier/data/.../{model}/silver/single_vs_multi/valid.json` | Clf2 validation — **unchanged** from IT1. |
| `classifier/data/.../predict.json` | Test set — **unchanged** from IT1. |

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
| Epochs (GPT) | `35, 40` | `35, 40` | No |
| Epochs (XL, XXL) | `15, 20, 25, 30, 35` | `15, 20, 25, 30, 35` | No |
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

Undersampling is performed **offline, before training starts**, as a static data preprocessing step. Each model's shell script (`run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh`) calls the undersampling script before entering the `for EPOCH in ...` loop:

```bash
python ./data_utils/make_no_ret_vs_ret_undersampled.py --model ${LLM_NAME}
```

This writes `train_undersampled.json` to disk; the training loop then reads that file via `--train_file`.

### 4.2 Algorithm — step by step

| Step | Code | Location |
|---|---|---|
| 1. Parse `--model {flan_t5_xl,flan_t5_xxl,gpt} --seed 42` | `parse_args()` | [make_no_ret_vs_ret_undersampled.py L8–26] |
| 2. Resolve input path to `.../{model}/silver/no_retrieval_vs_retrieval/train.json` and output to `train_undersampled.json` | `default_paths(model)` | [make_no_ret_vs_ret_undersampled.py L29–32] |
| 3. Load full training data as a Python list of dicts | `json.load(f)` | [make_no_ret_vs_ret_undersampled.py L43–44] |
| 4. Partition into `by_label["A"]` and `by_label["R"]` buckets | for-loop over items | [make_no_ret_vs_ret_undersampled.py L46–51] |
| 5. Compute `minority_size = min(len(A), len(R))` | `min()` | [make_no_ret_vs_ret_undersampled.py L53] |
| 6. Seed an independent RNG: `rng = random.Random(42)` | `random.Random(args.seed)` | [make_no_ret_vs_ret_undersampled.py L54] |
| 7. Identify minority class and keep all its samples | `if len(A) <= len(R): balanced = list(by_label["A"]) else: balanced = list(by_label["R"])` | [make_no_ret_vs_ret_undersampled.py L56–61] |
| 8. Sample `minority_size` from the **majority** class without replacement | `rng.sample(by_label[majority], minority_size)` | [make_no_ret_vs_ret_undersampled.py L57–62] |
| 9. Shuffle the combined list | `rng.shuffle(balanced)` | [make_no_ret_vs_ret_undersampled.py L63] |
| 10. Write to `train_undersampled.json` | `json.dump(balanced, f, indent=4)` | [make_no_ret_vs_ret_undersampled.py L66–67] |

### 4.3 Target ratio

**1:1 (perfect balance).** The script keeps all samples from the minority class and undersamples the majority class to `minority_size = min(|A|, |R|)`. This produces equal counts for all three models regardless of which class is larger.

### 4.4 Concrete numbers (all models)

| Metric | flan_t5_xl | flan_t5_xxl | GPT |
|---|---|---|---|
| Input total | 1 292 | 1 409 | 1 417 |
| Input A count (%) | 424 (32.8 %) | 511 (36.3 %) | 1 013 (71.5 %) |
| Input R count (%) | 868 (67.2 %) | 898 (63.7 %) | 404 (28.5 %) |
| Majority class | R | R | A |
| Minority class | A (424) | A (511) | R (404) |
| Output total | **848** | **1 022** | **808** |
| Output A count (%) | 424 (50.0 %) | 511 (50.0 %) | 404 (50.0 %) |
| Output R count (%) | 424 (50.0 %) | 511 (50.0 %) | 404 (50.0 %) |
| Samples discarded | **444 (34.4 %)** | **387 (27.5 %)** | **609 (43.0 %)** |
| Effective balancing? | **Yes** — 1:1 | **Yes** — 1:1 | **Yes** — 1:1 |

For XL and XXL where R is the majority class, the script keeps all A samples (minority) and undersamples R to `minority_size`. For GPT where A is the majority, it keeps all R samples and undersamples A.

### 4.5 Reproducibility

| Seed aspect | Value | Notes |
|---|---|---|
| Undersampling seed | `42` | Default value of `--seed` [make_no_ret_vs_ret_undersampled.py L13]. Passed to `random.Random(42)` — an **independent** RNG instance, not the global `random` module state. |
| Training seed | `42` | Passed as `--seed 42` in the shell script. Applied via `accelerate.utils.set_seed(42)` inside `run_classifier.py`. |
| Relationship | **Independent** | The undersampling RNG is created and consumed in a separate Python process that exits before training begins. The two seeds happen to share the same value `42` but are applied to different RNGs in different processes. |

### 4.6 Majority-class detection

The algorithm determines which class is larger via `len(by_label["A"]) <= len(by_label["R"])`. If A is the minority (XL, XXL), it keeps all A and undersamples R. If R is the minority (GPT), it keeps all R and undersamples A. This ensures 1:1 balancing regardless of which class dominates.

- **XL:** keeps all 424 A, samples 424 from 868 R → total 848.
- **XXL:** keeps all 511 A, samples 511 from 898 R → total 1 022.
- **GPT:** keeps all 404 R, samples 404 from 1 013 A → total 808.

---

## 5. Data Pipeline

### 5.1 Clf1 training data comparison

**GPT (undersampling effective):**

| Aspect | Iteration 1 | Iteration 2 |
|---|---|---|
| **Training file** | `.../gpt/silver/no_retrieval_vs_retrieval/train.json` | `.../gpt/silver/no_retrieval_vs_retrieval/train_undersampled.json` |
| **Total samples** | 1 417 | 808 |
| **A count (%)** | 1 013 (71.5 %) | 404 (50.0 %) |
| **R count (%)** | 404 (28.5 %) | 404 (50.0 %) |
| **A:R ratio** | 2.51:1 | 1:1 |
| **Data reduction** | — | 609 samples removed (43.0 %) |

**flan_t5_xl:**

| Aspect | Iteration 1 | Iteration 2 |
|---|---|---|
| **Training file** | `.../flan_t5_xl/silver/.../train.json` | `.../flan_t5_xl/silver/.../train_undersampled.json` |
| **Total samples** | 1 292 | 848 |
| **A count (%)** | 424 (32.8 %) | 424 (50.0 %) |
| **R count (%)** | 868 (67.2 %) | 424 (50.0 %) |
| **A:R ratio** | 1:2.05 | 1:1 |
| **Data reduction** | — | 444 samples removed (34.4 %) |

**flan_t5_xxl:**

| Aspect | Iteration 1 | Iteration 2 |
|---|---|---|
| **Training file** | `.../flan_t5_xxl/silver/.../train.json` | `.../flan_t5_xxl/silver/.../train_undersampled.json` |
| **Total samples** | 1 409 | 1 022 |
| **A count (%)** | 511 (36.3 %) | 511 (50.0 %) |
| **R count (%)** | 898 (63.7 %) | 511 (50.0 %) |
| **A:R ratio** | 1:1.76 | 1:1 |
| **Data reduction** | — | 387 samples removed (27.5 %) |

**Common to all models:**

| Aspect | Value |
|---|---|
| Inductive-bias labels? | No (silver only) |
| Validation file | `.../{model}/silver/no_retrieval_vs_retrieval/valid.json` — **unchanged** from IT1 |
| Predict file | `.../predict.json` (3 000 samples) — **unchanged** from IT1 |
| Tokenisation | Identical pipeline (§4.4 of DOCUMENTATION_IT1.md) |

### 5.2 Training steps per epoch

**GPT (epochs 35, 40):**

| Metric | Iteration 1 | Iteration 2 |
|---|---|---|
| Training samples | 1 417 | 808 |
| Batch size | 32 | 32 |
| Steps per epoch | ⌈1 417 / 32⌉ = 45 | ⌈808 / 32⌉ = 26 |
| Total steps (epoch 35) | 1 575 | 910 |
| Total steps (epoch 40) | 1 800 | 1 040 |

**flan_t5_xl (epochs 15, 20, 25, 30, 35):**

| Metric | Iteration 1 | Iteration 2 |
|---|---|---|
| Training samples | 1 292 | 848 |
| Batch size | 32 | 32 |
| Steps per epoch | ⌈1 292 / 32⌉ = 41 | ⌈848 / 32⌉ = 27 |
| Total steps (epoch 15) | 615 | 405 |
| Total steps (epoch 35) | 1 435 | 945 |

**flan_t5_xxl (epochs 15, 20, 25, 30, 35):**

| Metric | Iteration 1 | Iteration 2 |
|---|---|---|
| Training samples | 1 409 | 1 022 |
| Batch size | 32 | 32 |
| Steps per epoch | ⌈1 409 / 32⌉ = 45 | ⌈1 022 / 32⌉ = 32 |
| Total steps (epoch 15) | 675 | 480 |
| Total steps (epoch 35) | 1 575 | 1 120 |

---

## 6. Gate 2 — Completely Unchanged

Gate 2 (Clf2: B vs C) is **identical to Iteration 1** for all three models. The undersampling experiment changes only Gate 1. The standard Clf2 best-epoch outputs (from Phase 0 / IT1) are reused.

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
| Accuracy function | `calculate_accuracy()` [utils.py L231] | No |
| Per-class accuracy | `calculate_accuracy_perClass()` [utils.py L240] | No |
| Official evaluators | HotpotQA, 2WikiMultiHop, MuSiQue | No |
| SquadAnswerEmF1 | nq, trivia, squad | No |

See DOCUMENTATION_IT1.md §6 for full evaluation details.

---

## 8. Output Artifacts

### 8.1 Clf1 output directory tree (undersampled)

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/
  {model}/                                    ← flan_t5_xl, flan_t5_xxl, or gpt
    no_ret_vs_ret_undersampled/               ← SEPARATE from IT1's no_ret_vs_ret/
      epoch/
        {EPOCH}/                              ← 15,20,25,30,35 for xl/xxl; 35,40 for gpt
          {YYYY_MM_DD}/{HH_MM_SS}/
            config.json
            generation_config.json
            model.safetensors
            tokenizer_config.json
            valid/
              dict_id_pred_results.json
              final_eval_results.json
              final_eval_results_perClass.json
            predict/
              dict_id_pred_results.json
              final_eval_results.json
              final_eval_results_perClass.json
```

### 8.2 Separation from IT1

| Classifier | IT1 output path | IT2 output path | Risk of overwrite? |
|---|---|---|---|
| Clf1 (all models) | `.../{model}/no_ret_vs_ret/epoch/...` | `.../{model}/no_ret_vs_ret_undersampled/epoch/...` | **No** — different directory |
| Clf2 (all models) | `.../{model}/single_vs_multi/epoch/...` | `.../{model}/single_vs_multi/epoch/...` | **Shared** — but IT2 reuses the Clf2 from Phase 0; no new Clf2 training occurs |

### 8.3 Generated data files

| File | Location | Overwrites on re-run? |
|---|---|---|
| `train_undersampled.json` (xl) | `classifier/data/.../flan_t5_xl/silver/no_retrieval_vs_retrieval/train_undersampled.json` | **Yes** — regenerated unconditionally before each training run. |
| `train_undersampled.json` (xxl) | `classifier/data/.../flan_t5_xxl/silver/no_retrieval_vs_retrieval/train_undersampled.json` | **Yes** — regenerated unconditionally before each training run. |
| `train_undersampled.json` (gpt) | `classifier/data/.../gpt/silver/no_retrieval_vs_retrieval/train_undersampled.json` | **Yes** — regenerated unconditionally before each training run. |

---

## 9. Suspicious / Noteworthy Items

### 9.1 Aggressive data discard (all models)

| Issue | Detail |
|---|---|
| **What** | All three models discard majority-class samples: XL discards 444 R samples (34.4 %), XXL discards 387 R samples (27.5 %), GPT discards 609 A samples (43.0 %). |
| **Risk** | Reduced training set sizes (XL: 848, XXL: 1 022, GPT: 808). With batch size 32, steps per epoch are 27 (XL), 32 (XXL), 26 (GPT). Information from the majority-class samples is permanently lost. |
| **File** | [make_no_ret_vs_ret_undersampled.py L53–63] |
| **Mitigation** | None — no oversampling, SMOTE, or weighted loss is combined with the undersampling. |

### 9.2 High epoch count on reduced datasets

| Issue | Detail |
|---|---|
| **What** | All models train on reduced datasets: XL has 848 samples (27 steps/epoch, epochs 15–35), XXL has 1 022 samples (32 steps/epoch, epochs 15–35), GPT has 808 samples (26 steps/epoch, epochs 35–40). |
| **Risk** | Overfitting risk for all models. No early stopping is used — `run-all-iterations.sh` selects the best checkpoint via `find_best_epoch()`, providing coarse validation-based selection. |
| **File** | `run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh` |

### 9.3 Undersampling re-run behaviour

| Issue | Detail |
|---|---|
| **What** | The undersampling script is called **every time** each model's shell script runs, unconditionally. It overwrites `train_undersampled.json` on disk for that model. |
| **Risk** | Since the undersampling seed is fixed at `42` and the input files are deterministic, the output is reproducible across re-runs. However, if someone manually edits `train.json` or the undersampling script between runs, the generated file changes silently. There is no checksum or staleness check. |
| **File** | Each `run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh` L12 |

### 9.4 ~~XL/XXL undersampling is a no-op~~ (RESOLVED)

| Issue | Detail |
|---|---|
| **What** | An earlier version of the undersampling script always took all R samples first, which only produced correct 1:1 balancing when R was the minority (GPT). For XL/XXL where R was the majority, the output equalled the original dataset. |
| **Resolution** | Fixed. The script now identifies which class is larger and undersamples it to `minority_size`. All three models produce correctly balanced 1:1 outputs: XL → 848, XXL → 1 022, GPT → 808. |

### 9.5 ~~Active code defect: majority class always taken in full~~ (RESOLVED)

| Issue | Detail |
|---|---|
| **What** | The original code used `balanced = list(by_label["R"])` unconditionally, assuming R was always the minority. |
| **Resolution** | Fixed. The code now uses a conditional: `if len(A) <= len(R)` keeps all A and undersamples R; otherwise keeps all R and undersamples A. Correct 1:1 balancing is achieved for all models. |
| **File** | [make_no_ret_vs_ret_undersampled.py L56–62] |

### 9.6 Shell strictness difference from IT1

| Issue | Detail |
|---|---|
| **What** | All three IT2 scripts use `set -euo pipefail` and `GPU=${GPU:-0}`. The IT1 standard scripts lack `set -euo pipefail` and hardcode `GPU=0`. |
| **Risk** | `set -euo pipefail` means any non-zero exit code will abort the entire run. IT1 would silently continue past failures. This is **safer** behaviour in IT2 but means the two scripts have different failure semantics. |
| **File** | `run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh` L2 vs their IT1 counterparts |

### 9.7 Validation set remains imbalanced

| Issue | Detail |
|---|---|
| **What** | The validation files `valid.json` are unchanged for all models. All three models are now trained on 50/50 data but validated on their original imbalanced distributions (e.g., GPT: 72.5 % A / 27.5 % R; XL: 32.8 % A / 67.2 % R; XXL: 36.3 % A / 63.7 % R). |
| **Risk** | Overall validation accuracy may be misleading. A model that shifts its decision boundary (the intended goal of undersampling) may show lower overall accuracy even if recall on the previously-minority class improves. Per-class accuracy in `final_eval_results_perClass.json` should be used instead of overall accuracy for evaluation. |
| **File** | `run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh` — `--validation_file .../valid.json` |

### 9.8 No Clf2-side imbalance handling

| Issue | Detail |
|---|---|
| **What** | IT2 addresses Clf1 imbalance only. Clf2 training data is relatively balanced for all models. However, the Clf2 validation sets have B/C imbalance, creating train/validation distribution mismatches as documented in DOCUMENTATION_IT1.md §8.9. |
| **Impact** | Low for IT2 specifically — Clf2 is out of scope for this iteration. |
