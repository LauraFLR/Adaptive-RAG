# Iteration 2 — Random Undersampling for Gate 1

> **Design Science Research Artifact:** Apply random undersampling to the
> majority class in Clf1 (A vs R) training data to address class imbalance
> in the silver labels.  The undersampling script is run for all three model
> variants (flan_t5_xl, flan_t5_xxl, GPT).  Gate 2 (Clf2) remains identical
> to Iteration 1.
>
> **Important caveat:** Due to a code defect (§9.5), undersampling only
> takes effect for GPT where A is the majority class.  For XL and XXL where
> R is the majority class, the script produces the original dataset
> unmodified, so their IT2 results are near-identical to IT1.

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
| 7. Start balanced set with **all R samples** (full copy) | `balanced = list(by_label["R"])` | [make_no_ret_vs_ret_undersampled.py L56] |
| 8. Sample `minority_size` A samples **without replacement** | `rng.sample(by_label["A"], minority_size)` | [make_no_ret_vs_ret_undersampled.py L57] |
| 9. Shuffle the combined list | `rng.shuffle(balanced)` | [make_no_ret_vs_ret_undersampled.py L58] |
| 10. Write to `train_undersampled.json` | `json.dump(balanced, f, indent=4)` | [make_no_ret_vs_ret_undersampled.py L61–62] |

### 4.3 Target ratio

**Intended: 1:1 (perfect balance).** `minority_size` A samples + all R samples → equal counts. However, due to a code defect (§9.5), this only works when A is the majority class (GPT). When R is the majority class (XL, XXL), the output equals the original dataset.

### 4.4 Concrete numbers (all models)

| Metric | flan_t5_xl | flan_t5_xxl | GPT |
|---|---|---|---|
| Input total | 1 292 | 1 409 | 1 417 |
| Input A count (%) | 424 (32.8 %) | 511 (36.3 %) | 1 013 (71.5 %) |
| Input R count (%) | 868 (67.2 %) | 898 (63.7 %) | 404 (28.5 %) |
| Majority class | R | R | A |
| Minority class | A (424) | A (511) | R (404) |
| Output total | **1 292** | **1 409** | **808** |
| Output A count (%) | 424 (32.8 %) | 511 (36.3 %) | 404 (50.0 %) |
| Output R count (%) | 868 (67.2 %) | 898 (63.7 %) | 404 (50.0 %) |
| Samples discarded | **0 (0.0 %)** | **0 (0.0 %)** | **609 (43.0 %)** |
| Effective balancing? | **No** — defect §9.5 | **No** — defect §9.5 | **Yes** — 1:1 |

For XL and XXL, `balanced = list(by_label["R"])` takes all R (the majority), then `rng.sample(by_label["A"], minority_size)` takes all A (since `minority_size = len(A) = min(A, R)`). The output is the entire original dataset, shuffled. No undersampling occurs.

### 4.5 Reproducibility

| Seed aspect | Value | Notes |
|---|---|---|
| Undersampling seed | `42` | Default value of `--seed` [make_no_ret_vs_ret_undersampled.py L13]. Passed to `random.Random(42)` — an **independent** RNG instance, not the global `random` module state. |
| Training seed | `42` | Passed as `--seed 42` in the shell script. Applied via `accelerate.utils.set_seed(42)` inside `run_classifier.py`. |
| Relationship | **Independent** | The undersampling RNG is created and consumed in a separate Python process that exits before training begins. The two seeds happen to share the same value `42` but are applied to different RNGs in different processes. |

### 4.6 Code defect: majority class always taken in full

The algorithm at line 56 starts with `balanced = list(by_label["R"])` (all R samples), then samples from A. This **assumes R is the minority class**. For GPT data this is correct (R = 404 < A = 1 013). However, for `flan_t5_xl` and `flan_t5_xxl` where R is the **majority** class, the code produces the original dataset unmodified:

- **XL:** `balanced = list(by_label["R"])` takes all 868 R, then `rng.sample(by_label["A"], 424)` takes all 424 A → total 1 292 = original size.
- **XXL:** all 898 R + all 511 A → total 1 409 = original size.

The correct fix would be to undersample whichever class is larger to `minority_size`, keeping the smaller class intact. This defect is **actively triggered** in the current IT2 run for XL and XXL, making their IT2 results effectively a retrained IT1 (same data, different random epoch timestamps). See §9.5 for impact analysis.

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

**flan_t5_xl (undersampling ineffective — defect §9.5):**

| Aspect | Iteration 1 | Iteration 2 |
|---|---|---|
| **Training file** | `.../flan_t5_xl/silver/.../train.json` | `.../flan_t5_xl/silver/.../train_undersampled.json` |
| **Total samples** | 1 292 | 1 292 (unchanged) |
| **A count (%)** | 424 (32.8 %) | 424 (32.8 %) |
| **R count (%)** | 868 (67.2 %) | 868 (67.2 %) |
| **A:R ratio** | 1:2.05 | 1:2.05 (unchanged) |
| **Data reduction** | — | 0 samples removed |

**flan_t5_xxl (undersampling ineffective — defect §9.5):**

| Aspect | Iteration 1 | Iteration 2 |
|---|---|---|
| **Training file** | `.../flan_t5_xxl/silver/.../train.json` | `.../flan_t5_xxl/silver/.../train_undersampled.json` |
| **Total samples** | 1 409 | 1 409 (unchanged) |
| **A count (%)** | 511 (36.3 %) | 511 (36.3 %) |
| **R count (%)** | 898 (63.7 %) | 898 (63.7 %) |
| **A:R ratio** | 1:1.76 | 1:1.76 (unchanged) |
| **Data reduction** | — | 0 samples removed |

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
| Training samples | 1 292 | 1 292 (unchanged) |
| Batch size | 32 | 32 |
| Steps per epoch | ⌈1 292 / 32⌉ = 41 | 41 (unchanged) |
| Total steps (epoch 15) | 615 | 615 |
| Total steps (epoch 35) | 1 435 | 1 435 |

**flan_t5_xxl (epochs 15, 20, 25, 30, 35):**

| Metric | Iteration 1 | Iteration 2 |
|---|---|---|
| Training samples | 1 409 | 1 409 (unchanged) |
| Batch size | 32 | 32 |
| Steps per epoch | ⌈1 409 / 32⌉ = 45 | 45 (unchanged) |
| Total steps (epoch 15) | 675 | 675 |
| Total steps (epoch 35) | 1 575 | 1 575 |

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

### 9.1 Aggressive data discard (GPT only)

| Issue | Detail |
|---|---|
| **What** | 609 of 1 417 GPT samples (43.0 %) are discarded from Clf1 training. XL and XXL discard 0 samples due to defect §9.5. |
| **Risk** | The GPT model sees only 808 training samples. With batch size 32, that is just 26 steps per epoch. Information from 60 % of the A-class samples is permanently lost. |
| **File** | [make_no_ret_vs_ret_undersampled.py L53–58] |
| **Mitigation** | None — no oversampling, SMOTE, or weighted loss is combined with the undersampling. |

### 9.2 High epoch count on small dataset (GPT)

| Issue | Detail |
|---|---|
| **What** | 35–40 epochs on 808 GPT samples (26 steps/epoch). The model sees every training sample ~35–40 times. |
| **Risk** | Overfitting risk. No early stopping is used — only epochs 35 and 40 are checkpointed. `run-all-iterations.sh` selects the better of these two via `find_best_epoch()`, providing coarse validation-based selection. |
| **Note** | XL and XXL use epochs 15–35 on their full datasets (1 292 / 1 409 samples), which is the same regime as IT1 and carries similar overfitting risk. |
| **File** | [run_large_train_gpt_no_ret_vs_ret_undersampled.sh L14: `for EPOCH in 35 40`] |

### 9.3 Undersampling re-run behaviour

| Issue | Detail |
|---|---|
| **What** | The undersampling script is called **every time** each model's shell script runs, unconditionally. It overwrites `train_undersampled.json` on disk for that model. |
| **Risk** | Since the undersampling seed is fixed at `42` and the input files are deterministic, the output is reproducible across re-runs. However, if someone manually edits `train.json` or the undersampling script between runs, the generated file changes silently. There is no checksum or staleness check. |
| **File** | Each `run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh` L12 |

### 9.4 XL/XXL undersampling is a no-op

| Issue | Detail |
|---|---|
| **What** | Shell launchers now exist for all three models (`run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh`) and `run-all-iterations.sh` trains all three. However, due to the code defect in §9.5, the undersampling script produces the original dataset unmodified for XL and XXL. |
| **Impact** | XL and XXL IT2 results are near-identical to IT1 — any small differences are due to random training initialization from the different epoch timestamp directories. The IT2 experiment is only meaningful for GPT. |
| **Confirmed by data** | XL: `train_undersampled.json` = 1 292 samples (= original). XXL: 1 409 samples (= original). GPT: 808 samples (reduced from 1 417). |

### 9.5 Active code defect: majority class always taken in full

| Issue | Detail |
|---|---|
| **What** | `balanced = list(by_label["R"])` always takes **all** R samples, then `rng.sample(by_label["A"], minority_size)` samples from A. The code assumes R is always the smaller bucket. |
| **Triggered for** | **XL** (A=424, R=868): `minority_size = 424`, `balanced` = all 868 R + 424 A = 1 292 = original. **XXL** (A=511, R=898): 898 R + 511 A = 1 409 = original. In both cases the script silently produces the full dataset — no undersampling occurs. |
| **Not triggered for** | **GPT** (A=1 013, R=404): `minority_size = 404`, `balanced` = all 404 R + 404 sampled A = 808. Undersampling works correctly. |
| **Correct fix** | Should be: undersample whichever class is larger to `minority_size`, keeping the smaller class intact. |
| **Impact on IT2** | XL and XXL IT2 results are effectively IT1 retrained with a different timestamp. Only GPT IT2 results reflect actual undersampling. This is confirmed by the near-zero deltas: XL macro Δ EM = −0.002, XXL macro Δ EM = −0.002 vs IT1. |
| **File** | [make_no_ret_vs_ret_undersampled.py L56–57] |

### 9.6 Shell strictness difference from IT1

| Issue | Detail |
|---|---|
| **What** | All three IT2 scripts use `set -euo pipefail` and `GPU=${GPU:-0}`. The IT1 standard scripts lack `set -euo pipefail` and hardcode `GPU=0`. |
| **Risk** | `set -euo pipefail` means any non-zero exit code will abort the entire run. IT1 would silently continue past failures. This is **safer** behaviour in IT2 but means the two scripts have different failure semantics. |
| **File** | `run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh` L2 vs their IT1 counterparts |

### 9.7 Validation set remains imbalanced

| Issue | Detail |
|---|---|
| **What** | The validation files `valid.json` are unchanged for all models. For GPT: 1 038 A / 393 R (72.5 % / 27.5 %). The GPT model is trained on 50/50 data but validated on 72/28 data. For XL/XXL the training data is also unchanged (defect §9.5), so there is no train/validation mismatch. |
| **Risk** | For GPT, overall validation accuracy will be dominated by A-class performance. A model that learns to predict more R (the intended goal of undersampling) may show lower overall accuracy even if recall on R improves. Per-class accuracy in `final_eval_results_perClass.json` should be used instead of overall accuracy for evaluation. |
| **File** | `run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh` — `--validation_file .../valid.json` |

### 9.8 No Clf2-side imbalance handling

| Issue | Detail |
|---|---|
| **What** | IT2 addresses Clf1 imbalance only. Clf2 training data is relatively balanced for all models. However, the Clf2 validation sets have B/C imbalance, creating train/validation distribution mismatches as documented in DOCUMENTATION_IT1.md §8.9. |
| **Impact** | Low for IT2 specifically — Clf2 is out of scope for this iteration. |
