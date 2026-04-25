## 1. Files Involved

| File | Role |
|---|---|
| classifier/run/run_large_train_gpt_no_ret_vs_ret_undersampled.sh | Gate 1 training launcher (the only undersampled shell script; GPT-only) |
| classifier/data_utils/make_no_ret_vs_ret_undersampled.py | Utility that builds the balanced training JSON |
| classifier/run_classifier.py | Core training/eval entry point (shared by all experiments) |
| classifier/utils.py | Model loading, tokenisation, scheduler, accuracy metrics |
| classifier/run/run_large_train_gpt_single_vs_multi.sh | Gate 2 training launcher (Iteration 1's standard Clf2 — reused as-is) |
| classifier/postprocess/predict_complexity_split_classifiers.py | End-to-end routing: merges Clf1 + Clf2 predictions → QA answers |
| evaluate_final_acc.py | Computes EM/F1 on routed predictions |
| `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/gpt/silver/no_retrieval_vs_retrieval/train.json` | Input to undersampling (original imbalanced A/R file) |
| `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/gpt/silver/no_retrieval_vs_retrieval/train_undersampled.json` | Output of undersampling (balanced file) |
| `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/gpt/silver/no_retrieval_vs_retrieval/valid.json` | Validation file (unchanged from Iteration 1) |
| `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json` | Unlabelled 3,000-question test set (unchanged) |

**No undersampling scripts exist for `flan_t5_xl` or `flan_t5_xxl`.** This is a GPT-only experiment.

---

## 2. Model Architecture

Both Gate 1 and Gate 2 use **T5-Large** (`AutoModelForSeq2SeqLM`) — identical to Iteration 1. The shell script sets `MODEL=t5-large` ([line 5](Adaptive-RAG/classifier/run/run_large_train_gpt_no_ret_vs_ret_undersampled.sh#L5)), and run_classifier.py loads it via `AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)` in utils.py. Gate 1 predicts `A` vs `R`; Gate 2 predicts `B` vs `C`.

---

## 3. Training Parameters

All parameters are passed explicitly on run_large_train_gpt_no_ret_vs_ret_undersampled.sh of the undersampled script:

| Parameter | Value | Iteration 1 (standard `gpt no_ret_vs_ret`) | Difference? |
|---|---|---|---|
| `--learning_rate` | `3e-5` | `3e-5` | None |
| `--per_device_train_batch_size` | `32` | `32` | None |
| `--max_seq_length` | `384` | `384` | None |
| `--doc_stride` | `128` | `128` | None |
| `--num_train_epochs` | `35, 40` (loop) | `35, 40` | None |
| `--labels` | `A R` | `A R` | None |
| Optimizer | AdamW ([run_classifier.py L636](Adaptive-RAG/classifier/run_classifier.py#L636)) | AdamW | None |
| `--weight_decay` | `0.0` (default) | `0.0` (default) | None |
| Scheduler | `linear` with 0 warmup steps (defaults at run_classifier.py and run_classifier.py) | Same | None |
| Loss | Standard cross-entropy (model's built-in `outputs.loss`, run_classifier.py) — `--use_focal_loss` is **not** passed | Same | None |
| Early stopping | **None** — trains for the full epoch budget, no early stopping | Same | None |
| `--seed` | **Not passed** (defaults to `None` in run_classifier.py) | Same | None |

**The only difference from Iteration 1 is the training data file** (`train_undersampled.json` vs `train.json`). Every hyperparameter, loss function, optimizer, and scheduler is identical.

---

## 4. Random Undersampling Implementation

Implemented in make_no_ret_vs_ret_undersampled.py:

**When applied:** Before training begins. The shell script calls it at run_large_train_gpt_no_ret_vs_ret_undersampled.sh (`python ./data_utils/make_no_ret_vs_ret_undersampled.py --model gpt`) before the epoch loop starts. This is a **static, offline** preprocessing step — the balanced file is materialised once as JSON on disk, then consumed by the training loop unchanged.

**Library/logic:** Pure Python standard library — `random.Random(seed).sample()` at make_no_ret_vs_ret_undersampled.py. No external ML library (no imbalanced-learn, no scikit-learn).

**Algorithm ([lines 52–60](Adaptive-RAG/classifier/data_utils/make_no_ret_vs_ret_undersampled.py#L52-L60)):**
```python
minority_size = min(len(by_label["A"]), len(by_label["R"]))   # = 404 for GPT
balanced = list(by_label["R"])                                  # keep ALL minority (R=404)
balanced.extend(rng.sample(by_label["A"], minority_size))       # sample 404 from majority (A=1013)
rng.shuffle(balanced)                                           # shuffle
```

**Target ratio:** Exact 1:1 (50/50). For GPT data: **404 A + 404 R = 808 total** (down from 1,417 originals). This discards **609 A samples** (60.1% of the majority class).

**Reproducibility seed:** `--seed 42` (default at make_no_ret_vs_ret_undersampled.py), used via `random.Random(args.seed)` — a separate RNG instance that doesn't pollute global state.

**Input:** `classifier/data/.../gpt/silver/no_retrieval_vs_retrieval/train.json`
**Output:** `classifier/data/.../gpt/silver/no_retrieval_vs_retrieval/train_undersampled.json`

---

## 5. Data Pipeline

**Gate 1 training data differs from Iteration 1** in exactly one way: the undersampled file replaces the full file. Everything else is identical:

| Aspect | Iteration 1 (standard) | Iteration 2 (undersampled) |
|---|---|---|
| Training file | `silver/no_retrieval_vs_retrieval/train.json` (1,417 samples: A=1013, R=404) | `silver/no_retrieval_vs_retrieval/train_undersampled.json` (808 samples: A=404, R=404) |
| Source of silver labels | Same (`preprocess_silver_train.py` → `label_complexity()`) | Same — the undersampler reads the already-created silver file |
| Validation file | `silver/no_retrieval_vs_retrieval/valid.json` | **Same file, same path** ([line 43](Adaptive-RAG/classifier/run/run_large_train_gpt_no_ret_vs_ret_undersampled.sh#L43)) |
| Predict file | `predict.json` (3,000 unlabelled) | **Same file** ([line 56](Adaptive-RAG/classifier/run/run_large_train_gpt_no_ret_vs_ret_undersampled.sh#L56)) |
| Tokenisation | `preprocess_features_function` in utils.py | Same |

---

## 6. Gate 2 — Completely Unchanged

Gate 2 is **the standard Iteration 1 Clf2**. The undersampled experiment only touches Gate 1. Gate 2 is trained by run_large_train_gpt_single_vs_multi.sh with:
- Training data: `binary_silver_single_vs_multi/train.json`
- Validation data: `silver/single_vs_multi/valid.json`
- Labels: `B C`
- All hyperparameters identical

At routing time, predict_complexity_split_classifiers.py takes `--no_ret_vs_ret_file` and `--single_vs_multi_file` as arguments, so you point the first at the undersampled Clf1's predictions and the second at any Iteration 1 Clf2 checkpoint. Gate 2's weights, training procedure, and output directory are entirely separate.

---

## 7. Evaluation Setup

**Identical to Iteration 1.** The routing script (predict_complexity_split_classifiers.py) and final evaluator (evaluate_final_acc.py) are the same:

- Same 6 datasets (musique, hotpotqa, 2wikimultihopqa, nq, trivia, squad)
- Same backbone QA predictions from `predictions/test/` (pre-computed nor_qa, oner_qa, ircot_qa answers for GPT)
- Same BM25 retrieval counts: `ONER_BM25['gpt']=6`, `IRCOT_BM25['gpt']=3`
- Same EM/F1 computation via `DropAnswerEmAndF1` in evaluate_final_acc.py
- Validation accuracy uses the same `calculate_accuracy` / `calculate_accuracy_perClass` in utils.py

---

## 8. Output Artifacts

Checkpoints, validation, and predictions are written under a **separate directory** from Iteration 1:

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/gpt/
├── no_ret_vs_ret/epoch/{35,40}/{DATE}/          ← Iteration 1 Clf1
├── no_ret_vs_ret_undersampled/epoch/{35,40}/{DATE}/  ← Iteration 2 Clf1 ✓ SEPARATE
│   ├── model.safetensors, config.json, ...      (trained checkpoint)
│   ├── valid/
│   │   ├── dict_id_pred_results.json
│   │   ├── final_eval_results.json
│   │   └── final_eval_results_perClass.json
│   └── predict/
│       ├── dict_id_pred_results.json
│       └── final_eval_results.json
└── single_vs_multi/epoch/...                    ← Gate 2 (shared)
```

The critical difference is the subdirectory name `no_ret_vs_ret_undersampled` ([line 15](Adaptive-RAG/classifier/run/run_large_train_gpt_no_ret_vs_ret_undersampled.sh#L15)) vs `no_ret_vs_ret`. There is **no risk of overwriting Iteration 1 results** — date-stamped paths (`${DATE}`) add further protection.

Routed QA predictions would go to whichever `--output_path` is passed to predict_complexity_split_classifiers.py — typically something like `predictions/classifier/t5-large/gpt/split_undersampled/...`.

---

## 9. Suspicious / Noteworthy Items

1. **Aggressive majority-class discard.** For GPT, A=1,013 → 404 means **60% of A samples are thrown away**, and total training data drops from 1,417 → 808 (−43%). With only 404 samples per class, the model may underfit, especially given 35–40 epochs of training.

2. **No `--seed` passed to run_classifier.py.** The undersampling utility uses `seed=42` for the data, but the training script receives no `--seed` argument ([line 19–34](Adaptive-RAG/classifier/run/run_large_train_gpt_no_ret_vs_ret_undersampled.sh#L19-L34)). In run_classifier.py, `--seed` defaults to `None` (run_classifier.py), which skips `set_seed()` (run_classifier.py). This means training is **not fully reproducible** — DataLoader shuffling and weight initialisation differ across runs. (This is the same as Iteration 1, so it's not a regression.)

3. **High epoch count on tiny data.** 35–40 epochs on 808 samples (≈25 batches/epoch at batch=32) means only ~875–1,000 gradient steps total. This is relatively few steps, but the small dataset size raises overfitting risk — there is no early stopping, no dropout change, and no regularisation beyond AdamW with `weight_decay=0.0`.

4. **Undersampling re-runs every launch.** The shell script runs make_no_ret_vs_ret_undersampled.py at the top every time ([line 11](Adaptive-RAG/classifier/run/run_large_train_gpt_no_ret_vs_ret_undersampled.sh#L11)). Since the seed is fixed at 42, the output is deterministic — but this means if someone passes `--seed` with a different value, the file would silently change between runs without versioning.

5. **GPT-only experiment.** No undersampling scripts exist for `flan_t5_xl` (A=424, R=868) or `flan_t5_xxl` (A=511, R=898). In those models, **R is the majority class**, not A — the imbalance direction is reversed. The undersampling utility would still work (it always downsamples whichever is larger), but no shell scripts invoke it for those models.

6. **No premature termination risk.** The script uses `set -euo pipefail` ([line 2](Adaptive-RAG/classifier/run/run_large_train_gpt_no_ret_vs_ret_undersampled.sh#L2)), which will abort on any error. This is appropriate — but unlike the `*_weighted_ce.sh` scripts which downgrade validation failures to warnings, this script will terminate entirely if validation or prediction fails on any epoch.