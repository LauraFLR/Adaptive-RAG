# Training Scripts in `classifier/run`

This directory contains shell scripts that launch classifier training experiments through `run_classifier.py`.

Most scripts follow the same pattern:

1. Train a classifier for one epoch budget.
2. Run validation on a labeled validation file.
3. Run prediction on `data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json`.
4. Repeat for several epoch values.

## Important working-directory assumption

Run these scripts from the `classifier/` directory, not from the repository root.

Example:

```bash
cd /root/laura/Adaptive-RAG/classifier
bash run/run_large_train_xl_no_ret_vs_ret.sh
```

Most scripts call:

```bash
python run_classifier.py
```

and use relative paths like `./data/...`, which only resolve correctly when the current directory is `classifier/`.

## Common conventions

- Base classifier model: `t5-large`
- Main dataset bundle: `musique_hotpot_wiki2_nq_tqa_sqd`
- XL / XXL runs sweep epochs `15 20 25 30 35`
- GPT runs sweep epochs `35 40`
- Outputs are written under `classifier/outputs/...`
- Validation is done with `--do_eval`
- Prediction is also done with `--do_eval`, using the unlabeled `predict.json`

## Script families

### 1. Original 3-class classifier scripts

These are the original scripts for predicting the full label set `A / B / C` in one model.

| Script | Silver labels source | Training file | Validation file | Labels | Output subdir | Notes |
|---|---|---|---|---|---|---|
| `run_large_train_xl.sh` | `flan_t5_xl` | `data/.../flan_t5_xl/binary_silver/train.json` | `data/.../flan_t5_xl/silver/valid.json` | default `A B C` | `flan_t5_xl/epoch/...` | Original single-model classifier. GPU is hardcoded to `7`. |
| `run_large_train_xxl.sh` | `flan_t5_xxl` | `data/.../flan_t5_xxl/binary_silver/train.json` | `data/.../flan_t5_xxl/silver/valid.json` | default `A B C` | `flan_t5_xxl/epoch/...` | Same as XL version with XXL-derived silver labels. GPU is hardcoded to `7`. |
| `run_large_train_gpt.sh` | `gpt` | `data/.../gpt/binary_silver/train.json` | `data/.../gpt/silver/valid.json` | default `A B C` | `gpt/epoch/...` | Same setup, but only runs epochs `35` and `40`. GPU is hardcoded to `7`. |

Notes:

- These scripts do not pass `--labels`, so `run_classifier.py` uses its default label order `A B C`.
- `binary_silver/train.json` mixes inductive-bias binary data with silver labels.

### 2. Standard split-classifier scripts

These scripts implement the two-stage replacement for the original 3-class classifier.

#### Classifier 1: no retrieval vs retrieval

These predict `A` vs `R`, where `R` means retrieval is needed and merges original classes `B` and `C`.

| Script | Training file | Validation file | Labels | Output subdir | Notes |
|---|---|---|---|---|---|
| `run_large_train_xl_no_ret_vs_ret.sh` | `data/.../flan_t5_xl/silver/no_retrieval_vs_retrieval/train.json` | `data/.../flan_t5_xl/silver/no_retrieval_vs_retrieval/valid.json` | `A R` | `flan_t5_xl/no_ret_vs_ret/epoch/...` | Standard Clf1 run for XL silver labels. GPU hardcoded to `0`. |
| `run_large_train_xxl_no_ret_vs_ret.sh` | `data/.../flan_t5_xxl/silver/no_retrieval_vs_retrieval/train.json` | `data/.../flan_t5_xxl/silver/no_retrieval_vs_retrieval/valid.json` | `A R` | `flan_t5_xxl/no_ret_vs_ret/epoch/...` | Standard Clf1 run for XXL silver labels. |
| `run_large_train_gpt_no_ret_vs_ret.sh` | `data/.../gpt/silver/no_retrieval_vs_retrieval/train.json` | `data/.../gpt/silver/no_retrieval_vs_retrieval/valid.json` | `A R` | `gpt/no_ret_vs_ret/epoch/...` | Standard Clf1 run for GPT silver labels. Uses epochs `35` and `40`. |

#### Classifier 2: single-step vs multi-step retrieval

These predict `B` vs `C`. In the cascade, they are intended for samples that Clf1 already routed to retrieval.

| Script | Training file | Validation file | Labels | Output subdir | Notes |
|---|---|---|---|---|---|
| `run_large_train_xl_single_vs_multi.sh` | `data/.../flan_t5_xl/binary_silver_single_vs_multi/train.json` | `data/.../flan_t5_xl/silver/single_vs_multi/valid.json` | `B C` | `flan_t5_xl/single_vs_multi/epoch/...` | Standard Clf2 run for XL labels. Training uses merged binary + silver data. GPU hardcoded to `0`. |
| `run_large_train_xxl_single_vs_multi.sh` | `data/.../flan_t5_xxl/binary_silver_single_vs_multi/train.json` | `data/.../flan_t5_xxl/silver/single_vs_multi/valid.json` | `B C` | `flan_t5_xxl/single_vs_multi/epoch/...` | Standard Clf2 run for XXL labels. |
| `run_large_train_gpt_single_vs_multi.sh` | `data/.../gpt/binary_silver_single_vs_multi/train.json` | `data/.../gpt/silver/single_vs_multi/valid.json` | `B C` | `gpt/single_vs_multi/epoch/...` | Standard Clf2 run for GPT labels. Uses epochs `35` and `40`. |

### 3. Classifier 1 imbalance-handling experiments

These scripts are all variants of `no_ret_vs_ret` that try to handle the `A`/`R` imbalance differently.

#### Focal-loss runs

These pass `--use_focal_loss` with a nonzero gamma.

| Script | Default gamma | Default alpha | Output subdir | Notes |
|---|---|---|---|---|
| `run_large_train_xl_no_ret_vs_ret_focal.sh` | `2.0` | `0.33` | `flan_t5_xl/no_ret_vs_ret_focal/epoch/...` | Keeps only the latest `checkpoint-*` directory per epoch and evaluates that checkpoint. GPU can be overridden with `GPU=...`. |
| `run_large_train_xxl_no_ret_vs_ret_focal.sh` | `2.0` | `0.36` | `flan_t5_xxl/no_ret_vs_ret_focal/epoch/...` | Same focal-loss setup for XXL labels. GPU can be overridden with `GPU=...`. |
| `run_large_train_gpt_no_ret_vs_ret_focal.sh` | `2.0` | `0.71` | `gpt/no_ret_vs_ret_focal/epoch/...` | Same focal-loss setup for GPT labels. Uses epochs `35` and `40`, keeps the latest checkpoint, and downgrades validation/prediction failures to warnings. |

#### Weighted cross-entropy runs

These use the focal-loss implementation with `gamma=0`, which reduces focal loss to class-weighted cross-entropy.

| Script | Main difference |
|---|---|
| `run_large_train_xl_no_ret_vs_ret_weighted_ce.sh` | Starts at epoch `20` instead of `15`, cleans extra checkpoints, and tolerates validation/prediction failures with warning messages. |
| `run_large_train_xxl_no_ret_vs_ret_weighted_ce.sh` | Same cleanup and warning-tolerant behavior for XXL. |
| `run_large_train_gpt_no_ret_vs_ret_weighted_ce.sh` | GPT version of the same safer weighted-CE launcher. Uses epochs `35` and `40`. |

#### Undersampling run

| Script | Training file | Output subdir | Notes |
|---|---|---|---|
| `run_large_train_gpt_no_ret_vs_ret_undersampled.sh` | `data/.../gpt/silver/no_retrieval_vs_retrieval/train_undersampled.json` | `gpt/no_ret_vs_ret_undersampled/epoch/...` | Before training, it calls `data_utils/make_no_ret_vs_ret_undersampled.py --model gpt` to create a balanced `A/R` training set by undersampling the majority class. |

### 4. Classifier 2 ablations and feature experiments

These scripts explore alternatives to the standard `single_vs_multi` training setup.

#### Silver-only Clf2

| Script | Model argument | Training file | Validation file | Output subdir | Notes |
|---|---|---|---|---|---|
| `run_large_train_silver_only_single_vs_multi.sh` | `flan_t5_xl`, `flan_t5_xxl`, or `gpt` | `data/.../{model}/silver/single_vs_multi/train.json` | `data/.../{model}/silver/single_vs_multi/valid.json` | `{model}/single_vs_multi/epoch/.../silver_only` | Removes the inductive-bias binary data and trains Clf2 on silver labels only. This script expects the model name as its first positional argument. GPT uses epochs `35` and `40`; XL and XXL use `15 20 25 30 35`. |

Example:

```bash
cd /root/laura/Adaptive-RAG/classifier
bash run/run_large_train_silver_only_single_vs_multi.sh flan_t5_xl
```

#### Clf2 oracle ceiling

`classifier/postprocess/predict_complexity_oracle_ceiling.py` answers: *how much QA F1 could be gained by a perfect Clf2, given the current fixed Clf1?*

It holds Clf1 fixed as the agreement gate from Iteration 2a (questions where `nor_qa` and `oner_qa` agree get routed to A; all others are treated as R-routed). For each R-routed question it applies oracle Clf2 routing: route to B if `oner_qa` is correct, else to C if `ircot_qa` is correct, else default to B. The resulting QA predictions represent the theoretical ceiling that any real Clf2 improvement can close toward.

Outputs are written to `predictions/classifier/t5-large/{model}/split_agreement_oracle/` — one prediction file per dataset plus `oracle_stats.json`. Pass that path to `evaluate_final_acc.py` to get the F1 numbers.

Usage (run from the repo root, not from `classifier/`):

```bash
python classifier/postprocess/predict_complexity_oracle_ceiling.py flan_t5_xl
python classifier/postprocess/predict_complexity_oracle_ceiling.py flan_t5_xxl
```

#### Feature-augmented Clf2

| Script | Model argument | Training file | Validation file | Predict file | Output subdir | Notes |
|---|---|---|---|---|---|---|
| `run_large_train_feat_single_vs_multi.sh` | `flan_t5_xl`, `flan_t5_xxl`, or `gpt` | `data/.../{model}/binary_silver_feat_single_vs_multi/train.json` | `data/.../{model}/silver_feat_single_vs_multi/valid.json` | `data/.../feat_predict.json` | `{model}/feat_single_vs_multi/epoch/.../feat` | Retrains Clf2 using structural feature prefixes in the input text. |

Supporting probe and data-generation files:

- `classifier/postprocess/clf2_feature_probe.py` runs a lightweight feasibility experiment for Clf2 features. It extracts three hand-built features from each B/C question: token length, named-entity count, and a regex-based bridge flag for likely multi-hop phrasing. It then trains a logistic regression on an 80/20 stratified split and reports ROC-AUC, accuracy, a classification report, and per-feature coefficients. It also saves `clf2_feature_probe_data.csv` and `clf2_feature_probe_scatter.png`.
- `classifier/data_utils/add_feature_prefix.py` takes the same feature logic and rewrites the Clf2 train/valid/predict JSON files so each question is prefixed with tags like `[LEN:X] [ENT:Y] [BRIDGE:Z]`. Those generated files are what `run_large_train_feat_single_vs_multi.sh` consumes.

Example:

```bash
cd /root/laura/Adaptive-RAG/classifier
bash run/run_large_train_feat_single_vs_multi.sh gpt
```

## Which script should you use?

- Use `run_large_train_{xl,xxl,gpt}.sh` only if you want the original single 3-class classifier.
- Use `*_no_ret_vs_ret.sh` and `*_single_vs_multi.sh` if you want the standard two-stage split classifier.
- Use `*_focal.sh`, `*_weighted_ce.sh`, or `*_undersampled.sh` only for Clf1 imbalance experiments.
- Use `run_large_train_silver_only_single_vs_multi.sh` to test whether inductive-bias binary data is helping Clf2.
- Use `run_large_train_feat_single_vs_multi.sh` to test the feature-prefixed Clf2 input format.

## Related files

- `run_classifier.py`: actual training and evaluation entry point used by every script.
- `data_utils/make_no_ret_vs_ret_undersampled.py`: builds the balanced training file used by the GPT undersampling experiment.
- `classifier/postprocess/predict_complexity_oracle_ceiling.py`: measures the maximum QA F1 achievable with a perfect Clf2, given a fixed agreement-gate Clf1. Run from the repo root.
- `classifier/postprocess/clf2_feature_probe.py`: probes whether simple structural features can separate Clf2 labels `B` vs `C` before training a feature-augmented classifier.
- `classifier/data_utils/add_feature_prefix.py`: materializes the feature-prefixed Clf2 train/valid/predict files used by `run_large_train_feat_single_vs_multi.sh`.
- `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/SPLIT_CLASSIFIERS.md`: explains the split label files and folder layout.
- `classifier/SPLIT_CLASSIFIERS_CHANGES.md`: documents the move from one 3-class classifier to two binary classifiers.

## Practical cautions

- Several scripts hardcode `GPU=0` or `GPU=7` instead of using an environment override.
- The focal and weighted-CE scripts evaluate the latest saved checkpoint, not necessarily the top-level output directory.
- The `*_weighted_ce.sh` scripts keep the safer behavior that previously lived in the `*_fixed.sh` filenames.
- `run_large_train_silver_only_single_vs_multi.sh` and `run_large_train_feat_single_vs_multi.sh` require a positional model argument.