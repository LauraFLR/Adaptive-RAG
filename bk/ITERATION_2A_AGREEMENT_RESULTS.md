# Iteration 2a: Cross-Strategy Answer Agreement Gate

## Motivation

The trained Clf1 (A vs R binary classifier) suffers from low A-class recall (~42–45%) due to a 2:1 class imbalance in training data. The previous threshold experiment (Iteration 1) showed that adjusting the decision boundary to compensate for this imbalance **hurts** downstream QA accuracy because retrieval rarely harms and skipping it is costly.

This iteration takes a different approach: replace the trained Clf1 entirely with a **training-free uncertainty estimation (UE) proxy** based on cross-strategy answer agreement.

**Core idea**: If the LLM produces the same answer with no retrieval (`nor_qa`) and with single-step retrieval (`oner_qa`), it likely knows the answer from parametric memory → route to A (no retrieval). If the answers disagree → the LLM is uncertain → pass to Clf2 to decide B vs C.

---

## Background: Available Uncertainty Signals

Before implementing, we inspected what signals are available in the existing prediction files:

| Signal | Available? | Notes |
|---|---|---|
| Token/sequence probability | No | `output_scores=False` hardcoded in `llm_server/serve.py` |
| Predictive entropy | No | Needs token log-probs |
| Self-consistency (multi-sample) | No | Needs `do_sample=True` + multiple runs |
| **Cross-strategy agreement** | **Yes** | All `nor_qa` and `oner_qa` predictions already exist |
| Verbalized confidence | No | Requires live LLM |

The LLM client generator (`commaqa/models/llm_client_generator.py`) uses fake scores (`1/(index+1)`) and the `SearchState._score` is always initialized to `0.0` and never updated. The disk cache (`~/.cache/llmcalls/`, 5,305 entries) also stores no score data. Cross-strategy agreement is the only UE signal computable from existing artifacts.

---

## Implementation

### Script

**`classifier/postprocess/predict_complexity_agreement.py`**

Arguments:
- `model_name`: `flan_t5_xl` or `flan_t5_xxl`
- `--clf2_pred_file`: Path to Clf2's `dict_id_pred_results.json`
- `--predict_file`: Path to `predict.json` (3,000 test questions)
- `--output_path`: Where to write routed predictions

### Algorithm

```
For each question in predict.json:
    1. Load nor_qa answer and oner_qa answer for the question's dataset
    2. Apply answer_extractor (handle CoT chains)
    3. Normalize both (lowercase, strip punctuation/articles/whitespace)
    4. If nor_qa_normalized == oner_qa_normalized AND both non-empty:
         → Route to A (no retrieval)
       Else:
         → Use Clf2 argmax prediction (B or C)
    5. Fetch the actual QA answer from the chosen strategy's prediction file
```

### Prediction File Paths

For each dataset, the script loads predictions from:
```
predictions/test/nor_qa_{model}_{dataset}____prompt_set_1/
    prediction__{dataset}_to_{dataset}__test_subsampled.json

predictions/test/oner_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__15___distractor_count__1/
    prediction__{dataset}_to_{dataset}__test_subsampled.json
```

Both files are `dict[qid → answer_string]` with 500 entries per dataset, and the qid sets are identical.

---

## Results

### Routing Distribution

#### flan_t5_xl

| Dataset | Default (Clf1) A/B/C | Agreement A/B/C |
|---|---|---|
| NQ | 43 / 457 / 0 | 70 / 430 / 0 |
| TriviaQA | 50 / 412 / 38 | 131 / 343 / 26 |
| SQuAD | 14 / 437 / 49 | 32 / 423 / 45 |
| MuSiQue | 10 / 70 / 420 | 48 / 66 / 386 |
| HotpotQA | 65 / 125 / 310 | 120 / 116 / 264 |
| 2WikiMHQA | 223 / 29 / 248 | 180 / 51 / 269 |
| **Total** | **405 / 1530 / 1065** | **581 / 1429 / 990** |

Agreement rate: 581/3000 = 19.4%

#### flan_t5_xxl

| Dataset | Default (Clf1) A/B/C | Agreement A/B/C |
|---|---|---|
| NQ | 118 / 380 / 2 | 91 / 406 / 3 |
| TriviaQA | 181 / 292 / 27 | 153 / 320 / 27 |
| SQuAD | 62 / 386 / 52 | 51 / 398 / 51 |
| MuSiQue | 20 / 67 / 413 | 37 / 63 / 400 |
| HotpotQA | 91 / 106 / 303 | 131 / 100 / 269 |
| 2WikiMHQA | 140 / 74 / 286 | 166 / 70 / 264 |
| **Total** | **612 / 1305 / 1083** | **629 / 1357 / 1014** |

Agreement rate: 629/3000 = 21.0%

### End-to-End QA Accuracy

#### flan_t5_xl

| Dataset | Default (Clf1) | Agreement | Δ |
|---|---|---|---|
| NQ | 0.436 | 0.446 | **+0.010** |
| TriviaQA | 0.580 | 0.598 | **+0.018** |
| SQuAD | 0.334 | 0.336 | **+0.002** |
| MuSiQue | 0.260 | 0.252 | −0.008 |
| HotpotQA | 0.442 | 0.448 | **+0.006** |
| 2WikiMHQA | 0.444 | 0.526 | **+0.082** |
| **Average** | **0.416** | **0.434** | **+0.018** |

#### flan_t5_xxl

| Dataset | Default (Clf1) | Agreement | Δ |
|---|---|---|---|
| NQ | 0.434 | 0.474 | **+0.040** |
| TriviaQA | 0.540 | 0.620 | **+0.080** |
| SQuAD | 0.322 | 0.344 | **+0.022** |
| MuSiQue | 0.238 | 0.236 | −0.002 |
| HotpotQA | 0.430 | 0.468 | **+0.038** |
| 2WikiMHQA | 0.540 | 0.590 | **+0.050** |
| **Average** | **0.417** | **0.455** | **+0.038** |

#### EM Comparison

| Dataset | xl default | xl agree | Δ | xxl default | xxl agree | Δ |
|---|---|---|---|---|---|---|
| NQ | 0.368 | 0.378 | +0.010 | 0.378 | 0.412 | +0.034 |
| TriviaQA | 0.516 | 0.530 | +0.014 | 0.484 | 0.560 | +0.076 |
| SQuAD | 0.274 | 0.276 | +0.002 | 0.266 | 0.280 | +0.014 |
| MuSiQue | 0.232 | 0.228 | −0.004 | 0.210 | 0.210 | +0.000 |
| HotpotQA | 0.422 | 0.428 | +0.006 | 0.402 | 0.440 | +0.038 |
| 2WikiMHQA | 0.390 | 0.480 | +0.090 | 0.478 | 0.532 | +0.054 |
| **Average** | **0.367** | **0.387** | **+0.020** | **0.370** | **0.406** | **+0.036** |

#### F1 Comparison

| Dataset | xl default | xl agree | Δ | xxl default | xxl agree | Δ |
|---|---|---|---|---|---|---|
| NQ | 0.464 | 0.473 | +0.009 | 0.475 | 0.510 | +0.035 |
| TriviaQA | 0.603 | 0.619 | +0.016 | 0.569 | 0.645 | +0.076 |
| SQuAD | 0.389 | 0.390 | +0.001 | 0.384 | 0.402 | +0.018 |
| MuSiQue | 0.321 | 0.316 | −0.005 | 0.293 | 0.288 | −0.005 |
| HotpotQA | 0.538 | 0.545 | +0.006 | 0.512 | 0.554 | +0.042 |
| 2WikiMHQA | 0.478 | 0.568 | +0.090 | 0.575 | 0.631 | +0.056 |
| **Average** | **0.466** | **0.485** | **+0.020** | **0.468** | **0.505** | **+0.037** |

### Efficiency (Total Retrieval Steps)

| Model | Default (Clf1) | Agreement | Δ Steps | Δ% |
|---|---|---|---|---|
| flan_t5_xl | 6,155 | 5,617 | −538 | −8.7% |
| flan_t5_xxl | 3,586 | 3,488 | −98 | −2.7% |

---

## Classifier-Level Self-Knowledge Evaluation

To understand **why** agreement works better, we evaluated it as an A/R classifier on the validation set (silver labels from `no_retrieval_vs_retrieval/valid.json`), using `predictions/test/` answers (same questions as the validation set).

### flan_t5_xl (1,350 validation samples: 439 A, 911 R)

| Metric | Clf1 (trained, ep20) | Agreement |
|---|---|---|
| Overall Accuracy | 73.2% | **89.4%** |
| A-Precision | 0.630 | **0.921** |
| A-Recall | 0.426 | **0.738** |
| A-F1 | 0.508 | **0.819** |
| R-Recall | 87.9% | **96.9%** |
| ROC-AUC | — | **0.854** |

Confusion matrix: TP=324, FP=28, FN=115, TN=883

### flan_t5_xxl (1,415 validation samples: 519 A, 896 R)

| Metric | Clf1 (trained, ep35) | Agreement |
|---|---|---|
| Overall Accuracy | 66.9% | **88.7%** |
| A-Precision | 0.560 | **0.917** |
| A-Recall | 0.447 | **0.761** |
| A-F1 | 0.497 | **0.832** |
| R-Recall | 83.3% | **96.0%** |
| ROC-AUC | — | **0.861** |

Confusion matrix: TP=395, FP=36, FN=124, TN=860

### Why Agreement Outperforms Clf1

1. **Extremely high A-precision (92%)**: When nor_qa and oner_qa agree, the answer is almost certainly correct without retrieval. Only 28 (xl) or 36 (xxl) false positives out of ~350–430 agreement cases. This means the gate almost never incorrectly skips retrieval.

2. **A-recall nearly doubles** (42.6%→73.8% xl, 44.7%→76.1% xxl): Agreement captures most genuinely known answers, while Clf1 was biased by class imbalance toward predicting R.

3. **The signal is inherent to the model**: Agreement directly measures whether the LLM "knows" the answer (same answer with and without retrieval), whereas Clf1 must learn this indirectly from noisy silver labels.

4. **Robust R-recall (96–97%)**: When answers disagree, retrieval is almost always needed. This matches the observation from the threshold experiment that the cost of skipping retrieval is high.

---

## Comparison with Iteration 1 (Threshold Tuning)

| Method | xl Avg Acc | vs Default | xxl Avg Acc | vs Default |
|---|---|---|---|---|
| Default (Clf1 t=0.50) | 0.416 | — | 0.417 | — |
| Threshold 0.35 (Iter 1) | 0.388 | −0.028 | 0.380 | −0.037 |
| **Agreement (Iter 2a)** | **0.434** | **+0.018** | **0.455** | **+0.038** |

The agreement gate improves by the same magnitude that threshold tuning worsened — but in the positive direction. The gap between the two approaches is +4.6pp (xl) and +7.5pp (xxl).

---

## Analysis

### Why Threshold Harms but Agreement Helps

Both methods increase the number of A-routed questions, but with fundamentally different quality:

- **Threshold 0.35**: Forces more A predictions by lowering the bar uniformly. Many borderline questions get misrouted to A, hurting accuracy on datasets where retrieval helps (NQ: −5.6pp, TriviaQA: −6.6pp).

- **Agreement**: Only routes to A when there's strong evidence — the LLM gave the same answer with and without retrieval. This is a much more selective (19–21% vs 25–34% at threshold 0.35) but much more precise signal (A-precision 92% vs ~63%).

### Biggest Winners

- **2WikiMultiHopQA**: +8.2pp (xl), +5.0pp (xxl) — largest gains. Many questions in this dataset have answers the LLM knows without retrieval (e.g., factual dates, names), and the agreement signal correctly identifies these.

- **TriviaQA**: +1.8pp (xl), +8.0pp (xxl) — trivia facts are precisely the type of knowledge stored in parametric memory.

### Only Slight Loser

- **MuSiQue**: −0.8pp (xl), −0.2pp (xxl) — multi-hop compositional questions where answer agreement is coincidental rather than indicative of true knowledge. The agreement rate on MuSiQue is the lowest of all datasets (48/500 xl, 37/500 xxl).

### No Training Required

The agreement gate eliminates Clf1 entirely — no training data, no checkpoint selection, no class imbalance issues. It works purely from existing prediction artifacts.

---

## File Locations

| Artifact | Path |
|---|---|
| Agreement router script | `classifier/postprocess/predict_complexity_agreement.py` |
| xl predictions | `predictions/classifier/t5-large/flan_t5_xl/split_agreement/nor_oner_clf2ep35/` |
| xxl predictions | `predictions/classifier/t5-large/flan_t5_xxl/split_agreement/nor_oner_clf2ep30/` |
| xl routing stats | `predictions/classifier/t5-large/flan_t5_xl/split_agreement/nor_oner_clf2ep35/routing_stats.json` |
| xxl routing stats | `predictions/classifier/t5-large/flan_t5_xxl/split_agreement/nor_oner_clf2ep30/routing_stats.json` |
| Evaluation results | `{output_path}/{dataset}/eval_metic_result_acc.json` |
| Default Clf1 results | `predictions/classifier/t5-large/{model}/split/{epoch_combo}/` |
| Threshold results | `predictions/classifier/t5-large/{model}/split_thresh035/{epoch_combo}/` |

## Commands to Reproduce

```bash
# flan_t5_xl
python classifier/postprocess/predict_complexity_agreement.py flan_t5_xl \
  --clf2_pred_file classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/single_vs_multi/epoch/35/2026_03_26/14_42_47/predict/dict_id_pred_results.json \
  --predict_file classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json \
  --output_path predictions/classifier/t5-large/flan_t5_xl/split_agreement/nor_oner_clf2ep35/

python evaluate_final_acc.py --pred_path predictions/classifier/t5-large/flan_t5_xl/split_agreement/nor_oner_clf2ep35/

# flan_t5_xxl
python classifier/postprocess/predict_complexity_agreement.py flan_t5_xxl \
  --clf2_pred_file classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xxl/single_vs_multi/epoch/30/2026_03_26/16_43_14/predict/dict_id_pred_results.json \
  --predict_file classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json \
  --output_path predictions/classifier/t5-large/flan_t5_xxl/split_agreement/nor_oner_clf2ep30/

python evaluate_final_acc.py --pred_path predictions/classifier/t5-large/flan_t5_xxl/split_agreement/nor_oner_clf2ep30/
```
