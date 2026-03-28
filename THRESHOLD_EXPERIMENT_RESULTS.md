# Clf1 Threshold Experiment Results

## Motivation

The split-classifier architecture uses two cascaded classifiers:
- **Clf1** (A vs R): Determines whether a query needs retrieval (R) or not (A)
- **Clf2** (B vs C): For retrieval queries, determines single-step (B) vs multi-step (C)

Training data has a **2:1 imbalance** (R:A), causing Clf1 to favour predicting R.
A threshold sweep on the validation set found **0.35** as an optimal decision boundary
for both flan_t5_xl and flan_t5_xxl, significantly improving A-class recall.

**Hypothesis**: Lowering the threshold from 0.50 to 0.35 should route more queries to
the no-retrieval path (A), improving efficiency. If classifier recall for A was genuinely
too low, this may also improve or maintain downstream QA accuracy.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Clf1 threshold (default) | 0.50 (argmax) |
| Clf1 threshold (adjusted) | 0.35 |
| Clf2 threshold | 0.50 (argmax, unchanged) |
| Test set | 3000 questions (500 per dataset) |
| Datasets | NQ, TriviaQA, SQuAD, MuSiQue, HotpotQA, 2WikiMultiHopQA |
| QA backbone | flan_t5_xl, flan_t5_xxl |
| Metrics | EM, F1, Accuracy |

**Script**: `classifier/postprocess/predict_complexity_threshold.py`

---

## Clf1 Routing Distribution Changes

### flan_t5_xl (Clf1 epoch 20, Clf2 epoch 35)

| Threshold | A (total) | R (total) | A% |
|---|---|---|---|
| 0.50 (default) | 405 | 2595 | 13.5% |
| 0.35 (adjusted) | 743 | 2257 | 24.8% |
| **Change** | **+338** | **−338** | **+11.3pp** |

### flan_t5_xxl (Clf1 epoch 35, Clf2 epoch 30)

| Threshold | A (total) | R (total) | A% |
|---|---|---|---|
| 0.50 (default) | 612 | 2388 | 20.4% |
| 0.35 (adjusted) | 1029 | 1971 | 34.3% |
| **Change** | **+417** | **−417** | **+13.9pp** |

---

## Per-Dataset Routing Distributions

### flan_t5_xl

| Dataset | Threshold | A | B | C | Steps |
|---|---|---|---|---|---|
| NQ | 0.50 | 43 | 457 | 0 | 457 |
| NQ | **0.35** | **142** | **358** | **0** | **358** |
| TriviaQA | 0.50 | 50 | 412 | 38 | 635 |
| TriviaQA | **0.35** | **154** | **318** | **28** | **499** |
| SQuAD | 0.50 | 14 | 437 | 49 | 644 |
| SQuAD | **0.35** | **56** | **400** | **44** | **588** |
| MuSiQue | 0.50 | 10 | 70 | 420 | 1608 |
| MuSiQue | **0.35** | **33** | **67** | **400** | **1541** |
| HotpotQA | 0.50 | 65 | 125 | 310 | 1784 |
| HotpotQA | **0.35** | **122** | **108** | **270** | **1558** |
| 2WikiMHQA | 0.50 | 223 | 29 | 248 | 1027 |
| 2WikiMHQA | **0.35** | **236** | **29** | **235** | **967** |

### flan_t5_xxl

| Dataset | Threshold | A | B | C | Steps |
|---|---|---|---|---|---|
| NQ | 0.50 | 118 | 380 | 2 | 384 |
| NQ | **0.35** | **191** | **307** | **2** | **311** |
| TriviaQA | 0.50 | 181 | 292 | 27 | 350 |
| TriviaQA | **0.35** | **297** | **189** | **14** | **219** |
| SQuAD | 0.50 | 62 | 386 | 52 | 491 |
| SQuAD | **0.35** | **123** | **333** | **44** | **422** |
| MuSiQe | 0.50 | 20 | 67 | 413 | 931 |
| MuSiQue | **0.35** | **50** | **61** | **389** | **876** |
| HotpotQA | 0.50 | 91 | 106 | 303 | 734 |
| HotpotQA | **0.35** | **158** | **81** | **261** | **620** |
| 2WikiMHQA | 0.50 | 140 | 74 | 286 | 696 |
| 2WikiMHQA | **0.35** | **210** | **47** | **243** | **582** |

---

## End-to-End QA Results

### flan_t5_xl

| Dataset | Threshold | EM | F1 | Acc | Δ Acc |
|---|---|---|---|---|---|
| NQ | 0.50 | 0.368 | 0.464 | 0.436 | — |
| NQ | **0.35** | **0.318** | **0.413** | **0.380** | **−0.056** |
| TriviaQA | 0.50 | 0.516 | 0.603 | 0.580 | — |
| TriviaQA | **0.35** | **0.458** | **0.544** | **0.514** | **−0.066** |
| SQuAD | 0.50 | 0.274 | 0.389 | 0.334 | — |
| SQuAD | **0.35** | **0.254** | **0.371** | **0.312** | **−0.022** |
| MuSiQue | 0.50 | 0.232 | 0.321 | 0.260 | — |
| MuSiQue | **0.35** | **0.224** | **0.317** | **0.254** | **−0.006** |
| HotpotQA | 0.50 | 0.422 | 0.538 | 0.442 | — |
| HotpotQA | **0.35** | **0.404** | **0.520** | **0.426** | **−0.016** |
| 2WikiMHQA | 0.50 | 0.390 | 0.478 | 0.444 | — |
| 2WikiMHQA | **0.35** | **0.390** | **0.477** | **0.442** | **−0.002** |
| **Average** | **0.50** | **0.367** | **0.466** | **0.416** | **—** |
| **Average** | **0.35** | **0.341** | **0.440** | **0.388** | **−0.028** |

### flan_t5_xxl

| Dataset | Threshold | EM | F1 | Acc | Δ Acc |
|---|---|---|---|---|---|
| NQ | 0.50 | 0.378 | 0.475 | 0.434 | — |
| NQ | **0.35** | **0.348** | **0.441** | **0.398** | **−0.036** |
| TriviaQA | 0.50 | 0.484 | 0.569 | 0.540 | — |
| TriviaQA | **0.35** | **0.432** | **0.515** | **0.484** | **−0.056** |
| SQuAD | 0.50 | 0.266 | 0.384 | 0.322 | — |
| SQuAD | **0.35** | **0.238** | **0.350** | **0.288** | **−0.034** |
| MuSiQue | 0.50 | 0.210 | 0.293 | 0.238 | — |
| MuSiQue | **0.35** | **0.206** | **0.292** | **0.236** | **−0.002** |
| HotpotQA | 0.50 | 0.402 | 0.512 | 0.430 | — |
| HotpotQA | **0.35** | **0.358** | **0.473** | **0.386** | **−0.044** |
| 2WikiMHQA | 0.50 | 0.478 | 0.575 | 0.540 | — |
| 2WikiMHQA | **0.35** | **0.426** | **0.522** | **0.488** | **−0.052** |
| **Average** | **0.50** | **0.370** | **0.468** | **0.417** | **—** |
| **Average** | **0.35** | **0.335** | **0.432** | **0.380** | **−0.037** |

---

## Efficiency Comparison (Total Retrieval Steps)

| Model | Threshold | Total Steps | Δ Steps | Δ% |
|---|---|---|---|---|
| flan_t5_xl | 0.50 | 6,155 | — | — |
| flan_t5_xl | **0.35** | **5,511** | **−644** | **−10.5%** |
| flan_t5_xxl | 0.50 | 3,586 | — | — |
| flan_t5_xxl | **0.35** | **3,030** | **−556** | **−15.5%** |

---

## Analysis

### Key Finding: Threshold 0.35 Hurts QA Accuracy

Lowering the Clf1 threshold from 0.50 to 0.35 **decreases** downstream QA accuracy
across all datasets and both models:

- **flan_t5_xl**: Average accuracy drops from 0.416 → 0.388 (−2.8pp)
- **flan_t5_xxl**: Average accuracy drops from 0.417 → 0.380 (−3.7pp)

The largest drops occur on datasets where single-hop retrieval (B) is the dominant
correct strategy (NQ: −5.6pp xl, −3.6pp xxl; TriviaQA: −6.6pp xl, −5.6pp xxl).
These are precisely the datasets where the threshold shift moves the most B-questions
to A (no retrieval).

### Why This Happens

The threshold sweep on the validation set optimised for **Clf1 classification accuracy**
(A vs R as measured against silver labels). However:

1. **Silver labels may be noisy**: The A/B/C labels come from a heuristic (comparing
   QA strategy outputs), not gold annotations. Clf1 at 0.50 may actually be making
   better *downstream* decisions than the silver labels suggest.

2. **Retrieval rarely hurts**: Sending a question through single-step retrieval (B)
   when it could have been answered without retrieval (A) has low cost — the retrieval
   may confirm the answer or cause only minor degradation. The reverse (skipping
   retrieval for a question that needs it) is much more costly.

3. **Asymmetric error costs**: The threshold sweep treats A→R and R→A misclassifications
   symmetrically, but on QA accuracy, false positives for A (classifying R questions
   as A) are far more damaging than false negatives (classifying A questions as R).

### Efficiency vs Accuracy Trade-off

While the threshold adjustment reduces retrieval steps by 10–15%, the QA accuracy
cost is significant. The per-step accuracy cost is:

| Model | Δ Acc (avg) | Δ Steps | Acc Cost per 100 Steps Saved |
|---|---|---|---|
| flan_t5_xl | −0.028 | −644 | 0.43pp |
| flan_t5_xxl | −0.037 | −556 | 0.67pp |

### Recommendation

**Keep the default threshold (0.50 / argmax) for Clf1.** The classification-level
improvement from threshold adjustment does not translate to downstream QA improvements.
The underlying issue is that the silver labels used for Clf1 training may assign "A"
to questions where the LLM happened to guess correctly without retrieval, but these
questions still benefit from retrieval confirmation in practice.

Potential directions instead:
- **Re-examine silver labelling criteria** with stricter conditions for A assignment
- **Train with asymmetric loss** that penalises A→R errors less than R→A errors
- **Explore per-dataset thresholds** rather than a single global threshold

---

## File Locations

| Artifact | Path |
|---|---|
| Threshold routing script | `classifier/postprocess/predict_complexity_threshold.py` |
| XL predictions (thresh 0.35) | `predictions/classifier/t5-large/flan_t5_xl/split_thresh035/no_ret_ep20_single_ep35/` |
| XXL predictions (thresh 0.35) | `predictions/classifier/t5-large/flan_t5_xxl/split_thresh035/no_ret_ep35_single_ep30/` |
| XL predictions (default 0.50) | `predictions/classifier/t5-large/flan_t5_xl/split/no_ret_ep20_single_ep35/` |
| XXL predictions (default 0.50) | `predictions/classifier/t5-large/flan_t5_xxl/split/no_ret_ep35_single_ep30/` |
| Threshold sweep results (xl) | `classifier/outputs/.../flan_t5_xl/no_ret_vs_ret/epoch/20/.../threshold_sweep/` |
| Threshold sweep results (xxl) | `classifier/outputs/.../flan_t5_xxl/no_ret_vs_ret/epoch/35/.../threshold_sweep/` |
