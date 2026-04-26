# Experiment Results — All Models × Datasets × Iterations

## F1 Scores

### Iteration 1 — Cascade Baseline (Clf1 → Clf2)

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | **Avg** |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.464 | 0.603 | 0.389 | 0.321 | 0.538 | 0.478 | **0.466** |
| Flan-T5-XXL | 0.475 | 0.569 | 0.384 | 0.293 | 0.512 | 0.575 | **0.468** |
| GPT | 0.557 | 0.751 | 0.310 | 0.328 | 0.520 | 0.584 | **0.508** |

Best epochs — XL: Clf1 ep20 / Clf2 ep35, XXL: Clf1 ep35 / Clf2 ep30, GPT: Clf1 ep35 / Clf2 ep35.

### Iteration 2 — Agreement Gate (replaces Clf1) + Clf2

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | **Avg** |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.473 | 0.619 | 0.390 | 0.316 | 0.545 | 0.568 | **0.485** |
| Flan-T5-XXL | 0.510 | 0.645 | 0.402 | 0.288 | 0.554 | 0.631 | **0.505** |
| GPT | 0.469 | 0.662 | 0.337 | 0.330 | 0.581 | 0.646 | **0.504** |

### Iteration 2 — Oracle Ceiling (agreement gate + perfect Clf2)

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | **Avg** |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.520 | 0.660 | 0.413 | 0.328 | 0.573 | 0.557 | **0.509** |
| Flan-T5-XXL | 0.548 | 0.669 | 0.426 | 0.328 | 0.589 | 0.615 | **0.529** |
| GPT | 0.526 | 0.700 | 0.357 | 0.342 | 0.592 | 0.617 | **0.522** |

### Iteration 3 — Agreement Gate + Feature-Augmented Clf2

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | **Avg** |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.473 | 0.619 | 0.386 | 0.317 | 0.550 | 0.569 | **0.486** |
| Flan-T5-XXL | 0.512 | 0.645 | 0.402 | 0.292 | 0.557 | 0.632 | **0.507** |
| GPT | 0.468 | 0.665 | 0.339 | 0.338 | 0.573 | 0.647 | **0.505** |

Best feat Clf2 epochs — XL: ep25, XXL: ep35, GPT: ep25.

---

## EM Scores

### Iteration 1 — Cascade Baseline

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | **Avg** |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.368 | 0.516 | 0.274 | 0.232 | 0.422 | 0.390 | **0.367** |
| Flan-T5-XXL | 0.378 | 0.484 | 0.266 | 0.210 | 0.402 | 0.478 | **0.370** |
| GPT | 0.398 | 0.630 | 0.166 | 0.222 | 0.386 | 0.460 | **0.377** |

### Iteration 2 — Agreement Gate + Clf2

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | **Avg** |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.378 | 0.530 | 0.276 | 0.228 | 0.428 | 0.480 | **0.387** |
| Flan-T5-XXL | 0.412 | 0.560 | 0.280 | 0.210 | 0.440 | 0.532 | **0.406** |
| GPT | 0.324 | 0.544 | 0.178 | 0.232 | 0.458 | 0.510 | **0.374** |

### Iteration 2 — Oracle Ceiling

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | **Avg** |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.432 | 0.580 | 0.304 | 0.246 | 0.466 | 0.500 | **0.421** |
| Flan-T5-XXL | 0.454 | 0.590 | 0.312 | 0.248 | 0.484 | 0.548 | **0.439** |
| GPT | 0.400 | 0.598 | 0.206 | 0.252 | 0.494 | 0.514 | **0.411** |

### Iteration 3 — Agreement Gate + Feature-Augmented Clf2

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | **Avg** |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.378 | 0.530 | 0.270 | 0.230 | 0.432 | 0.478 | **0.386** |
| Flan-T5-XXL | 0.414 | 0.560 | 0.282 | 0.214 | 0.440 | 0.536 | **0.408** |
| GPT | 0.326 | 0.550 | 0.184 | 0.238 | 0.452 | 0.512 | **0.377** |

---

## Cross-Iteration Comparison (Avg F1)

| Model | Iter 1 | Iter 2 | Δ 1→2 | Oracle | Iter 3 | Δ 2→3 |
|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.466 | 0.485 | +1.9 pp | 0.509 | 0.486 | +0.1 pp |
| Flan-T5-XXL | 0.468 | 0.505 | +3.7 pp | 0.529 | 0.507 | +0.2 pp |
| GPT | 0.508 | 0.504 | −0.4 pp | 0.522 | 0.505 | +0.1 pp |

---

## Classifier Confusion Matrices (Validation Set)

### Clf1 — A (no retrieval) vs R (retrieval needed)

#### Flan-T5-XL (ep20, n=1350, acc=73.2%)

| Gold \ Pred | A | R | Total |
|---|---|---|---|
| **A** | 187 | 252 | 439 |
| **R** | 110 | 801 | 911 |
| Pred Total | 297 | 1053 | |

|  | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.630 | 0.426 | 0.508 |
| R | 0.761 | 0.879 | 0.816 |

#### Flan-T5-XXL (ep35, n=1415, acc=66.9%)

| Gold \ Pred | A | R | Total |
|---|---|---|---|
| **A** | 232 | 287 | 519 |
| **R** | 182 | 714 | 896 |
| Pred Total | 414 | 1001 | |

|  | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.560 | 0.447 | 0.497 |
| R | 0.713 | 0.797 | 0.753 |

#### GPT (ep35, n=1431, acc=72.5%)

| Gold \ Pred | A | R | Total |
|---|---|---|---|
| **A** | 934 | 104 | 1038 |
| **R** | 290 | 103 | 393 |
| Pred Total | 1224 | 207 | |

|  | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.763 | 0.900 | 0.826 |
| R | 0.498 | 0.262 | 0.343 |

> GPT's Clf1 is heavily biased toward A: it correctly identifies when no retrieval is needed (A-recall 90%) but misses 74% of questions that *do* need retrieval. This is the "self-knowledge blindness" that motivates the agreement gate in Iteration 2.

---

### Clf2 — B (single-step) vs C (multi-step retrieval)

#### Flan-T5-XL (ep35, n=911, acc=71.4%)

| Gold \ Pred | B | C | Total |
|---|---|---|---|
| **B** | 497 | 194 | 691 |
| **C** | 67 | 153 | 220 |
| Pred Total | 564 | 347 | |

|  | Precision | Recall | F1 |
|---|---|---|---|
| B | 0.881 | 0.719 | 0.792 |
| C | 0.441 | 0.695 | 0.540 |

#### Flan-T5-XXL (ep30, n=896, acc=68.1%)

| Gold \ Pred | B | C | Total |
|---|---|---|---|
| **B** | 473 | 228 | 701 |
| **C** | 58 | 137 | 195 |
| Pred Total | 531 | 365 | |

|  | Precision | Recall | F1 |
|---|---|---|---|
| B | 0.891 | 0.675 | 0.768 |
| C | 0.375 | 0.703 | 0.489 |

#### GPT (ep35, n=393, acc=53.2%)

| Gold \ Pred | B | C | Total |
|---|---|---|---|
| **B** | 128 | 144 | 272 |
| **C** | 40 | 81 | 121 |
| Pred Total | 168 | 225 | |

|  | Precision | Recall | F1 |
|---|---|---|---|
| B | 0.762 | 0.471 | 0.582 |
| C | 0.360 | 0.669 | 0.468 |

> GPT's Clf2 has the smallest validation set (393 vs ~900) and lowest accuracy (53.2%). All three models show a C-over-prediction bias (low C-precision), but GPT's is the most severe.

---

### Feature-Augmented Clf2 — B vs C (Iteration 3)

#### Flan-T5-XL (ep25, n=911, acc=71.7%)

| Gold \ Pred | B | C | Total |
|---|---|---|---|
| **B** | 501 | 190 | 691 |
| **C** | 68 | 152 | 220 |
| Pred Total | 569 | 342 | |

|  | Precision | Recall | F1 |
|---|---|---|---|
| B | 0.880 | 0.725 | 0.795 |
| C | 0.444 | 0.691 | 0.541 |

#### Flan-T5-XXL (ep35, n=896, acc=71.8%)

| Gold \ Pred | B | C | Total |
|---|---|---|---|
| **B** | 507 | 194 | 701 |
| **C** | 59 | 136 | 195 |
| Pred Total | 566 | 330 | |

|  | Precision | Recall | F1 |
|---|---|---|---|
| B | 0.896 | 0.723 | 0.800 |
| C | 0.412 | 0.697 | 0.518 |

#### GPT (ep25, n=393, acc=59.8%)

| Gold \ Pred | B | C | Total |
|---|---|---|---|
| **B** | 148 | 124 | 272 |
| **C** | 34 | 87 | 121 |
| Pred Total | 182 | 211 | |

|  | Precision | Recall | F1 |
|---|---|---|---|
| B | 0.813 | 0.544 | 0.652 |
| C | 0.412 | 0.719 | 0.524 |

> Feature augmentation has minimal effect on XL (+0.3 pp acc) but meaningfully helps GPT (+6.6 pp acc), primarily by improving B-recall (47.1% → 54.4%). Despite this validation improvement, the downstream QA F1 gain is only +0.1 pp, confirming that classifier accuracy does not translate linearly into end-to-end QA improvement.
>
> **Note on sample size:** GPT's Clf2 validation set (n=393) is less than half the size of XL's (n=911) and XXL's (n=896), because fewer GPT silver labels survive the B/C filtering. The 6.6 pp validation gain therefore has substantially higher variance — a 95% binomial CI at n=393 is ±4.8 pp vs ±3.2 pp at n=911. This makes the disconnect between validation accuracy and end-to-end QA F1 even more expected.
