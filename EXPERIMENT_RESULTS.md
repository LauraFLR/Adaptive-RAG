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
