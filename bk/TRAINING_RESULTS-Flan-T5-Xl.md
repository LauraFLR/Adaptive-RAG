# Split Classifier Training Results — flan_t5_xl

Trained on silver labels from the `flan_t5_xl` model. Base classifier model: `t5-large`.

---

## Classifier 1 — no_retrieval (A) vs retrieval (R)

- **Training data**: `silver/no_retrieval_vs_retrieval/train.json` — 1292 samples (A:424, R:868)
- **Validation data**: `silver/no_retrieval_vs_retrieval/valid.json` — 1350 samples (A:439, R:911)

| Epoch | Overall Acc | A acc | A pred / gold | R acc | R pred / gold |
|---|---|---|---|---|---|
| 15 | 72.37% | 41.69% | 300 / 439 | 87.16% | 1050 / 911 |
| **20** | **73.19%** | 42.60% | 297 / 439 | 87.93% | 1053 / 911 |
| 25 | 72.67% | 47.15% | 344 / 439 | 84.96% | 1006 / 911 |
| 30 | 72.22% | 48.75% | 364 / 439 | 83.53% | 986 / 911 |
| 35 | 72.22% | 50.57% | 380 / 439 | 82.66% | 970 / 911 |

**Best overall: epoch 20 (73.19%)**

Over training, the model shifts from over-predicting R to gradually recognizing more A queries. R accuracy drops while A accuracy rises — the overall optimum is at epoch 20.

### Confusion matrix (epoch 20)

|  | Pred A | Pred R | Total |
|---|---|---|---|
| **Gold A** | **187** | 252 | 439 |
| **Gold R** | 110 | **801** | 911 |
| Total | 297 | 1053 | 1350 |

|  | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.630 | 0.426 | 0.508 |
| R | 0.761 | 0.879 | 0.816 |

The model has high R recall (0.879) but low A recall (0.426) — it catches most retrieval-needed queries at the cost of misclassifying many no-retrieval queries as needing retrieval.

---

## Classifier 2 — single (B) vs multi (C)

- **Training data**: `binary_silver_single_vs_multi/train.json` — 3268 samples (B:1871, C:1397), merged from silver + binary inductive-bias labels
- **Validation data**: `silver/single_vs_multi/valid.json` — 911 samples (B:691, C:220)

| Epoch | Overall Acc | B acc | B pred / gold | C acc | C pred / gold |
|---|---|---|---|---|---|
| 15 | 65.86% | 64.40% | 510 / 691 | 70.45% | 401 / 220 |
| 20 | 67.62% | 66.86% | 528 / 691 | 70.00% | 383 / 220 |
| 25 | 68.61% | 68.31% | 539 / 691 | 69.55% | 372 / 220 |
| 30 | 70.91% | 70.62% | 550 / 691 | 71.82% | 361 / 220 |
| **35** | **71.35%** | 71.92% | 564 / 691 | 69.55% | 347 / 220 |

**Best overall: epoch 35 (71.35%)**

Accuracy was still climbing at epoch 35. May benefit from additional epochs (40+).

### Confusion matrix (epoch 35)

|  | Pred B | Pred C | Total |
|---|---|---|---|
| **Gold B** | **497** | 194 | 691 |
| **Gold C** | 67 | **153** | 220 |
| Total | 564 | 347 | 911 |

|  | Precision | Recall | F1 |
|---|---|---|---|
| B | 0.881 | 0.719 | 0.792 |
| C | 0.441 | 0.695 | 0.540 |

High B precision (0.881) but moderate C precision (0.441) — the model over-predicts C (347 predicted vs 220 gold), reflecting its tendency to err on the side of multi-step retrieval.

---

## Cascaded 3-class confusion matrix (includes propagated errors from classifier 1)

Using best epochs (clf1 epoch 20 + clf2 epoch 35), evaluated on the 1,098 validation samples where both classifiers have predictions (252 gold-A samples misrouted to clf2 at stage 1 are excluded since clf2 was not validated on them):

|  | Pred A | Pred B | Pred C | Total |
|---|---|---|---|---|
| **Gold A** | **187** | 0 | 0 | 187 |
| **Gold B** | 98 | **454** | 139 | 691 |
| **Gold C** | 12 | 63 | **145** | 220 |
| Total | 297 | 517 | 284 | 1098 |

**Overall accuracy: 786/1098 = 71.58%**

|  | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.630 | 1.000 | 0.773 |
| B | 0.878 | 0.657 | 0.752 |
| C | 0.511 | 0.659 | 0.575 |

Note: A recall appears as 1.000 because the 252 gold-A samples that clf1 misclassified as R are excluded (no clf2 prediction available). Including them would lower A recall to 0.426 (matching clf1).

---

## Test set QA results

Routed predictions using best checkpoints (clf1 epoch 20 + clf2 epoch 35) on 3,000 test questions (500 per dataset).

### Routing distribution

| Dataset | A (no retrieval) | B (single-step) | C (multi-step) | Total steps |
|---|---|---|---|---|
| musique | 10 | 70 | 420 | 1,608 |
| hotpotqa | 65 | 125 | 310 | 1,784 |
| 2wikimultihopqa | 223 | 29 | 248 | 1,027 |
| nq | 43 | 457 | 0 | 457 |
| trivia | 50 | 412 | 38 | 635 |
| squad | 14 | 437 | 49 | 644 |
| **Total** | **405** | **1,530** | **1,065** | **6,155** |

### QA accuracy

| Dataset | EM | F1 | Acc |
|---|---|---|---|
| musique | 0.232 | 0.321 | 0.260 |
| hotpotqa | 0.422 | 0.538 | 0.442 |
| 2wikimultihopqa | 0.390 | 0.478 | 0.444 |
| nq | 0.368 | 0.464 | 0.436 |
| trivia | 0.516 | 0.603 | 0.580 |
| squad | 0.274 | 0.389 | 0.334 |

---

## Output locations

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/
├── no_ret_vs_ret/epoch/{15,20,25,30,35}/2026_03_26/14_06_07/
│   ├── model.safetensors, config.json, ...   (trained model)
│   ├── valid/                                 (validation metrics)
│   └── predict/                               (test set predictions, unlabeled)
└── single_vs_multi/epoch/{15,20,25,30,35}/2026_03_26/14_42_47/
    ├── model.safetensors, config.json, ...
    ├── valid/
    └── predict/

predictions/classifier/t5-large/flan_t5_xl/split/no_ret_ep20_single_ep35/
├── {dataset}/
│   ├── {dataset}.json              (routed QA predictions)
│   ├── {dataset}_option.json       (predictions with routing labels + step counts)
│   └── eval_metic_result_acc.json  (EM, F1, Acc)
```

---

## Notes

- Labels are **silver** (noisy by construction — derived from which QA strategy happened to produce a correct answer for flan_t5_xl), so ~71–73% accuracy is reasonable.
- Classifier 1 has a 2:1 class imbalance (R vs A) in training data, which biases early predictions toward R.
- Classifier 2 used merged binary+silver data (3268 samples), which improved class balance compared to silver-only (868 samples with B:671 vs C:197).
