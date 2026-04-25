# Split Classifier Training Results — flan_t5_xxl

Trained on silver labels from the `flan_t5_xxl` model. Base classifier model: `t5-large`.

---

## Classifier 1 — no_retrieval (A) vs retrieval (R)

- **Training data**: `silver/no_retrieval_vs_retrieval/train.json` — 1409 samples (A:511, R:898)
- **Validation data**: `silver/no_retrieval_vs_retrieval/valid.json` — 1415 samples (A:519, R:896)

| Epoch | Overall Acc | A acc | A pred / gold | R acc | R pred / gold |
|---|---|---|---|---|---|
| 15 | 65.44% | 36.42% | 348 / 519 | 82.25% | 1067 / 896 |
| 20 | 66.64% | 42.20% | 391 / 519 | 80.80% | 1024 / 896 |
| 25 | 64.59% | 42.00% | 418 / 519 | 77.68% | 997 / 896 |
| 30 | 66.36% | 39.50% | 367 / 519 | 81.92% | 1048 / 896 |
| **35** | **66.86%** | 44.70% | 414 / 519 | 79.69% | 1001 / 896 |

**Best overall: epoch 35 (66.86%)**

The model shows a similar pattern to the flan_t5_xl version — heavy initial bias toward R, gradually learning to recognise A queries over training. Overall accuracy peaks at epoch 35, though gains are small past epoch 20. The model still over-predicts R (1001 predicted vs 896 gold at epoch 35).

### Confusion matrix (epoch 35)

|  | Pred A | Pred R | Total |
|---|---|---|---|
| **Gold A** | **232** | 287 | 519 |
| **Gold R** | 182 | **714** | 896 |
| Total | 414 | 1001 | 1415 |

|  | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.560 | 0.447 | 0.497 |
| R | 0.713 | 0.797 | 0.753 |

Similar to xl: high R recall (0.797) but low A recall (0.447). Both precision and recall are lower than xl across the board, reflecting noisier silver labels.

---

## Classifier 2 — single (B) vs multi (C)

- **Training data**: `binary_silver_single_vs_multi/train.json` — 3298 samples (B:1903, C:1395), merged from silver + binary inductive-bias labels
- **Validation data**: `silver/single_vs_multi/valid.json` — 896 samples (B:701, C:195)

| Epoch | Overall Acc | B acc | B pred / gold | C acc | C pred / gold |
|---|---|---|---|---|---|
| 15 | 65.63% | 64.05% | 505 / 701 | 71.28% | 391 / 195 |
| 20 | 65.74% | 64.48% | 510 / 701 | 70.26% | 386 / 195 |
| 25 | 67.63% | 66.90% | 527 / 701 | 70.26% | 369 / 195 |
| **30** | **68.08%** | 67.48% | 531 / 701 | 70.26% | 365 / 195 |
| 35 | 67.30% | 67.05% | 532 / 701 | 68.21% | 364 / 195 |

**Best overall: epoch 30 (68.08%)**

Accuracy peaks at epoch 30 and slightly drops at 35, suggesting mild overfitting. The model consistently over-predicts C (365–391 predicted vs 195 gold), reflecting the heavy class imbalance in validation data (B:701 vs C:195).

### Confusion matrix (epoch 30)

|  | Pred B | Pred C | Total |
|---|---|---|---|
| **Gold B** | **473** | 228 | 701 |
| **Gold C** | 58 | **137** | 195 |
| Total | 531 | 365 | 896 |

|  | Precision | Recall | F1 |
|---|---|---|---|
| B | 0.891 | 0.675 | 0.768 |
| C | 0.375 | 0.703 | 0.489 |

High B precision (0.891) but low C precision (0.375) — the model heavily over-predicts C (365 predicted vs 195 gold). C recall is decent (0.703), confirming the model errs toward multi-step retrieval.

---

## Cascaded 3-class confusion matrix (includes propagated errors from classifier 1)

Using best epochs (clf1 epoch 35 + clf2 epoch 30), evaluated on the 1,128 validation samples where both classifiers have predictions (287 gold-A samples misrouted to clf2 at stage 1 are excluded since clf2 was not validated on them):

|  | Pred A | Pred B | Pred C | Total |
|---|---|---|---|---|
| **Gold A** | **232** | 0 | 0 | 232 |
| **Gold B** | 162 | **369** | 170 | 701 |
| **Gold C** | 20 | 48 | **127** | 195 |
| Total | 414 | 417 | 297 | 1128 |

**Overall accuracy: 728/1128 = 64.54%**

|  | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.560 | 1.000 | 0.718 |
| B | 0.885 | 0.526 | 0.660 |
| C | 0.428 | 0.651 | 0.516 |

Note: A recall appears as 1.000 because the 287 gold-A samples that clf1 misclassified as R are excluded (no clf2 prediction available). Including them would lower A recall to 0.447 (matching clf1).

---

## Comparison with flan_t5_xl labels

| Classifier | flan_t5_xl best | flan_t5_xxl best | Δ |
|---|---|---|---|
| no_ret_vs_ret | 73.19% (ep 20) | 66.86% (ep 35) | −6.33 pp |
| single_vs_multi | 71.35% (ep 35) | 68.08% (ep 30) | −3.27 pp |
| Cascaded 3-class | 71.58% | 64.54% | −7.04 pp |

The classifiers trained on flan_t5_xxl silver labels perform **lower** than those trained on flan_t5_xl labels across all metrics. This likely reflects noisier silver labels from the xxl model — a different QA strategy may produce correct answers for xxl vs xl, making the label distribution harder for the classifier to learn.

---

## Test set QA results

Routed predictions using best checkpoints (clf1 epoch 35 + clf2 epoch 30) on 3,000 test questions (500 per dataset).

### Routing distribution

| Dataset | A (no retrieval) | B (single-step) | C (multi-step) | Total steps |
|---|---|---|---|---|
| musique | 20 | 67 | 413 | 931 |
| hotpotqa | 91 | 106 | 303 | 734 |
| 2wikimultihopqa | 140 | 74 | 286 | 696 |
| nq | 118 | 380 | 2 | 384 |
| trivia | 181 | 292 | 27 | 350 |
| squad | 62 | 386 | 52 | 491 |
| **Total** | **612** | **1,305** | **1,083** | **3,586** |

### QA accuracy

| Dataset | EM | F1 | Acc |
|---|---|---|---|
| musique | 0.210 | 0.293 | 0.238 |
| hotpotqa | 0.402 | 0.512 | 0.430 |
| 2wikimultihopqa | 0.478 | 0.575 | 0.540 |
| nq | 0.378 | 0.475 | 0.434 |
| trivia | 0.484 | 0.569 | 0.540 |
| squad | 0.266 | 0.384 | 0.322 |

### Comparison with flan_t5_xl QA results

| Dataset | xl F1 | xxl F1 | Δ |
|---|---|---|---|
| musique | 0.321 | 0.293 | −0.028 |
| hotpotqa | 0.538 | 0.512 | −0.026 |
| 2wikimultihopqa | 0.478 | 0.575 | **+0.097** |
| nq | 0.464 | 0.475 | +0.011 |
| trivia | 0.603 | 0.569 | −0.034 |
| squad | 0.389 | 0.384 | −0.005 |

Despite lower classifier accuracy, xxl outperforms xl on 2wikimultihopqa (+0.097 F1) and nq (+0.011 F1), likely due to the stronger underlying QA model. xl is better on the remaining four datasets.

---

## Output locations

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xxl/
├── no_ret_vs_ret/epoch/{15,20,25,30,35}/2026_03_26/16_07_15/
│   ├── model.safetensors, config.json, ...   (trained model)
│   ├── valid/                                 (validation metrics)
│   └── predict/                               (test set predictions, unlabeled)
└── single_vs_multi/epoch/{15,20,25,30,35}/2026_03_26/16_43_14/
    ├── model.safetensors, config.json, ...
    ├── valid/
    └── predict/

predictions/classifier/t5-large/flan_t5_xxl/split/no_ret_ep35_single_ep30/
├── {dataset}/
│   ├── {dataset}.json              (routed QA predictions)
│   ├── {dataset}_option.json       (predictions with routing labels + step counts)
│   └── eval_metic_result_acc.json  (EM, F1, Acc)
```

---

## Notes

- Labels are **silver** (noisy by construction — derived from which QA strategy happened to produce a correct answer for flan_t5_xxl), so ~66–68% accuracy is within a reasonable range given label noise.
- Classifier 1 has a ~1.76:1 class imbalance (R:898 vs A:511) in training, which biases predictions toward R.
- Classifier 2 used merged binary+silver data (3298 samples), improving class balance compared to silver-only validation (B:701 vs C:195 ≈ 3.6:1 ratio).
- All epochs (15–35) completed successfully for both classifiers, with model checkpoints, validation metrics, and test predictions saved.
