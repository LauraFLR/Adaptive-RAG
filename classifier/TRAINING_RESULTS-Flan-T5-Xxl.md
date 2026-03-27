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

---

## Comparison with flan_t5_xl labels

| Classifier | flan_t5_xl best | flan_t5_xxl best | Δ |
|---|---|---|---|
| no_ret_vs_ret | 73.19% (ep 20) | 66.86% (ep 35) | −6.33 pp |
| single_vs_multi | 71.35% (ep 35) | 68.08% (ep 30) | −3.27 pp |

The classifiers trained on flan_t5_xxl silver labels perform **lower** than those trained on flan_t5_xl labels (−6.3 pp and −3.3 pp respectively). This likely reflects noisier silver labels from the xxl model — a different QA strategy may produce correct answers for xxl vs xl, making the label distribution harder for the classifier to learn.

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
```

---

## Notes

- Labels are **silver** (noisy by construction — derived from which QA strategy happened to produce a correct answer for flan_t5_xxl), so ~66–68% accuracy is within a reasonable range given label noise.
- Classifier 1 has a ~1.76:1 class imbalance (R:898 vs A:511) in training, which biases predictions toward R.
- Classifier 2 used merged binary+silver data (3298 samples), improving class balance compared to silver-only validation (B:701 vs C:195 ≈ 3.6:1 ratio).
- All epochs (15–35) completed successfully for both classifiers, with model checkpoints, validation metrics, and test predictions saved.
