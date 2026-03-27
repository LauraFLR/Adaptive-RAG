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
```

---

## Notes

- Labels are **silver** (noisy by construction — derived from which QA strategy happened to produce a correct answer for flan_t5_xl), so ~71–73% accuracy is reasonable.
- Classifier 1 has a 2:1 class imbalance (R vs A) in training data, which biases early predictions toward R.
- Classifier 2 used merged binary+silver data (3268 samples), which improved class balance compared to silver-only (868 samples with B:671 vs C:197).
