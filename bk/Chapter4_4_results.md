# Chapter 4.4 Results

This summary uses the latest split-classifier runs completed on April 12-13, 2026:

- Clf1 (`no_ret_vs_ret`) runs from `2026_04_12`
- Clf2 (`single_vs_multi`) runs from `2026_04_13`

For end-to-end QA F1, each model uses the best Clf2 epoch from the April 13 run and pairs it with every Clf1 epoch from the April 12 run.

## Best epoch per model

| Model | Best Clf2 epoch | Clf2 valid acc | Best Clf1 epoch by QA F1 | Best overall QA F1 |
|---|---:|---:|---:|---:|
| `flan_t5_xl` | 35 | 70.47 | 20 | 46.26 |
| `flan_t5_xxl` | 35 | 67.63 | 20 | 47.31 |
| `gpt` | 40 | 54.20 | 40 | 51.22 |

QA F1 values are percentages.

## Per-epoch QA F1 and Clf1 A metrics

### flan_t5_xl

Best Clf2 epoch used for routing: `35`

| Clf1 epoch | Overall QA F1 | Clf1 A-recall | Clf1 A-F1 |
|---|---:|---:|---:|
| 15 | 46.25 | 41.91 | 49.80 |
| 20 | **46.26** | 42.82 | 50.67 |
| 25 | 46.05 | 46.92 | 53.02 |
| 30 | 45.86 | 48.75 | 53.70 |
| 35 | 45.40 | 51.03 | 54.17 |

### flan_t5_xxl

Best Clf2 epoch used for routing: `35`

| Clf1 epoch | Overall QA F1 | Clf1 A-recall | Clf1 A-F1 |
|---|---:|---:|---:|
| 15 | 46.92 | 34.30 | 41.49 |
| 20 | **47.31** | 37.96 | 45.50 |
| 25 | 45.81 | 46.05 | 49.18 |
| 30 | 46.12 | 44.89 | 48.90 |
| 35 | 46.13 | 43.16 | 47.66 |

### gpt

Best Clf2 epoch used for routing: `40`

| Clf1 epoch | Overall QA F1 | Clf1 A-recall | Clf1 A-F1 |
|---|---:|---:|---:|
| 35 | 50.73 | 89.60 | 82.45 |
| 40 | **51.22** | 89.50 | 82.69 |

## Training class ratios

Clf1 ratios come from `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/silver/no_retrieval_vs_retrieval/train.json`.

Clf2 ratios come from `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/binary_silver_single_vs_multi/train.json`.

| Model | Clf1 A count | Clf1 R count | Clf1 A:R split | Clf2 B count | Clf2 C count | Clf2 B:C split |
|---|---:|---:|---|---:|---:|---|
| `flan_t5_xl` | 424 | 868 | 32.82% : 67.18% | 1871 | 1397 | 57.25% : 42.75% |
| `flan_t5_xxl` | 511 | 898 | 36.27% : 63.73% | 1903 | 1395 | 57.70% : 42.30% |
| `gpt` | 1013 | 404 | 71.49% : 28.51% | 1475 | 1329 | 52.60% : 47.40% |

## Notes

- The Clf1 A-recall values come directly from the saved per-class validation summaries.
- The Clf1 A-F1 values are computed from the saved A recall, A predicted count, and A gold count in the same validation summaries.
- The QA F1 values reuse the existing Chapter 4.4 split-evaluation outputs already present under `predictions/classifier/t5-large/{model}/split/`.