# Official Experiment Results Documentation

---

# Iteration 1 - Cascade Baseline

## Clf1 Validation Results

- **Task:** Binary classification (A = no retrieval, R = retrieval)
- **Model:** t5-large, default cross-entropy loss, no imbalance handling
- **Training data:** Silver labels from Adaptive-RAG training set
- **Hyperparams:** lr=3e-5, batch_size=32, max_seq_length=384, doc_stride=128
- **Run date:** 2026-04-23

### flan_t5_xl (1,292 training samples)

| Epoch | Accuracy (%) | A acc (%) | A pred | A gold | R acc (%) | R pred | R gold |
|-------|-------------|-----------|--------|--------|-----------|--------|--------|
| 15    | 72.00       | 38.72     | 279    | 439    | 88.04     | 1071   | 911    |
| **20**| **73.19**   | 43.96     | 309    | 439    | 87.27     | 1041   | 911    |
| 25    | 72.67       | 47.61     | 348    | 439    | 84.74     | 1002   | 911    |
| 30    | 72.52       | 49.89     | 370    | 439    | 83.42     | 980    | 911    |
| 35    | 72.22       | 51.48     | 388    | 439    | 82.22     | 962    | 911    |

Best: **epoch 20 at 73.19%**. Heavy R-bias — R acc ~87% vs A acc ~44%.

<details>
<summary>Per-dataset breakdown (best epoch = 20)</summary>

| Dataset | Acc (%) | A acc (%) | A gold | R acc (%) | R gold |
|---------|---------|-----------|--------|-----------|--------|
| musique | 88.71 | 0.00 | 12 | 98.21 | 112 |
| hotpotqa | 78.12 | 51.81 | 83 | 90.75 | 173 |
| 2wikimultihopqa | 66.42 | 85.38 | 130 | 48.15 | 135 |
| nq | 70.04 | 15.49 | 71 | 93.37 | 166 |
| trivia | 62.99 | 20.00 | 125 | 92.35 | 183 |
| squad | 88.75 | 16.67 | 18 | 97.89 | 142 |

</details>

### flan_t5_xxl (1,413 training samples)

| Epoch | Accuracy (%) | A acc (%) | A pred | A gold | R acc (%) | R pred | R gold |
|-------|-------------|-----------|--------|--------|-----------|--------|--------|
| 15    | 65.58       | 35.65     | 338    | 519    | 82.92     | 1077   | 896    |
| **20**| **66.22**   | 37.38     | 347    | 519    | 82.92     | 1068   | 896    |
| 25    | 65.23       | 42.97     | 419    | 519    | 78.13     | 996    | 896    |
| 30    | 65.30       | 42.58     | 414    | 519    | 78.46     | 1001   | 896    |
| 35    | 65.02       | 41.43     | 406    | 519    | 78.68     | 1009   | 896    |

Best: **epoch 20 at 66.22%**. Similar R-bias pattern. Lowest overall accuracy of the three variants.

<details>
<summary>Per-dataset breakdown (best epoch = 20)</summary>

| Dataset | Acc (%) | A acc (%) | A gold | R acc (%) | R gold |
|---------|---------|-----------|--------|-----------|--------|
| musique | 83.47 | 0.00 | 17 | 97.12 | 104 |
| hotpotqa | 71.65 | 37.93 | 87 | 88.51 | 174 |
| 2wikimultihopqa | 62.84 | 73.77 | 122 | 55.17 | 174 |
| nq | 66.80 | 23.40 | 94 | 91.98 | 162 |
| trivia | 52.05 | 26.83 | 164 | 79.08 | 153 |
| squad | 77.44 | 14.29 | 35 | 94.57 | 129 |

</details>

### gpt (1,445 training samples)

| Epoch | Accuracy (%) | A acc (%) | A pred | A gold | R acc (%) | R pred | R gold |
|-------|-------------|-----------|--------|--------|-----------|--------|--------|
| **35**| **72.68**   | 89.69     | 1215   | 1038   | 27.74     | 216    | 393    |
| 40    | 72.19       | 88.34     | 1194   | 1038   | 29.52     | 237    | 393    |

Best: **epoch 35 at 72.68%**. Opposite bias — heavily over-predicts A (89.7% A acc vs 27.7% R acc), reflecting the A-dominant label distribution in GPT silver data.

<details>
<summary>Per-dataset breakdown (best epoch = 35)</summary>

| Dataset | Acc (%) | A acc (%) | A gold | R acc (%) | R gold |
|---------|---------|-----------|--------|-----------|--------|
| musique | 60.81 | 66.67 | 87 | 52.46 | 61 |
| hotpotqa | 63.67 | 85.56 | 187 | 18.68 | 91 |
| 2wikimultihopqa | 68.97 | 86.67 | 165 | 38.54 | 96 |
| nq | 79.20 | 97.99 | 199 | 5.88 | 51 |
| trivia | 88.42 | 97.19 | 320 | 5.88 | 34 |
| squad | 58.57 | 80.00 | 80 | 30.00 | 60 |

</details>

### Summary

All variants exhibit strong class imbalance bias under default cross-entropy. XL/XXL over-predict R; GPT over-predicts A. Minority class accuracy ranges from 28–44%.

## Clf2 Validation Results

## Full Pipeline Results

---

# Iteration 2 - Gate 1 with Random Undersampling

## Clf1 Validation Results

## Full Pipeline Results

---

# Iteration 3 - Gate 1 with Class-Weighted Loss

## Clf1 Validation Results

## Full Pipeline Results

---

# Iteration 4 - Gate 1 with Focal Loss

## Clf1 Validation Results

## Full Pipeline Results

---

# Iteration 5 - Training-Free Gate 1 (UE Answer Agreement)

## Clf2 Validation Results

## Full Pipeline Results
 
---

# Iteration 6 - Gate 2 Structural Features (Diagnostic)

## Clf2 Validation Results

## Full Pipeline Results

---

# Iteration 7 - Gate 2 with Structural Features Prepended

## Clf2 Validation Results

## Full Pipeline Results
