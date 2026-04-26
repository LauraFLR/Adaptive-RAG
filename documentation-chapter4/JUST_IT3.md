## Iteration 3 — Parameter Audit: Class-Weighted Cross-Entropy for Clf1

Again, since all base hyperparameters are identical to Iteration 1, the audit focuses on the **weighted CE-specific design decisions**.

---

### 1. Method: Class-Weighted Cross-Entropy via FocalLoss with γ=0

**Verdict: ⚠️ Implementation is correct but indirect — needs framing**

Using the FocalLoss class with γ=0 to implement weighted CE is mathematically sound: when γ=0, the focal modulation term (1−p_t)^γ = 1.0, reducing the loss to α-weighted cross-entropy. This is a well-known relationship — Lin et al. (2017) explicitly note that focal loss generalizes weighted CE as a special case.

However, from a thesis-writing perspective, you should make this explicit. A reviewer seeing `--use_focal_loss --focal_gamma 0.0` might assume focal loss is active. State clearly: "We implement class-weighted cross-entropy by setting γ=0 in the focal loss framework of Lin et al. (2017), which reduces to standard weighted CE."

**Cite:** Lin et al. (2017) for the focal loss framework; King & Zeng (2001) or Japkowicz & Stephen (2002) for cost-sensitive learning via class weighting as a general technique.

---

### 2. Alpha Values: Hardcoded 0.6 across all models

**Verdict: ❌ Incorrect for XL and XXL — this is a bug, not a design choice**

Your documentation already flags this, and it is the most critical issue in Iteration 3. Let me make the problem precise:

The alpha tensor is `[1 - focal_alpha, focal_alpha] = [0.4, 0.6]`, where index 0 = A, index 1 = R.

| Model | Minority class | Minority index | Weight assigned | Effect |
|---|---|---|---|---|
| flan_t5_xl | A (424/1292 = 32.8%) | 0 | **0.4** (lower) | ❌ Suppresses minority |
| flan_t5_xxl | A (511/1409 = 36.3%) | 0 | **0.4** (lower) | ❌ Suppresses minority |
| gpt | R (404/1417 = 28.5%) | 1 | **0.6** (higher) | ✅ Upweights minority |

For XL and XXL, the weighting is **inverted** — it makes the imbalance problem *worse*, not better. This undermines the experimental validity of Iteration 3 for two out of three models.

**What SOTA practice prescribes:** The standard approach is inverse-frequency weighting:

w_c = N_total / (K × N_c)

where K is the number of classes and N_c is the count of class c. This is the default in `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` (Pedregosa et al., 2011) and is the method described in King & Zeng (2001).

For your data, this would yield:

| Model | w_A | w_R |
|---|---|---|
| flan_t5_xl | 1292/(2×424) = **1.52** | 1292/(2×868) = **0.74** |
| flan_t5_xxl | 1409/(2×511) = **1.38** | 1409/(2×898) = **0.78** |
| gpt | 1417/(2×1013) = **0.70** | 1417/(2×404) = **1.75** |

**Recommendation — two options:**

1. **If you can re-run XL/XXL:** Compute weights from data using inverse-frequency weighting. This is the defensible SOTA approach.
2. **If you cannot re-run:** Report the results honestly, note the inversion as a discovered error, and interpret XL/XXL Iteration 3 results as an accidental "anti-weighting" ablation — which actually provides useful evidence that wrong-direction weighting hurts performance. This is salvageable as a negative result if framed correctly.

---

### 3. Weights Not Computed from Data (hardcoded constant)

**Verdict: ❌ Not SOTA — weights should be data-driven**

Independent of the inversion bug, using a hardcoded alpha across all three models violates the principle of class-weighted CE. The whole point is to calibrate the loss to each dataset's class distribution. Each model variant has a different A/R ratio, so each needs different weights.

**SOTA practice:** Compute weights automatically from training label counts, as in:
- `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` — Pedregosa et al. (2011)
- The "balanced" mode in PyTorch's `CrossEntropyLoss(weight=...)` — requires manual computation but follows the same formula

**Cite:** Japkowicz & Stephen (2002), "The class imbalance problem: A systematic study" — establishes that cost-sensitive weighting should reflect the empirical class distribution. Also: Cui et al. (2019), "Class-Balanced Loss Based on Effective Number of Samples" — for a more sophisticated data-driven weighting scheme.

---

### 4. Different Training Code Path (HF Trainer vs manual loop)

**Verdict: ⚠️ Confound — needs acknowledgment**

Iterations 1 and 2 use a manual training loop with HuggingFace Accelerate. Iteration 3 switches to `FocalLossTrainer(Trainer)`, which uses HuggingFace's Trainer internally. While the optimizer (AdamW), scheduler (linear), and hyperparameters are functionally equivalent, subtle differences exist:

- **Gradient clipping:** HF Trainer applies `max_grad_norm=1.0` by default. Your manual loop does not clip gradients. This is a confound.
- **Logging/evaluation timing:** Trainer manages its own step counting and evaluation schedule
- **Mixed precision:** Trainer may handle fp16/bf16 differently from your manual Accelerate setup

**Recommendation:** Acknowledge this code-path difference in your thesis. You can argue that the core training dynamics (optimizer, LR, schedule) are equivalent, but note gradient clipping as a potential confound. If you have access to Jeong et al.'s Trainer configuration, check whether they clip gradients.

---

### 5. Seed = 42 (via Trainer default)

**Verdict: ✅ Improvement over Iterations 1–2**

The Trainer path defaults to seed=42 when `args.seed is None`. This is actually better than Iterations 1–2 (which were unseeded at that time). Now that you've fixed the seed to 42 across all iterations, this is consistent.

---

### 6. XL Skips Epoch 15

**Verdict: ⚠️ Minor inconsistency — acknowledge**

The XL weighted-CE script sweeps `{20, 25, 30, 35}` while Iteration 1 XL sweeps `{15, 20, 25, 30, 35}`. This means you have one fewer comparison point for XL. Not a SOTA issue, but a completeness issue that should be noted if you do epoch-wise comparisons.

---

### 7. Tolerant Error Handling (`|| echo` instead of `set -euo pipefail`)

**Verdict: ⚠️ Operational risk — not a SOTA issue per se**

The weighted-CE scripts silently continue if validation or prediction fails. This means a corrupted checkpoint could produce missing results without any error signal. This is a code quality issue, not a methodological one, but worth being aware of when interpreting results. Verify that all expected output files exist.

---

### Summary Table

| Parameter / Decision | Value | SOTA? | Action |
|---|---|---|---|
| Method: weighted CE via focal(γ=0) | Mathematically correct | ✅ | Frame explicitly in thesis |
| Alpha for GPT (R=minority at idx 1) | 0.6 → [0.4, 0.6] | ✅ | Correct direction |
| Alpha for XL (A=minority at idx 0) | 0.6 → [0.4, 0.6] | ❌ | **Inverted — suppresses minority** |
| Alpha for XXL (A=minority at idx 0) | 0.6 → [0.4, 0.6] | ❌ | **Inverted — suppresses minority** |
| Data-driven vs hardcoded weights | Hardcoded constant | ❌ | Should compute from class frequencies |
| Training code path | HF Trainer (not manual loop) | ⚠️ | Confound (gradient clipping); acknowledge |
| Seed | 42 (via Trainer default) | ✅ | Consistent |
| XL epoch range | Skips epoch 15 | ⚠️ | Minor; acknowledge |
| Training data | Full (no undersampling) | ✅ | Correct for this iteration |
| Validation data | Unchanged | ✅ | Correct |

---

### Critical Action Items

3. **Document the code-path difference.** One sentence noting that Iteration 3 uses HF Trainer (with gradient clipping) while Iterations 1–2 use a manual loop.

### Key References to Cite

- **Lin et al. (2017)** — "Focal Loss for Dense Object Detection" — establishes that γ=0 reduces focal loss to weighted CE
- **King & Zeng (2001)** — "Logistic Regression in Rare Events Data" — foundational work on cost-sensitive reweighting
- **Japkowicz & Stephen (2002)** — "The class imbalance problem: A systematic study" — systematic evaluation of cost-sensitive methods
- **Cui et al. (2019)** — "Class-Balanced Loss Based on Effective Number of Samples" — modern data-driven weighting
- **Pedregosa et al. (2011)** — scikit-learn — for the `compute_class_weight("balanced")` formula as a reference implementation