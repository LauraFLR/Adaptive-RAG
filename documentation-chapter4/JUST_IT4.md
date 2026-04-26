## Iteration 4 — Parameter Audit: Focal Loss for Clf1

Since base hyperparameters are again identical to Iteration 1 and the code path is the same as Iteration 3 (HF Trainer with `FocalLossTrainer`), the audit focuses on the **focal loss-specific design decisions**.

---

### 1. Gamma = 2.0

**Verdict: ✅ Directly SOTA — the canonical default**

Lin et al. (2017) systematically evaluate γ ∈ {0, 0.5, 1, 2, 5} and find that γ=2 performs best across their experiments on dense object detection. This value has since become the de facto default in virtually all subsequent focal loss applications. No justification beyond citing the original paper is needed.

**Cite:** Lin et al. (2017), "Focal Loss for Dense Object Detection," Section 5.1 — γ=2 is the recommended setting.

---

### 2. Per-Model Alpha Values: xl=0.33, xxl=0.36, gpt=0.71

**Verdict: ⚠️ Correct direction, but justification needs care**

Unlike Iteration 3 (where alpha was hardcoded identically across models), Iteration 4 sets model-specific alphas that correctly upweight each model's minority class. The direction is right for all three models. The question is whether the specific values are justifiable.

Your documentation already notes the pattern: each alpha closely approximates the minority class fraction:

| Model | Minority fraction | Alpha | 1 − alpha (minority weight) |
|---|---|---|---|
| xl | A: 424/1292 = 0.328 | 0.33 | 0.67 |
| xxl | A: 511/1409 = 0.363 | 0.36 | 0.64 |
| gpt | R: 404/1417 = 0.285 | 0.71 | 0.29 |

Wait — this needs closer scrutiny. The `alpha` parameter in your implementation is `[1-α, α]`, so index 0 (A) gets weight `1-α` and index 1 (R) gets weight `α`. For xl: A gets 0.67, R gets 0.33. Since A is the minority, the minority gets the *higher* weight. ✅ Correct.

But what is the *justification* for setting alpha equal to the minority fraction? Lin et al. (2017) state: "In practice α may be set by inverse class frequency." They use α=0.25 in their best setting (with γ=2), noting that the optimal alpha depends on gamma — higher gamma reduces the need for aggressive alpha because the focal modulation already down-weights easy (majority) examples.

**Your alpha values follow the "inverse class frequency" heuristic** that Lin et al. describe. Specifically, you set alpha ≈ the minority fraction, which means `1-alpha` ≈ the majority fraction, giving the minority class a weight proportional to the majority's prevalence. This is a recognized approach.

However, there is a subtle issue: **Lin et al. found that the optimal alpha decreases as gamma increases.** At γ=2, they found α=0.25 optimal (for a highly imbalanced detection task at ~1:1000 ratio). Your imbalance ratios (~1:2 to 1:2.5) are much milder, so the optimal alpha likely differs from theirs. Your heuristic of using the class frequency is a reasonable starting point, but it is not *derived* from Lin et al.'s grid search.

**Justification path:** State that alpha values are set to approximate the inverse class frequency for each model, following the heuristic described in Lin et al. (2017, Section 3). Acknowledge that the optimal (α, γ) combination is task-dependent, but that you follow the standard default of γ=2 paired with frequency-based alpha.

---

### 3. Alpha Still Not Automatically Computed from Data

**Verdict: ⚠️ Same issue as Iteration 3 — manually set, not programmatically derived**

Although the alpha values are *correct* this time (unlike Iteration 3's inversion bug for XL/XXL), they are still manually specified in each shell script rather than computed from training label counts. The values happen to closely match the minority class fractions, but this is a coincidence of manual tuning, not a systematic computation.

If you implement the `--auto_class_weights` flag from the Copilot prompt I gave for Iteration 3, you could reuse it here too: compute `w_c = N/(K×N_c)` and then optionally normalize to sum to 1 for focal loss alpha. However, since the manually chosen values are already close to the data-driven values, the practical impact is minimal.

**Recommendation:** For the thesis, note that the alphas are "set to approximate the inverse class frequency" and cite the formula. This is more defensible than saying "manually tuned" even though the result is the same.

---

### 4. Focal Loss Implementation: Custom `torch.nn.Module`

**Verdict: ✅ Correct implementation — matches Lin et al. (2017)**

The formula `FL(p_t) = -α_t × (1-p_t)^γ × log(p_t)` is the standard focal loss from Lin et al. (2017), Equation 5. The `clamp(min=1e-8)` on softmax probabilities prevents log(0) — this is standard numerical practice.

One minor note: Lin et al. apply focal loss to sigmoid outputs (binary per-class), while your implementation applies it to softmax outputs (joint distribution over classes). For the 2-class case, softmax and sigmoid produce equivalent gradients, so this is functionally correct. If a reviewer asks, you can note this equivalence.

**Cite:** Lin et al. (2017) for the formula; note that the implementation uses softmax rather than per-class sigmoid, which is equivalent for K=2 classes.

---

### 5. Training Code Path: Same as Iteration 3 (HF Trainer)

**Verdict: ⚠️ Same confound as Iteration 3 — but consistent within Iterations 3–4**

The `FocalLossTrainer` code path (HF Trainer with gradient clipping, internal AdamW) is the same as Iteration 3. This means Iterations 3 and 4 are **directly comparable to each other** (same code path, same optimizer construction, only loss function differs: γ=0 vs γ=2). But both differ from Iterations 1–2 (manual loop, no gradient clipping).

This is actually a useful property: the Iteration 3 → 4 comparison is clean (only gamma changes), while the Iteration 1 → 3/4 comparison has the code-path confound.

**Recommendation:** Frame Iterations 3–4 as a paired comparison (weighted CE vs focal loss, same infrastructure). Acknowledge the code-path difference relative to Iterations 1–2.

---

### 6. Error Handling Inconsistency (GPT vs XL/XXL)

**Verdict: ⚠️ Operational risk — not methodological**

GPT scripts silently continue on validation failure; XL/XXL scripts abort. This means if an XL eval fails at epoch 15, you lose epochs 20–35. For GPT, you get partial results. Not a SOTA issue, but verify that all expected output files exist for all epoch-model combinations.

---

### 7. Comparison: Iteration 3 vs Iteration 4 Alpha Values

This comparison is important because it reveals the Iteration 3 bug more starkly:

| Model | It3 alpha → [A_wt, R_wt] | It4 alpha → [A_wt, R_wt] | Minority | It3 correct? | It4 correct? |
|---|---|---|---|---|---|
| xl | 0.6 → [0.4, 0.6] | 0.33 → [0.67, 0.33] | A | ❌ | ✅ |
| xxl | 0.6 → [0.4, 0.6] | 0.36 → [0.64, 0.36] | A | ❌ | ✅ |
| gpt | 0.6 → [0.4, 0.6] | 0.71 → [0.29, 0.71] | R | ✅ | ✅ |

Iteration 4 corrects the weighting direction for all three models. This is further evidence that the Iteration 3 XL/XXL results need re-running with corrected weights.

---

### Summary Table

| Parameter / Decision | Value | SOTA? | Justification Source |
|---|---|---|---|
| γ = 2.0 | Canonical default | ✅ | Lin et al. (2017), Section 5.1 |
| α per model (xl=0.33, xxl=0.36, gpt=0.71) | Matches inverse class frequency | ✅ | Lin et al. (2017), Section 3 heuristic |
| α manually set vs data-driven | Manual (but correct) | ⚠️ | Acceptable if framed as frequency-based; auto-compute would be cleaner |
| Implementation (custom nn.Module) | Correct formula | ✅ | Lin et al. (2017), Eq. 5 |
| Softmax vs sigmoid | Softmax (equivalent for K=2) | ✅ | Standard; note equivalence if questioned |
| Code path (HF Trainer) | Same as Iteration 3 | ⚠️ | Consistent within It3–4; confound vs It1–2 |
| Training data | Full (no undersampling) | ✅ | Correct for focal loss |
| Seed | 42 (via Trainer default) | ✅ | Reproducible |
| Epoch range | Same as Iteration 1 | ✅ | Consistent |

---

### Key References to Cite

- **Lin et al. (2017)** — "Focal Loss for Dense Object Detection" — the original focal loss paper; γ=2 as recommended default; α set by inverse class frequency
- **Mukhoti et al. (2020)** — "Calibrating Deep Neural Networks using Focal Loss" — extends focal loss analysis to classification tasks beyond object detection; validates γ=2 as broadly effective
- **Li et al. (2020)** — "Generalized Focal Loss" — discusses focal loss variants and the interaction between α and γ

---

### One Thesis-Writing Note

Iterations 2–4 form a clean **ablation triad** over imbalance-mitigation strategies for Gate 1:

| Iteration | Strategy | Retains all data? | Loss modification? | Per-sample adaptation? |
|---|---|---|---|---|
| 2 | Random undersampling | ❌ (discards majority) | No | No |
| 3 | Weighted CE (α only) | ✅ | Constant per-class weight | No |
| 4 | Focal loss (α + γ) | ✅ | Dynamic per-sample weight | ✅ (down-weights easy examples) |

This progression from data-level → loss-level (static) → loss-level (dynamic) is a textbook structure for an imbalance-mitigation evaluation. Frame it as such in your thesis — it follows the taxonomy from He & Garcia (2009) and Johnson & Khoshgoftaar (2019), "Survey on deep learning with class imbalance."

---

# Should I Change Iteration 4 Alphas and Re-Run?

**No, I do not recommend changing this.** Here is my reasoning:

The key point is that Lin et al.'s optimal α=0.25 was found for a **specific task** (dense object detection) with a **vastly different imbalance ratio** (~1:1000 foreground-to-background). Your setting is fundamentally different in two ways:

1. **Much milder imbalance** (~1:2 to 1:2.5). At these ratios, the focal modulation term (1−p_t)^γ has a weaker differential effect between easy majority and hard minority samples than it does at 1:1000. This means alpha needs to do more of the rebalancing work, so a *higher* alpha for the minority class (which is what your frequency-based values give) is defensible.

2. **Different task domain.** Lin et al.'s grid search over (α, γ) was optimized for RetinaNet on COCO. There is no reason to expect the same optimal pair for T5-Large text classification on silver QA labels.

The frequency-based heuristic you use is one of the two standard approaches in the literature. Lin et al. (2017) themselves describe it as a valid method in Section 3. The other approach is joint grid search over (α, γ), which is computationally expensive and rarely done in practice outside the original paper.

**What to write in the thesis:** "We set α to approximate the inverse class frequency for each model variant, following the heuristic described in Lin et al. (2017). We use the canonical γ=2. While the optimal (α, γ) pair is task-dependent, systematic hyperparameter search over both values was outside the scope of this work."

This is honest, defensible, and standard. No code change needed.