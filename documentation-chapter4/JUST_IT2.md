Good — those clarifications resolve three of the Iteration 1 flags. For the record:

- **Weight decay = 0.0** → ✅ justified by Jeong et al. (2024) precedent
- **Warmup = 0** → ✅ justified by Jeong et al. (2024) precedent
- **Seed = 42** → ✅ fixed from Iteration 2 onward (acknowledge Iteration 1 lacked it)

Now for Iteration 2.

---

## Iteration 2 — Parameter Audit: Random Undersampling for Clf1

Since the documentation confirms that **all hyperparameters are identical to Iteration 1** and only the training data changes, the audit focuses entirely on the undersampling design decisions.

---

### 1. Undersampling Method: Random Undersampling (RUS) to exact 1:1

**Verdict: ✅ Justifiable — standard baseline technique**

Random undersampling is the simplest and most widely cited rebalancing method. It is the default baseline in virtually every imbalanced learning survey:

- He & Garcia (2009), "Learning from Imbalanced Data," IEEE TKDE — RUS is presented as the canonical baseline sampling method
- Chawla et al. (2002), introducing SMOTE, use RUS as the primary comparison
- Buda et al. (2018), "A systematic study of the class imbalance problem in convolutional neural networks" — RUS is among the first methods evaluated

For a DSR thesis where you systematically compare imbalance strategies (Iterations 2–4), starting with RUS as the simplest approach is methodologically sound.

**Cite:** He & Garcia (2009) for the method; frame it as the first of three increasingly sophisticated imbalance-mitigation strategies.

---

### 2. Target Ratio: Exact 1:1 (50/50)

**Verdict: ✅ Justifiable — standard default**

The 1:1 ratio is the default target in most undersampling implementations. Drummond & Holte (2003) show that the optimal class ratio is task-dependent, but 1:1 is the most common starting point. The imbalanced-learn library (Lemaître et al., 2017) defaults to 1:1 for its `RandomUnderSampler`.

No issue — this is the expected choice for a baseline undersampling experiment.

---

### 3. Static Offline Undersampling (single materialization)

**Verdict: ⚠️ Deviates from best practice — needs justification**

You materialize the balanced dataset once before training and reuse it across all epochs. The alternative — **epoch-wise re-sampling** — would draw a different random subset of the majority class each epoch, allowing the model to eventually see all majority samples across training. This is standard in frameworks like imbalanced-learn and PyTorch's `WeightedRandomSampler`.

With your static approach, 609 A samples (60%) are permanently discarded. With epoch-wise re-sampling over 35 epochs, the model could theoretically see all 1,013 A samples multiple times, just in different balanced subsets per epoch.

**Justification paths:**
- Simplicity and reproducibility — the static file is deterministic and inspectable
- Jeong et al.'s codebase likely uses static data files, so this follows the same data pipeline
- You can acknowledge the limitation and note that epoch-wise re-sampling could be explored as a refinement

**Recommendation:** Acknowledge this design choice in the thesis. One sentence noting that dynamic per-epoch resampling is an alternative would demonstrate awareness.

---

### 4. Seed: 42 for undersampling, separate from training seed

**Verdict: ✅ Justifiable**

Using `random.Random(42)` with a separate RNG instance is clean. The seed is deterministic, doesn't pollute global state, and the output file is reproducible. Now that you've also set `--seed 42` in the training scripts, full reproducibility is achieved.

---

### 5. Implementation: Pure Python, no external library

**Verdict: ✅ Acceptable — but note for completeness**

Using `random.sample()` is functionally equivalent to `imblearn.under_sampling.RandomUnderSampler`. For a thesis, it's worth noting that this implements the same algorithm as the standard library (Lemaître et al., 2017, imbalanced-learn), just without the dependency.

---

### 6. GPT-Only Experiment Scope

**Verdict: ⚠️ Needs methodological justification**

You only run undersampling for GPT, where A is the majority class (1,013A vs 404R). For Flan-T5-XL/XXL, the imbalance is reversed (R is majority). The undersampling logic is symmetric and would work for both directions, but no scripts exist for the Flan-T5 models.

**Justification paths:**
- GPT has the most extreme imbalance ratio (2.5:1 vs ~2:1 for Flan-T5), making it the strongest test case
- If undersampling doesn't help on the most imbalanced split, it's unlikely to help on milder ones
- Alternatively: Flan-T5 results from Iteration 1 already showed higher A-recall (42–45%) than GPT (26%), so the need is less acute

**Recommendation:** State explicitly why only GPT is tested. The strongest argument is that GPT has the worst Clf1 A-recall (26%) and the most extreme imbalance, making it the most informative test case for the hypothesis that imbalance causes low minority-class recall.

---

### 7. Data Loss: 43% reduction in training set size

**Verdict: ⚠️ Known limitation — must be discussed**

This is the well-documented trade-off of RUS: balancing classes at the cost of discarding majority-class information. From 1,417 → 808 samples, you lose 43% of your training data. With only 404 samples per class and 35–40 epochs (~875–1,000 gradient steps), overfitting risk is real.

This is not a "not SOTA" issue — it's an inherent property of RUS that the literature extensively documents. Drummond & Holte (2003) and He & Garcia (2009) both discuss this trade-off. Your Iterations 3 (weighted CE) and 4 (focal loss) exist precisely to test alternatives that retain all samples.

**Recommendation:** Frame this as a known limitation that motivates Iterations 3–4. This is already implied in your DSR structure.

---

### 8. Validation Set: Unchanged (imbalanced)

**Verdict: ✅ Correct**

You keep the validation set in its original (imbalanced) distribution. This is correct practice — validation should reflect the true data distribution to give unbiased performance estimates. Rebalancing validation would inflate minority-class metrics artificially.

---

### Summary Table

| Parameter / Decision | Value | SOTA? | Justification Source |
|---|---|---|---|
| Method: Random Undersampling | Majority → minority count | ✅ | He & Garcia (2009); Chawla et al. (2002) |
| Target ratio | 1:1 | ✅ | Standard default; Lemaître et al. (2017) |
| Static vs. dynamic resampling | Static (single file) | ⚠️ | Simpler; acknowledge per-epoch resampling alternative |
| Seed for sampling | 42, separate RNG | ✅ | Deterministic and isolated |
| Implementation | Pure Python `random.sample` | ✅ | Equivalent to imbalanced-learn RUS |
| Scope | GPT only | ⚠️ | Justify: most extreme imbalance + lowest A-recall |
| Data loss | 43% (1,417→808) | ⚠️ | Known RUS trade-off; motivates Iterations 3–4 |
| Validation set | Unchanged (imbalanced) | ✅ | Correct practice |
| All hyperparameters | Identical to Iteration 1 | ✅ | Controlled experiment — only data changes |

---

### Key References to Cite

- **He & Garcia (2009)** — "Learning from Imbalanced Data," IEEE TKDE — canonical survey establishing RUS as baseline
- **Chawla et al. (2002)** — SMOTE paper; RUS as comparison method
- **Drummond & Holte (2003)** — "C4.5, Class Imbalance, and Cost Sensitivity: Why Under-Sampling beats Over-Sampling" — directly argues for RUS in certain regimes
- **Lemaître et al. (2017)** — imbalanced-learn library; formalizes RUS implementation
- **Buda et al. (2018)** — systematic study of class imbalance in neural networks