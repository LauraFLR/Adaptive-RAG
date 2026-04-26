## Iteration 7 — Parameter Audit: Structural Feature Prefix for Gate 2 (Clf2)

Since base hyperparameters are identical to Iteration 1's Clf2 (with one exception), and the architecture is unchanged, the audit focuses on the **feature injection design decisions**.

---

### 1. Feature Injection Method: Plain-Text Prefix

**Verdict: ⚠️ Justifiable but needs careful framing — alternatives exist**

Prepending structured features as text tokens (`[LEN:14] [ENT:2] [BRIDGE:1]`) before the question is a recognized approach in the T5/seq2seq literature. The model must learn to associate these token patterns with the classification decision during fine-tuning.

**Precedents:**
- **Raffel et al. (2020)** — T5 uses task-specific text prefixes ("translate English to German:", "summarize:") to condition the model on the task. Your feature prefix follows the same principle: conditioning the encoder on structured metadata via natural-language-like tokens.
- **Schick & Schütze (2021)** — Pattern-Exploiting Training (PET) shows that reformulating classification inputs with structured textual patterns can improve few-shot performance on pre-trained language models.
- **Chai et al. (2025)** — use feature-enriched inputs for complexity classification, though with different features and injection methods.

**Alternatives not explored:**
- **Binned categorical prefixes** (e.g., `[LEN:short]` instead of `[LEN:14]`) — reduces the token vocabulary the model must learn. Your documentation flags this in item 6 of the suspicious items. More on this below.
- **Separate embedding concatenation** — inject features as a learned embedding vector concatenated to the encoder hidden states. This would require architectural changes and is out of scope for your thesis.
- **Multi-task / auxiliary loss** — train with a secondary objective predicting the features. Also out of scope.

The plain-text prefix is the simplest injection method that requires zero architectural changes, which is a legitimate design choice for a DSR iteration testing whether the signal transfers at all.

**Cite:** Raffel et al. (2020) for the T5 prefix-conditioning paradigm; Schick & Schütze (2021) for structured textual pattern injection.

---

### 2. Raw Integer Values (No Binning, No Normalization)

**Verdict: ⚠️ Suboptimal — should be discussed, possibly changed**

This is the most substantive methodological concern. `[LEN:7]` and `[LEN:42]` produce entirely different subword sequences. T5 has no built-in understanding that 42 > 7 — it must learn the ordinal relationship between every observed integer from the training data. With only ~3,200 training samples, many integer values will appear rarely or never, fragmenting the signal.

**What the literature suggests:**
- **Thawani et al. (2021)** — "Representing Numbers in NLP: A Survey and a Vision" — documents that transformer language models struggle with numerical reasoning from raw digit tokens. Discretizing numbers into categorical bins significantly improves performance on tasks requiring numerical understanding.
- **Wallace et al. (2019)** — show that BERT-family models fail to generalize numerically; binning or verbalizing numbers is recommended for classification tasks.

**Binning would produce:**

| Feature | Raw | Binned (example) |
|---|---|---|
| `[LEN:7]` | 7 | `[LEN:short]` |
| `[LEN:14]` | 14 | `[LEN:medium]` |
| `[LEN:28]` | 28 | `[LEN:long]` |
| `[ENT:0]` | 0 | `[ENT:none]` |
| `[ENT:1]` | 1 | `[ENT:one]` |
| `[ENT:3]` | 3 | `[ENT:many]` |

This reduces the number of unique prefix token sequences from dozens to a handful, making each pattern appear frequently enough in training for T5 to learn a robust association.

However — and this is the critical counterpoint — **Iteration 7's end-to-end result is +0.1–0.2pp across all three models**, which is within noise. The question is whether binning would change this conclusion. Given that the Iteration 6 probe showed AUC=0.676 (weak signal to begin with) and that the ~2.4pp recoverable ceiling is small, it is plausible that even perfect feature injection would yield only marginal gains. Binning might help, but it is unlikely to transform a null result into a significant one.

**Recommendation:** Acknowledge the raw-integer limitation in your thesis and propose binning as a potential refinement. Whether you re-run is a judgment call — see my assessment at the end.

---

### 3. Feature Set: Same as Iteration 6

**Verdict: ✅ Methodologically correct**

Using the exact same three features (token_len, entity_count, bridge_flag) that were validated in the diagnostic probe ensures consistency between the go/no-go decision and the full experiment. If you changed the features between iterations, the probe's AUC would no longer justify the experiment.

---

### 4. GPT Epoch Range Mismatch

**Verdict: ❌ Confound — needs acknowledgment or re-run**

The standard GPT Clf2 uses epochs `{35, 40}`. The feature-augmented script uses `{15, 20, 25, 30, 35}` for all models. This means:

- The GPT epoch-40 checkpoint (the standard best) is never produced for the feature-augmented variant
- You cannot do a controlled comparison between standard Clf2 (epoch 40) and feature-augmented Clf2 (epoch 40) for GPT
- The feature-augmented GPT trains at epochs 15–25 that were never tested for standard GPT Clf2, introducing an additional variable

**This is a confound.** If feature-augmented GPT Clf2 underperforms at epoch 35, you cannot tell whether it's because features hurt or because GPT needed epoch 40.

**Recommendation:** Either re-run with `{35, 40}` for GPT to match the baseline, or acknowledge this mismatch explicitly and note that the epoch-35 comparison is valid (both variants produce an epoch-35 checkpoint).

---

### 5. Training Code Path: Manual Loop (Same as Iteration 1)

**Verdict: ✅ Consistent with Iteration 1 Clf2 baseline**

Unlike Iterations 3–4 (which used FocalLossTrainer / HF Trainer), Iteration 7 uses the same manual training loop as Iteration 1. This means the Iteration 1 Clf2 → Iteration 7 Clf2 comparison is clean: same optimizer construction, same scheduler, same gradient handling. The only difference is the input data (feature-prefixed vs. plain). This is the correct controlled comparison.

---

### 6. Feature Prefix Token Overhead

**Verdict: ✅ Negligible risk**

A prefix like `[LEN:14] [ENT:2] [BRIDGE:1] ` adds approximately 15–20 subword tokens. With `max_seq_length=384` and typical question lengths well under 100 tokens, truncation is extremely unlikely. No issue.

---

### 7. No Normalization or Standardization of Features

**Verdict: ✅ Not applicable**

Unlike the logistic regression probe (where feature scaling matters), the T5 encoder processes features as text tokens. There is no numeric feature space to normalize. The relevant concern is binning (covered above), not standardization.

---

### 8. Seed Not Set

**Verdict: ⚠️ Same as Iteration 1 — you said you fixed this**

The documentation says `--seed` is not passed and defaults to `None`. If you've fixed the seed to 42 across all scripts (as you mentioned after Iteration 1), ensure the feature-augmented script also passes `--seed 42`. Otherwise this is a regression.

---

### Summary Table

| Parameter / Decision | Value | SOTA? | Justification Source |
|---|---|---|---|
| Feature injection: text prefix | `[LEN:X] [ENT:Y] [BRIDGE:Z] question` | ✅ | Raffel et al. (2020); Schick & Schütze (2021) |
| Raw integer values (no binning) | `[LEN:14]` not `[LEN:medium]` | ⚠️ | Thawani et al. (2021); Wallace et al. (2019) suggest binning |
| Feature set | Same 3 features as Iteration 6 probe | ✅ | Consistent with go/no-go decision |
| GPT epoch range | {15,20,25,30,35} vs baseline {35,40} | ❌ | Confound; epoch 40 missing |
| Training code path | Manual loop (same as Iteration 1) | ✅ | Clean controlled comparison |
| Token overhead | ~15–20 tokens on 384 budget | ✅ | Negligible |
| Architecture | Unmodified T5-Large | ✅ | Zero-change injection |
| Loss | Standard CE (no focal/weighting) | ✅ | Matches Iteration 1 Clf2 baseline |
| Seed | Not passed (should be 42) | ⚠️ | Verify consistency with other scripts |

---

### Should You Change Anything?

Given that Iteration 7 is a **null result** (+0.1–0.2pp, within noise), the strategic question is whether fixes would change the conclusion:

**GPT epoch mismatch — Yes, fix this.** It's a simple shell script edit (change the epoch loop to `35 40` for GPT) and removes a confound from your reporting. Even if the result stays null, a reviewer cannot question the comparison.

**Binning — No, discuss instead.** The null result is interpretable either way: the features carry weak signal (AUC 0.676) that does not transfer to end-to-end F1 regardless of injection method. Binning *might* help marginally, but the ~2.4pp recoverable ceiling means even a perfect Gate 2 improvement is bounded. Proposing binning as future work is cleaner than re-running an experiment that is likely to remain null.

**Seed — Yes, add `--seed 42`.** One-line fix in the shell script.

---

### Key References to Cite

- **Raffel et al. (2020)** — T5; prefix-conditioning paradigm for seq2seq models
- **Schick & Schütze (2021)** — PET; structured textual patterns for classification
- **Thawani et al. (2021)** — "Representing Numbers in NLP" — numerical reasoning limitations of transformers; binning recommendation
- **Wallace et al. (2019)** — numerical generalization failures in pre-trained LMs
- **Dong et al. (2025)** — surface features vs. fine-tuned encoders for complexity (46.6% vs. 85.9%)

---

### Thesis-Level Framing for the Null Result

The Iteration 7 null result is actually one of your thesis's more valuable findings. Frame it as:

"Structural query features that pass the discriminative feasibility threshold in isolation (AUC 0.676, Iteration 6) do not transfer to end-to-end QA F1 improvement when injected as text prefixes into the T5-Large classifier (+0.1–0.2pp across all three model variants, within noise). This result is consistent with Dong et al. (2025), who find that surface-level structural features achieve only 46.6% accuracy on complexity classification compared to 85.9% for fine-tuned encoders, suggesting that the B/C decision requires deeper semantic representations of reasoning structure that shallow features cannot capture. The ~2.4pp recoverable ceiling (Section 4.8.4) further bounds the maximum achievable gain, implying that even substantially better Gate 2 features would yield modest end-to-end improvements."
---

### The Core Question: Would More Features Change the Conclusion?

Your Iteration 7 null result (+0.1–0.2pp) exists within a **hard ceiling of ~2.4pp recoverable F1** from a perfect Clf2. This ceiling is computed from your oracle gap analysis (Iteration 5, Section 4.8.4) and is independent of feature quality. Even if you found features that perfectly separated B from C, the maximum end-to-end gain is 2.4pp — and realistically, any improvement would be a fraction of that.

Adding more features to the probe could raise the AUC from 0.676 to, say, 0.72 or 0.75. But Iteration 7 already showed that the existing signal (AUC 0.676) does not transfer to end-to-end gain. A higher AUC on the probe does not guarantee transfer, and the ceiling limits the payoff even if it does transfer.

---

### What More Features Would Cost You

1. **Scope expansion in a Bachelor's thesis.** You have 7 iterations with a clean narrative arc. Adding a new feature engineering cycle means re-running the probe, potentially re-running Iteration 7 with the new features, and writing up the analysis. This is at least several days of work.

2. **Risk of inconclusive results.** If the expanded probe shows AUC 0.73 but end-to-end F1 stays at +0.2pp, you have the same null result with more complexity. If it shows AUC 0.68, you've spent time confirming what you already know.

3. **Narrative coherence.** Your current arc is: "shallow structural features pass feasibility but don't transfer → semantic-depth representations needed as future work." This is a clean conclusion. Adding more shallow features and getting the same null result makes the story longer, not stronger.

---

### What You Should Do Instead

Use the concept matrix features as **future work recommendations**, not as additional experiments. Your thesis already has the Dong et al. (2025) result showing that surface features achieve 46.6% accuracy vs. 85.9% for fine-tuned encoders. This directly supports your conclusion that the B/C boundary requires deeper representations.

In your discussion section, you could write something like:

"The concept matrix (Section 2.4) identifies additional feature families for Gate 2 that were not tested in this work: embedding-based semantic similarity, decomposition-based complexity scores, and LLM-annotated reasoning depth labels. These represent progressively richer representations that could close the gap between the shallow structural features tested here (AUC 0.676) and the fine-tuned encoder ceiling reported by Dong et al. (2025). However, the ~2.4pp recoverable ceiling (Section 4.8.4) suggests that even substantial Gate 2 improvements would yield modest end-to-end gains in the current evaluation setup, and that the primary source of remaining error lies in the irreducible portion of the oracle gap (~5.3pp) where no routing strategy succeeds."

---

### The One Exception

If your concept matrix contains a feature that is **qualitatively different** from the three you tested — specifically, something that accesses semantic depth rather than surface structure — then testing it could strengthen your thesis by showing a clear contrast between surface and semantic features. For example, if you could compute an LLM-based decomposition score (e.g., prompting the LLM to decompose the question and counting sub-questions), that would be a genuinely different signal class. But that crosses into Iteration 8 territory and is almost certainly out of scope for a Bachelor's thesis.

**My recommendation: do not add more features to the probe. Your current 7-iteration structure is complete and the null result is well-explained by the literature. Use the concept matrix features as future work.**