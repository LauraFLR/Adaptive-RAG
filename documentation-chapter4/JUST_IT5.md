## Iteration 5 — Parameter Audit: Training-Free Agreement Gate (UE-Based Gate 1)

Iteration 5 is fundamentally different from Iterations 1–4: there are no training hyperparameters to audit. The entire Gate 1 is replaced by a deterministic heuristic. The audit therefore focuses on **the design decisions of the agreement method itself** and whether each is grounded in literature.

---

### 1. Method: Cross-Strategy Answer Agreement as UE Proxy

**Verdict: ✅ Grounded in literature — but terminology needs precision**

Your Thesis_Proposal.md correctly identifies this as a **consistency-based UE method**, analogous to Lex-Similarity in the Moskvoretskii et al. (2025) taxonomy. The core logic — "if retrieval doesn't change the answer, the LLM already knows it" — is a behavioral consistency signal that directly measures self-knowledge at inference time.

However, the documentation flags an important distinction: this is **not** classical sampling-based consistency (Wang et al., 2023b, self-consistency via temperature sampling). Both `nor_qa` and `oner_qa` are generated with `do_sample=False` (greedy decoding). The "diversity" comes from the two different **inference pipelines** (no retrieval vs. one-step retrieval), not from stochastic sampling.

**Closest precedents in the literature:**
- **Moskvoretskii et al. (2025)** — Lex-Similarity: compare answers across perturbations; your "perturbation" is the addition of retrieval context. The taxonomy classifies this as consistency-based UE.
- **Adaptive Retrieval (Mallen et al., 2023)** — the pop-score / entity frequency trigger is also a training-free routing signal, though feature-based rather than consistency-based.
- **FLARE (Jiang et al., 2023)** — uses token-level logit confidence to trigger retrieval mid-generation; different mechanism (logit-based vs. consistency-based) but same goal (training-free Gate 1).

**What to cite:** Moskvoretskii et al. (2025) for the taxonomic placement; Wang et al. (2023b) for the self-consistency principle; note explicitly that your variant uses cross-strategy agreement rather than multi-sample agreement.

---

### 2. Number of Comparison Strategies: 2 (nor_qa vs oner_qa)

**Verdict: ⚠️ Minimal — should be discussed**

You compare exactly 2 deterministic outputs. Classical consistency methods typically use N=5–20 stochastic samples to estimate agreement rates (Wang et al., 2023b use N=40 for self-consistency; Kuhn et al., 2023 use N=5–10 for semantic entropy; Chen et al., 2025 analyze cost-accuracy tradeoffs at N=1–8).

With N=2 deterministic outputs, you have a binary signal (agree/disagree) with no soft gradation. This is the most information-sparse form of consistency estimation possible.

**Justification paths (all valid, from your Thesis_Proposal.md):**
- `output_scores=False` is hardcoded in the LLM server, so logit-based methods are infeasible
- The prediction cache stores no token probabilities, so sampling-based methods would require full re-inference
- The two strategies (no retrieval vs. single-step retrieval) represent a semantically meaningful perturbation — adding external knowledge — rather than a statistical noise perturbation (temperature). This means even N=2 carries high signal-to-noise
- Zero additional inference cost: both predictions are already pre-computed

**Recommendation:** Frame this as a pragmatic design constraint, not a free choice. State that the method achieves strong results despite using the minimum possible sample count, and that richer consistency signals (multi-sample, semantic similarity) are a natural extension.

---

### 3. Agreement Function: Exact String Match After Normalization

**Verdict: ⚠️ Conservative but defensible — alternatives should be acknowledged**

The normalization pipeline (lowercase → strip articles → strip punctuation → collapse whitespace) follows the standard SQuAD evaluation normalization (Rajpurkar et al., 2016). The final comparison is exact string equality.

This is **conservative**: near-matches like "New York City" vs. "New York" are treated as disagreement, routing to retrieval. This conservatism is safe because your Thesis_Proposal.md notes that retrieval rarely hurts — so false disagreements (unnecessary retrieval) have low cost, while false agreements (skipping needed retrieval) have high cost.

**Alternatives in the literature:**
- **Token-level F1 with threshold** — the SQuAD evaluation metric itself; would catch partial matches
- **Semantic similarity** — Kuhn et al. (2023) use bidirectional NLI entailment with DeBERTa-large to cluster semantically equivalent answers; this is the semantic entropy approach
- **Lexical overlap** (Lex-Similarity from Moskvoretskii et al., 2025) — uses ROUGE or token overlap rather than exact match

**Recommendation:** Acknowledge exact match as the simplest agreement function. Note that fuzzy matching (token F1 threshold, semantic similarity) could increase the agreement rate and route more questions to A, but at the risk of more false agreements. The conservative exact-match choice is defensible given the asymmetric cost structure (false agreement is costlier than false disagreement).

---

### 4. No Tunable Threshold

**Verdict: ✅ Acceptable — and arguably a strength**

There is no threshold to tune. The decision is binary: strings match or they don't. This means **zero hyperparameter sensitivity** for Gate 1, which is a genuine advantage over trained classifiers (which have implicit decision thresholds at 0.5 probability).

Compare with FLARE (Jiang et al., 2023), which requires tuning a logit confidence threshold τ, or Voloshyn (2026), who formalize the threshold design problem for UE-based routing. Your method sidesteps this entirely.

**Cite:** Frame as a design advantage. "The agreement gate has no tunable hyperparameters, eliminating threshold sensitivity — a known limitation of UE-based routing (Voloshyn, 2026)."

---

### 5. Normalization Pipeline: SQuAD-Standard

**Verdict: ✅ Standard and citable**

The normalization (`normalize_answer()`: lowercase, strip articles, strip punctuation, collapse whitespace) is the standard SQuAD evaluation normalization from Rajpurkar et al. (2016). Additionally, `answer_extractor()` handles CoT-style outputs by extracting the final answer from "... the answer is: X" patterns.

**Cite:** Rajpurkar et al. (2016) for the normalization procedure.

---

### 6. Gate 2: Unchanged from Iteration 1

**Verdict: ✅ Correct controlled experiment design**

Only Gate 1 changes; Gate 2 is consumed as a fixed artifact from Iteration 1. This isolates the effect of the agreement gate. Methodologically clean.

---

### 7. Zero Training Cost

**Verdict: ✅ — and this is a key thesis contribution**

The agreement gate requires no model training, no GPU time, and no additional LLM inference (all predictions are pre-computed). This is a genuine practical advantage worth highlighting. In the taxonomy of He & Garcia (2009), this is an "algorithm-level" solution that requires no data modification or model retraining.

---

### 8. Model-Dependent Effectiveness (GPT Regression)

**Verdict: Not a SOTA issue — but a critical finding to discuss**

The GPT regression (−0.4pp F1) is not a parameter choice problem; it's an inherent limitation of consistency-based UE when the LLM produces false agreements (same wrong answer with and without retrieval). Your Thesis_Proposal.md already identifies the mechanism: GPT's higher output variability causes coincidental agreement on incorrect answers.

This aligns with findings from Moskvoretskii et al. (2025), who show that consistency methods have the highest routing quality but are not immune to distribution shift and model-specific behavior.

---

### Summary Table

| Design Decision | Choice | SOTA? | Justification Source |
|---|---|---|---|
| Method: cross-strategy agreement | nor_qa vs oner_qa | ✅ | Moskvoretskii et al. (2025), Lex-Similarity family |
| Number of comparisons | N=2 (deterministic) | ⚠️ | Pragmatic constraint; discuss vs. multi-sample |
| Agreement function | Exact string match after normalization | ⚠️ | Conservative; acknowledge fuzzy alternatives |
| Normalization | SQuAD-standard | ✅ | Rajpurkar et al. (2016) |
| Threshold | None (binary match) | ✅ | Advantage: zero hyperparameter sensitivity |
| Training cost | Zero | ✅ | Key contribution |
| Gate 2 | Unchanged from Iteration 1 | ✅ | Controlled experiment |
| Greedy decoding (do_sample=False) | Pre-existing constraint | ✅ | Not a design choice; inherited from pipeline |

---

### Key References to Cite

- **Moskvoretskii et al. (2025)** — UE taxonomy; Lex-Similarity as the closest named method; consistency-based UE family
- **Wang et al. (2023b)** — self-consistency principle (majority vote over sampled outputs); your method is a minimal-sample variant
- **Rajpurkar et al. (2016)** — SQuAD; the normalization procedure
- **Kuhn et al. (2023)** — semantic entropy; for discussing the fuzzy-matching alternative
- **Chen et al. (2025)** — cost-accuracy tradeoff at varying N; contextualizes your N=2 choice
- **Voloshyn (2026)** — threshold design problem; your method avoids it entirely
- **Mallen et al. (2023)** — Adaptive Retrieval; precedent for training-free routing signals


---

This is a good strategic question. Let me assess it from both angles.

---

### Can You Justify the Current Design with Literature?

**Yes — comfortably.** You have strong coverage:

1. **The method itself** maps to Moskvoretskii et al. (2025)'s Lex-Similarity family. It is a named category in the most comprehensive UE taxonomy available for RAG.

2. **The semantic-perturbation framing** (comparing outputs across different inference conditions rather than stochastic samples) has precedent in TARG (Khaliq et al., 2024), which uses sampling-based variance across retrieval conditions, and in Mallen et al. (2023), who use entity frequency as a training-free routing signal. Your method is a simpler, deterministic variant of the same principle.

3. **The constraint justification** is airtight: `output_scores=False` is hardcoded, the prediction cache stores no logits, and re-running inference would change the experimental setup you inherited from Jeong et al. (2024). You are not *choosing* N=2 over N=40 — you are working within the constraints of an existing pipeline. This is a legitimate and common situation in systems research.

4. **The results speak for themselves.** +2.0pp / +3.7pp F1 for Flan-T5 with zero training cost, A-F1 jumping from ~0.50 to ~0.83 — these are large, interpretable improvements. A reviewer would need a strong reason to demand a different method when this one works well and is well-grounded.

5. **The GPT regression is a feature, not a bug.** It reveals a genuine boundary condition (output determinism as a prerequisite for consistency-based UE), which is itself a contribution. Moskvoretskii et al. (2025) note that consistency methods are sensitive to model-specific behavior — your GPT result is empirical confirmation.

---

### Should You Change the UE Gate?

**No.** Here is why:

- **Scope risk.** Implementing a different UE method (e.g., semantic entropy with DeBERTa NLI clustering, or multi-sample temperature sampling) would require either modifying the LLM server to enable `output_scores=True` and `do_sample=True`, or adding an entirely new inference pipeline. This is a substantial engineering effort that changes the experimental setup — in a Bachelor's thesis with 7 iterations, this scope expansion is risky.

- **Your DSR structure is already complete.** Iterations 1–4 establish the imbalance/signal-quality diagnosis. Iteration 5 provides the positive result. Iterations 6–7 explore Gate 2. The narrative arc is coherent. Swapping Iteration 5's method would ripple through everything downstream (the oracle gap analysis, the Gate 2 ceiling, the Iteration 6 go/no-go decision — all depend on Iteration 5's Gate 1 being fixed).

- **A more sophisticated UE method is better framed as future work.** Your thesis identifies the *principle* (UE-based Gate 1 outperforms trained Clf1) and demonstrates it with the simplest feasible instantiation. Proposing richer UE methods (semantic entropy, multi-sample consistency, logit-based gating) as extensions is a stronger thesis conclusion than attempting and potentially botching a more complex implementation.

---

### What to Write

In the discussion/limitations section, something along these lines:

"The agreement gate implements the simplest feasible consistency-based UE signal within the constraints of the existing inference pipeline. Richer UE methods — such as semantic entropy (Kuhn et al., 2023), multi-sample self-consistency (Wang et al., 2023b), or logit-based confidence (Duan et al., 2024) — could potentially improve Gate 1 precision, particularly for models with lower output determinism (e.g., GPT). However, these methods require either access to token-level probabilities (not available in the current pipeline) or additional stochastic inference passes (increasing computational cost). Investigating whether richer UE signals can close the remaining false-agreement gap constitutes a natural extension of this work."

This positions your contribution correctly: you proved the *principle* works, identified *when* it fails, and scoped the *next step*. That is a complete Bachelor's thesis contribution.