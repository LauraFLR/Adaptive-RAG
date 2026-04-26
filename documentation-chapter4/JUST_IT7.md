Problem: You Cannot Replicate SymRAG Exactly
SymRAG's κ(q) includes A(q) — mean attention values from the LLM. Your pipeline has output_scores=False hardcoded, meaning you have no access to attention values or logits. You would need to run a forward pass through T5 or another model just to compute attention scores for each query — which is not "training-free" in the way your Gate 1 is.
Additionally, SymRAG routes to three paths (symbolic / neural / hybrid), not binary B/C. Their thresholds are dynamically adjusted based on system load, which is irrelevant to your offline evaluation. And the paper does not publish their weight values (w_A, w_L, etc.).
What You Can Do: The Structural Heuristic Component S_H(q)
The sub-component you can replicate is S_H(q), which uses only the structural features — entity count and multi-hop indicators — with no attention values required. This is the part of κ(q) that maps to your existing features. Combined with L(q) (query length), this gives you a threshold-based complexity score using exactly the features from your Iteration 6 probe.
This is implementable as a training-free Gate 2 that mirrors SymRAG's structural heuristic.

---

Having read the full paper, I need to correct several things in my prompt. Here are the key findings from the paper:

**What the paper actually specifies (Appendix A.1.1, page 12):**

1. **A(q)** uses `prajjwal1/bert-tiny` (not the main LLM) — this is a tiny 4.4M parameter model, not an expensive forward pass
2. **Published weights:** w_A = 1.0, w_L = 1.0, w_sh1 = 0.05, w_sh2 = 0.1
3. **L(q)** is normalized by max query length in the dataset
4. **S_H(q)** normalizes by query length: S_H(q) = w_sh1 · N_ents(q)/|q| + w_sh2 · N_hops(q)/|q|
5. **N_hops(q)** counts multi-hop keywords (their equivalent of your bridge_flag, but as a count not binary)
6. **Initial thresholds:** T_low_κ = 0.4, T_high_κ = 0.8 (Table 7, page 23)

This changes the feasibility picture. Since bert-tiny is a 4.4M parameter model, computing A(q) is cheap — it doesn't require your main LLM. You could include it. But for your thesis, I'd recommend **omitting A(q)** and using only the structural heuristic S_H(q) + L(q), for two reasons: (a) you don't need to introduce a new model dependency, and (b) it gives you a clean "structural features only" comparison against your T5-Large Clf2.

Here is the corrected prompt:

---

```
Create a new script `classifier/postprocess/predict_complexity_kappa.py` that 
implements the structural heuristic component of SymRAG's κ(q) complexity 
score (Hakim et al., 2025) for the Gate 2 (B vs C) decision, as a 
training-free alternative to the T5-Large Clf2 classifier.

REFERENCE: SymRAG (Hakim et al., 2025, Definition 1, Appendix A.1.1):
  κ(q) = (w_A · A(q) + w_L · L(q)) · (1 + S_H(q))
  where S_H(q) = w_sh1 · N_ents(q)/|q| + w_sh2 · N_hops(q)/|q|

We omit the attention-based A(q) term (which requires a bert-tiny forward 
pass) and use only the structural components L(q) and S_H(q). This gives:
  κ(q) = w_L · L(q) · (1 + S_H(q))

With SymRAG's published weights: w_L = 1.0, w_sh1 = 0.05, w_sh2 = 0.1.

The script should:

1. ACCEPT THESE ARGUMENTS:
   - model_name (choices: flan_t5_xl, flan_t5_xxl, gpt)
   - --clf1_pred_file (path to Gate 1 predictions, OR:)
   - --use_agreement_gate (flag: if set, compute agreement gate internally 
     using the same logic as predict_complexity_agreement.py)
   - --clf2_pred_file (path to a T5-Large Clf2 predictions file — used ONLY 
     as a comparison baseline, not for routing)
   - --predict_file (default: the standard predict.json path)
   - --output_path (required)
   - --kappa_threshold (float, default: 0.5)
   - --tune_threshold (flag: if set, tune threshold on validation data)
   - --valid_file (path to Clf2 validation JSON with "answer" field 
     containing B/C labels — used only when --tune_threshold is set)

2. FEATURE EXTRACTION:
   Reuse the same feature extraction logic from clf2_feature_probe.py:
   - token_len: len(question.split()) — whitespace word count
   - entity_count: len(doc.ents) via spaCy en_core_web_sm with only NER 
     enabled (disable parser, lemmatizer)
   - hop_indicator_count: count of how many of the 7 BRIDGE_PATTERNS match 
     the question (NOT binary — count all matching patterns). This maps to 
     SymRAG's N_hops(q), which counts multi-hop keyword indicators.
   
   Copy the BRIDGE_PATTERNS list from clf2_feature_probe.py (the 7 regex 
   patterns for relative clauses, double possessives, temporal 
   subordination, demonstrative back-references, both/and, between/and, 
   nested wh-questions).
   
   Process all questions via nlp.pipe(questions, batch_size=256).

3. COMPLEXITY SCORE κ(q) — following SymRAG's formula exactly 
   (minus A(q)):
   
   # Normalize query length by max in dataset
   L(q) = token_len / max_token_len_in_dataset
   
   # Structural heuristic (SymRAG Appendix A.1.1)
   # Note: SymRAG normalizes by query length |q| (in tokens)
   S_H(q) = 0.05 * (entity_count / token_len) + 0.10 * (hop_count / token_len)
   
   # Final score (with w_L = 1.0, omitting w_A * A(q))
   kappa(q) = L(q) * (1 + S_H(q))

4. THRESHOLD TUNING (when --tune_threshold is set):
   - Load --valid_file (the Clf2 silver validation JSON)
   - Filter to entries with label in {"B", "C"} only
   - Compute κ(q) for each validation question
   - Search thresholds from the 5th to 95th percentile of κ values, 
     in 100 steps
   - For each threshold: if κ(q) >= threshold → predict C, else → predict B
   - Select the threshold that maximizes accuracy on the validation set
   - Log: "Tuned threshold: {t:.4f}, validation accuracy: {acc:.4f}, 
     val B-acc: {b_acc:.4f}, val C-acc: {c_acc:.4f}"
   - Use this threshold for prediction (overrides --kappa_threshold)
   - Also report: best F1 (macro), and the threshold that maximizes 
     macro-F1 (note if different from accuracy-optimal)

5. ROUTING LOGIC for each question in predict.json:
   a. Determine Gate 1 decision:
      - If --use_agreement_gate: compute agreement between nor_qa and 
        oner_qa using the same logic as predict_complexity_agreement.py 
        (normalize_answer, answer_extractor, exact string match)
      - Else: load --clf1_pred_file and use its predictions
   b. If Gate 1 = A (agree / no retrieval) → use nor_qa answer
   c. If Gate 1 = R (disagree / needs retrieval):
      - Compute κ(q)
      - If κ(q) >= threshold → route C (use ircot_qa answer)
      - If κ(q) < threshold → route B (use oner_qa answer)

6. PREDICTION FILE LOADING:
   Use the same file paths and BM25 retrieval counts as 
   predict_complexity_agreement.py:
   
   DATASETS = ["musique", "hotpotqa", "2wikimultihopqa", "nq", "trivia", "squad"]
   
   ONER_BM25 = {"flan_t5_xl": 15, "flan_t5_xxl": 15, "gpt": 6}
   IRCOT_BM25 = {"flan_t5_xl": 6, "flan_t5_xxl": 6, "gpt": 3}
   
   Load nor_qa, oner_qa, ircot_qa predictions from:
   predictions/test/{strategy}_{model}_{dataset}____prompt_set_1.../prediction__*.json

7. OUTPUT FORMAT (same as predict_complexity_agreement.py):
   - {dataset}/{dataset}.json — routed QA predictions
   - {dataset}/{dataset}_option.json — predictions with routing labels 
     and step counts
   - routing_stats.json containing:
     {
       "method": "symrag_kappa_structural",
       "threshold_used": ...,
       "threshold_tuned": true/false,
       "kappa_stats": {"mean": ..., "std": ..., "min": ..., "max": ...},
       "routing_counts": {"A": ..., "B": ..., "C": ...},
       "per_dataset": { ... },
       "symrag_weights": {"w_L": 1.0, "w_sh1": 0.05, "w_sh2": 0.10},
       "note": "A(q) attention term omitted; structural heuristic only"
     }

8. Use the same answer_extractor() and normalize_answer() functions as 
   predict_complexity_agreement.py. Import or copy them.

IMPORTANT CONSTRAINTS:
- Do NOT import or use torch, transformers, or any model loading
- The script should be fully training-free (only spaCy for NER)
- Use the same output directory structure conventions as other iterations
- The script must work for all three model variants
```

---

The key corrections from the original prompt:

- **S_H(q) normalizes by query length |q|**, not raw counts — this is explicitly in Appendix A.1.1
- **hop_count should be a count**, not binary — SymRAG's N_hops(q) counts multi-hop keywords, so I changed bridge_flag from binary to a count of how many patterns match
- **The multiplicative structure** κ = L(q) · (1 + S_H(q)) is preserved, not an additive weighted sum as I had before
- **Published weights** (w_sh1=0.05, w_sh2=0.1) are used directly instead of arbitrary tuneable weights
- **Threshold search range** uses percentiles of actual κ values rather than arbitrary 0.1–0.9 range