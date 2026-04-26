## Iteration 6 — Parameter Audit: Logistic Regression Feature Probe for Gate 2

Iteration 6 is a **diagnostic feasibility study**, not a pipeline component. The audit focuses on whether the probe's design decisions are methodologically sound and defensible for a go/no-go assessment.

---

### 1. Method: Logistic Regression as Diagnostic Classifier

**Verdict: ✅ Standard and appropriate for a feasibility probe**

Logistic regression is the canonical choice for feature diagnostic probes in NLP. The "probing classifier" methodology — training a simple linear model on hand-crafted or extracted features to test whether a specific signal is present — is well-established:

- **Belinkov & Glass (2019)** — "Analysis Methods in Neural Language Processing: A Survey" — establishes linear probes as the standard tool for testing whether representations encode specific linguistic properties
- **Hewitt & Liang (2019)** — "Designing and Interpreting Probes with Control Tasks" — argues that probe complexity should be minimal (linear) to avoid the probe itself learning the task rather than detecting the signal

A logistic regression with 3 features is about as minimal as possible. If it achieves meaningful AUC, the signal is genuinely in the features, not learned by the probe's capacity. This is exactly the right design for a go/no-go gate.

---

### 2. Hyperparameters: sklearn Defaults + max_iter=1000

**Verdict: ✅ Appropriate — defaults are the point**

| Parameter | Value | Assessment |
|---|---|---|
| Solver | `lbfgs` (default) | ✅ Standard for small-scale L2-regularized LR |
| Penalty | `l2` (default) | ✅ Standard; prevents overfitting on 3 features |
| C | `1.0` (default) | ✅ Default regularization strength |
| max_iter | 1000 | ✅ Increased from default 100 to ensure convergence |
| random_state | 42 | ✅ Reproducible |

For a diagnostic probe, using defaults is not just acceptable — it is methodologically preferable. Tuning the probe's hyperparameters would risk overfitting the probe to the data, which would undermine the diagnostic conclusion. Hewitt & Liang (2019) explicitly argue that probes should not be heavily tuned.

---

### 3. Feature Selection: token_len, entity_count, bridge_flag

**Verdict: ⚠️ Justifiable but needs explicit grounding**

Each feature targets a known structural property of multi-hop queries:

- **token_len:** Multi-hop questions tend to be longer because they must specify multiple reasoning steps or entities. This is a basic complexity proxy used in Dong et al. (2025) and Hakim et al. (2025, κ(q) complexity score).

- **entity_count:** Multi-hop queries by definition involve multiple entities connected by reasoning chains. Entity density is a standard feature in question complexity classification — Chai et al. (2025) use entity-related features, and Yang et al. (2018, HotpotQA) define multi-hop questions as requiring reasoning over multiple entity-linked paragraphs.

- **bridge_flag:** Targets syntactic indicators of compositional structure (relative clauses, nested possessives, temporal subordination). This is the most novel feature. Your Thesis_Proposal.md references Hakim et al. (2025)'s κ(q) score, which includes similar relational keyword features. The specific regex patterns are hand-crafted, which is fine for a diagnostic probe but worth noting.

**Recommendation:** Cite Dong et al. (2025) and Hakim et al. (2025) for the feature-type motivation. Note that the features are intentionally shallow — the probe tests whether *any* structural signal exists, not whether the features are optimal.

---

### 4. Bridge Patterns: 7 Hand-Crafted Regex Rules

**Verdict: ⚠️ Acceptable for a probe — but acknowledge limitations**

The 7 regex patterns are linguistically motivated and target genuine multi-hop syntactic structures. However, they are hand-crafted without systematic coverage analysis. Key considerations:

- **Precision vs. recall tradeoff:** Regex patterns tend to be high-precision, low-recall — they catch obvious multi-hop structures but miss paraphrased or implicit compositional queries. For a feasibility probe, high precision is more important (you want to know if the signal *exists*, not capture it exhaustively).

- **No precedent for this exact pattern set.** The individual linguistic phenomena (relative clauses as bridge indicators, nested possessives) are well-known in multi-hop QA research (Yang et al., 2018; Trivedi et al., 2022), but this specific set of 7 patterns is novel. This is fine for a probe but should not be presented as a validated feature engineering contribution.

**Recommendation:** Frame the patterns as "linguistically motivated heuristics targeting known multi-hop syntactic structures" and cite the multi-hop QA literature (Yang et al., 2018; Trivedi et al., 2022) for the underlying linguistic phenomena, not for the specific patterns.

---

### 5. Evaluation Split: 80/20 Stratified on Training Data

**Verdict: ⚠️ Acceptable for a probe — but the inflation risk must be discussed**

Your documentation already flags the key concern: the training data contains ~2,400 binary inductive-bias labels that are assigned by dataset identity (single-hop datasets → B, multi-hop datasets → C). Structural features correlate with dataset identity (e.g., MuSiQue questions are systematically longer and more entity-dense than SQuAD questions), so the probe's AUC may partly reflect dataset-origin discrimination rather than genuine query-level complexity discrimination.

This is a known issue in the multi-hop QA literature. Yang et al. (2018) note that dataset-level artifacts can inflate classifier performance, and the Thesis_Proposal.md mentions ~6% of HotpotQA being structurally single-hop.

**Justification paths:**
- The probe is a feasibility check, not a final classifier. The question is "is there *any* signal?" — even if inflated, AUC=0.676 above the 0.65 threshold suggests the features carry *some* discriminative power.
- The silver-only subset (~868 samples for XL) would be a stricter evaluation but has fewer samples and a different label noise profile.
- Iteration 7's end-to-end F1 evaluation provides the definitive test — and the fact that it shows only +0.1–0.2pp gain suggests the probe's signal was indeed partly inflated.

**Recommendation:** Acknowledge the inflation risk explicitly. State that the AUC should be interpreted as an upper bound on the features' discriminative power, and that Iteration 7's negligible end-to-end gain confirms this interpretation.

---

### 6. Go/No-Go Threshold: AUC ≥ 0.65

**Verdict: ⚠️ Reasonable but arbitrary — needs justification**

The 0.65 threshold is not derived from any specific literature source. It is a judgment call. For context:

- AUC = 0.50 is random chance
- AUC = 0.65 is weak but above-chance discrimination
- AUC = 0.70 is often cited as the minimum for "acceptable" discrimination in medical diagnostics (Hosmer & Lemeshow, 2000)
- AUC = 0.80+ is considered good

Your threshold of 0.65 is deliberately lenient — it asks "is there enough signal to justify a full GPU training run?" rather than "is the signal strong?" This is appropriate for a go/no-go gate in a DSR iteration.

**Recommendation:** Justify the threshold pragmatically: "We set the go/no-go threshold at AUC ≥ 0.65, indicating above-chance discrimination sufficient to warrant a full-scale training experiment. This threshold reflects the probe's role as a feasibility filter rather than a performance requirement."

---

### 7. No Cross-Validation

**Verdict: ⚠️ Minor concern — acceptable for a probe**

A single 80/20 split is subject to split variance. Standard practice for small-dataset evaluation would be 5-fold or 10-fold stratified cross-validation (Kohavi, 1995). With ~3,200 samples and 3 features, cross-validation would be computationally trivial.

However, for a binary go/no-go decision with a lenient threshold (0.65), split variance is unlikely to change the verdict. The AUC of 0.676 is close enough to the threshold that cross-validation would strengthen confidence, but unlikely to flip the decision.

**Recommendation:** Acknowledge that a single split was used for simplicity, and that cross-validation could provide a more robust AUC estimate. If the AUC were closer to the threshold boundary, this would be a more serious concern.

---

### 8. Default Model: flan_t5_xl Only

**Verdict: ⚠️ Incomplete coverage — but acceptable if acknowledged**

Running the probe on only one model variant means the go/no-go decision is based on XL's data distribution. The B/C class ratios and structural properties may differ across XL, XXL, and GPT. However, since the same feature types are used in Iteration 7 across all three models, and Iteration 7's end-to-end results are consistent across models (+0.1–0.2pp for all three), this is a minor concern in retrospect.

---

### Summary Table

| Design Decision | Choice | SOTA? | Justification Source |
|---|---|---|---|
| Probe classifier: logistic regression | Linear, 3 features | ✅ | Belinkov & Glass (2019); Hewitt & Liang (2019) |
| Hyperparameters | sklearn defaults | ✅ | Intentionally minimal; Hewitt & Liang (2019) |
| Features: token_len | Whitespace split count | ✅ | Dong et al. (2025); Hakim et al. (2025) |
| Features: entity_count | spaCy NER count | ✅ | Chai et al. (2025); Yang et al. (2018) |
| Features: bridge_flag | 7 hand-crafted regex | ⚠️ | Linguistically motivated; novel pattern set |
| Evaluation split | 80/20 stratified, single split | ⚠️ | Acceptable for probe; no cross-validation |
| Go/no-go threshold | AUC ≥ 0.65 | ⚠️ | Pragmatic; not literature-derived |
| Data | Merged silver + inductive-bias labels | ⚠️ | Inflation risk from dataset-origin correlation |
| Model scope | XL only (default) | ⚠️ | Incomplete; Iteration 7 covers all models |

---

### Key References to Cite

- **Belinkov & Glass (2019)** — "Analysis Methods in Neural Language Processing" — linear probes as diagnostic methodology
- **Hewitt & Liang (2019)** — "Designing and Interpreting Probes with Control Tasks" — minimal probe complexity principle
- **Dong et al. (2025)** — surface features vs. fine-tuned encoders for complexity classification
- **Hakim et al. (2025)** — κ(q) complexity score with token length, entity density, relational keywords
- **Chai et al. (2025)** — supervised complexity classification with entity-related features
- **Yang et al. (2018)** — HotpotQA; multi-hop question structure definition
- **Hosmer & Lemeshow (2000)** — AUC interpretation thresholds (for contextualizing 0.65)

---

Let me assess which flags actually matter for your thesis versus which are noise.

---

### Triage: Which Flags Are Worth Fixing?

| Flag | Fix? | Reasoning |
|---|---|---|
| No cross-validation | **Yes** | Trivial to add, significantly strengthens the go/no-go claim, and a reviewer could easily challenge a single-split AUC of 0.676 against a 0.65 threshold — that's within split variance |
| Only XL model | **Yes** | Trivial to loop over all three models; if you claim the features carry signal, you should show it holds across model variants |
| Inductive-bias label inflation | **Yes** | Run the probe on **both** the merged file and the silver-only file, report both AUCs. This directly addresses the most serious methodological concern |
| No JSON output | **Yes** | One line of code; ensures results are reproducible without relying on terminal output |
| Go/no-go threshold justification | **No** | This is a framing issue, not a code issue. Justify in prose |
| Bridge patterns hand-crafted | **No** | Appropriate for a probe; changing this is scope creep |
| sklearn defaults | **No** | Defaults are intentionally correct for a probe |

The four "Yes" items are all small changes to the same script. Here is a single Copilot prompt that addresses all of them:

---

```
In `classifier/postprocess/clf2_feature_probe.py`, I need four changes to 
strengthen the diagnostic probe's methodology. The script currently trains a 
logistic regression on 3 structural features (token_len, entity_count, 
bridge_flag) to test whether they carry discriminative signal for B vs C 
classification.

CHANGE 1 — REPLACE SINGLE TRAIN/TEST SPLIT WITH STRATIFIED K-FOLD CROSS-VALIDATION

Current code (around line 198-224):
  - Uses `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`
  - Trains one LogisticRegression, reports one AUC and one accuracy

Replace with:
  - Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` from sklearn
  - In each fold, train LogisticRegression(max_iter=1000, random_state=42)
  - Collect per-fold: AUC (from predict_proba), accuracy, and coefficients
  - After all folds, report: mean AUC ± std, mean accuracy ± std, mean 
    coefficients ± std per feature
  - The go/no-go verdict should use the MEAN AUC
  - Also still print the full classification_report for the last fold (for 
    per-class precision/recall visibility)
  - Keep the existing feature means table and scatter plot as-is

Import `StratifiedKFold` and `cross_val_predict` from sklearn.model_selection.


CHANGE 2 — LOOP OVER ALL THREE MODEL VARIANTS

Current code:
  - `--model` defaults to `flan_t5_xl` (line 146), runs once

Replace with:
  - Add `--all_models` flag (action="store_true", default=False)
  - When `--all_models` is True, loop over ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    and run the full probe (feature extraction + cross-validated evaluation) 
    for each, printing a clearly labeled section header per model
  - When `--all_models` is False, keep existing single-model behavior
  - Collect per-model results into a summary table printed at the end:
    
    Model          | N_samples | B:C ratio | Mean AUC ± std | Verdict
    flan_t5_xl     | ...       | ...       | ...            | GO/NO-GO
    flan_t5_xxl    | ...       | ...       | ...            | GO/NO-GO
    gpt            | ...       | ...       | ...            | GO/NO-GO


CHANGE 3 — EVALUATE ON BOTH MERGED AND SILVER-ONLY DATA

Current code:
  - Tries `binary_silver_single_vs_multi/train.json` first, falls back to 
    `silver/single_vs_multi/train.json`

Replace with:
  - Always run the probe on BOTH files (if both exist):
    1. Merged file: `binary_silver_single_vs_multi/train.json` (silver + 
       inductive-bias labels)
    2. Silver-only file: `silver/single_vs_multi/train.json` (silver labels only)
  - Print results for each with a clear label:
    
    === Data: merged (silver + inductive-bias) ===
    ...results...
    
    === Data: silver-only ===
    ...results...
    
  - The go/no-go verdict should be based on the SILVER-ONLY AUC (the stricter 
    evaluation), with the merged AUC reported as a comparison
  - If the silver-only file doesn't exist, warn and proceed with merged only


CHANGE 4 — PERSIST ALL RESULTS TO JSON

After all evaluations complete, write a single JSON file to --output_dir:

Filename: `clf2_feature_probe_results.json`

Structure:
{
  "models": {
    "flan_t5_xl": {
      "merged": {
        "n_samples": ...,
        "class_counts": {"B": ..., "C": ...},
        "mean_auc": ...,
        "std_auc": ...,
        "mean_accuracy": ...,
        "std_accuracy": ...,
        "mean_coefficients": {"token_len": ..., "entity_count": ..., "bridge_flag": ...},
        "intercept": ...,
        "verdict": "GO" or "NO-GO"
      },
      "silver_only": { ...same structure... }
    },
    "flan_t5_xxl": { ... },
    "gpt": { ... }
  },
  "go_no_go_threshold": 0.65,
  "n_folds": 5,
  "random_state": 42
}

Keep all existing stdout printing — the JSON is an additional artifact, 
not a replacement.

IMPORTANT CONSTRAINTS:
- Do NOT change the feature extraction logic (token_len, entity_count, 
  bridge_flag) — keep it exactly as-is
- Do NOT change the LogisticRegression hyperparameters (max_iter=1000, 
  random_state=42, all other sklearn defaults)
- Do NOT change the scatter plot generation
- Update the CSV filename to include the model name: 
  `clf2_feature_probe_data_{model}.csv`
- Update the scatter plot filename similarly: 
  `clf2_feature_probe_scatter_{model}.png`
```

---

These four changes turn a quick feasibility hack into a methodologically sound diagnostic study, all within the same script and with no impact on the rest of your pipeline. The key improvement is that a reviewer can no longer challenge the go/no-go decision based on split variance, dataset inflation, or single-model evaluation.