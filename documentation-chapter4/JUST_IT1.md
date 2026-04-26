I'll start by reviewing the documentation you provided for Iteration 1, then check the project knowledge for relevant context on what constitutes SOTA practices for these parameters.Excellent — I found the key reference. The original Adaptive-RAG paper (Jeong et al., 2024) specifies its classifier training setup. Now let me search for additional context on T5 fine-tuning best practices to cross-reference each parameter.Now let me search for the original T5 paper's recommended fine-tuning setup and the original Adaptive-RAG's specific hyperparameters.Now I have enough information to give you a thorough parameter-by-parameter audit. Let me compile the analysis.

---

## Iteration 1 — Parameter Audit Against Literature

I evaluate each parameter from your DOCUMENTATION_IT1.md against (a) the original Adaptive-RAG paper (Jeong et al., 2024), (b) the T5 fine-tuning literature, and (c) general transformer fine-tuning best practices. I flag each as **✅ Justifiable**, **⚠️ Deviates — needs justification**, or **❌ Not SOTA — should be addressed or defended**.

---

### 1. Model: T5-Large (770M), generative classification head

**Verdict: ✅ Justifiable**

Jeong et al. (2024) explicitly state they use T5-Large for their query-complexity classifier. Your choice directly follows the original Adaptive-RAG implementation. The generative decoding approach (argmax over label token logits) is the same as theirs. No issue here — you can cite Jeong et al. (2024) directly.

---

### 2. Learning Rate: 3e-5

**Verdict: ⚠️ Deviates from the original — needs explicit justification**

Jeong et al. (2024) report using a learning rate of **3e-5** in their implementation details section. So your value matches theirs. However, the original T5 paper (Raffel et al., 2020) pre-trained T5 with Adafactor and an inverse-square-root schedule. For **downstream fine-tuning** of T5, the community convention splits:

- **Adafactor** with relative step size (lr=None) — the T5 authors' recommended approach
- **AdamW** at 1e-4 to 3e-5 — common in HuggingFace practice

Your value of 3e-5 with AdamW falls within the standard range for transformer fine-tuning (1e-5 to 5e-5). **You can justify this by citing Jeong et al. (2024) as the direct precedent**, and the HuggingFace fine-tuning conventions (Devlin et al., 2019, BERT paper recommends {2e-5, 3e-5, 5e-5} for fine-tuning).

---

### 3. Optimizer: AdamW

**Verdict: ⚠️ Minor deviation — easily justifiable**

The T5 authors recommend **Adafactor** for T5 models. However, Jeong et al. (2024) explicitly use **AdamW** (Loshchilov & Hutter, 2019) for their classifier. Since you are replicating and extending their system, using AdamW is directly justifiable. Additionally, the HuggingFace documentation notes that for original T5 pre-trained models (as opposed to T5-v1.1), AdamW works well.

**Cite:** Jeong et al. (2024) for AdamW choice; Loshchilov & Hutter (2019) for the optimizer itself.

---

### 4. Weight Decay: 0.0

**Verdict: ⚠️ Deviates from best practice — needs justification**

Standard transformer fine-tuning practice uses weight decay of **0.01** (as established by Devlin et al., 2019 for BERT and widely adopted since). Setting weight decay to 0.0 effectively disables regularization, which is unusual. The PyTorch AdamW default is 0.01.

**Potential justification paths:**
- If Jeong et al. (2024) also used 0.0, cite them (their paper says AdamW but doesn't specify weight decay — you may need to check their code)
- Small dataset sizes (~1.3K–3.3K samples) and few epochs could make weight decay less critical
- Your epoch sweep (15–35) already acts as implicit regularization via early stopping by selection

**Recommendation:** Acknowledge this explicitly in your thesis. If you cannot confirm Jeong et al.'s weight decay setting, note it as a deviation and argue that the small dataset/few-epoch regime makes it less consequential.

---

### 5. Warmup Steps: 0

**Verdict: ❌ Not SOTA — needs strong justification**

This is the most significant deviation. Learning rate warmup is considered near-standard practice for transformer fine-tuning:

- Devlin et al. (2019) use warmup for BERT fine-tuning
- The T5 HuggingFace documentation explicitly states: "Training without LR warmup or clip_threshold is not recommended"
- The warmup literature (Goyal et al., 2017; Vaswani et al., 2017) establishes warmup as essential for stabilizing Adam-family optimizers on pre-trained models

With 0 warmup, you start at full learning rate (3e-5) immediately. For a 770M-parameter model, this risks destabilizing early training, especially with the small dataset sizes you have (~1.3K samples for Clf1).

**Justification paths:**
- If Jeong et al. (2024) also used 0 warmup, cite them as precedent
- The learning rate (3e-5) is already conservative, reducing the need for warmup
- Empirically, your model converges and matches the original Adaptive-RAG F1 — evidence that warmup was not critical in this regime

**Recommendation:** Explicitly acknowledge this and state the justification. If you have access to Jeong et al.'s code, check whether they used warmup.

---

### 6. LR Scheduler: Linear decay (to 0)

**Verdict: ✅ Justifiable**

Linear decay is one of the standard schedules for transformer fine-tuning (the other common choice being cosine annealing). Devlin et al. (2019) use linear warmup + linear decay. Your choice is standard.

---

### 7. Batch Size (train): 32

**Verdict: ✅ Justifiable**

For T5-Large fine-tuning, batch size 32 is reasonable and within typical ranges (16–64). Given dataset sizes of ~1.3K–3.3K, each epoch is ~40–100 steps, which is very short. A smaller batch size would increase noise per step; a larger one would make each epoch even shorter. 32 is a sensible choice.

---

### 8. Epochs: Sweep over {15, 20, 25, 30, 35} (XL/XXL) and {35, 40} (GPT)

**Verdict: ⚠️ Unusual — needs justification**

The epoch counts are high by transformer fine-tuning standards (where 2–10 epochs is typical). However, your dataset sizes are exceptionally small (~1.3K samples), so 15–35 epochs translate to roughly 600–1,400 gradient steps at batch size 32. This is actually a modest number of total steps.

Jeong et al. (2024) state: "the classifier is trained using the epoch that shows the best performance until 100 training iterations." This is ambiguous — it could mean 100 epochs or 100 iterations. Your sweep approach is reasonable as a brute-force substitute for proper early stopping.

**Key concern:** Each epoch value trains a **fresh model from scratch**, not a continuation. This means you're running 5 (or 2) independent training runs, which is computationally wasteful but methodologically clean.

**Justification:** Frame it as a systematic hyperparameter search over training duration, substituting for the absence of early stopping. Cite that with ~40 steps per epoch, 35 epochs ≈ 1,400 steps, which is within normal fine-tuning ranges.

---

### 9. No Early Stopping / No Best-Model Checkpointing

**Verdict: ❌ Not SOTA — needs justification**

Standard practice is to use validation-based early stopping or at least save the best checkpoint. Your setup saves only the **last epoch's** weights and selects the best model post-hoc by comparing validation results across independent runs. This is equivalent to manual early stopping but:

- Wastes compute (5 full training runs instead of 1 with checkpointing)
- Risks missing the optimal checkpoint within a run (only epoch-end checkpoints, not step-level)

**Justification paths:**
- The epoch sweep is effectively a coarse-grained form of model selection
- Post-hoc selection on validation accuracy is methodologically equivalent to early stopping
- Jeong et al. (2024) describe a similar post-hoc selection approach

---

### 10. No Seed Set → Non-reproducibility

**Verdict: ❌ Not SOTA — needs justification**

Reproducibility is a basic scientific requirement. Not setting a seed means your results cannot be exactly reproduced, and you cannot disentangle random variation from design choices across iterations.

**Justification paths:**
- If Jeong et al.'s code also lacks seed setting, cite for consistency
- You run multiple epoch sweeps per configuration, which provides some robustness
- In a DSR thesis with iterative design, exact reproducibility may be secondary to demonstrating trends

**Recommendation:** Acknowledge this limitation explicitly in your threats-to-validity section.

---

### 11. No Imbalance Handling (vanilla CE, no weighting/sampling)

**Verdict: ✅ Correct for Iteration 1**

This is intentionally the baseline. The entire point of Iteration 1 is to establish that the vanilla setup suffers from class imbalance, motivating Iterations 2–4. No issue — this is by design.

---

### 12. Max Sequence Length: 384, Doc Stride: 128

**Verdict: ⚠️ Unusual for classification — needs justification**

These parameters (`max_seq_length=384`, `doc_stride=128`, `return_overflowing_tokens=True`) are SQuAD-style extractive QA settings, not typical for text classification. The doc stride + overflowing tokens mechanism creates **multiple training examples from a single input** if the question exceeds 384 tokens. For short questions (which most of yours are), this is harmless but unnecessary overhead.

**Justification:** Likely inherited from the Adaptive-RAG codebase, which may have been built on a SQuAD template. Since most questions are well under 384 tokens, the stride mechanism rarely activates and has no practical effect. State this explicitly.

---

### Summary Table

| Parameter | Value | SOTA? | Justification Source |
|---|---|---|---|
| Model (T5-Large, generative) | T5-Large | ✅ | Jeong et al. (2024) |
| Learning rate | 3e-5 | ✅ | Jeong et al. (2024); Devlin et al. (2019) |
| Optimizer | AdamW | ✅ | Jeong et al. (2024); Loshchilov & Hutter (2019) |
| Weight decay | 0.0 | ⚠️ | Needs justification; standard is 0.01 |
| Warmup steps | 0 | ❌ | Deviates from HF/T5 guidance; justify via LR conservatism or Jeong et al. precedent |
| LR scheduler | Linear decay | ✅ | Standard practice |
| Batch size (train) | 32 | ✅ | Reasonable for dataset size |
| Epochs | 15–35 sweep | ⚠️ | High count but low total steps; frame as systematic model selection |
| Early stopping | None | ❌ | Post-hoc selection is a substitute; acknowledge as limitation |
| Seed | Not set | ❌ | Reproducibility gap; acknowledge in threats to validity |
| Imbalance handling | None | ✅ | Intentional baseline design |
| Max seq length / doc stride | 384 / 128 | ⚠️ | Inherited from SQuAD template; harmless but should be noted |
| Loss function | Cross-entropy | ✅ | Standard; Jeong et al. (2024) |

---

### Critical Action Items

1. **Warmup = 0** is your hardest parameter to justify. Check Jeong et al.'s open-source code — if they also use 0 warmup, you have a direct precedent. Otherwise, argue that the conservative LR (3e-5) and short training runs mitigate warmup's absence.

2. **Weight decay = 0.0** — try to confirm Jeong et al.'s setting. If unconfirmable, argue small-dataset regime + epoch selection as implicit regularization.

3. **Seed** — acknowledge as a limitation; if you have time, consider fixing a seed in later iterations and noting the change.

4. **Max seq length / doc stride** — one sentence acknowledging this is inherited and functionally inert for your question lengths.