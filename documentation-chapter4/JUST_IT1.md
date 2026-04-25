Here is the full SOTA assessment for Iteration 1.

***

## Overall Framing

Iteration 1 is a faithful extension of the Adaptive-RAG baseline, replacing the single ternary classifier (A/B/C) with two cascaded binary classifiers (Clf1: A vs R; Clf2: B vs C). The Adaptive-RAG paper explicitly states it uses T5-Large, a learning rate of 3e-5, and AdamW, so your core hyperparameters are directly traceable to peer-reviewed literature. Below is a parameter-by-parameter verdict. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_15b04dae-a132-4275-be82-5199023cbeae/9452e2bd-d072-4d09-bad5-c5073d24b5d6/Adaptive-RAG.pdf)

***

## ✅ Justified Parameters

| Parameter | Value | Verdict | Citation to use |
|---|---|---|---|
| Base model | T5-Large (770M) | Direct match to Adaptive-RAG baseline | Raffel et al., 2020 (T5); Jeong et al., 2024 (Adaptive-RAG)  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_15b04dae-a132-4275-be82-5199023cbeae/9452e2bd-d072-4d09-bad5-c5073d24b5d6/Adaptive-RAG.pdf) |
| Learning rate | `3e-5` | Matches Adaptive-RAG exactly; also used in FLAN instruction tuning  [blog.paperspace](https://blog.paperspace.com/instruction-tuning/) | Jeong et al., 2024  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_15b04dae-a132-4275-be82-5199023cbeae/9452e2bd-d072-4d09-bad5-c5073d24b5d6/Adaptive-RAG.pdf) |
| Optimizer | AdamW | Standard for transformer fine-tuning; decoupled weight decay is theoretically motivated | Loshchilov & Hutter, 2019 (cited in Adaptive-RAG)  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_15b04dae-a132-4275-be82-5199023cbeae/9452e2bd-d072-4d09-bad5-c5073d24b5d6/Adaptive-RAG.pdf) |
| LR scheduler | Linear decay | Standard and widely used in NLP fine-tuning  [mbrenndoerfer](https://mbrenndoerfer.com/writing/fine-tuning-learning-rates-llrd-warmup-decay-transformers) | Devlin et al., 2019 (BERT) |
| Classification via generative decoding | Constrained argmax over label token logits | The canonical T5 text-to-text approach; avoids adding a non-pretrained classification head | Raffel et al., 2020  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_15b04dae-a132-4275-be82-5199023cbeae/9452e2bd-d072-4d09-bad5-c5073d24b5d6/Adaptive-RAG.pdf) |
| Epoch sweep + manual best-epoch selection | 15/20/25/30/35 epochs | Matches Adaptive-RAG: "trained using the epoch that shows the best performance"  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_15b04dae-a132-4275-be82-5199023cbeae/9452e2bd-d072-4d09-bad5-c5073d24b5d6/Adaptive-RAG.pdf) | Jeong et al., 2024 |
| Batch size | 32 | Standard for seq2seq fine-tuning  [datacamp](https://www.datacamp.com/tutorial/flan-t5-tutorial) | Fine-tuning convention |
| Max seq length | 384 | Conservative but sufficient for short questions; above any realistic query length | T5 fine-tuning convention |
| Silver labels + binary inductive-bias labels (Clf2) | Merged training set | Directly inherited from Adaptive-RAG's two-strategy labeling scheme  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_15b04dae-a132-4275-be82-5199023cbeae/9452e2bd-d072-4d09-bad5-c5073d24b5d6/Adaptive-RAG.pdf) | Jeong et al., 2024 |

***

## ⚠️ Parameters That Need Justification or Are Non-Standard

### 1. Warmup steps = 0
**Issue:** The consensus in transformer fine-tuning is to use a warmup phase of roughly 5–10% of total training steps. Without warmup, the model starts from a high learning rate immediately, which can destabilize early training. A guide specifically targeting fine-tuning small LLMs recommends at least a 1% linear warmup. [github](https://github.com/huggingface/transformers/issues/6673)

**Your defense:** The Adaptive-RAG paper does not explicitly mention warmup either, and it uses the same `3e-5 + AdamW` setup. Since IT1 is a direct faithful re-implementation of that baseline, the 0-warmup can be justified as matching the baseline exactly. You should note it explicitly in your thesis as an inherited limitation, not a deliberate design choice.

***

### 2. Weight decay = 0.0
**Issue:** For AdamW, the whole point of using AdamW over plain Adam is the decoupled weight decay regularization (Loshchilov & Hutter, 2019). Setting it to 0.0 makes AdamW and Adam equivalent. Standard practice for transformer fine-tuning is weight decay of 0.01–0.1. [mbrenndoerfer](https://mbrenndoerfer.com/writing/adamw-optimizer-decoupled-weight-decay)

**Your defense:** Again, this matches the Adaptive-RAG code (which also does not report a non-zero weight decay) and is the HuggingFace default. Defensible as faithful baseline reproduction, but worth acknowledging. For later iterations that introduce new components (e.g., Focal Loss), consider setting `weight_decay=0.01`. [discuss.huggingface](https://discuss.huggingface.co/t/does-the-default-weight-decay-of-0-0-in-transformers-adamw-make-sense/1180)

***

### 3. No random seed
**Issue:** Not setting a random seed means results cannot be reproduced exactly. This is a methodological concern in DSR, where you want to isolate the effect of each design change across iterations.

**Your defense:** The Adaptive-RAG paper itself states "we perform experiments with a single run" due to computational cost. This is an acceptable practical limitation you can explicitly disclose. You should state in your thesis that results may vary by ±X% due to random initialization and data shuffling. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_15b04dae-a132-4275-be82-5199023cbeae/9452e2bd-d072-4d09-bad5-c5073d24b5d6/Adaptive-RAG.pdf)

***

### 4. No early stopping — epoch sweep instead
**Issue:** Early stopping is the standard practice to prevent overfitting and reduce compute. Running 5 independent full training runs from scratch (epoch sweep) is computationally more expensive than training once with early stopping. [arxiv](https://arxiv.org/html/2412.13337v1)

**Your defense:** This is not a SOTA violation. The approach is equivalent to a manual grid search over epochs and produces the same outcome (best-validation model) as early stopping, while also giving you a learning curve for free. It is directly aligned with Adaptive-RAG's methodology. The tradeoff is compute vs. simplicity — acceptable for a thesis context. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_15b04dae-a132-4275-be82-5199023cbeae/9452e2bd-d072-4d09-bad5-c5073d24b5d6/Adaptive-RAG.pdf)

***

### 5. Cascaded binary vs. single ternary classifier (the core IT1 claim)
**Issue:** You need literature to justify *why* a cascaded binary architecture should outperform the single ternary classifier.

**Justification to use:**
- Binary classification is generally easier to learn than multiclass because it involves simpler decision boundaries and requires less data per class. [reddit](https://www.reddit.com/r/deeplearning/comments/gqqmhm/binary_classification_vs_multi_classification/)
- Adaptive-RAG's own confusion matrix shows frequent B↔C confusion (~31% of C misclassified as B) and A↔B confusion (~47% of A misclassified as B), directly motivating a decomposed approach. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_15b04dae-a132-4275-be82-5199023cbeae/9452e2bd-d072-4d09-bad5-c5073d24b5d6/Adaptive-RAG.pdf)
- Two-stage cascade routing is an established pattern in NLP routing systems — see the EMNLP 2025 "SELECT-THEN-ROUTE" paper, which empirically validates that two-stage cascades outperform single-stage routers. [aclanthology](https://aclanthology.org/2025.emnlp-industry.28.pdf)
- Decomposing a multiclass problem into a hierarchy of binary problems is theoretically grounded in the error-correcting output codes (ECOC) literature (Dietterich & Bakiri, 1995).

***

## Summary of Actions

| Action | Priority |
|---|---|
| Cite Raffel et al. (2020) and Jeong et al. (2024) for all inherited hyperparameters | Required |
| Acknowledge warmup=0 and weight_decay=0.0 as inherited from baseline, not deliberate design | Required |
| Cite Loshchilov & Hutter (2019) for AdamW | Required |
| Add seed in subsequent iterations for reproducibility | Recommended |
| Cite the A↔B and B↔C confusion rates from Adaptive-RAG's Figure 3 to motivate the cascade | Required for IT1 claim |
| Add EMNLP 2025 "SELECT-THEN-ROUTE" and/or ECOC literature to further justify binary cascade | Strongly recommended |

Ready for Iteration 2 whenever you are.