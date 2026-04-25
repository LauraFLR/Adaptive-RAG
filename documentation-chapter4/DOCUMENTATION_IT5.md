## Iteration 5 — Training-Free Gate 1 (UE Answer Agreement): Full Implementation Summary

The documentation numbering in OFFICIAL_EXP_RESULTS.md labels this "Iteration 5". In the codebase itself, the script's docstring calls it "Iteration 2a" (predict_complexity_agreement.py). It is the same experiment; the documentation uses a linear numbering across all experiments.

---

### 1. Files Involved

| File | Role |
|---|---|
| classifier/postprocess/predict_complexity_agreement.py | **Core script**: implements the agreement gate + routing. This is the only new script. |
| classifier/postprocess/postprocess_utils.py | Utility: `load_json()`, `save_json()` |
| evaluate_final_acc.py | Computes EM/F1 on routed predictions (shared across all iterations) |
| `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json` | 3,000 unlabelled test questions (qid→dataset mapping) |
| `predictions/test/nor_qa_{model}_{dataset}____prompt_set_1/prediction__*.json` | Pre-computed nor_qa answers (6 datasets × 3 models) |
| `predictions/test/oner_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__*___distractor_count__1/prediction__*.json` | Pre-computed oner_qa answers |
| `predictions/test/ircot_qa_{model}_{dataset}____prompt_set_1___bm25_retrieval_count__*___distractor_count__1/prediction__*.json` | Pre-computed ircot_qa answers (used for C-routed questions) |
| Clf2 checkpoint's `predict/dict_id_pred_results.json` | Gate 2 predictions (from Iteration 1 training) — passed via `--clf2_pred_file` |

No shell wrapper script exists; the Python script is invoked directly.

---

### 2. Gate 1 Replacement

Gate 1 is replaced by a **completely separate script** — predict_complexity_agreement.py. There is no flag or config switch within the trained-classifier pipeline. The script is standalone: it takes the Clf2 predictions file as input and produces the final routed predictions directly, bypassing any trained Clf1 entirely.

**No fine-tuning of Gate 1 occurs anywhere.** The script:
- Has no `import transformers`, no model loading, no `torch` usage
- Accepts only `model_name`, `--clf2_pred_file`, `--predict_file`, and `--output_path` ([lines 108–113](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L108-L113))
- The A/R decision is computed purely from pre-existing prediction files on disk

---

### 3. Uncertainty Estimation Implementation

This is **not** multi-sample temperature sampling. It is **cross-strategy answer agreement** between two already-computed, deterministic inference runs.

**Number of "samples":** Exactly 2 — one from `nor_qa` (no retrieval) and one from `oner_qa` (single-step retrieval). Both were generated with `do_sample=False` ([llm_client_generator.py L17](Adaptive-RAG/commaqa/models/llm_client_generator.py#L17), serve.py), meaning **greedy decoding**. No temperature sampling is involved.

**Sampling strategy:** None. The predictions are pre-computed artifacts from Phase 2/4 of the pipeline. The LLM server generates with `do_sample=False` and `temperature=1.0` ([llm_client_generator.py L17–18](Adaptive-RAG/commaqa/models/llm_client_generator.py#L17-L18)), but since `do_sample=False`, temperature has no effect — it's pure greedy/beam search.

**Aggregation logic** — `compute_agreement()` ([lines 86–107](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L86-L107)):

```
1. For each qid, load nor_qa raw answer and oner_qa raw answer
2. Handle list answers: take first element if list (lines 93–96)
3. Cast to str (lines 97–98)
4. Apply answer_extractor(): strip quotes, extract from "... answer is: X" CoT pattern (lines 100–101)
5. Apply normalize_answer(): lowercase → remove punctuation → remove articles → collapse whitespace (lines 103–104)
6. agree = both normalized answers are non-empty AND equal (line 106)
```

**Routing** ([lines 122–132](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L122-L132)):
- If `agree == True` → route to **A** (no retrieval)
- If `agree == False` → use Clf2's argmax prediction (**B** or **C**)

---

### 4. Decision Threshold

There is **no tunable threshold**. The decision is a hard binary: exact string match after normalization, or not ([line 106](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L106)):

```python
agree = bool(nor_norm and oner_norm and nor_norm == oner_norm)
```

This is effectively a threshold of 1.0 on agreement (100% of samples must agree). Since there are only 2 answers being compared, it's all-or-nothing. It is **hardcoded** — not tuned on a validation set and not inherited from any prior iteration.

---

### 5. No Residual Fine-Tuning Artifacts

Confirmed: **no trained Gate 1 checkpoint is loaded or referenced.**

- The script's imports are: `argparse`, `json`, `os`, `re`, `string`, `sys`, `Counter`, and `postprocess_utils` ([lines 16–23](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L16-L23)). No `torch`, no `transformers`, no model loading.
- The only external model artifact is the Clf2 predictions file (`--clf2_pred_file`), which is for Gate 2, not Gate 1.
- No reference to any `no_ret_vs_ret` output directory, checkpoint, or model weights appears anywhere in the script.

---

### 6. Data Pipeline

**How queries are fed:** The script loads the same pre-computed QA prediction files that were generated in Phases 2 and 4 of the main pipeline. Specifically, `load_strategy_predictions()` ([lines 68–80](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L68-L80)) reads `predictions/test/nor_qa_{model}_{dataset}____prompt_set_1/prediction__*.json` and the corresponding `oner_qa` files for all 6 datasets. Each file is a `dict[qid → answer_string]`.

**Cross-model consistency:** The script's `model_name` argument accepts `flan_t5_xl`, `flan_t5_xxl`, and `gpt` ([line 109](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L109)). BM25 retrieval counts are model-specific ([lines 32–33](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L32-L33)):

| Model | ONER_BM25 | IRCOT_BM25 |
|---|---|---|
| flan_t5_xl | 15 | 6 |
| flan_t5_xxl | 15 | 6 |
| gpt | 6 | 3 |

The pipeline is identical across all 3 backbones — the only difference is the prediction file paths and BM25 counts, which correctly reflect the original experimental configuration. All 3 models are evaluable.

---

### 7. Gate 2 — Completely Unchanged

Gate 2 is **consumed as-is** from a previously trained Iteration 1 checkpoint. The script takes `--clf2_pred_file` ([line 111](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L111)), which points to a `dict_id_pred_results.json` produced by a standard Clf2 training run (e.g., `single_vs_multi/epoch/35/.../predict/dict_id_pred_results.json`).

The Clf2 predictions are loaded at predict_complexity_agreement.py and used only for disagree cases at predict_complexity_agreement.py:
```python
clf2[qid]["prediction"]  # B or C
```

No retraining, fine-tuning, or modification of Gate 2 occurs. Same weights, same training procedure, same checkpoint as Iteration 1.

---

### 8. Evaluation Setup

**Identical to Iteration 1.** The script writes routed predictions in exactly the same format — `{dataset}/{dataset}.json` and `{dataset}/{dataset}_option.json` — ([lines 198–199](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L198-L199)), and the user runs `evaluate_final_acc.py --pred_path {output_path}` ([line 244](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L244)).

- Same 6 datasets: musique, hotpotqa, 2wikimultihopqa, nq, trivia, squad ([line 29](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L29))
- Same ground truth: `processed_data/{dataset}/test_subsampled.jsonl` ([evaluate_final_acc.py L103](Adaptive-RAG/evaluate_final_acc.py#L103))
- Same EM/F1: `SquadAnswerEmF1Metric` ([evaluate_final_acc.py L101](Adaptive-RAG/evaluate_final_acc.py#L101))
- All 3 backbones are evaluable (the `choices` tuple includes all three)

---

### 9. Output Artifacts

Predictions are written to the `--output_path` directory, which is user-specified. The typical convention is:

```
predictions/classifier/t5-large/{model}/split_agreement/nor_oner_clf2ep{N}/
├── {dataset}/
│   ├── {dataset}.json              (routed QA predictions)
│   └── {dataset}_option.json       (predictions with routing labels + step counts)
└── routing_stats.json              (agreement rate, A/B/C counts, per-dataset stats)
```

This is in the `split_agreement/` subdirectory, **separate from**:
- Iteration 1: `split/`
- Iteration 2 (undersampled): `no_ret_vs_ret_undersampled/` (in outputs)
- Iterations 3–4 (weighted CE / focal): `no_ret_vs_ret_focal/`, `no_ret_vs_ret_weighted_ce/` (in outputs)

No risk of overwriting any prior iteration's results.

---

### 10. Suspicious / Noteworthy Items

1. **Not true uncertainty estimation — it's answer agreement.** The name "UE Answer Agreement" in the documentation is somewhat misleading. Classic UE methods involve multiple stochastic forward passes (temperature sampling, MC dropout). Here, both `nor_qa` and `oner_qa` predictions were generated with **`do_sample=False`** ([llm_client_generator.py L17](Adaptive-RAG/commaqa/models/llm_client_generator.py#L17)), so they are deterministic greedy outputs. The "uncertainty signal" is not from sampling variance but from **whether retrieval changes the LLM's answer**. This is semantically meaningful but is a different kind of signal than sampling-based UE.

2. **`output_scores=False` confirmed.** At serve.py, scores are explicitly disabled (`output_scores=False`), confirming that no token-level probabilities are available. The agreement method is the only UE proxy possible with these cached predictions.

3. **No temperature/sampling pathology.** Since the method doesn't rely on sampling, there is no risk of temperature=0 collapsing the signal. The two predictions being compared come from genuinely different inference pipelines (no retrieval vs. one-step retrieval), so the answers can and do diverge.

4. **Binary threshold with no soft margin.** The exact-match-after-normalization check means a near-match (e.g., "New York City" vs "New York") would be counted as disagreement and routed to retrieval. This is conservative (favors retrieval), which is safe given the finding that retrieval rarely hurts.

5. **GPT model has different BM25 counts** (6 vs 15 for oner). This is consistent with the original experimental configuration, not a bug, but it means the oner_qa answers for GPT are based on fewer retrieved passages, which could affect agreement rates.

6. **`routing_stats.json` saved** ([lines 206–221](Adaptive-RAG/classifier/postprocess/predict_complexity_agreement.py#L206-L221)): the script self-documents its decisions with agreement rate, A/B/C distribution, and per-dataset breakdowns — good for reproducibility.

7. **No issues that could cause premature termination or silent failure.** The script has no `try/except` blocks that swallow errors, no timeouts, and no conditional exits.