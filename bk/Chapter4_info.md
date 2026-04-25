# Message 1 — Training hyperparameters:

Here's the complete hyperparameter breakdown for T5-Large fine-tuning as Clf1 and Clf2, from run_classifier.py and the shell scripts in `classifier/run/`:

## Shared Hyperparameters (Clf1 & Clf2 are identical)

| Parameter | Value | CLI Arg |
|---|---|---|
| **Epochs** | Sweep over `[15, 20, 25, 30, 35]` | `--num_train_epochs` |
| **Learning Rate** | `3e-5` | `--learning_rate` |
| **Train Batch Size** | `32` (per device) | `--per_device_train_batch_size` |
| **Eval Batch Size** | `100` (per device) | `--per_device_eval_batch_size` |
| **Warmup Steps** | `0` | `--num_warmup_steps` |
| **Weight Decay** | `0.0` | `--weight_decay` |
| **Optimizer** | AdamW (hardcoded, L529) | — |
| **LR Scheduler** | Linear (warmup + decay) | `--lr_scheduler_type` |
| **Max Sequence Length** | `384` | `--max_seq_length` |
| **Doc Stride** | `128` | `--doc_stride` |
| **Gradient Accumulation** | `1` | `--gradient_accumulation_steps` |
| **Seed** | Not set (non-deterministic) | `--seed` |

## Checkpoint Selection

No automated early stopping. The model is **saved after each epoch**, and the best epoch is selected **post-hoc** by evaluating all checkpoints on the validation set:

| Classifier | flan_t5_xl | flan_t5_xxl |
|---|---|---|
| **Clf1** (labels `A R`) | Epoch 20 → 73.19% | Epoch 35 → 66.86% |
| **Clf2** (labels `B C`) | Epoch 35 → 71.35% | Epoch 30 → 68.08% |

Results documented in TRAINING_RESULTS-Flan-T5-Xl.md and TRAINING_RESULTS-Flan-T5-Xxl.md.

## Clf1 vs Clf2 Differences

Only the **labels and training data** differ — all hyperparameters are identical:

| | **Clf1** (no-retrieval vs retrieval) | **Clf2** (single vs multi-step) |
|---|---|---|
| **Labels** | `--labels A R` | `--labels B C` |
| **Train data** | `silver/no_retrieval_vs_retrieval/train.json` | `binary_silver_single_vs_multi/train.json` |

## Key Scripts

- **Training script**: classifier/run_classifier.py (all argparse defs + AdamW at L523–537)
- **Clf1 shells**: `run/run_large_train_xl.sh`, `run/run_large_train_xxl.sh`, `run/run_large_train_xl_no_ret_vs_ret.sh`
- **Clf2 shells**: `run/run_large_train_xl_single_vs_multi.sh`, `run/run_large_train_xxl_single_vs_multi.sh`
- **Preprocessing/tokenization**: classifier/utils.py (L91–107)

------------------
# Message 2 — Label construction:

## Clf1 (A vs R) — Training Set Construction

**Clf1 uses silver labels only** — no binary inductive-bias labels are merged.

The script preprocess_silver_train.py calls `label_complexity()` (preprocess_utils.py) on the dev_500 split of each dataset. For each question, it checks which QA strategy (nor/oner/ircot) answered correctly, assigning the **simplest succeeding** label:

| Strategy succeeded | Label |
|---|---|
| Zero-shot (nor_qa) correct | A |
| Single-step (oner_qa) correct | B |
| Multi-step (ircot_qa) correct | C |

If multiple strategies work, the **last assignment wins** (A overwrites C/B), so the label reflects the *simplest* effective strategy. The 3-class silver labels (A/B/C) are then **relabelled** A→A, B→R, C→R for the binary Clf1 file at `silver/no_retrieval_vs_retrieval/train.json`.

The shell scripts (e.g. run_large_train_xl_no_ret_vs_ret.sh) confirm:
```
--train_file ./data/.../silver/no_retrieval_vs_retrieval/train.json
--labels A R
```

## Clf2 (B vs C) — Training Set Construction

Clf2 **does** merge binary + silver. The pipeline is:

1. **Binary inductive-bias labels** (preprocess_binary_train.py): 400 questions × 6 datasets = **2,400 samples** (B=1200, C=1200). Multi-hop datasets (MuSiQue, HotpotQA, 2WikiMHQA) → C; single-hop (NQ, TriviaQA, SQuAD) → B. These are **shared across all models** (pure dataset-level heuristic, not LLM-dependent).

2. **Merge** (concat_binary_silver_train.py): binary (2,400) + model-specific silver → `binary_silver/train.json`. Overlap is checked but **turns out to be zero** (binary uses dev_train IDs, silver uses dev_500 IDs), so it's a straight concatenation.

3. **Filter to B+C**: `binary_silver_single_vs_multi/train.json` = `binary_silver/train.json` with all A-labelled samples removed. Verified empirically:

| Model | binary_silver | minus A | = binary_silver_single_vs_multi |
|---|---|---|---|
| flan_t5_xl | 3,692 | −424 A | **3,268** (B=1,871 / C=1,397) |
| flan_t5_xxl | 3,809 | −511 A | **3,298** (B=1,903 / C=1,395) |
| gpt | 3,817 | −1,013 A | **2,804** (B=1,475 / C=1,329) |

The shell scripts confirm: `--train_file ./data/.../binary_silver_single_vs_multi/train.json --labels B C`

## All Split Sizes

| | **flan_t5_xl** | **flan_t5_xxl** | **gpt** |
|---|---|---|---|
| **Clf1 train** (silver-only, A/R) | 1,292 (424A / 868R) | 1,409 (511A / 898R) | 1,417 (1013A / 404R) |
| **Clf1 valid** | 1,350 (439A / 911R) | 1,415 (519A / 896R) | 1,431 (1038A / 393R) |
| **Clf2 train** (binary+silver, B/C) | 3,268 (1871B / 1397C) | 3,298 (1903B / 1395C) | 2,804 (1475B / 1329C) |
| **Clf2 valid** (silver-only, B/C) | 911 (691B / 220C) | 896 (701B / 195C) | 393 (272B / 121C) |
| **Test** (shared, unlabelled) | 3,000 | 3,000 | 3,000 |

## Are Labels LLM-Specific?

**Silver labels: yes, fully LLM-specific.** Different backbone LLMs answer different questions correctly, producing different A/B/C assignments. GPT's distribution is dramatically different (71% A vs ~33% for Flan-T5) because GPT solves more questions without retrieval.

**Binary inductive-bias labels: no, shared.** The same 2,400 samples (from `binary/total_data_train.json`) are used for all three models. These are pure dataset-level heuristics (multi-hop dataset → C, single-hop → B) independent of any LLM.

**Merged Clf2 training set: partially LLM-specific.** The binary portion (2,400) is shared; the silver portion (~400–900 B/C samples depending on model) varies. The final Clf2 training size differs across models because the A-count removed differs.






------------------
# Message 3 — Oracle ceiling and inference pipeline:


## predict_complexity_oracle_ceiling.py

This script measures the **maximum F1 achievable** if Clf2 were perfect, keeping the agreement gate fixed. It constructs the oracle label per question as follows:

**For agree questions** (nor_qa answer == oner_qa answer after normalization): route **A** — same as the real agreement gate.

**For disagree (R-routed) questions**, it uses the **silver test-set labels** from `silver/valid.json` to pick the cheapest correct retrieval strategy (predict_complexity_oracle_ceiling.py):

| Condition | Oracle label |
|---|---|
| `"one"` in `total_answer` → oner_qa got it right | **B** (cheaper) |
| `"multiple"` in `total_answer` → only ircot_qa got it right | **C** |
| Neither retrieval strategy worked (or question missing from silver) | **B** (default — answer will be wrong regardless) |

The priority is **cheapest-first**: B is preferred over C when both work, because single-step retrieval is cheaper. The default-B fallback on "no correct retrieval" is a no-op for F1 — neither B nor C would produce the right answer.

After constructing the oracle A/B/C label for all 3,000 test questions, the script loads the corresponding **pre-computed QA predictions** from `predictions/test/` and writes per-dataset output files to `predictions/classifier/t5-large/{model}/split_agreement_oracle/`.

---

## How `predict.json` is Produced

preprocess_predict.py builds the 3,000-question test file. It reads the 6 `test_subsampled.jsonl` files from `processed_data/{dataset}/`, extracts `question_id` and `question_text` via `prepare_predict_file()`, concatenates all 6 datasets (500 each), and writes `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json`. This file is **shared across all models** — it contains no labels, just question IDs, text, and dataset names.

---

## How nor_qa / oner_qa / ircot_qa Predictions Are Retrieved

**From disk, not a live LLM server.** All postprocess scripts (predict_complexity_agreement.py, predict_complexity_oracle_ceiling.py, predict_complexity_split_classifiers.py) load QA answers directly from pre-computed JSON files on disk:

```
predictions/test/nor_qa_{model}_{dataset}____prompt_set_1/prediction__{ds}_to_{ds}__test_subsampled.json
predictions/test/oner_qa_{model}_{dataset}____...___bm25_retrieval_count__{N}___distractor_count__1/prediction__...json
predictions/test/ircot_qa_{model}_{dataset}____...___bm25_retrieval_count__{N}___distractor_count__1/prediction__...json
```

These were generated once upstream by `run_retrieval_test.sh` (which requires the live LLM + retriever servers). The routing/evaluation pipeline is entirely offline — it just reads from disk and selects which pre-computed answer to use per question based on the classifier's A/B/C prediction. BM25 retrieval counts differ per model family: Flan-T5 uses oner=15/ircot=6, GPT uses oner=6/ircot=3 (lookup tables at predict_complexity_oracle_ceiling.py).





-----------------
# Message 4 — Feasibility probe and embedding probe:

## Feature Probe: clf2_feature_probe.py

**Script:** classifier/postprocess/clf2_feature_probe.py

**Three features:**

| Feature | Computation |
|---|---|
| `token_len` | `len(doc.text.split())` — whitespace-split word count (not spaCy tokens) |
| `entity_count` | `len(doc.ents)` — named entity count via **spaCy `en_core_web_sm`**, loaded with `disable=["parser", "lemmatizer"]` (only NER + tok kept) |
| `bridge_flag` | 0/1 from a compiled regex union of 7 patterns (clf2_feature_probe.py) |

**Bridge regex patterns** (case-insensitive):
1. Relative-clause bridges: `\b(?:who|where|which|that)\s+(?:was|were|is|are|did|had|has|does)\b`
2. Double possessive: `\w+'s\s+\w+(?:\s+\w+){0,5}\s+\w+'s`
3. Temporal subordination before a wh-word: `\b(?:before|after|when|while)\b.{3,60}\b(?:who|what|where|which)\b`
4. Demonstrative back-reference: `\b(?:that|this|those|these)\s+(?:country|city|person|team|...)\b`
5. Explicit "both…and": `\b(?:both)\b.{1,40}\band\b`
6. "between…and": `\bbetween\b.{1,40}\band\b`
7. Nested wh-question: `\bof\s+the\s+\w+\s+(?:who|that|which|where)\b`

**Train/test split:** `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)` — 80/20 stratified from the `binary_silver_single_vs_multi/train.json` data (all B/C items). No separate held-out split; it's a feasibility probe, not a production model.

**Classifier:** `LogisticRegression(max_iter=1000, random_state=42)` with target `y = 1` for C and `0` for B. Reports AUC, accuracy, per-feature coefficients, and a go/no-go verdict at **AUC ≥ 0.65** threshold.

---

## Embedding Probe: clf2_embedding_probe.py

**Script:** classifier/future_work/clf2_embedding_probe.py

**Encoder:** `all-MiniLM-L6-v2` (384-dim) via `sentence-transformers`, with `paraphrase-MiniLM-L3-v2` as a fallback. Encoding: `encoder.encode(questions, batch_size=64, normalize_embeddings=False)` — raw (unnormalized) embeddings.

**Same train/test split:** 80/20 stratified, `random_state=42`, same `binary_silver_single_vs_multi/train.json` data filtered to B/C.

**Classifier on top:** `LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)` on the raw 384-dim embeddings. Same binary target (C=1, B=0).

**Outputs:** AUC, accuracy, classification report, a **PCA scatter plot** (2 components), and a comparison table against the surface-feature probe AUC (0.676). This probe uses a higher go/no-go threshold of **AUC ≥ 0.75**. Result was AUC ≈ 0.818 — well above threshold, but the downstream MLP classifier built from these embeddings (clf2_embedding_clf.py) caused a small F1 regression, hence "future work."