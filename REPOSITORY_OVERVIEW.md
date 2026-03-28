# Adaptive-RAG Repository Overview

## What This Project Does

Adaptive-RAG is an **adaptive retrieval-augmented generation** system for question answering. Instead of using a single retrieval strategy for all questions, it trains a **query complexity classifier** that routes each incoming question to the most appropriate retrieval strategy:

| Class | Label | Strategy | System Name | Description |
|-------|-------|----------|-------------|-------------|
| A | No retrieval | Zero-step | `nor_qa` | LLM answers directly from parametric knowledge |
| B | Single retrieval | One-step | `oner_qa` | One round of BM25 retrieval, then QA |
| C | Multi-step retrieval | Multi-step | `ircot_qa` | Iterative Retrieval with Chain-of-Thought (IRCoT) |

The core idea: simple factoid questions don't need retrieval; single-hop questions need one retrieval step; complex multi-hop questions need iterative retrieval and reasoning.

---

## Repository Structure

```
Adaptive-RAG/
├── run.py                    # Main orchestrator: instantiates configs, runs experiments
├── runner.py                 # CLI wrapper around run.py
├── predict.py                # Runs QA inference using a config
├── evaluate.py               # Computes EM/F1 metrics on predictions
├── evaluate_final_acc.py     # Evaluates final accuracy after classifier routing
├── lib.py                    # Shared utilities (JSON I/O, server addresses, etc.)
│
├── classifier/               # ** Query complexity classifier (t5-large) **
│   ├── run_classifier.py     #    Training/eval script (seq2seq fine-tuning)
│   ├── utils.py              #    Model loading, preprocessing, metric helpers
│   ├── preprocess/           #    Scripts to create training/validation/test labels
│   ├── postprocess/          #    Routes QA answers based on classifier predictions
│   ├── run/                  #    Shell scripts to launch training
│   ├── data/                 #    Pre-computed label data (see Data Splits below)
│   └── outputs/              #    Trained checkpoints + prediction outputs
│
├── commaqa/                  # Core inference engine (state machine architecture)
│   ├── inference/            #    configurable_inference.py, ircot.py, participant_execution.py
│   ├── models/               #    gpt3generator.py, llm_client_generator.py
│   └── configs/              #    Default config templates
│
├── retriever_server/         # FastAPI server wrapping Elasticsearch BM25 retrieval
│   ├── serve.py              #    REST API (/retrieve/ endpoint, port 8000)
│   ├── elasticsearch_retriever.py   # BM25 search implementation
│   └── build_index.py        #    Index builder for each corpus
│
├── llm_server/               # FastAPI server for LLM inference
│   ├── serve.py              #    REST API (/generate/ endpoint, e.g. port 8010)
│   └── client.py             #    Client wrapper for inference requests
│
├── base_configs/             # Jsonnet config templates (one per system×model×dataset)
├── instantiated_configs/     # Generated configs with specific hyperparameters
├── prompts/                  # Generated few-shot prompt files
├── prompt_generator/         # Generates prompt files from annotated demonstrations
├── processing_scripts/       # Raw dataset → processed JSONL converters
├── metrics/                  # EM/F1 metric implementations (DROP, SQuAD, support)
├── official_evaluation/      # Official eval scripts for HotpotQA, 2Wiki, MuSiQue
│
├── processed_data/           # Processed datasets (JSONL format)
│   └── {dataset}/
│       ├── train.jsonl
│       ├── dev.jsonl
│       ├── dev_500_subsampled.jsonl
│       └── test_subsampled.jsonl
│
├── raw_data/                 # Original dataset files before processing
├── predictions/              # All QA predictions
│   ├── dev_500/              #    Predictions on dev_500 split (used for silver train labels)
│   ├── test/                 #    Predictions on test split (used for silver valid labels + final eval)
│   └── classifier/           #    Routed predictions after classifier
│
├── run_retrieval_dev.sh      # Runs write→predict→evaluate on dev_500
├── run_retrieval_test.sh     # Runs write→predict→evaluate on test
└── requirements.txt          # Python dependencies
```

---

## Datasets

Six QA datasets are used, split into two categories:

**Multi-hop** (require reasoning across multiple documents):
- **HotpotQA** — Multi-hop questions over Wikipedia
- **2WikiMultiHopQA** — Multi-hop questions from two Wikipedia articles  
- **MuSiQue** — Multi-step compositional questions

**Single-hop** (answerable from a single passage):
- **Natural Questions (NQ)** — Google search questions with Wikipedia answers
- **TriviaQA** — Trivia questions with evidence documents
- **SQuAD** — Stanford reading comprehension on Wikipedia paragraphs

Each dataset is subsampled to **500 questions** for the dev and test splits used in experiments.

---

## End-to-End Pipeline

The system operates in 8 phases:

### Phase 1: Data Preparation
```
raw_data/{dataset}/*.json
    → processing_scripts/process_{dataset}.py
    → processed_data/{dataset}/train.jsonl, dev.jsonl
    → processing_scripts/subsample_dataset_and_remap_paras.py
    → processed_data/{dataset}/dev_500_subsampled.jsonl, test_subsampled.jsonl
```

### Phase 2: Run All Three QA Strategies on dev_500
```
For each of {nor_qa, oner_qa, ircot_qa} × {flan_t5_xl, flan_t5_xxl, gpt} × {6 datasets}:
    ./run_retrieval_dev.sh $SYSTEM $MODEL $DATASET $PORT
    → predictions/dev_500/{system}_{model}_{dataset}/prediction__*.json
```

### Phase 3: Generate Silver Training Labels (from dev_500 predictions)
```
classifier/preprocess/preprocess_silver_train.py {model_name}
    → classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/silver/train.json
```

### Phase 4: Run All Three QA Strategies on test
```
For each strategy × model × dataset:
    ./run_retrieval_test.sh $SYSTEM $MODEL $DATASET $PORT
    → predictions/test/{system}_{model}_{dataset}/prediction__*.json
```

### Phase 5: Generate Silver Validation Labels (from test predictions)
```
classifier/preprocess/preprocess_silver_valid.py {model_name}
    → classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/silver/valid.json
```

### Phase 6: Create Binary Labels + Merge
```
classifier/preprocess/preprocess_binary_train.py
    → classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/binary/total_data_train.json

classifier/preprocess/concat_binary_silver_train.py
    → classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/binary_silver/train.json
```

### Phase 7: Train the Classifier

Two approaches exist:

**Original (single 3-class classifier)**:
```
classifier/run/run_large_train_{xl,xxl,gpt}.sh
    → classifier/outputs/.../pytorch_model.bin
```

**Current (split/cascaded — two binary classifiers)**:
```
Train Classifier 1 (A vs R) on silver/no_retrieval_vs_retrieval/train.json
Train Classifier 2 (B vs C) on binary_silver_single_vs_multi/train.json
    → classifier/outputs/.../no_ret_vs_ret/epoch/{N}/...
    → classifier/outputs/.../single_vs_multi/epoch/{N}/...
```

### Phase 8: Classify Test Questions + Route + Evaluate
```
classifier/postprocess/predict_complexity_on_classification_results.py
    → For each question: pick the answer from the strategy the classifier chose
    → (Split version: Clf1 decides A vs R; if R, Clf2 decides B vs C)
evaluate_final_acc.py
    → Final EM/F1 scores
```

---

## Classifier Data Splits: Training vs. Validation vs. Testing

This is the most important section. The classifier has three distinct data roles, each derived differently.

### Overview Table

| Role | File | Source | Size | Labels |
|------|------|--------|------|--------|
| **Training** | `classifier/data/.../{model}/binary_silver/train.json` | Silver (dev_500) + Binary (raw train) | ~3,700 | A, B, C |
| **Validation** | `classifier/data/.../{model}/silver/valid.json` | Silver labels from test predictions | ~1,350 | A, B, C |
| **Test (predict)** | `classifier/data/.../predict.json` | test_subsampled questions (unlabeled) | 3,000 | None |

### Training Data: `binary_silver/train.json`

The training set is a **merge of two label sources**: silver labels and binary labels. Silver labels take priority — if a question appears in both, the silver label is kept and the binary version is removed.

#### Silver Labels (`silver/train.json`, ~1,300 samples)

- **Source questions**: `processed_data/{dataset}/dev_500_subsampled.jsonl` — 500 questions per dataset from the **dev split**
- **Source predictions**: `predictions/dev_500/` — predictions from all three strategies (nor_qa, oner_qa, ircot_qa) on those dev_500 questions
- **Labeling logic** (in `preprocess_silver_train.py` → `label_complexity()`):
  - For each question, check which strategy's classification file marks it as correctly answered
  - If `nor_qa` (no retrieval) got it right → label **A**
  - Else if `oner_qa` (single retrieval) got it right → label **B**  
  - Else if `ircot_qa` (multi-step retrieval) got it right → label **C**
  - The priority order is A > B > C (prefer simpler strategies when multiple are correct)
- **Model-specific**: Different for flan_t5_xl, flan_t5_xxl, and gpt because different LLMs get different questions right
- **Not all 3,000 questions get labels**: Only questions where at least one strategy succeeded are included

#### Binary Labels (`binary/total_data_train.json`, 2,400 samples)

- **Source questions**: `raw_data/{dataset}/train.json` — questions from the **original full training sets**
- **Labeling logic** (in `preprocess_binary_train.py`):
  - Multi-hop datasets (MuSiQue, HotpotQA, 2WikiMultiHopQA) → **C** (multi-step)
  - Single-hop datasets (NQ, TriviaQA, SQuAD) → **B** (single-step)
  - **Never assigns A** — only silver labels produce A labels
- **400 samples per dataset** (6 datasets × 400 = 2,400 total), taken as the first 400 from each
- **Model-independent**: Same binary labels used for all three model variants
- **NQ deduplication**: Questions overlapping with MuSiQue's single-hop sub-questions are removed

#### Merge Logic (`concat_binary_silver_train.py`)

```python
# Silver labels take priority over binary labels
silver_ids = {item['id'] for item in silver_data}
binary_filtered = [item for item in binary_data if item['id'] not in silver_ids]
training_data = binary_filtered + silver_data
```

Typical resulting distribution (for flan_t5_xl):
- Class A: ~424 (all from silver)
- Class B: ~1,871 (silver + binary)
- Class C: ~1,397 (silver + binary)
- **Total: ~3,692**

### Validation Data: `silver/valid.json`

- **Source questions**: `processed_data/{dataset}/test_subsampled.jsonl` — 500 questions per dataset from the **test split**
- **Source predictions**: `predictions/test/` — predictions from all three strategies on those test questions
- **Labeling logic**: Identical to silver training labels (same `label_complexity()` function)
- **Model-specific**: Different labels per LLM variant
- **Size**: ~1,350 samples (not all 3,000 get labels; only questions answered correctly by ≥1 strategy)
- **Used for**: Selecting the best training epoch/checkpoint during classifier training

### Test Data (Prediction): `predict.json`

- **Source questions**: `processed_data/{dataset}/test_subsampled.jsonl` — **same 500 questions per dataset** as validation
- **No labels**: The `answer` field is empty
- **Size**: 3,000 (500 × 6 datasets, all included regardless of correctness)
- **Used for**: Running the trained classifier to get A/B/C predictions, which then route to the appropriate QA strategy

### Critical Observations About Data Splits

1. **Validation and test use the same questions.** The silver validation labels (`silver/valid.json`) and the unlabeled prediction file (`predict.json`) are both derived from `test_subsampled.jsonl`. The validation set is a labeled subset (~1,350 of the 3,000), while the prediction set is the full 3,000 without labels. This means **the classifier's validation set overlaps with its test set**.

2. **Training uses dev, validation/test use test.** Silver training labels come from `dev_500_subsampled.jsonl` predictions. Silver validation labels and predictions come from `test_subsampled.jsonl`.

3. **Silver labels are model-specific.** A question labeled A for flan_t5_xl might be labeled B or C for flan_t5_xxl, because different LLMs have different parametric knowledge. This means separate classifiers are trained per LLM.

4. **Binary labels add data volume but no A labels.** The binary labels increase training set size (~2,400 additional samples) but only contribute B and C labels. All A labels must come from silver labels.

5. **Label priority favors simpler strategies.** In `label_complexity()`, labels are assigned in order C → B → A, with each overwriting the previous. Since A is checked last, if no-retrieval works for a question it gets A even if other strategies also work.

### Data Flow Diagram

```
                         TRAINING DATA
                         ═════════════
  processed_data/{dataset}/dev_500_subsampled.jsonl (500 × 6 = 3,000 questions)
           │
           ├─── nor_qa predictions  ──┐
           ├─── oner_qa predictions ──┼── preprocess_silver_train.py
           └─── ircot_qa predictions ─┘          │
                                          silver/train.json (~1,300)
                                                 │
  raw_data/{dataset}/train.json                  │
           │                                     │
           └── preprocess_binary_train.py        │
                      │                          │
               binary/total_data_train.json      │
               (400/dataset × 6 = 2,400)         │
                      │                          │
                      └────── concat ────────────┘
                                │
                      binary_silver/train.json (~3,700)
                      ═══════════════════════════════


                       VALIDATION DATA
                       ═══════════════
  processed_data/{dataset}/test_subsampled.jsonl (500 × 6 = 3,000 questions)
           │
           ├─── nor_qa predictions  ──┐
           ├─── oner_qa predictions ──┼── preprocess_silver_valid.py
           └─── ircot_qa predictions ─┘          │
                                          silver/valid.json (~1,350)
                                     (used for checkpoint selection)


                          TEST DATA
                          ═════════
  processed_data/{dataset}/test_subsampled.jsonl (same 3,000 questions)
           │
           └── preprocess_predict.py
                      │
               predict.json (3,000, unlabeled)
               (classifier predicts A/B/C → routes to QA strategy answers)
```

---

## Inference Architecture (CommaQA)

The inference engine in `commaqa/` implements a **state machine** where each state is a "participant model" that produces an output and specifies the next state.

### IRCoT (Multi-step) — `ircot_qa`
```
[BM25 Retriever] → retrieve top-k paragraphs for query
        ↓
[CoT Generator] → generate one reasoning sentence given context
        ↓
[Exit Controller] → can we answer now?
        ↓ No           ↓ Yes
  (loop back to      [Answer Extractor] → extract final answer
   Retriever with
   new query from
   generated sentence)
```

### One-step — `oner_qa`
```
[BM25 Retriever] → retrieve top-k paragraphs (k=15)
        ↓
[QA Model] → answer directly from retrieved context
```

### No retrieval — `nor_qa`
```
[QA Model] → answer directly from parametric knowledge (no retrieval)
```

### Configuration

Each experiment is defined by a Jsonnet config file in `base_configs/` that specifies:
- The state machine graph (start state, transitions, end state)
- The retrieval parameters (corpus name, BM25 count, max paragraphs)
- The LLM parameters (model name, prompt file, generation settings)
- The evaluation parameters (prediction type, metrics)

Configs are **parameterized** via Jsonnet local variables and external variables:
- `RETRIEVER_HOST`, `RETRIEVER_PORT` — Elasticsearch connection
- `LLM_SERVER_ADDRESS` — LLM server URL
- `bm25_retrieval_count`, `distractor_count`, `rc_context_type` — hyperparameters

---

## Infrastructure Services

### Elasticsearch (Retriever)
- Version 7.10.2, runs on port 9200
- Separate index per corpus: `hotpotqa`, `2wikimultihopqa`, `musique`, `wiki` (shared by NQ/TriviaQA/SQuAD)
- Wrapped by FastAPI server in `retriever_server/serve.py` (port 8000)

### LLM Server
- FastAPI server in `llm_server/serve.py` (configurable port, typically 8010)
- Loads model specified by `MODEL_NAME` environment variable
- Supports Flan-T5 family (base through xxl), GPT-J, OPT, GPT-NeoX
- Uses `diskcache` for prompt caching to avoid redundant LLM calls
- For GPT-3.5: uses OpenAI API directly via `commaqa/models/gpt3generator.py`

---

## Evaluation

### QA Metrics (`evaluate.py`)
- **EM (Exact Match)**: After normalization (lowercase, remove articles/punctuation/whitespace)
- **F1**: Token-level precision/recall
- **Support EM/F1**: For multi-hop datasets, evaluates supporting fact identification
- **Answer Support Recall**: Whether retrieved paragraphs contain the answer

### Official Evaluation (`official_evaluation/`)
Dataset-specific official scripts for HotpotQA, 2WikiMultiHopQA, and MuSiQue with stricter metrics.

### Final Classifier Evaluation (`evaluate_final_acc.py`)
After classifier routing, evaluates how the routed answers (from the strategy the classifier chose) compare to ground truth, with per-class (A/B/C) accuracy breakdowns.

---

## Key Entry Points for Running Experiments

```bash
# 1. Start Elasticsearch
./elasticsearch-7.10.2/bin/elasticsearch -d -p es.pid

# 2. Start retriever server
uvicorn serve:app --port 8000 --app-dir retriever_server

# 3. Start LLM server
MODEL_NAME=flan-t5-xl uvicorn serve:app --port 8010 --app-dir llm_server

# 4. Run a strategy on dev_500
./run_retrieval_dev.sh ircot_qa flan-t5-xl hotpotqa 8010

# 5. Run a strategy on test
./run_retrieval_test.sh ircot_qa flan-t5-xl hotpotqa 8010

# 6. Generate classifier labels
cd classifier/preprocess
python preprocess_silver_train.py flan_t5_xl
python preprocess_silver_valid.py flan_t5_xl
python preprocess_binary_train.py
python concat_binary_silver_train.py

# 7. Train classifier
cd classifier && bash run/run_large_train_xl.sh

# 8. Route test predictions
cd classifier && python postprocess/predict_complexity_on_classification_results.py flan_t5_xl

# 9. Evaluate final accuracy
python evaluate_final_acc.py
```

---

## Classifier Model Details

- **Architecture**: T5-Large (seq2seq), fine-tuned as a classification task
- **Input**: Question text, tokenized up to 384 tokens, with prefix for T5
- **Loss**: Cross-entropy over the label vocabulary
- **Training**: Learning rate 3e-5, batch size 32, tested at epochs 15/20/25/30/35
- **Checkpoint selection**: Best epoch chosen based on accuracy on validation set
- **One classifier per LLM**: Separate classifiers trained for flan_t5_xl, flan_t5_xxl, and gpt because silver labels differ per model

### Approach 1: Single 3-Class Classifier (Original)

- **Output**: Single token — "A", "B", or "C"
- **Training data**: `binary_silver/train.json` (~3,700 samples)
- **Validation data**: `silver/valid.json` (~1,350 samples)
- **Training scripts**: `classifier/run/run_large_train_{xl,xxl,gpt}.sh`

### Approach 2: Split (Cascaded) Classifier (Current)

Instead of one 3-class classifier, two binary classifiers run in cascade:

```
Question → [Classifier 1: A vs R?]
                 │             │
              A (no retrieval)  R (retrieval needed)
                               │
                        [Classifier 2: B vs C?]
                              │          │
                       B (single-step)  C (multi-step)
```

**Classifier 1 — No-retrieval (A) vs Retrieval (R)**:
- Decides whether the question needs any retrieval at all
- Labels: `"A"` (no retrieval) vs `"R"` (retrieval needed, merging original B+C)
- Training data: `silver/no_retrieval_vs_retrieval/train.json`
- Validation data: `silver/no_retrieval_vs_retrieval/valid.json`

**Classifier 2 — Single (B) vs Multi (C)**:
- Only runs on questions classified as R by Classifier 1
- Labels: `"B"` (single-step) vs `"C"` (multi-step)
- Training data: `binary_silver_single_vs_multi/train.json` (silver + binary inductive-bias labels merged)
- Validation data: `silver/single_vs_multi/valid.json`
- Derived from original data by dropping all A-labelled samples

#### Split Classifier Label Files

```
classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/{model}/silver/
├── train.json                          # Original 3-class (A/B/C)
├── valid.json                          # Original 3-class (A/B/C)
├── no_retrieval_vs_retrieval/          # Classifier 1 labels
│   ├── train.json
│   └── valid.json
└── single_vs_multi/                    # Classifier 2 labels
    ├── train.json
    └── valid.json
```

#### Split Classifier Sample Counts

| Model | Split | Clf1: no_ret_vs_ret (A / R) | Clf2: single_vs_multi (B / C) |
|---|---|---|---|
| flan_t5_xl | train | 1,292 (424 / 868) | 868 (671 / 197) |
| flan_t5_xl | valid | 1,350 (439 / 911) | 911 (691 / 220) |
| flan_t5_xxl | train | 1,409 (511 / 898) | 898 (703 / 195) |
| flan_t5_xxl | valid | 1,415 (519 / 896) | 896 (701 / 195) |
| gpt | train | 1,417 (1,013 / 404) | 404 (275 / 129) |
| gpt | valid | 1,431 (1,038 / 393) | 393 (272 / 121) |

Note: Classifier 2 training uses merged `binary_silver_single_vs_multi/train.json` (~3,268–3,298 samples) which adds binary inductive-bias labels to improve class balance. Validation uses silver-only labels.

---

## Training Results

### flan_t5_xl

**Classifier 1 (A vs R)** — Best: epoch 20 (73.19% accuracy)

| Epoch | Overall Acc | A acc | R acc |
|---|---|---|---|
| 15 | 72.37% | 41.69% | 87.16% |
| **20** | **73.19%** | 42.60% | 87.93% |
| 25 | 72.67% | 47.15% | 84.96% |
| 30 | 72.22% | 48.75% | 83.53% |
| 35 | 72.22% | 50.57% | 82.66% |

**Classifier 2 (B vs C)** — Best: epoch 35 (71.35% accuracy)

| Epoch | Overall Acc | B acc | C acc |
|---|---|---|---|
| 15 | 65.86% | 64.40% | 70.45% |
| 20 | 67.62% | 66.86% | 70.00% |
| 25 | 68.61% | 68.31% | 69.55% |
| 30 | 70.91% | 70.62% | 71.82% |
| **35** | **71.35%** | 71.92% | 69.55% |

**Cascaded 3-class** (clf1 ep20 + clf2 ep35): **71.58% overall accuracy**

| | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.630 | 0.426 | 0.508 |
| B | 0.878 | 0.657 | 0.752 |
| C | 0.511 | 0.659 | 0.575 |

### flan_t5_xxl

**Classifier 1 (A vs R)** — Best: epoch 35 (66.86% accuracy)

**Classifier 2 (B vs C)** — Best: epoch 30 (68.08% accuracy)

**Cascaded 3-class** (clf1 ep35 + clf2 ep30): **64.54% overall accuracy**

| | Precision | Recall | F1 |
|---|---|---|---|
| A | 0.560 | 0.447 | 0.497 |
| B | 0.885 | 0.526 | 0.660 |
| C | 0.428 | 0.651 | 0.516 |

The xxl classifiers perform lower than xl (−7.04 pp cascaded), likely due to noisier silver labels from the xxl model.

### Test Set QA Accuracy (After Routing)

| Dataset | xl EM | xl F1 | xxl EM | xxl F1 |
|---|---|---|---|---|
| musique | 0.232 | 0.321 | 0.210 | 0.293 |
| hotpotqa | 0.422 | 0.538 | 0.402 | 0.512 |
| 2wikimultihopqa | 0.390 | 0.478 | 0.478 | 0.575 |
| nq | 0.368 | 0.464 | 0.378 | 0.475 |
| trivia | 0.516 | 0.603 | 0.484 | 0.569 |
| squad | 0.274 | 0.389 | 0.266 | 0.384 |

### Output Locations

```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/{model}/
├── no_ret_vs_ret/epoch/{15,20,25,30,35}/{DATE}/
│   ├── model.safetensors, config.json   (trained classifier 1)
│   ├── valid/                           (validation metrics)
│   └── predict/                         (test set predictions)
└── single_vs_multi/epoch/{15,20,25,30,35}/{DATE}/
    ├── model.safetensors, config.json   (trained classifier 2)
    ├── valid/
    └── predict/

predictions/classifier/t5-large/{model}/split/{best_epoch_combo}/
├── {dataset}/
│   ├── {dataset}.json              (routed QA predictions)
│   ├── {dataset}_option.json       (predictions with routing labels + step counts)
│   └── eval_metic_result_acc.json  (EM, F1, Acc)
```

---

## Threshold Experiment: Clf1 Decision Boundary Tuning

### Background

Clf1 (A vs R) has a **2:1 class imbalance** in training data (R is twice as frequent as A), causing the model to under-predict A. A threshold sweep on the validation set (`classifier/threshold_sweep.py`) found that lowering the Clf1 decision boundary from the default 0.50 (argmax) to **0.35** significantly improves A-class recall:

| Model | Threshold | A-recall | R-recall | Overall Acc |
|---|---|---|---|---|
| flan_t5_xl | 0.50 | 42.6% | 87.9% | 73.2% |
| flan_t5_xl | **0.35** | **59.0%** | **76.5%** | **71.3%** |
| flan_t5_xxl | 0.50 | 44.7% | 83.3% | 66.9% |
| flan_t5_xxl | **0.35** | **66.7%** | **64.1%** | **65.0%** |

### End-to-End QA Impact

A threshold-adjusted routing script (`classifier/postprocess/predict_complexity_threshold.py`) was used to re-run the full QA routing and evaluation pipeline at threshold 0.35.

**Routing distribution changes** (total across 3,000 test questions):

| Model | Threshold | A | B | C | Total Steps |
|---|---|---|---|---|---|
| flan_t5_xl | 0.50 | 405 | 1,530 | 1,065 | 6,155 |
| flan_t5_xl | **0.35** | **743** | **1,280** | **977** | **5,511 (−10.5%)** |
| flan_t5_xxl | 0.50 | 612 | 1,305 | 1,083 | 3,586 |
| flan_t5_xxl | **0.35** | **1,029** | **1,018** | **953** | **3,030 (−15.5%)** |

**QA Accuracy (per dataset)**:

| Dataset | xl (0.50) | xl (0.35) | Δ | xxl (0.50) | xxl (0.35) | Δ |
|---|---|---|---|---|---|---|
| NQ | 0.436 | 0.380 | −0.056 | 0.434 | 0.398 | −0.036 |
| TriviaQA | 0.580 | 0.514 | −0.066 | 0.540 | 0.484 | −0.056 |
| SQuAD | 0.334 | 0.312 | −0.022 | 0.322 | 0.288 | −0.034 |
| MuSiQue | 0.260 | 0.254 | −0.006 | 0.238 | 0.236 | −0.002 |
| HotpotQA | 0.442 | 0.426 | −0.016 | 0.430 | 0.386 | −0.044 |
| 2WikiMHQA | 0.444 | 0.442 | −0.002 | 0.540 | 0.488 | −0.052 |
| **Average** | **0.416** | **0.388** | **−0.028** | **0.417** | **0.380** | **−0.037** |

### Conclusion

Lowering the Clf1 threshold from 0.50 to 0.35 **hurts downstream QA accuracy** across all datasets and both models (−2.8pp for xl, −3.7pp for xxl), despite improving Clf1's classification-level A-recall. The reason: **retrieval rarely hurts** — routing a question through single-step retrieval when it could have been answered without retrieval is low-cost, while skipping retrieval for a question that needs it is very costly. The error costs are asymmetric, so the conservative default threshold (0.50) is preferable.

The efficiency gain (10–15% fewer retrieval steps) does not justify the accuracy drop.

### Threshold Experiment File Locations

```
classifier/threshold_sweep.py                    # Threshold sweep analysis script
classifier/postprocess/predict_complexity_threshold.py  # Threshold-adjusted routing script

# Threshold sweep outputs
classifier/outputs/.../flan_t5_xl/no_ret_vs_ret/epoch/20/.../threshold_sweep/
classifier/outputs/.../flan_t5_xxl/no_ret_vs_ret/epoch/35/.../threshold_sweep/

# Threshold-adjusted QA predictions
predictions/classifier/t5-large/flan_t5_xl/split_thresh035/no_ret_ep20_single_ep35/
predictions/classifier/t5-large/flan_t5_xxl/split_thresh035/no_ret_ep35_single_ep30/
```

Full details: [`THRESHOLD_EXPERIMENT_RESULTS.md`](Adaptive-RAG/THRESHOLD_EXPERIMENT_RESULTS.md)

---

## File Format Reference

### Processed Data (JSONL)
```json
{
  "dataset": "hotpotqa",
  "question_id": "5ab92dba554299131ca422a2",
  "question_text": "Are Sportsworld and Film Magazine both from the UK?",
  "answers_objects": [{"spans": ["yes"]}],
  "contexts": [
    {"idx": 0, "title": "Sportsworld", "paragraph_text": "...", "is_supporting": true},
    {"idx": 1, "title": "Film Magazine", "paragraph_text": "...", "is_supporting": true}
  ]
}
```

### Classifier Data (JSON)
```json
[
  {
    "id": "5ab92dba554299131ca422a2",
    "question": "Are Sportsworld and Film Magazine both from the UK?",
    "answer": "C",
    "answer_description": "multi",
    "dataset_name": "hotpotqa",
    "total_answer": ["multiple"]
  }
]
```

### Predictions (JSON)
```json
{
  "5ab92dba554299131ca422a2": "yes",
  "5a7e01d555429930b01aba3c": "Daniel Robey"
}
```
