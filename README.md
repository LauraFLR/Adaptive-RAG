# Adaptive-RAG: Cascaded Binary Routing

**Laura Ehlert Moreno**
DHBW Stuttgart — Business Information Systems (Data Science)

B.Sc. thesis extension of [Adaptive-RAG](https://arxiv.org/pdf/2403.14403.pdf) (NAACL 2024). Replaces the original 3-class query complexity classifier (A/B/C) with a **cascaded binary routing** architecture: **Gate 1** decides *A vs. R* (no retrieval vs. retrieval needed), then **Gate 2** decides *B vs. C* (single-step vs. multi-step retrieval) for R-routed questions.

Five iterations (plus the baseline) progressively refine both gates — from a trained cascade through class-imbalance mitigation, a training-free agreement gate, a diagnostic κ(q) feature probe, to a **fully training-free cascade** that requires no fine-tuned model, no checkpoint, and no GPU at routing time.

Three backbone LLMs are evaluated: **Flan-T5-XL**, **Flan-T5-XXL**, and **GPT-3.5**. All experiments use **pre-computed QA predictions** (`nor_qa`, `oner_qa`, `ircot_qa`) from the original Adaptive-RAG repository — routing is entirely offline. The base model for all trained classifiers is **T5-Large (770M params)** with seq2seq generative decoding.

---

> **Quick start — pre-computed data included.**
> This repository ships with all pre-computed QA predictions (`predictions/`), classifier training labels (`classifier/data/`), and processed datasets (`processed_data/`). If you only want to reproduce the classifier training and evaluation (Iterations 0–5), you can **skip Setup sections 2–6** entirely and jump straight from §1 (environment) to the Iteration sections. Sections 2–6 document how the upstream data was originally generated and are only needed if you want to regenerate it from scratch.

---

## Results Summary

| Iteration | Gate 1 | Gate 2 | Training Required | XL F1 | XXL F1 | GPT F1 | vs. Baseline |
|---|---|---|---|---|---|---|---|
| IT0 (Baseline) | 3-class clf | (single clf) | Yes | 46.94 | 48.62 | 50.91 | — |
| IT1 | Trained Clf1 (CE) | Trained Clf2 (CE) | Yes | 46.10 | 47.51 | 51.56 | Mixed |
| IT2 | Trained Clf1 (weighted CE) | Trained Clf2 (CE) | Yes | 45.76 | 45.97 | 51.61 | Mixed |
| IT3 | Agreement gate (TF) | Trained Clf2 (CE) | Partial | **48.71** | **50.56** | 50.33 | ✓ XL/XXL |
| IT4 | — | κ(q) probe (diagnostic) | N/A | — | — | — | Diagnostic |
| IT5 | Agreement gate (TF) | κ(q) threshold (TF) | **No** | **48.50** | **50.18** | **51.33** | **✓ All 3** |

TF = training-free. **Bold** F1 values exceed the baseline.

---

## Repository Structure

```
Adaptive-RAG/
├── evaluate_final_acc.py                  # End-to-end QA evaluation (EM/F1) after routing
├── run-all-iterations.sh                  # Master script: trains + routes + evaluates all iterations
├── run_retrieval_test.sh                  # Runs write → predict → evaluate for one strategy
├── run_retrieval_dev.sh                   # Same, on the dev_500 split
│
├── classifier/
│   ├── run_classifier.py                  # T5-Large fine-tuning (supports focal loss, weighted CE)
│   ├── utils.py                           # Model loading, tokenization, metrics
│   │
│   ├── run/                               # Training shell scripts
│   │   ├── run_large_train_{xl,xxl,gpt}.sh                          # IT0: 3-class baseline
│   │   ├── run_large_train_{xl,xxl,gpt}_no_ret_vs_ret.sh            # IT1: Clf1 (A vs R)
│   │   ├── run_large_train_{xl,xxl,gpt}_single_vs_multi.sh          # IT1: Clf2 (B vs C)
│   │   ├── run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_weighted_ce.sh # IT2: Clf1 weighted CE
│   │   ├── run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_focal.sh      # IT2: Clf1 focal loss
│   │   ├── run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh # IT2: Clf1 undersampled
│   │   ├── run_large_train_feat_single_vs_multi.sh                   # Feature-augmented Clf2
│   │   ├── run_large_train_silver_only_single_vs_multi.sh            # Silver-only Clf2 ablation
│   │   └── README.md                      # Detailed documentation of all shell scripts
│   │
│   ├── postprocess/                       # Routing + analysis scripts
│   │   ├── predict_complexity_on_classification_results.py  # IT0: baseline routing
│   │   ├── predict_complexity_split_classifiers.py          # IT1/IT2: cascade routing
│   │   ├── predict_complexity_agreement.py                  # IT3: agreement gate + Clf2
│   │   ├── predict_complexity_kappa.py                      # IT5: κ(q) threshold gate
│   │   ├── predict_complexity_oracle_ceiling.py             # Oracle ceiling analysis
│   │   ├── clf2_feature_probe.py                            # IT4: logistic regression B/C probe
│   │   ├── clf2_kappa_feature_probe.py                      # IT4: κ(q)-aligned feature probe
│   │   └── postprocess_utils.py                             # Shared I/O and routing helpers
│   │
│   ├── preprocess/                        # Label generation scripts
│   │   ├── preprocess_silver_train.py     # Generate silver A/B/C labels from QA predictions
│   │   ├── preprocess_silver_valid.py     # Silver validation labels
│   │   ├── preprocess_binary_train.py     # Inductive-bias labels (B/C by dataset)
│   │   ├── concat_binary_silver_train.py  # Merge silver + IB → binary_silver/train.json
│   │   ├── preprocess_predict.py          # Prepare test prediction file
│   │   └── preprocess_utils.py            # Shared preprocessing helpers
│   │
│   ├── data_utils/
│   │   ├── add_feature_prefix.py          # Prepend [LEN:X] [ENT:Y] [BRIDGE:Z] to questions
│   │   └── make_no_ret_vs_ret_undersampled.py  # IT2: undersample majority class for Clf1
│   │
│   ├── analysis/                          # Post-hoc analysis
│   │   ├── squad_verbosity_analysis.py    # SQuAD verbosity analysis
│   │   ├── flan_focal_checkpoint_rankings.json
│   │   └── flan_focal_vs_ce_eval.json
│   │
│   ├── data/                              # Classifier training/validation/prediction labels
│   │   └── musique_hotpot_wiki2_nq_tqa_sqd/
│   │       ├── predict.json               # 3,000 unlabelled test questions
│   │       ├── {model}/silver/no_retrieval_vs_retrieval/   # Clf1 train/valid (A/R)
│   │       ├── {model}/silver/single_vs_multi/             # Clf2 valid (silver-only, B/C)
│   │       ├── {model}/binary_silver/                      # 3-class merged (IT0)
│   │       └── {model}/binary_silver_single_vs_multi/      # Clf2 train (silver+IB, B/C)
│   │
│   ├── outputs/                           # Trained checkpoints + prediction outputs
│   │
│   └── future_work/                       # Out-of-scope experiments
│       ├── clf2_embedding_probe.py        # Sentence-embedding probe (AUC 0.818)
│       └── clf2_embedding_clf.py          # MLP on sentence embeddings
│
├── results/
│   ├── collect_all_results.py             # Consolidate all iteration metrics into JSON
│   ├── generate_charts.py                 # Generate result charts
│   ├── all_iterations.json                # Collected results (all iterations)
│   ├── EVALUATION_OVERVIEW.md             # Evaluation overview
│   └── iter5/
│       ├── compute_gate1_metrics.py       # Gate 1 classification metrics for IT5
│       └── {model}/gate1_metrics.json     # Per-model Gate 1 precision/recall/F1
│
├── predictions/
│   ├── test/                              # Pre-computed QA predictions (all 3 strategies)
│   │   ├── nor_qa_{model}_{dataset}____prompt_set_1/
│   │   ├── oner_qa_{model}_{dataset}____...___bm25_.../
│   │   └── ircot_qa_{model}_{dataset}____...___bm25_.../
│   └── classifier/                        # Routed predictions after classifier
│       └── t5-large/{model}/iter*_*/...
│
├── processed_data/                        # JSONL datasets (500 test Qs per dataset)
├── llm_server/                            # FastAPI LLM inference server
├── retriever_server/                      # FastAPI BM25 retrieval server
└── commaqa/                               # Core QA inference engine
```

---

## Setup

### 1. Create Environment

```bash
conda create -n adaptiverag python=3.8
conda activate adaptiverag
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### 2. Prepare Retriever Server *(skip if using pre-computed data)*

```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.10.2-linux-x86_64.tar.gz.sha512
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
cd elasticsearch-7.10.2/
./bin/elasticsearch   # start the server
# pkill -f elasticsearch   # to stop the server
```

Start the elasticsearch server on port 9200 (default), then start the retriever server:

```bash
uvicorn serve:app --port 8000 --app-dir retriever_server
```

### 3. Datasets *(skip if using pre-computed data)*

**Multi-hop datasets** (MuSiQue, HotpotQA, 2WikiMultiHopQA) — download from https://github.com/StonyBrookNLP/ircot:

```bash
bash ./download/processed_data.sh

bash ./download/raw_data.sh
python processing_scripts/subsample_dataset_and_remap_paras.py musique dev_diff_size 500
python processing_scripts/subsample_dataset_and_remap_paras.py hotpotqa dev_diff_size 500
python processing_scripts/subsample_dataset_and_remap_paras.py 2wikimultihopqa dev_diff_size 500

python retriever_server/build_index.py {dataset_name}   # hotpotqa, 2wikimultihopqa, musique
```

**Single-hop datasets** (NQ, TriviaQA, SQuAD) — download from https://github.com/facebookresearch/DPR:

```bash
# Natural Questions
mkdir -p raw_data/nq && cd raw_data/nq
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz && gzip -d biencoder-nq-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz && gzip -d biencoder-nq-train.json.gz

# TriviaQA
cd .. && mkdir -p trivia && cd trivia
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz && gzip -d biencoder-trivia-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz && gzip -d biencoder-trivia-train.json.gz

# SQuAD
cd .. && mkdir -p squad && cd squad
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz && gzip -d biencoder-squad1-dev.json.gz
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz && gzip -d biencoder-squad1-train.json.gz

# Wikipedia corpus (shared by NQ/TriviaQA/SQuAD)
cd .. && mkdir -p wiki && cd wiki
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz && gzip -d psgs_w100.tsv.gz

# Process raw data
python ./processing_scripts/process_nq.py
python ./processing_scripts/process_trivia.py
python ./processing_scripts/process_squad.py

# Subsample
python processing_scripts/subsample_dataset_and_remap_paras.py {dataset_name} test 500       # nq, trivia, squad
python processing_scripts/subsample_dataset_and_remap_paras.py {dataset_name} dev_diff_size 500  # nq, trivia, squad

# Build index
python retriever_server/build_index.py wiki
```

Verify index sizes: `curl localhost:9200/_cat/indices` — expect HotpotQA (5,233,329), 2WikiMultiHopQA (430,225), MuSiQue (139,416), Wiki (21,015,324).

### 4. Prepare LLM Server *(skip if using pre-computed data)*

```bash
MODEL_NAME=flan-t5-xl uvicorn serve:app --port 8010 --app-dir llm_server
```

### 5. Run All Three QA Strategies *(skip if using pre-computed data)*

Pre-computed predictions are provided in `predictions/`. To regenerate:

```bash
SYSTEM=ircot_qa   # ircot_qa (multi), oner_qa (single), nor_qa (zero)
MODEL=flan-t5-xl  # flan-t5-xl, flan-t5-xxl
DATASET=nq        # nq, squad, trivia, 2wikimultihopqa, hotpotqa, musique
LLM_PORT_NUM=8010

# Dev set (used for silver training labels):
bash run_retrieval_dev.sh $SYSTEM $MODEL $DATASET $LLM_PORT_NUM

# Test set (used for evaluation + silver validation labels):
bash run_retrieval_test.sh $SYSTEM $MODEL $DATASET $LLM_PORT_NUM
```

### 6. Generate Classifier Labels *(skip if using pre-computed data)*

```bash
python ./classifier/preprocess/preprocess_silver_train.py flan_t5_xl    # or flan_t5_xxl
python ./classifier/preprocess/preprocess_silver_valid.py flan_t5_xl
python ./classifier/preprocess/preprocess_binary_train.py
python ./classifier/preprocess/concat_binary_silver_train.py
python ./classifier/preprocess/preprocess_predict.py
```

---

## Data

All classifier data lives under `classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/`. Silver labels are model-specific (different LLMs answer different questions correctly). `{model}` is one of `flan_t5_xl`, `flan_t5_xxl`, or `gpt`.

| File | Purpose | Size |
|------|---------|------|
| `{model}/silver/no_retrieval_vs_retrieval/train.json` | Clf1 training labels (A / R) | ~1,300–1,400 |
| `{model}/silver/no_retrieval_vs_retrieval/valid.json` | Clf1 validation labels | ~1,350–1,430 |
| `{model}/binary_silver_single_vs_multi/train.json` | Clf2 training labels (B / C), silver + binary merged | ~2,800–3,300 |
| `{model}/silver/single_vs_multi/valid.json` | Clf2 validation labels (silver-only) | ~400–900 |
| `predict.json` | 3,000 unlabelled test questions (500 × 6 datasets) | 3,000 |

Six datasets (500 test questions each): **NQ**, **TriviaQA**, **SQuAD**, **HotpotQA**, **2WikiMultiHopQA**, **MuSiQue**.

Pre-computed QA predictions used by the agreement gate and routing:

| Directory pattern | Content |
|---|---|
| `predictions/test/nor_qa_{model}_{dataset}____prompt_set_1/` | No-retrieval answers |
| `predictions/test/oner_qa_{model}_{dataset}____...___bm25_retrieval_count__{N}___distractor_count__1/` | Single-step retrieval answers |
| `predictions/test/ircot_qa_{model}_{dataset}____...___bm25_retrieval_count__{N}___distractor_count__1/` | Multi-step retrieval answers |

> **Note:** BM25 retrieval counts differ by model: Flan-T5 uses oner=15/ircot=6, GPT uses oner=6/ircot=3. The scripts handle this automatically via lookup tables.

---

## Running All Iterations

The master script `run-all-iterations.sh` orchestrates training, routing, and evaluation for all iterations end-to-end:

```bash
# Run everything (IT1–IT5, mapped to internal IT1–IT7 in the script)
bash run-all-iterations.sh

# Run only specific iterations (by internal numbering: 1=IT1, 5=IT3-agreement, 7=IT5-kappa)
bash run-all-iterations.sh 5 7

# Skip training if checkpoints already exist
SKIP_TRAINING=true bash run-all-iterations.sh 5 7
```

> **Internal numbering note:** The `run-all-iterations.sh` script uses internal iteration numbers 1–7 (which include sub-experiments). The mapping to thesis iterations is: 1→IT1, 2–4→IT2 (undersampled/weighted CE/focal), 5→IT3, 6→IT4, 7→IT5.

All commands below assume `cwd` is the **repo root** (`Adaptive-RAG/`). Training shell scripts use relative paths and must be run from inside `classifier/`, so they are wrapped in subshells.

---

## Iteration 0 — Adaptive-RAG Baseline

**What it is:** The unmodified original Adaptive-RAG system — a single 3-class T5-Large classifier trained on merged `binary_silver/train.json` that routes each question to A, B, or C.

**Key results (macro-avg F1):** XL: 46.94 | XXL: 48.62 | GPT: 50.91

### Key Files

| File | Purpose |
|---|---|
| `classifier/run/run_large_train_{xl,xxl,gpt}.sh` | Train the 3-class classifier (labels: A B C) |
| `classifier/run_classifier.py` | T5-Large fine-tuning engine (shared by all iterations) |
| `classifier/postprocess/predict_complexity_on_classification_results.py` | Route test questions using 3-class predictions |
| `classifier/preprocess/concat_binary_silver_train.py` | Merge silver + IB labels → `binary_silver/train.json` |
| `classifier/preprocess/preprocess_binary_train.py` | Generate inductive-bias B/C labels from dataset structure |
| `classifier/preprocess/preprocess_silver_train.py` | Generate silver A/B/C labels from dev-set QA comparisons |

### Train & Evaluate

```bash
# Train (e.g. XL)
(cd classifier && bash run/run_large_train_xl.sh)

# Route test questions using 3-class predictions
python classifier/postprocess/predict_complexity_on_classification_results.py flan_t5_xl

# Evaluate
python evaluate_final_acc.py \
    --pred_path predictions/classifier/t5-large/flan_t5_xl/
```

---

## Iteration 1 — Cascade Baseline

**What it is:** Decomposes the 3-class classifier into two independent binary classifiers (Clf1: A vs. R, Clf2: B vs. C), both trained with standard cross-entropy loss. Tests whether cascade structure alone improves routing.

**What changed (single variable):** Architecture — from one 3-class classifier to two binary classifiers.

- **Clf1 (A vs. R):** Silver labels only. A stays A; B and C are collapsed to R. Training file: `{model}/silver/no_retrieval_vs_retrieval/train.json`.
- **Clf2 (B vs. C):** Merged silver + inductive-bias labels, filtered to remove all A-labeled samples. Training file: `{model}/binary_silver_single_vs_multi/train.json`.

**Key results (macro-avg F1):** XL: 46.10 | XXL: 47.51 | GPT: 51.56

**Diagnostic finding:** Clf1 exhibits severe majority-class bias. For Flan-T5 models (R-majority), A-recall is only 35–48%. For GPT (A-majority), R-recall is only 35%.

### Key Files

| File | Purpose |
|---|---|
| `classifier/run/run_large_train_{xl,xxl,gpt}_no_ret_vs_ret.sh` | Train Clf1 (A vs R) with `--labels A R` |
| `classifier/run/run_large_train_{xl,xxl,gpt}_single_vs_multi.sh` | Train Clf2 (B vs C) with `--labels B C` |
| `classifier/postprocess/predict_complexity_split_classifiers.py` | Cascade routing: merge Clf1 + Clf2 predictions |
| `classifier/SPLIT_CLASSIFIERS_CHANGES.md` | Documents the 3-class → binary refactoring |

### Train Clf1 (A vs. R)

```bash
(cd classifier && bash run/run_large_train_xl_no_ret_vs_ret.sh)
(cd classifier && bash run/run_large_train_xxl_no_ret_vs_ret.sh)
(cd classifier && bash run/run_large_train_gpt_no_ret_vs_ret.sh)
```

Each script trains T5-Large at epochs 15–35 (35–40 for GPT) with `seed=42`. Outputs to:
`classifier/outputs/.../no_ret_vs_ret/epoch/{N}/{DATE}/predict/dict_id_pred_results.json`

### Train Clf2 (B vs. C)

```bash
(cd classifier && bash run/run_large_train_xl_single_vs_multi.sh)
(cd classifier && bash run/run_large_train_xxl_single_vs_multi.sh)
(cd classifier && bash run/run_large_train_gpt_single_vs_multi.sh)
```

### Run Cascade Inference

Select the epoch with highest validation accuracy for each classifier, then merge predictions:

```bash
# From classifier/postprocess/ directory (required for local imports):
(cd classifier/postprocess && python predict_complexity_split_classifiers.py flan_t5_xl \
    --no_ret_vs_ret_file  ../outputs/.../no_ret_vs_ret/epoch/{BEST_EP}/{DATE}/predict/dict_id_pred_results.json \
    --single_vs_multi_file ../outputs/.../single_vs_multi/epoch/{BEST_EP}/{DATE}/predict/dict_id_pred_results.json \
    --output_path ../../predictions/classifier/t5-large/flan_t5_xl/iter1_standard/)
```

> **Note:** `predict_complexity_split_classifiers.py` must be run from `classifier/postprocess/` due to a local import.

### Evaluate

```bash
python evaluate_final_acc.py \
    --pred_path predictions/classifier/t5-large/flan_t5_xl/iter1_standard/
```

---

## Iteration 2 — Clf1 with Class-Weighted Cross-Entropy

**What it is:** Addresses the class imbalance diagnosed in IT1 by replacing Clf1's loss function with class-weighted cross-entropy (inverse-frequency weighting).

**What changed (single variable):** Clf1 loss function — standard CE → weighted CE. Clf2 is unchanged from IT1.

**Implementation detail:** Uses `FocalLoss` in `classifier/run_classifier.py` with γ=0 (which reduces to weighted CE: −α_t · log(p_t)). Per-class weights computed via inverse-frequency formula (`w_c = N_total / (K · N_c)`) when `--auto_class_weights` is set.

**Weights:**

| Model | w_A | w_R | Minority class |
|---|---|---|---|
| Flan-T5-XL | 1.524 | 0.744 | A |
| Flan-T5-XXL | 1.379 | 0.785 | A |
| GPT | 0.699 | 1.754 | R |

**Key results (macro-avg F1):** XL: 45.76 | XXL: 45.97 | GPT: 51.61

**Diagnostic finding:** Weighted CE improves minority-class recall but does NOT improve downstream QA F1. A focal loss variant (γ=2.0) and random undersampling also fail. The consistent failure across three rebalancing strategies proves that **class imbalance is not the root cause** — the silver labels themselves are unreliable for the A/R boundary.

### Key Files

| File | Purpose |
|---|---|
| `classifier/run/run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_weighted_ce.sh` | Train Clf1 with `--use_focal_loss --focal_gamma 0 --auto_class_weights` |
| `classifier/run/run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_focal.sh` | Train Clf1 with `--use_focal_loss --focal_gamma 2.0 --focal_alpha 0.33` |
| `classifier/run/run_large_train_{xl,xxl,gpt}_no_ret_vs_ret_undersampled.sh` | Train Clf1 on undersampled data |
| `classifier/data_utils/make_no_ret_vs_ret_undersampled.py` | Undersample majority class in Clf1 training data |
| `classifier/run_classifier.py` | Contains `FocalLoss` class and `FocalLossTrainer` (HF Trainer subclass) |

### Train Clf1 with Weighted CE

```bash
(cd classifier && bash run/run_large_train_xl_no_ret_vs_ret_weighted_ce.sh)
(cd classifier && bash run/run_large_train_xxl_no_ret_vs_ret_weighted_ce.sh)
(cd classifier && bash run/run_large_train_gpt_no_ret_vs_ret_weighted_ce.sh)
```

### Alternative: Focal Loss (γ=2.0)

```bash
(cd classifier && bash run/run_large_train_xl_no_ret_vs_ret_focal.sh)
```

### Alternative: Random Undersampling

```bash
# Generate undersampled training file, then train
(cd classifier && bash run/run_large_train_xl_no_ret_vs_ret_undersampled.sh)
```

The undersampling script (`make_no_ret_vs_ret_undersampled.py`) reads `silver/no_retrieval_vs_retrieval/train.json` and writes `train_undersampled.json` with balanced class sizes.

### Route & Evaluate

Same cascade inference as IT1 using `predict_complexity_split_classifiers.py`, substituting the weighted-CE Clf1 predictions:

```bash
# Route (from classifier/postprocess/)
(cd classifier/postprocess && python predict_complexity_split_classifiers.py flan_t5_xl \
    --no_ret_vs_ret_file  ../outputs/.../no_ret_vs_ret_weighted_ce/epoch/{BEST_EP}/{DATE}/predict/dict_id_pred_results.json \
    --single_vs_multi_file ../outputs/.../single_vs_multi/epoch/{BEST_EP}/{DATE}/predict/dict_id_pred_results.json \
    --output_path ../../predictions/classifier/t5-large/flan_t5_xl/iter3_weighted_ce/)

# Evaluate
python evaluate_final_acc.py \
    --pred_path predictions/classifier/t5-large/flan_t5_xl/iter3_weighted_ce/
```

---

## Iteration 3 — Gate 1 as Cross-Strategy Agreement Gate (Training-Free)

**What it is:** Replaces the trained Clf1 entirely with a **training-free heuristic**: if the LLM's no-retrieval answer (`nor_qa`) and single-step retrieval answer (`oner_qa`) are identical after normalization, the question is routed to A. Otherwise, it's routed to R and passed to Clf2 (unchanged from IT1).

**What changed (single variable):** Gate 1 mechanism — trained T5-Large classifier → deterministic answer-agreement gate. Clf2 consumed as-is from IT1 checkpoints.

**Agreement logic (3 steps):**
1. **Answer extraction:** Regex `(.* answer is:? (.*)\\.?)` strips chain-of-thought reasoning.
2. **Normalization:** `normalize_answer()` — lowercase, strip articles/punctuation, collapse whitespace.
3. **Exact match:** Agree iff both normalized answers are non-empty and identical.

**Routing rule:** Agree → A (use `nor_qa` answer). Disagree → Clf2 decides B or C.

**Agreement rates:** XL: 19.4% | XXL: 21.0% | GPT: 36.9%

**Key results (macro-avg F1):** XL: 48.71 | XXL: 50.56 | GPT: 50.33

**This is the first iteration to beat the Adaptive-RAG baseline** for Flan-T5 models (+1.77 pp XL, +1.94 pp XXL). GPT regresses by −0.58 pp (caused by Clf2's poor GPT accuracy of 52.9%, not by the agreement gate itself).

### Key Files

| File | Purpose |
|---|---|
| `classifier/postprocess/predict_complexity_agreement.py` | Agreement gate routing: compares nor_qa vs oner_qa, then applies Clf2 |
| `classifier/postprocess/predict_complexity_oracle_ceiling.py` | Oracle ceiling: measures max F1 with perfect Clf2 |
| `results/iter5/compute_gate1_metrics.py` | Compute Gate 1 precision/recall/F1 metrics |

### Run the Agreement Gate

```bash
python classifier/postprocess/predict_complexity_agreement.py flan_t5_xl \
    --clf2_pred_file  classifier/outputs/.../single_vs_multi/epoch/{BEST_EP}/{DATE}/predict/dict_id_pred_results.json \
    --predict_file    classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json \
    --output_path     predictions/classifier/t5-large/flan_t5_xl/iter5_agreement/
```

Repeat with `flan_t5_xxl` and `gpt`, using the corresponding Clf2 prediction files.

### Oracle Ceiling Analysis

Measures the maximum F1 achievable if Clf2 were perfect:

```bash
python classifier/postprocess/predict_complexity_oracle_ceiling.py flan_t5_xl
python classifier/postprocess/predict_complexity_oracle_ceiling.py flan_t5_xxl
python classifier/postprocess/predict_complexity_oracle_ceiling.py gpt
```

### Gate 1 Metrics

```bash
python results/iter5/compute_gate1_metrics.py \
    --model_name flan_t5_xl \
    --routed_pred_path predictions/classifier/t5-large/flan_t5_xl/iter5_agreement/ \
    --output_file results/iter5/xl/gate1_metrics.json
```

### Evaluate

```bash
python evaluate_final_acc.py \
    --pred_path predictions/classifier/t5-large/flan_t5_xl/iter5_agreement/
```

---

## Iteration 4 — Diagnostic κ(q) Feature Probe for Gate 2

**What it is:** A **standalone diagnostic experiment** (not deployed in the routing pipeline) that tests whether structural query features carry enough signal to distinguish B from C. Uses logistic regression on features derived from the SymRAG κ(q) complexity score.

**This is a diagnostic probe — no routing pipeline change. No end-to-end QA results.**

**Four features:**
1. `token_len_norm` = token count / max(token count)
2. `entity_density` = spaCy NER entity count / token count
3. `hop_density` = regex bridging-pattern match count / token count
4. `κ(q) = W_L · token_len_norm · (1 + W_SH1 · entity_density + W_SH2 · hop_density)` with weights W_L=1.0, W_SH1=0.05, W_SH2=0.10

**Probe setup:** scikit-learn `LogisticRegression` with `class_weight="balanced"`, `max_iter=1000`, `random_state=42`. 5-fold stratified cross-validation. Seven bridging-pattern regexes for hop_count detection.

**Key results:**

| Data source | Macro-F1 range | Verdict |
|---|---|---|
| Merged (silver + IB) | 0.629–0.650 | GO |
| Silver-only | 0.532–0.540 | NO-GO |
| IB-only | 0.685 | GO |

**Diagnostic finding:** κ(q) features encode **dataset-level structural complexity** (multi-hop datasets have longer questions with more bridging patterns) rather than per-question complexity from silver labels.

### Key Files

| File | Purpose |
|---|---|
| `classifier/postprocess/clf2_feature_probe.py` | Original 3-feature probe (token_len, entity_count, bridge_flag) |
| `classifier/postprocess/clf2_kappa_feature_probe.py` | κ(q)-aligned probe (token_len_norm, entity_density, hop_density, kappa) |

Both scripts use spaCy `en_core_web_sm` for NER and 7 bridging-phrase regex patterns:

1. Relative-clause bridges (`who/where/which/that + verb`)
2. Double possessive chains (`X's … Y's …`)
3. Temporal subordination before wh-words (`before/after/when + who/what/where`)
4. Demonstrative back-references (`that country`, `this person`)
5. Explicit comparisons (`both … and`, `between … and`)
6. Nested wh-questions (`of the X who/that/which`)

### Run the Probes

```bash
# κ(q)-aligned probe across all models
python classifier/postprocess/clf2_kappa_feature_probe.py --all_models

# Or per model
python classifier/postprocess/clf2_kappa_feature_probe.py --model flan_t5_xl
python classifier/postprocess/clf2_kappa_feature_probe.py --model flan_t5_xxl
python classifier/postprocess/clf2_kappa_feature_probe.py --model gpt

# Original 3-feature probe
python classifier/postprocess/clf2_feature_probe.py --model flan_t5_xl
```

Outputs: CSV feature data, scatter/histogram PNGs, and JSON summary under the output directory.

---

## Iteration 5 — Fully Training-Free κ(q) Threshold Gate

**What it is:** Replaces both trained classifiers with training-free mechanisms. Gate 1 = IT3's agreement gate. Gate 2 = a single κ(q) threshold tuned on merged data. This is the **first fully training-free cascade** — no fine-tuned model, no checkpoint, no GPU needed at routing time.

**What changed (single variable):** Gate 2 mechanism — trained Clf2 → κ(q) threshold gate. Gate 1 is unchanged from IT3.

**Gate 2 routing rule:** κ(q) ≥ τ → C (multi-step retrieval). κ(q) < τ → B (single-step retrieval).

**Threshold tuning:**
- Tuned on merged IB+silver Clf2 training files (`binary_silver_single_vs_multi/train.json`), NOT silver-only validation.
- Objective: maximize macro-F1 (not accuracy).
- Sweep: 100 candidates between 5th and 95th percentiles of κ(q) distribution.
- Resulting thresholds: τ ≈ 0.17 for all models.

**Dependencies:** Only spaCy `en_core_web_sm` (≈12 MB) + standard library. No torch/transformers imported.

**Key results (macro-avg F1):** XL: 48.50 | XXL: 50.18 | GPT: 51.33

**All three models beat the Adaptive-RAG baseline** (XL: +1.56 pp, XXL: +1.56 pp, GPT: +0.42 pp).

**Trade-off:** Aggressive C-routing increases retrieval steps by 31–64% vs baseline. Eliminates all model training/GPU dependencies in exchange for more retrieval computation.

### Key Files

| File | Purpose |
|---|---|
| `classifier/postprocess/predict_complexity_kappa.py` | Combined agreement gate + κ(q) threshold routing |

This single script implements:
- `normalize_answer()` / `answer_extractor()` — answer comparison for Gate 1
- `compute_agreement()` — nor_qa vs oner_qa agreement check
- `extract_features()` — spaCy NER + bridging-phrase regex features
- `compute_kappa()` — κ(q) = w_L · L(q) · (1 + S_H(q))
- `tune_threshold()` — sweep thresholds on validation data, optimizing macro-F1

### Run the Fully Training-Free Pipeline

```bash
# With automatic threshold tuning on merged data (recommended)
python classifier/postprocess/predict_complexity_kappa.py flan_t5_xl \
    --use_agreement_gate \
    --tune_threshold \
    --valid_file classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/binary_silver_single_vs_multi/train.json \
    --output_path predictions/classifier/t5-large/flan_t5_xl/iter7_kappa/

# Repeat for other models
python classifier/postprocess/predict_complexity_kappa.py flan_t5_xxl \
    --use_agreement_gate \
    --tune_threshold \
    --valid_file classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xxl/binary_silver_single_vs_multi/train.json \
    --output_path predictions/classifier/t5-large/flan_t5_xxl/iter7_kappa/

python classifier/postprocess/predict_complexity_kappa.py gpt \
    --use_agreement_gate \
    --tune_threshold \
    --valid_file classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/gpt/binary_silver_single_vs_multi/train.json \
    --output_path predictions/classifier/t5-large/gpt/iter7_kappa/

# Or with a fixed threshold (no tuning)
python classifier/postprocess/predict_complexity_kappa.py flan_t5_xl \
    --use_agreement_gate \
    --kappa_threshold 0.17 \
    --output_path predictions/classifier/t5-large/flan_t5_xl/kappa_fixed/
```

### Evaluate

```bash
python evaluate_final_acc.py \
    --pred_path predictions/classifier/t5-large/flan_t5_xl/iter7_kappa/
```

---

## Shared Infrastructure

### Training Engine — `classifier/run_classifier.py`

The core T5-Large fine-tuning script used by all training iterations. Key arguments:

| Argument | Purpose |
|---|---|
| `--labels A B C` | Label set (default: 3-class). Use `A R` for Clf1, `B C` for Clf2 |
| `--use_focal_loss` | Enable focal loss (requires `FocalLossTrainer`) |
| `--focal_gamma` | Focal loss γ parameter (0.0 = weighted CE, 2.0 = focal) |
| `--focal_alpha` | Fixed per-class weight (use with `--focal_gamma 2.0`) |
| `--auto_class_weights` | Compute inverse-frequency weights from training data |
| `--seed` | Random seed for reproducibility (default: unset; IT1+ use 42) |
| `--num_train_epochs` | Number of training epochs |

### Evaluation — `evaluate_final_acc.py`

Evaluates routed QA predictions against ground truth. Loads predictions from `{pred_path}/{dataset}/{dataset}.json` and ground truth from `processed_data/{dataset}/test_subsampled.jsonl`. Computes EM, F1, accuracy per dataset plus official evaluation for multi-hop datasets.

```bash
python evaluate_final_acc.py --pred_path predictions/classifier/t5-large/{model}/iter{N}_{tag}/
```

### Results Collection — `results/collect_all_results.py`

Consolidates all iteration metrics (classifier accuracy, QA F1/EM, routing distribution, deltas vs. baseline) into a single JSON file.

```bash
python results/collect_all_results.py
# Output: results/all_iterations.json
```

### Label Distribution — `print_label_distribution_per_dataset.py`

Prints per-dataset, per-class label distributions for all data stages (silver 3-class, Clf1 A/R, Clf2 B/C, IB-only, merged, undersampled).

```bash
python print_label_distribution_per_dataset.py
# Output: label_distribution_per_dataset.json
```

---

## Out of Scope / Future Work

The directory `classifier/future_work/` contains two scripts exploring semantic-embedding-based Clf2 improvements beyond the scope of this thesis:

- **`clf2_embedding_probe.py`** — Logistic regression on frozen all-MiniLM-L6-v2 sentence embeddings (384-dim). Achieved AUC 0.818 on B/C classification, well above the 0.75 go/no-go threshold.
- **`clf2_embedding_clf.py`** — MLP classifier on the same embeddings, producing a drop-in `dict_id_pred_results.json`. End-to-end evaluation showed a small regression (−0.5 to −0.7 pp F1), suggesting that while embeddings separate B/C better in isolation, the T5-Large Clf2 makes errors that are more benign for downstream QA accuracy.

These are discussed in the thesis Chapter 6 (Future Work) as a potential direction for improving Gate 2 with richer representations.

---

## Acknowledgements

This repository is a B.Sc. thesis extension that builds upon the original [Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG) codebase by Jeong et al. (NAACL 2024). The upstream code — including the QA inference engine (`commaqa/`), retriever/LLM servers, dataset processing scripts, evaluation logic, and pre-computed QA predictions — is taken from their repository. All modifications and additions in this fork (the cascaded binary routing architecture, agreement gate, κ(q) threshold gate, and associated training/analysis scripts) were developed as part of this thesis.

## Citation

If you use or reference the original Adaptive-RAG system, please cite:

```BibTex
@inproceedings{jeong2024adaptiverag,
  author       = {Soyeong Jeong and Jinheon Baek and Sukmin Cho and Sung Ju Hwang and Jong Park},
  title        = {Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity},
  booktitle    = {NAACL},
  year         = {2024},
  url          = {https://arxiv.org/abs/2403.14403}
}
```
