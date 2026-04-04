# Adaptive-RAG: Cascaded Binary Routing

B.Sc. thesis extension of [Adaptive-RAG](https://arxiv.org/pdf/2403.14403.pdf) (NAACL 2024). Replaces the original 3-class query complexity classifier (A/B/C) with a **cascaded binary routing** architecture: **Gate 1** decides *A vs. R* (no retrieval vs. retrieval needed), then **Gate 2** decides *B vs. C* (single-step vs. multi-step retrieval) for R-routed questions. Three iterations progressively refine both gates: (1) a trained cascade baseline, (2) a training-free agreement gate for Gate 1, and (3) structural feature augmentation for Gate 2.

---

> **Quick start — pre-computed data included.**
> This repository ships with all pre-computed QA predictions (`predictions/`), classifier training labels (`classifier/data/`), and processed datasets (`processed_data/`). If you only want to reproduce the classifier training and evaluation (Iterations 1–3), you can **skip Setup sections 2–6** entirely and jump straight from §1 (environment) to the Iteration sections. Sections 2–6 document how the upstream data was originally generated and are only needed if you want to regenerate it from scratch.

---

## Repository Structure

```
Adaptive-RAG/
├── evaluate_final_acc.py                  # End-to-end QA evaluation (EM/F1) after routing
├── run_retrieval_test.sh                  # Runs write → predict → evaluate for one strategy
├── run_retrieval_dev.sh                   # Same, on the dev_500 split
│
├── classifier/
│   ├── run_classifier.py                  # T5-Large fine-tuning script (used by all training shells)
│   │
│   ├── run/                               # Training shell scripts
│   │   ├── run_large_train_xl_no_ret_vs_ret.sh       # Iter 1: Clf1 (A vs R), XL silver labels
│   │   ├── run_large_train_xxl_no_ret_vs_ret.sh      # Iter 1: Clf1 (A vs R), XXL silver labels
│   │   ├── run_large_train_gpt_no_ret_vs_ret.sh      # Iter 1: Clf1 (A vs R), GPT silver labels
│   │   ├── run_large_train_xl_single_vs_multi.sh     # Iter 1: Clf2 (B vs C), XL
│   │   ├── run_large_train_xxl_single_vs_multi.sh    # Iter 1: Clf2 (B vs C), XXL
│   │   ├── run_large_train_gpt_single_vs_multi.sh    # Iter 1: Clf2 (B vs C), GPT
│   │   ├── run_large_train_feat_single_vs_multi.sh   # Iter 3: Feature-augmented Clf2
│   │   └── ...                            # Original 3-class scripts (not used by thesis)
│   │
│   ├── postprocess/                       # Routing + evaluation scripts
│   │   ├── predict_complexity_split_classifiers.py    # Iter 1: cascade inference (Clf1 → Clf2)
│   │   ├── predict_complexity_agreement.py            # Iter 2: agreement gate + Clf2
│   │   ├── predict_complexity_oracle_ceiling.py       # Iter 2: oracle ceiling (perfect Clf2)
│   │   ├── clf2_feature_probe.py                      # Iter 3: logistic regression B/C probe
│   │   └── postprocess_utils.py                       # Shared I/O helpers
│   │
│   ├── data_utils/
│   │   └── add_feature_prefix.py          # Iter 3: prepend [LEN:X] [ENT:Y] [BRIDGE:Z]
│   │
│   ├── data/                              # Classifier training/validation/prediction labels
│   │   └── musique_hotpot_wiki2_nq_tqa_sqd/
│   │       ├── predict.json               # 3,000 unlabelled test questions
│   │       ├── flan_t5_xl/                # Model-specific silver labels
│   │       │   ├── silver/no_retrieval_vs_retrieval/   # Clf1 train/valid
│   │       │   ├── silver/single_vs_multi/             # Clf2 valid (silver-only)
│   │       │   └── binary_silver_single_vs_multi/      # Clf2 train (silver + binary)
│   │       ├── flan_t5_xxl/               # Same structure for XXL
│   │       └── gpt/                       # Same structure for GPT (GPT-3.5)
│   │
│   ├── outputs/                           # Trained checkpoints + prediction outputs
│   │
│   └── future_work/                       # Out-of-scope experiments (see §8)
│       ├── clf2_embedding_probe.py
│       └── clf2_embedding_clf.py
│
├── predictions/
│   ├── test/                              # Pre-computed QA predictions (all 3 strategies)
│   │   ├── nor_qa_{model}_{dataset}____prompt_set_1/
│   │   ├── oner_qa_{model}_{dataset}____...___bm25_.../
│   │   └── ircot_qa_{model}_{dataset}____...___bm25_.../
│   └── classifier/                        # Routed predictions after classifier
│       └── t5-large/{model}/split_agreement/...
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

## Iteration 1 — Cascade Baseline

**Question:** Does decomposing the 3-class classifier into two binary stages (Gate 1: A vs. R, Gate 2: B vs. C) improve over the original single Adaptive-RAG classifier?

All commands below assume `cwd` is the **repo root** (`Adaptive-RAG/`). The training shell scripts use relative paths (`./data/`, `./outputs/`) and must be run from inside `classifier/`, so those are wrapped in a subshell.

### Train Clf1 (A vs. R)

```bash
# Flan-T5-XL
(cd classifier && bash run/run_large_train_xl_no_ret_vs_ret.sh)

# Flan-T5-XXL
(cd classifier && bash run/run_large_train_xxl_no_ret_vs_ret.sh)

# GPT
(cd classifier && bash run/run_large_train_gpt_no_ret_vs_ret.sh)
```

Each script trains T5-Large at epochs 15–35 (35–40 for GPT) and writes checkpoints + predictions to:
`classifier/outputs/.../no_ret_vs_ret/epoch/{N}/{DATE}/predict/dict_id_pred_results.json`

### Train Clf2 (B vs. C)

```bash
# Flan-T5-XL
(cd classifier && bash run/run_large_train_xl_single_vs_multi.sh)

# Flan-T5-XXL
(cd classifier && bash run/run_large_train_xxl_single_vs_multi.sh)

# GPT
(cd classifier && bash run/run_large_train_gpt_single_vs_multi.sh)
```

Outputs to: `classifier/outputs/.../single_vs_multi/epoch/{N}/{DATE}/predict/dict_id_pred_results.json`

### Run Cascade Inference

Select the epoch with highest validation accuracy for each classifier. For the XL experiments reported in the thesis, ep20 (Clf1) and ep35 (Clf2) were used:

```bash
(cd classifier/postprocess && python predict_complexity_split_classifiers.py flan_t5_xl \
    --no_ret_vs_ret_file  ../../classifier/outputs/.../no_ret_vs_ret/epoch/20/{DATE}/predict/dict_id_pred_results.json \
    --single_vs_multi_file ../../classifier/outputs/.../single_vs_multi/epoch/35/{DATE}/predict/dict_id_pred_results.json \
    --output_path ../../predictions/classifier/t5-large/flan_t5_xl/split/no_ret_ep20_single_ep35/)
```

> **Note:** `predict_complexity_split_classifiers.py` must be run from `classifier/postprocess/` due to a local import. The other postprocess scripts handle `sys.path` internally and work from the repo root.

### Evaluate

```bash
python evaluate_final_acc.py \
    --pred_path predictions/classifier/t5-large/flan_t5_xl/split/no_ret_ep20_single_ep35/
```

### Expected Results

The cascade matches the original 3-class Adaptive-RAG within noise:

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | Avg F1 |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.464 | 0.603 | 0.389 | 0.321 | 0.538 | 0.478 | 0.466 |
| Flan-T5-XXL | 0.475 | 0.569 | 0.384 | 0.293 | 0.512 | 0.575 | 0.468 |
| GPT | 0.557 | 0.751 | 0.310 | 0.328 | 0.520 | 0.584 | 0.508 |

Best Clf1/Clf2 epochs: XL ep20/ep35, XXL ep35/ep30, GPT ep35/ep35.

---

## Iteration 2 — Agreement Gate (UE-Based Gate 1)

**Question:** Does replacing the trained Clf1 with a training-free cross-strategy answer agreement test fix Gate 1's self-knowledge blindness?

**Method:** If `normalize(nor_qa_answer) == normalize(oner_qa_answer)` and both non-empty → route **A**; otherwise → pass to Clf2 for B/C classification. Normalization: lowercase, strip articles (a/an/the), strip punctuation, collapse whitespace.

This requires **no new inference** — it uses the pre-computed nor_qa and oner_qa predictions already in `predictions/test/`.

### Run the Agreement Gate

```bash
python classifier/postprocess/predict_complexity_agreement.py flan_t5_xl \
    --clf2_pred_file  classifier/outputs/.../single_vs_multi/epoch/35/{DATE}/predict/dict_id_pred_results.json \
    --predict_file    classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json \
    --output_path     predictions/classifier/t5-large/flan_t5_xl/split_agreement/nor_oner_clf2ep35/
```

Repeat with `flan_t5_xxl` and `gpt`, using the corresponding Clf2 prediction files.

### Run the Oracle Ceiling Analysis

Measures the maximum F1 achievable if Clf2 were perfect (agreement gate fixed):

```bash
python classifier/postprocess/predict_complexity_oracle_ceiling.py flan_t5_xl
python classifier/postprocess/predict_complexity_oracle_ceiling.py flan_t5_xxl
python classifier/postprocess/predict_complexity_oracle_ceiling.py gpt
```

### Evaluate

```bash
python evaluate_final_acc.py \
    --pred_path predictions/classifier/t5-large/flan_t5_xl/split_agreement/nor_oner_clf2ep35/
```

### Expected Results

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | Avg F1 |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.473 | 0.619 | 0.390 | 0.316 | 0.545 | 0.568 | 0.485 |
| Flan-T5-XXL | 0.510 | 0.645 | 0.402 | 0.288 | 0.554 | 0.631 | 0.505 |
| GPT | 0.469 | 0.662 | 0.337 | 0.330 | 0.581 | 0.646 | 0.504 |

| Model | Iter 1 Avg F1 | Iter 2 Avg F1 | Δ |
|---|---|---|---|
| Flan-T5-XL | 0.466 | 0.485 | +1.9 pp |
| Flan-T5-XXL | 0.468 | 0.505 | +3.7 pp |
| GPT | 0.508 | 0.504 | −0.4 pp |

The agreement gate's high A-precision (~92%) means it almost never incorrectly skips retrieval for XL/XXL. For GPT, the agreement rate is lower (36.9% vs ~50% for Flan-T5) and the gate provides no improvement over the cascade, suggesting GPT's nor_qa and oner_qa answers diverge more frequently.

**Oracle ceiling** (perfect Clf2 with agreement gate):

| Model | Iter 2 Avg F1 | Oracle Avg F1 | Headroom |
|---|---|---|---|
| Flan-T5-XL | 0.485 | 0.509 | +2.4 pp |
| Flan-T5-XXL | 0.505 | 0.529 | +2.4 pp |
| GPT | 0.504 | 0.522 | +1.8 pp |

---

## Iteration 3 — Feature-Augmented Clf2

**Question:** Do structural query features (token length, entity count, bridging phrase flag) improve Gate 2's B/C routing?

### Step 1 — Feasibility Probe

Run logistic regression on three features to assess B/C separability:

```bash
python classifier/postprocess/clf2_feature_probe.py
python classifier/postprocess/clf2_feature_probe.py --model flan_t5_xxl
python classifier/postprocess/clf2_feature_probe.py --model gpt
```

Reports ROC-AUC, accuracy, per-feature coefficients, and a scatter plot. Go/no-go threshold: **AUC ≥ 0.65**.

Expected: AUC ≈ 0.676 → **GO** (above 0.65, proceed to retraining).

### Step 2 — Add Feature Prefix

Prepend `[LEN:X] [ENT:Y] [BRIDGE:Z]` to each question in the Clf2 train, valid, and predict files:

```bash
python classifier/data_utils/add_feature_prefix.py
```

This writes feature-prefixed copies:
- `classifier/data/.../binary_silver_feat_single_vs_multi/train.json`
- `classifier/data/.../silver_feat_single_vs_multi/valid.json`
- `classifier/data/.../feat_predict.json`

### Step 3 — Retrain Clf2 with Features

```bash
# Flan-T5-XL
(cd classifier && bash run/run_large_train_feat_single_vs_multi.sh flan_t5_xl)

# Flan-T5-XXL
(cd classifier && bash run/run_large_train_feat_single_vs_multi.sh flan_t5_xxl)

# GPT
(cd classifier && bash run/run_large_train_feat_single_vs_multi.sh gpt)
```

Outputs to: `classifier/outputs/.../feat_single_vs_multi/epoch/{N}/feat/predict/dict_id_pred_results.json`

### Step 4 — Run Agreement Gate with Feature-Augmented Clf2

```bash
python classifier/postprocess/predict_complexity_agreement.py flan_t5_xl \
    --clf2_pred_file  classifier/outputs/.../feat_single_vs_multi/epoch/25/feat/predict/dict_id_pred_results.json \
    --predict_file    classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json \
    --output_path     predictions/classifier/t5-large/flan_t5_xl/split_agreement/nor_oner_featclf2ep25/
```

### Step 5 — Evaluate

```bash
python evaluate_final_acc.py \
    --pred_path predictions/classifier/t5-large/flan_t5_xl/split_agreement/nor_oner_featclf2ep25/
```

### Expected Results

| Model | NQ | TriviaQA | SQuAD | MuSiQue | HotpotQA | 2WikiMHQA | Avg F1 |
|---|---|---|---|---|---|---|---|
| Flan-T5-XL | 0.473 | 0.619 | 0.386 | 0.317 | 0.550 | 0.569 | 0.486 |
| Flan-T5-XXL | 0.512 | 0.645 | 0.402 | 0.292 | 0.557 | 0.632 | 0.507 |
| GPT | 0.468 | 0.665 | 0.339 | 0.338 | 0.573 | 0.647 | 0.505 |

| Model | Iter 2 Avg F1 | Iter 3 Avg F1 | Δ |
|---|---|---|---|
| Flan-T5-XL | 0.485 | 0.486 | +0.1 pp |
| Flan-T5-XXL | 0.505 | 0.507 | +0.2 pp |
| GPT | 0.504 | 0.505 | +0.1 pp |

Best feat Clf2 epochs: XL ep25, XXL ep35, GPT ep25.

**Null result.** The feature prefix improves F1 by ≤ 0.2 pp across all three models, within noise. Surface-level structural features carry insufficient semantic signal to meaningfully improve B/C routing beyond what T5-Large already captures from the question text alone.

---

## Out of Scope / Future Work

The directory `classifier/future_work/` contains two scripts exploring semantic-embedding-based Clf2 improvements beyond the scope of this thesis:

- **`clf2_embedding_probe.py`** — Logistic regression on frozen all-MiniLM-L6-v2 sentence embeddings (384-dim). Achieved AUC 0.818 on B/C classification, well above the 0.75 go/no-go threshold.
- **`clf2_embedding_clf.py`** — MLP classifier on the same embeddings, producing a drop-in `dict_id_pred_results.json`. End-to-end evaluation showed a small regression (−0.5 to −0.7 pp F1), suggesting that while embeddings separate B/C better in isolation, the T5-Large Clf2 makes errors that are more benign for downstream QA accuracy.

These are discussed in the thesis Chapter 6 (Future Work) as a potential direction for improving Gate 2 with richer representations.

---

## Citation

```BibTex
@inproceedings{jeong2024adaptiverag,
  author       = {Soyeong Jeong and Jinheon Baek and Sukmin Cho and Sung Ju Hwang and Jong Park},
  title        = {Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity},
  booktitle    = {NAACL},
  year         = {2024},
  url          = {https://arxiv.org/abs/2403.14403}
}
```
