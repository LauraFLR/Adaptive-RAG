# Classifier Training Guide (Using Pre-computed Data)

All QA prediction results and classifier datasets from the original paper are already provided. You do **not** need Elasticsearch, the retriever server, or the LLM server. This guide covers only what's needed to train, validate, and run the classifier.

---

## Step 0: Extract Pre-computed Archives

```bash
cd /root/laura/Adaptive-RAG

# QA predictions from all 3 strategies × 3 models × 6 datasets × 2 splits
tar xzf predictions.tar.gz

# Pre-built classifier training/validation/prediction data
tar xzf data.tar.gz
```

Both are already extracted in the current workspace.

---

## Step 1: Understand the Data

### Data Location

All classifier data lives under:  
```
classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/
```

### Label Scheme

The classifier predicts query complexity as one of three classes:

| Label | Meaning | Routed Strategy |
|-------|---------|-----------------|
| `A` | No retrieval needed | `nor_qa` (zero-step) |
| `B` | Single retrieval sufficient | `oner_qa` (single-step) |
| `C` | Multi-step retrieval needed | `ircot_qa` (multi-step) |

### Data Files

#### Training Data: `{model}/binary_silver/train.json`

This is the **final training set** — a merge of silver labels + binary labels.

| Variant | File | Records | A (zero) | B (single) | C (multi) |
|---------|------|---------|----------|------------|-----------|
| flan_t5_xl | `flan_t5_xl/binary_silver/train.json` | 3,692 | 424 | 1,871 | 1,397 |
| flan_t5_xxl | `flan_t5_xxl/binary_silver/train.json` | 3,809 | 511 | 1,903 | 1,395 |
| gpt | `gpt/binary_silver/train.json` | 3,817 | 1,013 | 1,475 | 1,329 |

Each record has: `id`, `question`, `answer` (A/B/C), `dataset_name`, and optionally `answer_description`.

This data is composed of:
- **Silver labels** (from `{model}/silver/train.json`): ~1,300 questions from the dev_500 split, labeled by which strategy correctly answered them. These are model-specific because different LLMs answer different questions correctly.
- **Binary labels** (from `binary/total_data_train.json`): 2,400 questions labeled by dataset type — single-hop datasets (NQ, TriviaQA, SQuAD) → `B`, multi-hop datasets (MuSiQue, HotpotQA, 2WikiMultiHopQA) → `C`. These are the same across all models. Note: binary labels never include `A` (that only comes from silver data).

When merging, silver labels take priority — any binary-labeled question whose ID already exists in silver data is removed.

#### Validation Data: `{model}/silver/valid.json`

Silver labels computed from the **test set** predictions. Used to select the best epoch checkpoint.

| Variant | File | Records | A | B | C |
|---------|------|---------|---|---|---|
| flan_t5_xl | `flan_t5_xl/silver/valid.json` | 1,350 | 439 | 691 | 220 |
| flan_t5_xxl | `flan_t5_xxl/silver/valid.json` | 1,415 | 519 | 701 | 195 |
| gpt | `gpt/silver/valid.json` | 1,431 | 1,038 | 272 | 121 |

#### Prediction Data: `predict.json`

3,000 test questions (500 per dataset × 6 datasets) with **no labels**. The trained classifier predicts complexity labels on this file.

| File | Records | Datasets |
|------|---------|----------|
| `predict.json` | 3,000 | musique (500), hotpotqa (500), 2wikimultihopqa (500), nq (500), trivia (500), squad (500) |

---

## Step 2: Train the Classifier

The classifier is a **`t5-large`** model fine-tuned as a seq2seq task (question → "A"/"B"/"C").

### Training Scripts

```bash
cd /root/laura/Adaptive-RAG/classifier
```

There is one script per LLM variant:

| Script | Uses silver labels from | Training data |
|--------|------------------------|---------------|
| `run/run_large_train_xl.sh` | flan_t5_xl | `flan_t5_xl/binary_silver/train.json` (3,692 records) |
| `run/run_large_train_xxl.sh` | flan_t5_xxl | `flan_t5_xxl/binary_silver/train.json` (3,809 records) |
| `run/run_large_train_gpt.sh` | gpt | `gpt/binary_silver/train.json` (3,817 records) |

### What Each Script Does

Each script loops over **epochs {15, 20, 25, 30, 35}** and for each epoch runs three stages:

**1. Train** — Fine-tune `t5-large` on the binary_silver training set:
```bash
python run_classifier.py \
    --model_name_or_path t5-large \
    --train_file ./data/musique_hotpot_wiki2_nq_tqa_sqd/{LLM_NAME}/binary_silver/train.json \
    --question_column question \
    --answer_column answer \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_train_batch_size 32 \
    --num_train_epochs {EPOCH} \
    --do_train \
    --output_dir {TRAIN_OUTPUT_DIR}
```

**2. Validate** — Evaluate on the silver validation set (to pick the best epoch):
```bash
python run_classifier.py \
    --model_name_or_path {TRAIN_OUTPUT_DIR} \
    --validation_file ./data/musique_hotpot_wiki2_nq_tqa_sqd/{LLM_NAME}/silver/valid.json \
    --do_eval \
    --output_dir {VALID_OUTPUT_DIR}
```

**3. Predict** — Run inference on the unlabeled test questions:
```bash
python run_classifier.py \
    --model_name_or_path {TRAIN_OUTPUT_DIR} \
    --validation_file ./data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json \
    --do_eval \
    --output_dir {PREDICT_OUTPUT_DIR}
```

### Output Structure

Outputs are saved under `classifier/outputs/`:
```
classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/{LLM_NAME}/epoch/{N}/{DATE}/
├── pytorch_model.bin          # Trained model checkpoint
├── config.json                # Model config
├── valid/
│   └── dict_id_pred_results.json   # Validation predictions + accuracy
└── predict/
    └── dict_id_pred_results.json   # Test set complexity predictions
```

### GPU Note

The scripts hardcode `CUDA_VISIBLE_DEVICES=7`. Change this to match your setup:
```bash
# Edit the GPU variable in the script, e.g.:
GPU=0
```

### Running a Single Variant

To train just one (e.g., flan_t5_xl):
```bash
cd /root/laura/Adaptive-RAG/classifier
bash ./run/run_large_train_xl.sh
```

---

## Step 3: Route QA Predictions Using Classifier Output

After training, identify your best checkpoint (by validation accuracy), then update the hardcoded path in the postprocessing script and run:

```bash
cd /root/laura/Adaptive-RAG

# Edit the classification_result_file path in this script to point to your best checkpoint's predict/dict_id_pred_results.json
python ./classifier/postprocess/predict_complexity_on_classification_results.py {model_name}
```

This script:
1. For each test question, reads the classifier's predicted label (A/B/C)
2. Picks the QA answer from the corresponding strategy's pre-computed test predictions
3. Saves the routed predictions to `predictions/classifier/.../{dataset}/{dataset}.json`
4. Prints per-dataset step counts (for efficiency analysis)

---

## Step 4: Evaluate Final QA Accuracy

```bash
cd /root/laura/Adaptive-RAG

# Edit base_pred_path in this script to point to your classifier output directory
python ./evaluate_final_acc.py
```

This computes EM, F1, and accuracy for each dataset by comparing the routed predictions against the ground truth in `processed_data/{dataset}/test_subsampled.jsonl`.

---

## Summary: Files You Need

| Purpose | File | Pre-computed? |
|---------|------|:---:|
| Training data (flan_t5_xl) | `classifier/data/.../flan_t5_xl/binary_silver/train.json` | Yes |
| Training data (flan_t5_xxl) | `classifier/data/.../flan_t5_xxl/binary_silver/train.json` | Yes |
| Training data (gpt) | `classifier/data/.../gpt/binary_silver/train.json` | Yes |
| Validation data (flan_t5_xl) | `classifier/data/.../flan_t5_xl/silver/valid.json` | Yes |
| Validation data (flan_t5_xxl) | `classifier/data/.../flan_t5_xxl/silver/valid.json` | Yes |
| Validation data (gpt) | `classifier/data/.../gpt/silver/valid.json` | Yes |
| Test prediction input | `classifier/data/.../predict.json` | Yes |
| QA predictions (for routing) | `predictions/test/...` (108 experiment dirs) | Yes |
| QA predictions (step counts) | `predictions/test/ircot_qa_*/..._chains.txt` | Yes |
| Ground truth (for eval) | `processed_data/{dataset}/test_subsampled.jsonl` | Yes |
| Pre-trained base model | `t5-large` (downloaded from HuggingFace) | Auto-download |

### What You Must Do Yourself

1. Extract the tarballs (`predictions.tar.gz`, `data.tar.gz`) — if not already done
2. Train the classifier (`bash ./run/run_large_train_xl.sh` etc.) — requires a GPU
3. Update hardcoded paths in postprocess script and `evaluate_final_acc.py`
4. Run routing + evaluation
