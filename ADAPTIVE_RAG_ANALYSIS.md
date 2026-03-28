# Adaptive-RAG Repository Analysis

## Overview

**Adaptive-RAG** is an implementation of the paper ["Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"](https://arxiv.org/pdf/2403.14403.pdf) (NAACL 2024).

The system is a **query-adaptive question-answering framework** that dynamically selects the most suitable strategy for retrieval-augmented LLMs based on input query complexity. Instead of using a one-size-fits-all approach, it chooses between:
- **No-retrieval methods** (for simple queries)
- **Single-step retrieval-augmented LLMs** (for moderately complex queries)
- **Iterative multi-step retrieval-augmented LLMs** (for complex multi-hop queries)

---

## Core Architecture

### 1. **Query Complexity Classifier**
- A **trained classifier** (smaller LM) predicts query complexity levels
- Automatically labeled using actual model predictions and dataset inductive biases
- Routes queries to the appropriate strategy

### 2. **Multi-Strategy System**
The system supports multiple reasoning approaches:
- **IRCoT (IR-based Chain-of-Thought)**: Iterative retrieval with reasoning
- **Direct QA**: Single-step question answering
- **CoT (Chain-of-Thought)**: Step-by-step reasoning

### 3. **Datasets Supported**
**Multi-hop (complex) datasets:**
- MuSiQue
- HotpotQA
- 2WikiMultiHopQA

**Single-hop (simple) datasets:**
- Natural Questions (NQ)
- TriviaQA
- SQuAD

---

## System Components

### Directory Structure

```
Adaptive-RAG/
├── commaqa/                    # Core inference/reasoning engine
│   ├── inference/              # Configurable inference pipeline
│   │   ├── configurable_inference.py    # Main inference orchestrator
│   │   ├── ircot.py                    # IRCoT strategy
│   │   ├── participant_execution.py    # Execution logic
│   │   └── prompt_reader.py            # Prompt/query processing
│   ├── dataset/                # Dataset handling
│   ├── models/                 # Model definitions
│   └── configs/                # Configuration schemas
│
├── retriever_server/           # FastAPI-based retrieval service
│   ├── serve.py               # REST API server
│   ├── elasticsearch_server.py # ES integration
│   ├── build_index.py         # Index building
│   └── unified_retriever.py   # Retrieval abstraction layer
│
├── classifier/                 # Query complexity classifier
│   └── run_classifier.py      # Classifier training/inference
│
├── base_configs/              # Jsonnet configuration templates
│   └── ircot_qa_*.jsonnet    # Strategy configurations
│
├── metrics/                    # Evaluation metrics
│   ├── drop_answer_em_f1.py   # BLEU-like metrics
│   └── support_em_f1.py       # Supporting evidence metrics
│
├── runner.py                  # High-level experiment runner
├── predict.py                 # Prediction execution wrapper
├── run.py                     # Main orchestration script
├── evaluate.py                # Evaluation pipeline
└── lib.py                     # Utility functions
```

### Key Files

**Entry Points:**
- **`runner.py`**: User-friendly CLI wrapper
  - Usage: `python runner.py {system} {model} {dataset} {command}`
  - Supports systems: `ircot`, `ircot_qa`, `oner`, `oner_qa`, `nor_qa`
  - Models: `flan-t5-xxl`, `flan-t5-xl`, `gpt`, `none`

- **`predict.py`**: Executes predictions on test sets
  - Loads config, sets up retriever/LLM servers
  - Calls commaqa inference engine
  - Handles multi-dataset evaluation

- **`run.py`**: Main orchestration
  - Instantiates configs from templates
  - Manages prediction → evaluation pipeline
  - Tracks experiment results

**Configuration:**
- **`.retriever_address.jsonnet`**: Elasticsearch server location
- **`.llm_server_address.jsonnet`**: LLM inference server location
- **`base_configs/*.jsonnet`**: Strategy templates
  - Define state machines for reasoning pipelines
  - Can be instantiated with different hyperparameters

**Example Configuration Structure (ircot_qa_flan_t5_xl_hotpotqa.jsonnet):**
```jsonnet
{
  "start_state": "step_by_step_bm25_retriever",
  "end_state": "[EOQ]",
  "models": {
    "step_by_step_bm25_retriever": {
      "name": "retrieve_and_reset_paragraphs",
      "next_model": "step_by_step_cot_reasoning_gen",
      "retrieval_type": "bm25",
      "retrieval_count": 6,
      // ... more parameters
    },
    // ... more state transitions
  }
}
```

---

## Execution Flow

### 1. **Setup Phase**
```bash
# Create environment
conda create -n adaptiverag python=3.8
pip install -r requirements.txt

# Setup Elasticsearch (retrieval backend)
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.2-linux-x86_64.tar.gz
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
cd elasticsearch-7.10.2
./bin/elasticsearch

# Start retriever server
uvicorn serve:app --port 8000 --app-dir retriever_server

# Download & prepare datasets
bash ./download/processed_data.sh
python retriever_server/build_index.py hotpotqa
```

### 2. **Configuration Generation**
```
runner.py arguments
    ↓
runner.py resolves to run.py command
    ↓
run.py loads base config (jsonnet file)
    ↓
run.py instantiates with hyperparameters
    ↓
Instantiated config saved to instantiated_configs/
```

### 3. **Prediction Phase**
```
predict.py
    ↓
Loads config + evaluation dataset (test/dev set)
    ↓
Sets up environment:
    - RETRIEVER_HOST/PORT (Elasticsearch)
    - LLM_SERVER_HOST/PORT (inference server)
    ↓
Calls commaqa.inference.configurable_inference
    ↓
State machine executes:
    [State 1: Retrieve] → [State 2: Reason] → ... → [State N: Output]
    ↓
Predictions saved to predictions/{set_name}/{experiment_name}/
```

### 4. **Evaluation Phase**
```
evaluate.py
    ↓
Reads predictions + ground truth
    ↓
Applies metrics:
    - Answer EM/F1 (for answer predictions)
    - Support EM/F1 (for supporting evidence)
    - Answer Support Recall (for full paragraphs)
    ↓
Metrics saved to predictions/{set_name}/{experiment_name}/evaluation_metrics_*.json
```

---

## Core Inference Engine (commaqa)

### Configurable Inference Pipeline
**File:** `commaqa/inference/configurable_inference.py`

The inference system uses a **state machine** architecture:
1. **State Transitions**: Each state produces output → triggers next state
2. **Model Execution**: Each state can execute different models/strategies
3. **Data Flow**: Questions → retrieval → reasoning → answers

### Example State Sequence (IRCoT):
```
START
  ↓
[step_by_step_bm25_retriever] ---retrieves paragraphs--→
  ↓
[step_by_step_cot_reasoning_gen] ---generates reasoning--→
  ↓
[check_if_can_extract_answer] ---extracts answer--→
  ↓
[answer_extractor] ---finalizes answer--→
  ↓
END [EOQ]
```

### Participant Models
Different participant types handle different tasks:
- **Retriever** (`retrieve_and_reset_paragraphs`): BM25/ES retrieval
- **Reasoner** (`step_by_step_cot_reasoning_gen`): CoT generation using LLM
- **Answer Extractor** (`answer_extractor`): Final answer extraction

---

## Retriever Server

**File:** `retriever_server/serve.py`

FastAPI application providing `/retrieve/` endpoint:
- **Retrieval Method**: Elasticsearch BM25 search
- **Request Format**: JSON with corpus name, query, max hits
- **Response**: Retrieved paragraphs + retrieval time

**Integration:**
- Uses **UnifiedRetriever** abstraction layer
- Currently supports Elasticsearch 7.10.2
- Parallelizing queries across multiple datasets

---

## Classifier Component

**File:** `classifier/run_classifier.py`

Trains a smaller LM to predict query complexity:
- **Input**: Query text
- **Output**: Complexity level (simple/moderate/complex)
- **Training**: Labels derived from actual model predictions
- **Usage**: Routes queries to appropriate strategy

---

## Data Flow

### Input Processing
```
Raw Question
    ↓
[prompt_reader.py] - Tokenization, truncation
    ↓
[ircot.py or similar] - Strategy selection
    ↓
Split into reasoning steps
```

### Output Processing
```
Generated text (possibly with CoT)
    ↓
[answer_extractor] - Extract final answer
    ↓
Normalize answer (lowercase, remove punctuation)
    ↓
Save to predictions JSON
```

---

## Dependencies

**Key Python Packages:**
- **torch** (≥2.0): Neural computation
- **transformers** (≥4.36.0): HuggingFace models (Flan-T5, GPT)
- **fastapi + uvicorn**: Web server for retriever
- **elasticsearch** (≥7.17, <8): Document retrieval
- **jsonnet**: Configuration templating
- **datasets**: Dataset handling
- **nltk**: Text processing

---

## Supported Models

- **Flan-T5 (XL/XXL)**: Instruction-tuned T5
- **GPT (via API)**: OpenAI GPT models

---

## Key Hyperparameters (in base configs)

```jsonnet
llm_retrieval_count           # LLM-based retrieval steps
bm25_retrieval_count = 6      # BM25 retrieval paragraphs
rc_context_type               # Context mode (gold/distractors)
distractor_count = 2          # Number of distractor paragraphs
rc_qa_type = "direct"         # QA type (direct/cot)
multi_step_show_titles        # Show paragraph titles
multi_step_show_paras         # Show full paragraphs
multi_step_show_cot           # Show reasoning chain
```

---

## Workflow Example

### Running a Complete Experiment
```bash
# 1. Start services
cd elasticsearch-7.10.2 && ./bin/elasticsearch &
uvicorn serve:app --port 8000 --app-dir retriever_server &

# 2. Run prediction
python runner.py ircot flan-t5-xl hotpotqa predict --sample_size 500

# 3. Evaluate
python runner.py ircot flan-t5-xl hotpotqa evaluate --sample_size 500

# 4. View results
cat predictions/dev_500/ircot_qa_flan_t5_xl_hotpotqa/evaluation_metrics_*.json

# 5. Cross-dataset evaluation
python runner.py ircot flan-t5-xl hotpotqa_to_2wikimultihopqa predict --sample_size 500
```

---

## Featured Implementations

The Adaptive-RAG framework is also featured in:
- **LlamaIndex**: Reference implementation
- **LangGraph**: LangChain's agentic framework
- **Cohere Notebooks**: ReAct agent pattern

---

## Recent Setup Modifications

The original project used Python 3.8 and conda. Recent setup documents indicate migration to:
- **Python 3.12** with venv (conda not required)
- **PyTorch 2.x** (modern CUDA support via pip)
- **Updated dependencies** for compatibility

---

## Summary

Adaptive-RAG is a sophisticated **retrieval-augmented QA system** that:
1. **Classifies query complexity** using a trained classifier
2. **Routes to appropriate strategy** (no-retrieval, single-step, multi-step)
3. **Executes reasoning pipeline** via state machine (IRCoT/CoT/Direct)
4. **Retrieves context** using Elasticsearch BM25
5. **Generates answers** using LLMs (Flan-T5 or GPT)
6. **Evaluates results** using multiple metrics (EM/F1, support recall)

The modular architecture allows experimentation with different models, datasets, and reasoning strategies through JSON configuration files.
