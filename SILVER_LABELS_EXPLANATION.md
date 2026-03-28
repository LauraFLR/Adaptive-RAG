# Silver labels
    
    Here's the structure and what each part means:
    
    ```
    classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/
    ├── predict.json                    # Test set questions for final QA evaluation
    ├── binary/                         # Labels from dataset inductive bias
    │   ├── musique_train.json          #   Multi-hop dataset → label C (complex)
    │   ├── hotpotqa_train.json         #   Multi-hop dataset → label C
    │   ├── 2wikimultihopqa_train.json  #   Multi-hop dataset → label C
    │   ├── nq_train.json               #   Single-hop dataset → label B (simple)
    │   ├── trivia_train.json           #   Single-hop dataset → label B
    │   ├── squad_train.json            #   Single-hop dataset → label B
    │   └── total_data_train.json       #   All of the above combined
    ├── flan_t5_xl/
    │   ├── silver/
    │   │   ├── train.json              # Silver training labels (from dev predictions)
    │   │   └── valid.json              # Silver validation labels (from test predictions)
    │   └── binary_silver/
    │       └── train.json              # Silver + binary combined (final training set)
    ├── flan_t5_xxl/                    # Same structure as flan_t5_xl
    │   ├── silver/
    │   └── binary_silver/
    └── gpt/                            # Same structure as flan_t5_xl
        ├── silver/
        └── binary_silver/
    ```
    
    ### The three label types:
    
    - **`answer: "A"`** — question correctly answered with **no retrieval** (simplest)
    - **`answer: "B"`** — question correctly answered with **single-step retrieval**
    - **`answer: "C"`** — question correctly answered with **multi-step retrieval** (most complex)
    
    ### The two labeling strategies:
    
    - **Silver labels** — Empirical: based on which retrieval strategy *actually* got the answer right for a given LLM. Different per model (flan_t5_xl vs xxl vs gpt).
    - **Binary labels** — Heuristic: based on the dataset's inherent bias (multi-hop datasets like MuSiQue → C, single-hop datasets like NQ → B). Same for all models.
    
    ### What's used for training:
    
    `binary_silver/train.json` is the **final training set** — the concatenation of silver + binary labels. This is what the classifier training scripts (`run_large_train_xl.sh`, etc.) actually read.