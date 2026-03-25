# Setup Changes from README Defaults

This document describes the modifications made to the original README installation steps to run on an Ubuntu 24.04 VM with Python 3.12.

## Python Environment

The README specifies conda with Python 3.8:
```bash
conda create -n adaptiverag python=3.8
conda activate adaptiverag
```

**What we did instead:** Used Python 3.12 (pre-installed on the system) with `venv`, since conda was not available and Python 3.8 is EOL (since October 2024).

```bash
python3.12 -m venv /root/laura/adaptiverag
source /root/laura/adaptiverag/bin/activate
```

The venv was created at `/root/laura/adaptiverag/` (sibling to `Adaptive-RAG/`) to avoid polluting the git repo.

**Why this works:** The project's Python code has no 3.8-specific syntax or deprecated stdlib usage. All compatibility issues were limited to pinned dependency versions.

## requirements.txt

The README's separate torch install step was skipped:
```bash
# SKIPPED — no longer needed
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

PyTorch 2.x includes CUDA support from the default PyPI index, so it installs directly via `pip install -r requirements.txt`.

The following dependency versions were updated in `requirements.txt` for Python 3.12 compatibility:

| Package | Original | Updated | Reason |
|---|---|---|---|
| torch | `>=1.7,!=1.12.0,<2.0` | `>=2.0` | PyTorch 1.x has no Python 3.12 wheels |
| transformers | pinned git commit `@8637316e...` | `>=4.36.0` | Old commit may not build on 3.12; released versions are more reliable |
| accelerate | `==0.15.0` | `>=0.25.0` | Old version incompatible with Python 3.12 |
| protobuf | `==3.19.0` | `>=3.20.0` | 3.19.x lacks 3.12 wheels |
| spacy | `==3.4.1` | `>=3.7.0` | Better Python 3.12 support |
| typing_extensions | `<4.6.0` | *(unconstrained)* | Constraint was unnecessarily restrictive |
| elasticsearch | `==7.9.1` | `>=7.17,<8` | 7.9.1 used `np.float_`, removed in NumPy 2.0. 7.17.x fixes this while staying compatible with the ES 7.10 server |

Packages that remained unchanged: `jsonnet`, `sentencepiece`, `nltk`, `scipy`, `openai`, `diskcache`, `rapidfuzz`, `datasets`, `pandas`, `requests`, `tqdm`, `ftfy`, `ujson`, `fastapi`, `uvicorn[standard]`, `dill`, `base58`, `pygments`, `beautifulsoup4`, `blingfire`, `wget`, `black`, `ruff`, `jsonlines`.

### Installed versions

Key packages as installed:

- torch 2.11.0+cu130
- transformers 5.3.0
- spacy 3.8.13
- accelerate 1.13.0
- elasticsearch 7.17.13
- numpy 2.4.3
- scipy 1.17.1

## Elasticsearch

The README steps are:
```bash
tar -xzf elasticsearch-7.10.2-linux-x86_64.tar.gz
cd elasticsearch-7.10.2/
./bin/elasticsearch
```

**Issue:** Elasticsearch refuses to run as the root user, which is the default user on this VM.

**What we did:**
1. Extracted the tarball to `/root/laura/elasticsearch-7.10.2/`
2. Changed ownership to the `ubuntu` user: `chown -R ubuntu:ubuntu elasticsearch-7.10.2`
3. Made `/root` and `/root/laura` traversable: `chmod 755 /root /root/laura`
4. Started ES as the `ubuntu` user in daemon mode:
   ```bash
   su ubuntu -c '/root/laura/elasticsearch-7.10.2/bin/elasticsearch -d -p /root/laura/elasticsearch-7.10.2/es.pid'
   ```

**Managing the server:**
- Verify it's running: `curl localhost:9200`
- Stop: `kill $(cat /root/laura/elasticsearch-7.10.2/es.pid)`
- Start again: `su ubuntu -c '/root/laura/elasticsearch-7.10.2/bin/elasticsearch -d -p /root/laura/elasticsearch-7.10.2/es.pid'`

source /root/laura/adaptiverag/bin/activate
uvicorn serve:app --port 8000 --app-dir /root/laura/Adaptive-RAG/retriever_server
