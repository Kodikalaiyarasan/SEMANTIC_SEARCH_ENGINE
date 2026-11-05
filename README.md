# 🔍 Semantic Search & Code Retrieval Engine

> **Dense Vector Retrieval** powered by a **Fine-Tuned SentenceTransformer** + **FAISS** with **sub-50ms** query latency

A production-grade semantic search microservice featuring a custom domain-adapted embedding model ([`Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search-v2`](https://huggingface.co/Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search-v2)) fine-tuned on CodeSearchNet and 100,000+ Python anchor-positive pairs. The engine bridges natural language technical queries to concrete code implementations and documentation, performing exact cosine similarity search via FAISS `IndexFlatIP`.

---

## 🚀 Key Highlights & What's New

- 🎯 **Fine-Tuned Code-Search Model**: Adapted `all-MiniLM-L6-v2` using `MultipleNegativesRankingLoss` on Python technical pairs, dramatically enhancing top-k retrieval for colloquial developer queries (e.g., *"How to resume a function from where it left in python"* ➔ `yield` generators).
- 🔄 **Dynamic Model Switching**: Hot-swap between the base zero-shot model and the fine-tuned checkpoint at runtime via API or UI dropdown to compare performance in real time.
- ⚡ **Sub-50ms Inference**: Dense vector representation with L2-normalized unit embeddings and FAISS `IndexFlatIP` inner-product search.
- 📊 **100k Record Dataset & Training Pipeline**: Includes automated dataset generation scripts (`generate_100k_dataset.py`) and an end-to-end training pipeline (`train_fine_tune.py`).
- 🖥️ **Full-Stack UI**: Interactive React frontend with 3 dedicated tabs (Search, Add Documents, View Corpus), dynamic model switcher, and real-time similarity metrics.

---

## 🧠 Fine-Tuning: From General Text to Code Semantics

### The Problem
General-purpose embedding models like baseline `all-MiniLM-L6-v2` excel at conversational English but struggle with technical programming syntax and vocabulary mismatch. A query such as *"How to pause and resume a function"* often retrieves unrelated generic functions (e.g., retry decorators, stack classes) rather than Python `yield` generator constructs.

### The Solution: Contrastive Domain Adaptation
We fine-tuned `all-MiniLM-L6-v2` into `Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search-v2`:
- **Architecture**: 6-layer Transformer, 384-dimensional dense embeddings, 33M parameters (maintaining zero latency overhead).
- **Loss Function**: `MultipleNegativesRankingLoss` (MNRL). For each `(anchor, positive)` pair in a batch, all other $N-1$ positives act as hard in-batch negatives.
- **Dataset**: Trained on CodeSearchNet (Python docstrings $\leftrightarrow$ implementations) enriched with a curated 100,000 anchor-positive dataset covering Python generators, dunder methods, concurrency (`asyncio`, `threading`), decorators, data structures, and algorithms.

```
Query (Anchor): "How to resume a function from where it left in python"
                         │
        [Fine-Tuned Transformer Encoder]
                         ▼
Dense Query Vector (384-dim) ── Inner Product ──▶ Top Match: def gen(): yield ...
```

---

## 📐 System Architecture

```
┌─────────────────────────────────┐
│       Client / Web UI           │
│  (React 18 / Tailwind / Nginx)  │
└────────────────┬────────────────┘
                 │ HTTP (port 9092 ➔ 9093)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Microservice (Uvicorn)              │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Dynamic Model Manager                  │   │
│   │  • Base: sentence-transformers/all-MiniLM-L6-v2     │   │
│   │  • Fine-Tuned: all-MiniLM-L6-v2-tech-code-search-v2 │   │
│   └──────────────────────────┬──────────────────────────┘   │
│                              │ Embeddings (384-dim)         │
│                              ▼                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │             FAISS Dense Vector Index                │   │
│   │     IndexFlatIP (Exact Cosine Similarity via IP)    │   │
│   └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **L2-Normalized Embeddings** | Inner Product on unit vectors mathematically equals exact Cosine Similarity ($\hat{e}_1 \cdot \hat{e}_2 = \cos\theta$). |
| **`IndexFlatIP`** | Brute-force exact search with zero approximation error for corpora $<100\text{K}$ vectors. |
| **Model Hot-Swapping** | Re-indexes corpus in memory on model change without server restarts. |
| **Multi-Stage Docker** | Model weights pre-cached in image layers to eliminate cold-start download delays. |

---

## 📁 Project Structure

```
NLP/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app, lifespan, model-switch & search routes
│   ├── models.py                   # Pydantic v2 schemas (Search, Index, SwitchModel, etc.)
│   └── search_engine.py             # SentenceTransformer + FAISS core logic
├── data/
│   ├── corpus.json                 # Clean document corpus (concepts + code implementations)
│   ├── python_code_search_500.json # 601 seed anchor-positive pairs
│   └── python_code_search_100k.json# 100,000 anchor-positive pairs for fine-tuning
├── frontend/
│   ├── Dockerfile                  # Nginx-based container for React UI (port 9092)
│   ├── nginx.conf                  # Nginx proxy config
│   ├── standalone.html             # Self-contained React 18 frontend (CDN-based)
│   ├── serve_ui.py                 # Local Python HTTP server + API reverse proxy
│   └── src/                        # Vite/React source components
├── scripts/
│   ├── generate_dataset.py         # Generates initial 600+ pairs
│   └── generate_100k_dataset.py    # Generates 100k diverse anchor-positive dataset
├── train_fine_tune.py              # PyTorch + SentenceTransformer fine-tuning script
├── benchmark.py                    # Latency benchmarking script (P50/P95/P99)
├── Dockerfile                      # Multi-stage backend container (port 9093)
├── docker-compose.yml              # Multi-container orchestration (Backend + Frontend)
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Option A: Run with Docker Compose (Recommended)

Build and run both the FastAPI backend and React frontend with a single command:

```bash
# Build and run containers in background
docker compose up -d --build

# Inspect logs
docker compose logs -f
```

- **Web UI**: [http://localhost:9092](http://localhost:9092)
- **FastAPI API**: [http://localhost:9093](http://localhost:9093)
- **Interactive Swagger Docs**: [http://localhost:9093/docs](http://localhost:9093/docs)

---

### Option B: Run Locally

#### 1. Setup Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

#### 2. Start the Backend API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 9093 --reload
```

#### 3. Start the Web UI
```bash
# Runs the local proxy server serving standalone.html on port 9092
python frontend/serve_ui.py
```
Open [http://localhost:9092](http://localhost:9092) in your browser.

---

## 📡 API Endpoints

### 1. `POST /search` — Semantic Search
Execute semantic search over the corpus. Optionally specify which model to query with.

```bash
curl -X POST http://localhost:9093/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to resume a function from where it left in python",
    "top_k": 3,
    "model_id": "Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search-v2"
  }'
```

**Response:**
```json
{
  "query": "How to resume a function from where it left in python",
  "top_k": 3,
  "model_used": "Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search-v2",
  "results": [
    {
      "rank": 1,
      "document_id": "code-001",
      "text": "def resume_execution():\n    yield 'Paused at 1'\n    yield 'Paused at 2'",
      "score": 0.8412
    }
  ],
  "latency_ms": 14.82
}
```

### 2. `GET /models` — List Available Models
```bash
curl http://localhost:9093/models
```

**Response:**
```json
{
  "active_model": "Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search-v2",
  "available_models": [
    {
      "id": "Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search-v2",
      "name": "Fine-Tuned Code Search (v2)",
      "is_active": true
    },
    {
      "id": "sentence-transformers/all-MiniLM-L6-v2",
      "name": "Vanilla all-MiniLM-L6-v2 (Base)",
      "is_active": false
    }
  ]
}
```

### 3. `POST /model/switch` — Hot-Swap Model
Switch the active model on the fly. Automatically re-encodes and updates the FAISS index.

```bash
curl -X POST http://localhost:9093/model/switch \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sentence-transformers/all-MiniLM-L6-v2"}'
```

### 4. `GET /documents` — View Indexed Corpus
```bash
curl http://localhost:9093/documents
```

### 5. `POST /index` — Add New Documents
```bash
curl -X POST http://localhost:9093/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "doc-custom-01",
        "text": "FastAPI uses Starlette for the web parts and Pydantic for the data parts."
      }
    ]
  }'
```

---

## 🏋️ Fine-Tuning Your Own Model

We provide a complete pipeline to generate datasets and fine-tune your own embedding model:

### 1. Generate 100,000 Python Anchor-Positive Pairs
```bash
python scripts/generate_100k_dataset.py
```
Outputs `data/python_code_search_100k.json` with 100k unique `{"anchor": "...", "positive": "..."}` pairs.

### 2. Run Fine-Tuning
```bash
# Requires PyTorch and sentence-transformers
pip install sentence-transformers datasets torch accelerate
python train_fine_tune.py
```
This trains using `MultipleNegativesRankingLoss`, evaluates on a holdout split, and saves the fine-tuned model checkpoint under `models/all-MiniLM-L6-v2-tech-code-search-v3`.

---

## 📊 Latency Benchmarking

Run the automated latency benchmark to verify sub-50ms query speeds:

```bash
python benchmark.py --url http://localhost:9093 --queries 100 --top-k 5
```

**Sample Benchmark Result:**
```
============================================================
  Semantic Search Engine — Latency Benchmark
============================================================
  Target:   http://localhost:9093
  Queries:  100
  Top-K:    5
============================================================

✓ Service healthy — index size: 100 vectors
  Model: Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search-v2

  RESULTS (Server-side latency):
    Min:      9.14 ms
    Mean:    15.21 ms
    P50:     14.30 ms
    P95:     23.18 ms
    P99:     31.42 ms
    Max:     38.70 ms

  Server P99 vs Target (<50 ms): 31.42 ms ➔ PASS ✓
```

---

## 🛠️ Tech Stack Summary

- **Embeddings & NLP**: `SentenceTransformers`, `PyTorch`, `HuggingFace Hub`
- **Fine-Tuned Checkpoint**: `Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search-v2`
- **Vector Search Engine**: `FAISS (IndexFlatIP)`
- **Backend Framework**: `FastAPI`, `Uvicorn`, `Pydantic v2`
- **Frontend UI**: `React 18`, `Tailwind CSS`, `Lucide Icons`
- **Containerization**: `Docker` multi-stage build, `Docker Compose`, `Nginx`
