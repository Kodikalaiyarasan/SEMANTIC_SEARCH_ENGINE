<<<<<<< HEAD
# SEMANTIC_SEARCH_ENGINE
NLP Project
=======
# 🔍 Semantic Search Engine

> **Dense Vector Retrieval** powered by **SentenceTransformers** + **FAISS** with **sub-50ms** query latency

A production-ready semantic search microservice that encodes documents and queries into dense 384-dimensional embeddings using the `all-MiniLM-L6-v2` model, then performs exact cosine similarity search via FAISS `IndexFlatIP`.

---

## 📐 Architecture

```
┌─────────────┐       ┌──────────────────────────────────────────────┐
│   Client     │──────▶│  FastAPI  (uvicorn ASGI)                     │
│  (httpx)     │◀──────│                                              │
└─────────────┘       │  ┌────────────────┐   ┌──────────────────┐   │
                      │  │ SentenceTransf. │   │  FAISS IndexFlatIP│   │
                      │  │ all-MiniLM-L6-v2│──▶│  (Cosine Sim.)   │   │
                      │  │ (384-dim)       │   │  80+ vectors     │   │
                      │  └────────────────┘   └──────────────────┘   │
                      └──────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **L2-normalized embeddings** | Inner Product on unit vectors = exact Cosine Similarity |
| **`IndexFlatIP`** (not `IndexFlatL2`) | Brute-force exact search — no approximation error for <100K vectors |
| **Lifespan handler** | Model + index initialized once at startup, shared across all requests |
| **Multi-stage Docker** | Model weights baked in at build time — zero cold-start downloads |

---

## 📁 Project Structure

```
NLP/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, lifespan, routes
│   ├── models.py             # Pydantic v2 request/response schemas
│   └── search_engine.py      # SentenceTransformer + FAISS core
├── data/
│   └── corpus.json            # 80 real-world technical passages
├── benchmark.py               # Latency benchmark (P50/P95/P99)
├── Dockerfile                 # Multi-stage, non-root, cached weights
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option A: Run Locally

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will:
- Load the `all-MiniLM-L6-v2` model (~80MB, downloaded once)
- Auto-index 80 documents from `data/corpus.json`
- Serve on `http://localhost:8000`

### Option B: Run with Docker

```bash
# Build (downloads model weights during build — ~5 min first time)
docker-compose build

# Start
docker-compose up -d

# Check logs
docker-compose logs -f semantic-search
```

---

## 📡 API Endpoints

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "index_size": 80,
  "model_name": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### `POST /index` — Index Documents

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "custom-001",
        "text": "Reinforcement learning trains agents through reward signals.",
        "metadata": {"topic": "RL"}
      }
    ]
  }'
```

```json
{
  "indexed": 1,
  "total_index_size": 81
}
```

### `POST /search` — Semantic Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How do neural networks learn from data?", "top_k": 3}'
```

```json
{
  "query": "How do neural networks learn from data?",
  "top_k": 3,
  "results": [
    {
      "rank": 1,
      "document_id": "doc-012",
      "text": "Gradient descent is an optimisation algorithm used to minimise ...",
      "metadata": {"category": "machine-learning", "source": "ml-textbook"},
      "score": 0.6842
    },
    ...
  ],
  "latency_ms": 12.34
}
```

### Interactive Docs

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📊 Run the Benchmark

```bash
# Make sure the server is running, then:
python benchmark.py --url http://localhost:8000 --queries 100 --top-k 5
```

**Expected output:**

```
============================================================
  Semantic Search Engine — Latency Benchmark
============================================================
  Target:   http://localhost:8000
  Queries:  100
  Top-K:    5
============================================================

✓ Service healthy — index size: 80 vectors
  Model: sentence-transformers/all-MiniLM-L6-v2

Running 100 queries...
  ... completed 25/100
  ... completed 50/100
  ... completed 75/100
  ... completed 100/100

────────────────────────────────────────────────────────────
  RESULTS  (all values in milliseconds)
────────────────────────────────────────────────────────────

  Server-side (encode+search):
    Min:      8.12 ms
    Mean:    14.56 ms
    P50:     13.82 ms
    P95:     22.41 ms
    P99:     28.73 ms
    Max:     35.19 ms

────────────────────────────────────────────────────────────
  Server P99 vs Target (<50 ms): 28.73 ms → PASS ✓
────────────────────────────────────────────────────────────
```

---

## 🧮 How Cosine Similarity Works Here

```
1. Encode text → embedding ∈ ℝ³⁸⁴
2. L2-normalize:  ê = e / ‖e‖₂       (so ‖ê‖₂ = 1)
3. Inner Product: ê₁ · ê₂ = cos(θ)   (for unit vectors)
4. FAISS IndexFlatIP stores unit vectors and computes IP = cosine sim
```

This means:
- **Score = 1.0** → identical meaning
- **Score = 0.0** → orthogonal (unrelated)
- **Score < 0.0** → semantically opposite (rare with trained embeddings)

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL_PATH` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID or local path |
| `DATA_DIR` | `data` | Directory containing the seed corpus |
| `AUTO_INDEX_FILE` | `corpus.json` | Filename to auto-index on startup |

---

## 🔧 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **NLP Model** | `all-MiniLM-L6-v2` (384-dim) | Sentence encoding |
| **Vector Index** | FAISS `IndexFlatIP` | Exact cosine similarity search |
| **Web Framework** | FastAPI + Pydantic v2 | Async API with validation |
| **Server** | Uvicorn (ASGI) | High-performance HTTP |
| **Runtime** | Python 3.11 | Modern Python features |
| **Container** | Docker (multi-stage) | Production deployment |

---

## 📜 License

MIT
>>>>>>> ea5ce0e (First Commit)
