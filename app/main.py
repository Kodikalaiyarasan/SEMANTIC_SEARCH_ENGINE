"""
FastAPI application — Semantic Search Engine.

Lifecycle
─────────
• The SentenceTransformer model and FAISS index are initialised **once**
  during the ``lifespan`` context manager and stored on ``app.state``.
• All endpoints read from ``app.state`` — zero per-request model reloads.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    ErrorResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SearchHit,
)
from app.search_engine import SemanticSearchEngine

# ── Logging ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────

MODEL_PATH: str = os.getenv(
    "MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"
)
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
AUTO_INDEX_FILE: str = os.getenv("AUTO_INDEX_FILE", "corpus.json")


# ── Lifespan ──────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager (FastAPI ≥ 0.109).

    Startup:
        1. Instantiate ``SemanticSearchEngine`` (loads model + creates index).
        2. If ``data/corpus.json`` exists, auto-index its contents so the
           service is immediately queryable after boot.
    Shutdown:
        Cleanup (currently a no-op; FAISS resources are freed with GC).
    """
    logger.info("── Startup: initialising Semantic Search Engine ──")
    engine = SemanticSearchEngine(model_path=MODEL_PATH)
    app.state.engine = engine

    # Auto-index seed corpus if available
    corpus_path = DATA_DIR / AUTO_INDEX_FILE
    if corpus_path.exists():
        logger.info("Auto-indexing seed corpus from %s", corpus_path)
        with open(corpus_path, "r", encoding="utf-8") as fh:
            docs = json.load(fh)
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metas = [d.get("metadata", {}) for d in docs]
        count = engine.add_documents(ids, texts, metas)
        logger.info("Auto-indexed %d documents from seed corpus", count)

    logger.info("── Startup complete ──")
    yield
    logger.info("── Shutdown: cleaning up ──")


# ── Application ───────────────────────────────────────────────

app = FastAPI(
    title="Semantic Search Engine",
    description=(
        "Dense-vector retrieval API powered by SentenceTransformers "
        "(all-MiniLM-L6-v2) and FAISS with exact cosine similarity."
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={500: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ────────────────────────────────────────────────────


def _get_engine(request: Request) -> SemanticSearchEngine:
    """Retrieve the engine singleton from application state."""
    engine: SemanticSearchEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not initialised")
    return engine


# ── Endpoints ─────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"],
    summary="Health check",
)
async def health(request: Request) -> HealthResponse:
    """Return service health status, loaded model, and current index size."""
    engine = _get_engine(request)
    return HealthResponse(
        status="healthy",
        index_size=engine.index_size,
        model_name=engine.model_name,
    )


@app.post(
    "/index",
    response_model=IndexResponse,
    tags=["Indexing"],
    summary="Index documents",
    status_code=201,
)
async def index_documents(
    request: Request,
    body: IndexRequest,
) -> IndexResponse:
    """
    Ingest a batch of documents.

    Each document is encoded via SentenceTransformers, L2-normalised,
    and inserted into the FAISS inner-product index.
    """
    engine = _get_engine(request)

    ids = [doc.id for doc in body.documents]
    texts = [doc.text for doc in body.documents]
    metas = [doc.metadata for doc in body.documents]

    count = engine.add_documents(ids, texts, metas)

    return IndexResponse(indexed=count, total_index_size=engine.index_size)


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Semantic search",
)
async def search(
    request: Request,
    body: SearchRequest,
) -> SearchResponse:
    """
    Run a top-k semantic similarity search against the indexed corpus.

    Returns ranked results with cosine similarity scores and end-to-end
    latency (model inference + FAISS retrieval) in milliseconds.
    """
    engine = _get_engine(request)

    if engine.index_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Index is empty — index documents before searching.",
        )

    results, latency_ms = engine.search(query=body.query, top_k=body.top_k)

    return SearchResponse(
        query=body.query,
        top_k=body.top_k,
        results=[SearchHit(**r) for r in results],
        latency_ms=round(latency_ms, 2),
    )
