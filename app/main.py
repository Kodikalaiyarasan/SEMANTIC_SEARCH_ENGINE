"""
FastAPI application — Semantic Search Engine.

Supports:
• Choosing between the fine-tuned code search model and the base model
• Exact cosine similarity retrieval using FAISS IndexFlatIP
• Clean document store without metadata overhead
• Full document inspection endpoint (GET /documents)
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    DocumentItem,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    ModelInfo,
    ModelListResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SwitchModelRequest,
    SwitchModelResponse,
)
from app.search_engine import SemanticSearchEngine

# ── Logging ───────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration & Available Models ──────────────────────────

AVAILABLE_MODELS: dict[str, dict[str, str]] = {
    "fine_tuned": {
        "id": "fine_tuned",
        "name": "Fine-Tuned Code Search (Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search)",
        "path": os.getenv(
            "MODEL_PATH", "Kodikalaiyarasan/all-MiniLM-L6-v2-tech-code-search"
        ),
        "description": "Fine-tuned on Python CodeSearchNet for programming code & technical retrieval",
    },
    "base": {
        "id": "base",
        "name": "Base Model (sentence-transformers/all-MiniLM-L6-v2)",
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Original general-domain sentence embeddings model",
    },
}

DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
AUTO_INDEX_FILE: str = os.getenv("AUTO_INDEX_FILE", "corpus.json")


def _get_or_create_engine(app: FastAPI, model_id: str) -> SemanticSearchEngine:
    """Retrieve or initialize an engine instance for the requested model."""
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_id}'. Available: {list(AVAILABLE_MODELS.keys())}",
        )

    engines: dict[str, SemanticSearchEngine] = app.state.engines
    if model_id not in engines:
        model_info = AVAILABLE_MODELS[model_id]
        logger.info("Initializing search engine for model '%s' (%s)", model_id, model_info["path"])
        engine = SemanticSearchEngine(model_path=model_info["path"])

        # Index all existing documents into the new engine
        all_docs: list[dict[str, str]] = getattr(app.state, "all_documents", [])
        if all_docs:
            ids = [d["id"] for d in all_docs]
            texts = [d["text"] for d in all_docs]
            engine.add_documents(ids, texts)
            logger.info("Indexed %d existing documents into model '%s'", len(all_docs), model_id)

        engines[model_id] = engine

    return engines[model_id]


# ── Lifespan ──────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager (FastAPI ≥ 0.109).

    Startup:
        1. Initialize engines dictionary and document storage.
        2. Load corpus.json seed documents.
        3. Pre-load default fine-tuned engine and index seed documents.
    """
    logger.info("── Startup: initialising Semantic Search Engine ──")
    app.state.engines = {}
    app.state.active_model_id = "fine_tuned"
    app.state.all_documents = []

    # Load seed corpus
    corpus_path = DATA_DIR / AUTO_INDEX_FILE
    if corpus_path.exists():
        logger.info("Loading seed corpus from %s", corpus_path)
        with open(corpus_path, "r", encoding="utf-8") as fh:
            raw_docs = json.load(fh)
        app.state.all_documents = [
            {"id": str(d["id"]), "text": str(d["text"])} for d in raw_docs
        ]
        logger.info("Loaded %d documents from seed corpus", len(app.state.all_documents))

    # Initialize default model
    _get_or_create_engine(app, app.state.active_model_id)

    logger.info("── Startup complete ──")
    yield
    logger.info("── Shutdown: cleaning up ──")


# ── Application ───────────────────────────────────────────────

app = FastAPI(
    title="Semantic Search Engine",
    description=(
        "Dense-vector retrieval API with fine-tuned code search & base model selection, "
        "powered by SentenceTransformers and FAISS."
    ),
    version="1.1.0",
    lifespan=lifespan,
    responses={500: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────


def _get_active_engine(request: Request) -> SemanticSearchEngine:
    """Retrieve the currently active search engine."""
    active_id = getattr(request.app.state, "active_model_id", "fine_tuned")
    return _get_or_create_engine(request.app, active_id)


# ── Endpoints ─────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"],
    summary="Health check",
)
async def health(request: Request) -> HealthResponse:
    """Return service health status, loaded model, and current index size."""
    engine = _get_active_engine(request)
    return HealthResponse(
        status="healthy",
        index_size=engine.index_size,
        model_name=engine.model_name,
        model_id=request.app.state.active_model_id,
    )


@app.get(
    "/models",
    response_model=ModelListResponse,
    tags=["Model Management"],
    summary="List available models",
)
async def list_models(request: Request) -> ModelListResponse:
    """Return available models and the currently active model."""
    active_id = request.app.state.active_model_id
    active_info = AVAILABLE_MODELS.get(active_id, {})
    return ModelListResponse(
        active_model_id=active_id,
        active_model_name=active_info.get("name", active_id),
        available_models=[
            ModelInfo(
                id=m["id"],
                name=m["name"],
                path=m["path"],
                description=m["description"],
            )
            for m in AVAILABLE_MODELS.values()
        ],
    )


@app.post(
    "/model/switch",
    response_model=SwitchModelResponse,
    tags=["Model Management"],
    summary="Switch active model",
)
async def switch_model(request: Request, body: SwitchModelRequest) -> SwitchModelResponse:
    """Switch active embedding model between the fine-tuned and base model."""
    if body.model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_id '{body.model_id}'. Allowed: {list(AVAILABLE_MODELS.keys())}",
        )

    engine = _get_or_create_engine(request.app, body.model_id)
    request.app.state.active_model_id = body.model_id

    return SwitchModelResponse(
        status="success",
        active_model_id=body.model_id,
        active_model_name=AVAILABLE_MODELS[body.model_id]["name"],
        index_size=engine.index_size,
    )


@app.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["Corpus"],
    summary="List all indexed documents",
)
async def get_documents(request: Request) -> DocumentListResponse:
    """Return all documents currently in the corpus."""
    docs = getattr(request.app.state, "all_documents", [])
    return DocumentListResponse(total=len(docs), documents=[DocumentItem(**d) for d in docs])


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
    Ingest a batch of documents into the search engine.
    """
    ids = [doc.id for doc in body.documents]
    texts = [doc.text for doc in body.documents]

    # Save to global documents store
    for doc in body.documents:
        request.app.state.all_documents.append({"id": doc.id, "text": doc.text})

    # Add to all initialized engines
    engines: dict[str, SemanticSearchEngine] = request.app.state.engines
    count = 0
    for eng in engines.values():
        count = eng.add_documents(ids, texts)

    active_engine = _get_active_engine(request)
    return IndexResponse(indexed=len(ids), total_index_size=active_engine.index_size)


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
    """
    target_model_id = body.model_id or request.app.state.active_model_id
    engine = _get_or_create_engine(request.app, target_model_id)

    if engine.index_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Index is empty — index documents before searching.",
        )

    results, latency_ms = engine.search(query=body.query, top_k=body.top_k)

    return SearchResponse(
        query=body.query,
        top_k=body.top_k,
        model_name=engine.model_name,
        results=[SearchHit(**r) for r in results],
        latency_ms=round(latency_ms, 2),
    )
