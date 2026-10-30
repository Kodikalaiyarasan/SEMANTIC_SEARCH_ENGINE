"""
Pydantic v2 schemas for the Semantic Search Engine API.

All request/response payloads are validated and documented via these models,
which FastAPI auto-exposes in the OpenAPI spec.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Request Schemas ───────────────────────────────────────────


class DocumentPayload(BaseModel):
    """A single document to be indexed."""

    id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., min_length=1, description="Raw text content to embed")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to the document (optional)",
    )


class IndexRequest(BaseModel):
    """Batch of documents to ingest into the FAISS index."""

    documents: list[DocumentPayload] = Field(
        ..., min_length=1, description="List of documents to index"
    )


class SearchRequest(BaseModel):
    """Query payload for semantic search."""

    query: str = Field(..., min_length=1, description="Natural-language query")
    top_k: int = Field(
        default=5, ge=1, le=100, description="Number of results to return"
    )
    model_id: str | None = Field(
        default=None,
        description="Optional model ID to search with ('fine_tuned' or 'base')",
    )


class SwitchModelRequest(BaseModel):
    """Request to switch active model."""

    model_id: str = Field(..., description="Target model ID ('fine_tuned' or 'base')")


# ── Response Schemas ──────────────────────────────────────────


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str = Field(default="healthy")
    index_size: int = Field(..., description="Number of vectors currently in the FAISS index")
    model_name: str = Field(..., description="Loaded SentenceTransformer model identifier")
    model_id: str = Field(default="fine_tuned", description="Identifier of the active model")


class SearchHit(BaseModel):
    """A single search result."""

    rank: int
    document_id: str
    text: str
    score: float = Field(..., description="Cosine similarity score (0-1)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata (optional)"
    )


class SearchResponse(BaseModel):
    """Response payload for a semantic search query."""

    query: str
    top_k: int
    model_name: str = Field(..., description="Model used for this search query")
    results: list[SearchHit]
    latency_ms: float = Field(..., description="End-to-end query latency in milliseconds")


class IndexResponse(BaseModel):
    """Response payload after indexing documents."""

    indexed: int = Field(..., description="Number of documents successfully indexed")
    total_index_size: int = Field(..., description="Total vectors in the index after ingestion")


class DocumentItem(BaseModel):
    """A document item in the corpus."""

    id: str
    text: str


class DocumentListResponse(BaseModel):
    """List of all documents currently in the index."""

    total: int
    documents: list[DocumentItem]


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    name: str
    path: str
    description: str


class ModelListResponse(BaseModel):
    """List of available models and the currently active model."""

    active_model_id: str
    active_model_name: str
    available_models: list[ModelInfo]


class SwitchModelResponse(BaseModel):
    """Response after switching active model."""

    status: str = Field(default="success")
    active_model_id: str
    active_model_name: str
    index_size: int


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
