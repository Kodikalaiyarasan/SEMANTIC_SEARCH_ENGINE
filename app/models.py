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
        description="Arbitrary metadata attached to the document",
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


# ── Response Schemas ──────────────────────────────────────────


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str = Field(default="healthy")
    index_size: int = Field(..., description="Number of vectors currently in the FAISS index")
    model_name: str = Field(..., description="Loaded SentenceTransformer model identifier")


class SearchHit(BaseModel):
    """A single search result."""

    rank: int
    document_id: str
    text: str
    metadata: dict[str, Any]
    score: float = Field(..., description="Cosine similarity score (0-1)")


class SearchResponse(BaseModel):
    """Response payload for a semantic search query."""

    query: str
    top_k: int
    results: list[SearchHit]
    latency_ms: float = Field(..., description="End-to-end query latency in milliseconds")


class IndexResponse(BaseModel):
    """Response payload after indexing documents."""

    indexed: int = Field(..., description="Number of documents successfully indexed")
    total_index_size: int = Field(..., description="Total vectors in the index after ingestion")


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
