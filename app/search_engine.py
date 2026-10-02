"""
Semantic Search Engine — Core Vector Retrieval Module.

Encapsulates:
  • SentenceTransformer model loading & inference
  • L2 unit-normalization of embeddings → FAISS IndexFlatIP = exact cosine similarity
  • Document store for id ↔ text/metadata mapping
  • Top-k retrieval with similarity scores
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """In-memory record associated with a FAISS vector slot."""

    document_id: str
    text: str
    metadata: dict[str, Any]


class SemanticSearchEngine:
    """
    Dense-vector retrieval engine backed by SentenceTransformers + FAISS.

    Design decisions
    ────────────────
    • All embeddings are L2-normalised before insertion so that
      ``faiss.IndexFlatIP`` computes **exact** cosine similarity
      (inner product of unit vectors == cosine similarity).
    • The document store is an append-only list aligned by FAISS
      internal IDs (sequential integers starting at 0).
    • Thread-safety: FAISS ``IndexFlatIP`` is safe for concurrent
      reads; writes (``add``) are serialised by the GIL in CPython.
    """

    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_path: str | None = None) -> None:
        """
        Parameters
        ----------
        model_path
            Local path or Hugging Face hub identifier.  Falls back to
            ``MODEL_NAME`` when *None*.
        """
        resolved = model_path or self.MODEL_NAME
        logger.info("Loading SentenceTransformer model: %s", resolved)

        t0 = time.perf_counter()
        self._model = SentenceTransformer(resolved)
        load_ms = (time.perf_counter() - t0) * 1_000
        logger.info("Model loaded in %.1f ms", load_ms)

        # Embedding dimensionality (384 for all-MiniLM-L6-v2)
        self._dim: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]

        # FAISS inner-product index — with unit-normalised vectors this
        # is mathematically equivalent to cosine similarity.
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._dim)

        # Parallel document store (positional alignment with FAISS IDs)
        self._documents: list[DocumentRecord] = []

        logger.info(
            "FAISS IndexFlatIP initialised — dim=%d, metric=InnerProduct (cosine)",
            self._dim,
        )

    # ── Properties ────────────────────────────────────────────

    @property
    def index_size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @property
    def embedding_dim(self) -> int:
        return self._dim

    # ── Encoding ──────────────────────────────────────────────

    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts into L2-unit-normalised embeddings.

        Returns
        -------
        np.ndarray
            Shape ``(len(texts), dim)`` with ``dtype=float32``.
        """
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalise inside ST
        )
        # Belt-and-suspenders: re-normalise in float32
        faiss.normalize_L2(embeddings)
        return embeddings.astype(np.float32)

    # ── Indexing ──────────────────────────────────────────────

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """
        Encode and index a batch of documents.

        Parameters
        ----------
        ids
            Unique document identifiers.
        texts
            Raw document texts.
        metadatas
            Per-document metadata dicts.

        Returns
        -------
        int
            Number of documents successfully indexed.
        """
        if not texts:
            return 0

        embeddings = self._encode(texts)
        self._index.add(embeddings)

        for doc_id, text, meta in zip(ids, texts, metadatas):
            self._documents.append(
                DocumentRecord(document_id=doc_id, text=text, metadata=meta)
            )

        logger.info(
            "Indexed %d documents — total index size: %d",
            len(texts),
            self.index_size,
        )
        return len(texts)

    # ── Search ────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Run a top-k cosine similarity search.

        Parameters
        ----------
        query
            Natural-language query string.
        top_k
            Number of nearest neighbours to retrieve.

        Returns
        -------
        tuple[list[dict], float]
            ``(results, latency_ms)`` where each result dict contains
            ``rank``, ``document_id``, ``text``, ``metadata``, ``score``.
        """
        t0 = time.perf_counter()

        # Clamp top_k to available documents
        effective_k = min(top_k, self.index_size)
        if effective_k == 0:
            return [], 0.0

        query_vec = self._encode([query])
        scores, indices = self._index.search(query_vec, effective_k)

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]), start=1
        ):
            if idx == -1:
                continue  # FAISS sentinel for missing results
            doc = self._documents[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "document_id": doc.document_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
            )

        latency_ms = (time.perf_counter() - t0) * 1_000
        logger.debug("Search completed in %.2f ms — %d hits", latency_ms, len(results))
        return results, latency_ms
