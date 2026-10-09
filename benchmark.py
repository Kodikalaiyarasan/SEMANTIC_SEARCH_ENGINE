#!/usr/bin/env python3
"""
Benchmark script for the Semantic Search Engine.

Fires 100 diverse queries at the /search endpoint, measures wall-clock
latency per request, and reports P50, P95, P99, mean, min, and max.

Usage
─────
    # Start the server first:
    #   uvicorn app.main:app --host 0.0.0.0 --port 8000
    #
    # Then run:
    python benchmark.py --url http://localhost:8000 --queries 100 --top-k 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import httpx

# ── Diverse benchmark queries covering the corpus topics ──────

BENCHMARK_QUERIES: list[str] = [
    "How do transformers process sequential data?",
    "What is FAISS and how does it work?",
    "Explain Docker containerization benefits",
    "How does FastAPI handle data validation?",
    "What is cosine similarity used for in NLP?",
    "How does Sentence-BERT generate sentence embeddings?",
    "Kubernetes container orchestration",
    "all-MiniLM-L6-v2 model characteristics",
    "Redis in-memory data store use cases",
    "Approximate nearest neighbor search algorithms",
    "PostgreSQL relational database features",
    "How does gradient descent optimize neural networks?",
    "RESTful API design principles",
    "Pydantic data validation in Python",
    "Attention mechanism in neural networks",
    "CI/CD pipeline automation benefits",
    "Word2Vec word embeddings technique",
    "Microservices architecture patterns",
    "TF-IDF text feature extraction",
    "PyTorch dynamic computational graphs",
    "Load balancing algorithms for web servers",
    "BERT pre-trained language model",
    "OAuth 2.0 authorization framework",
    "Batch normalization in deep learning",
    "GraphQL vs REST API comparison",
    "Elasticsearch full-text search engine",
    "LSTM and GRU recurrent networks",
    "Apache Kafka event streaming",
    "Cross-entropy loss function for classification",
    "Uvicorn ASGI server for Python",
    "Feature engineering best practices",
    "gRPC high-performance RPC framework",
    "PCA and t-SNE dimensionality reduction",
    "JWT token-based authentication",
    "Convolutional neural networks for images",
    "Prometheus monitoring and alerting",
    "Named entity recognition in text",
    "Message queue systems like RabbitMQ",
    "Regularization to prevent overfitting",
    "Nginx reverse proxy configuration",
    "Transfer learning with pre-trained models",
    "MongoDB document database",
    "Tokenization methods in NLP",
    "Terraform infrastructure as code",
    "Generative adversarial networks",
    "API rate limiting strategies",
    "Retrieval-augmented generation RAG",
    "Git version control system",
    "Hyperparameter tuning techniques",
    "WebSocket real-time communication",
    "Vector space model for information retrieval",
    "Celery distributed task queue Python",
    "Data augmentation for machine learning",
    "Service mesh with Istio sidecar proxies",
    "Knowledge distillation model compression",
    "SQLAlchemy ORM for Python databases",
    "Semantic search vs keyword search",
    "Helm Kubernetes package manager",
    "Variational autoencoders generative models",
    "Caching strategies Redis Memcached",
    "Sentence embeddings for semantic similarity",
    "Blue-green deployment strategy",
    "Self-supervised learning from unlabeled data",
    "OpenAPI Swagger specification FastAPI",
    "Contrastive learning representation",
    "Apache Airflow workflow orchestration",
    "Model quantization for inference speed",
    "Inverted index search engine data structure",
    "Multi-head attention transformer architecture",
    "Infrastructure as Code IaC tools",
    "Embedding space geometric similarity",
    "Circuit breaker pattern distributed systems",
    "Positional encoding in transformers",
    "Pytest Python testing framework",
    "Dense passage retrieval dual encoders",
    "Observability logging metrics tracing",
    "Learning rate scheduling strategies",
    "ASGI async Python web applications",
    "Cross-encoder re-ranking models",
    "ML model serving deployment frameworks",
    # Pad to ensure we always have enough for --queries=100
    "How to build a search engine with Python?",
    "What are dense vector representations?",
    "Explain the inner product similarity metric",
    "How does FAISS handle billion-scale vectors?",
    "What is the difference between HNSW and IVF?",
    "Real-time inference optimization techniques",
    "Sentence transformers pre-training data",
    "How to deploy ML models in production?",
    "Database indexing and query optimization",
    "Asynchronous programming in Python",
    "How to measure search engine relevance?",
    "Neural network weight initialization",
    "Distributed computing frameworks comparison",
    "Natural language understanding vs generation",
    "Container security best practices",
    "Text classification with deep learning",
    "Embedding similarity search applications",
    "Microservice inter-process communication",
    "Deep learning training on GPU clusters",
    "Python type hints and static analysis",
    "Production ML system monitoring",
]


def run_benchmark(
    base_url: str,
    num_queries: int,
    top_k: int,
) -> None:
    """Execute benchmark queries and report latency statistics."""

    # ── Preflight: health check ──
    print(f"\n{'='*60}")
    print(f"  Semantic Search Engine — Latency Benchmark")
    print(f"{'='*60}")
    print(f"  Target:   {base_url}")
    print(f"  Queries:  {num_queries}")
    print(f"  Top-K:    {top_k}")
    print(f"{'='*60}\n")

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        # Health check
        try:
            resp = client.get("/health")
            resp.raise_for_status()
            health = resp.json()
            print(f"✓ Service healthy — index size: {health['index_size']} vectors")
            print(f"  Model: {health['model_name']}\n")
        except httpx.HTTPError as exc:
            print(f"✗ Health check failed: {exc}")
            sys.exit(1)

        if health["index_size"] == 0:
            print("✗ Index is empty. Index documents before benchmarking.")
            sys.exit(1)

        # ── Warm-up (1 query, discarded) ──
        print("Warming up model (1 query)...")
        client.post("/search", json={"query": "warm-up query", "top_k": top_k})
        print("Warm-up complete.\n")

        # ── Benchmark loop ──
        queries = (BENCHMARK_QUERIES * ((num_queries // len(BENCHMARK_QUERIES)) + 1))[
            :num_queries
        ]
        latencies_client: list[float] = []
        latencies_server: list[float] = []

        print(f"Running {num_queries} queries...")
        for i, query in enumerate(queries):
            t0 = time.perf_counter()
            resp = client.post("/search", json={"query": query, "top_k": top_k})
            client_ms = (time.perf_counter() - t0) * 1_000
            resp.raise_for_status()
            data = resp.json()

            latencies_client.append(client_ms)
            latencies_server.append(data["latency_ms"])

            if (i + 1) % 25 == 0:
                print(f"  ... completed {i + 1}/{num_queries}")

    # ── Report ──
    print(f"\n{'─'*60}")
    print("  RESULTS  (all values in milliseconds)")
    print(f"{'─'*60}")

    for label, lat in [
        ("Server-side (encode+search)", latencies_server),
        ("Client-side (end-to-end)", latencies_client),
    ]:
        p50 = statistics.median(lat)
        p95 = sorted(lat)[int(len(lat) * 0.95)]
        p99 = sorted(lat)[int(len(lat) * 0.99)]
        mean = statistics.mean(lat)
        print(f"\n  {label}:")
        print(f"    Min:  {min(lat):>8.2f} ms")
        print(f"    Mean: {mean:>8.2f} ms")
        print(f"    P50:  {p50:>8.2f} ms")
        print(f"    P95:  {p95:>8.2f} ms")
        print(f"    P99:  {p99:>8.2f} ms")
        print(f"    Max:  {max(lat):>8.2f} ms")

    # ── Pass / Fail ──
    server_p99 = sorted(latencies_server)[int(len(latencies_server) * 0.99)]
    target = 50.0
    status = "PASS ✓" if server_p99 < target else "FAIL ✗"
    print(f"\n{'─'*60}")
    print(f"  Server P99 vs Target (<{target:.0f} ms): {server_p99:.2f} ms → {status}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the Semantic Search Engine API"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:9093",
        help="Base URL of the search service (default: http://localhost:9093)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of queries to run (default: 100)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K results per query (default: 5)",
    )
    args = parser.parse_args()
    run_benchmark(base_url=args.url, num_queries=args.queries, top_k=args.top_k)
