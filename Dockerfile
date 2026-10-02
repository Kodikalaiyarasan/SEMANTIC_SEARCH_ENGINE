# ── Stage 1: Download model weights ──────────────────────────
FROM python:3.11-slim AS model-downloader

RUN pip install --no-cache-dir sentence-transformers==3.0.1 torch==2.3.1 transformers==4.42.4

# Download and cache model weights at build time
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/model_cache')"

# ── Stage 2: Production image ────────────────────────────────
FROM python:3.11-slim AS production

# System deps for numpy/faiss
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded model weights from stage 1
COPY --from=model-downloader /model_cache /home/appuser/.cache/torch/sentence_transformers

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY benchmark.py .

# Fix ownership
RUN chown -R appuser:appuser /app /home/appuser

USER appuser

# Environment
ENV PYTHONUNBUFFERED=1 \
    MODEL_PATH="sentence-transformers/all-MiniLM-L6-v2" \
    DATA_DIR="data" \
    AUTO_INDEX_FILE="corpus.json"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
