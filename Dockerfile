# Single-stage lightweight build using CPU-only PyTorch (reduces image from ~8GB CUDA packages to <1GB)
FROM python:3.11-slim

# System deps for numpy/faiss
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# 1. Install lightweight CPU-only PyTorch first (~150MB instead of 4GB CUDA/NVIDIA blobs)
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir sentence-transformers==3.0.1 transformers==4.42.4 faiss-cpu==1.8.0 numpy==1.26.4 fastapi==0.111.1 "uvicorn[standard]==0.30.3" pydantic==2.8.2 httpx==0.27.0

# Copy application files
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

EXPOSE 9093

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:9093/health').raise_for_status()"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9093", "--workers", "1"]
