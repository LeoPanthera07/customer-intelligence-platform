# ── Stage 1: Build & Dependency Installation ─────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies needed for compiling extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake RAG Embedding model weights into the image for offline-readiness and instant startup
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Stage 2: Final Minimal Run Stage ─────────────────────────────
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy installed dependencies from the builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface

# Ensure local bin is in PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Copy project files
COPY src/ /app/src/
COPY data/ /app/data/
COPY monitoring/ /app/monitoring/
COPY mlflow.db /app/mlflow.db
COPY mlruns/ /app/mlruns/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI serving application with uvicorn
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
