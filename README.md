# Customer Intelligence Platform

A production-minded mini-platform that combines a classical ML service for campaign conversion prediction with an LLM/RAG complaint assistant, wired together behind a single spine with tracking, CI/CD, monitoring, and an integration endpoint.[file:1]

## Prerequisites

- Python 3.10
- Docker and Docker Compose
- Git

## Quick start (from a fresh clone)

\`\`\bash
# 1. Clone the repository
git clone https://github.com/<your-username>/customer-intelligence-platform.git
cd customer-intelligence-platform

# 2. Create and activate a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy env template and edit as needed
cp .env.example .env

# 5. Ingest and validate data (UCI + CFPB samples)
python src/data_pipeline/ingest.py
python src/data_pipeline/validate.py
\`\`\

## Running the ML service

### Local (development)

\`\`\bash
uvicorn src.serving.serve:app --host 0.0.0.0 --port 8000 --reload
\`\`\

### Docker (production-like)

\`\`\bash
# Build image
docker build -t customer-intel-ml-api .

# Run container
docker run --rm -p 8000:8000 --env-file .env customer-intel-ml-api
\`\`\

## Running the RAG complaint service

Once the RAG scripts are implemented:

\`\`\bash
# Build the FAISS index (complaint narratives)
python src/rag/build_index.py

# Run the RAG API (local)
uvicorn src.rag.answer:app --host 0.0.0.0 --port 8001 --reload
\`\`\

## Running tests

\`\`\bash
pytest tests/ -v
\`\`\

## Generating monitoring reports

\`\`\bash
# ML drift report (Evidently)
python monitoring/ml_drift.py

# RAG monitoring metrics
python monitoring/rag_monitor.py
\`\`\

## Endpoint reference

| Service | Endpoint           | Method | Description                                                   |
|---------|--------------------|--------|---------------------------------------------------------------|
| ML      | \`/health\`        | GET    | Health check, model and index versions                        |
| ML      | \`/predict\`       | POST   | Single customer conversion prediction                         |
| ML      | \`/batch-score\`   | POST   | Batch scoring for multiple customers                          |
| ML      | \`/metrics\`       | GET    | Latency, request counts, error counts, prediction distribution|
| RAG     | \`/ask-complaints\`| POST   | Complaint intelligence Q&A with cited evidence IDs            |
| Spine   | \`/customer-intel\`| POST   | Combined ML band + complaint themes for a customer segment   |

Refer to \`docs/\` for architecture, decisions, and monitoring details once populated.
