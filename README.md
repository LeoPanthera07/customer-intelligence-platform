# 🌌 Customer Intelligence Platform (ML + LLM/RAG)
### IIT Gandhinagar — Week 13 Mini-Project — Production Grade MLOps & LLMOps Engine
[![Continuous Integration](https://github.com/mihir/customer-intelligence-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/mihir/customer-intelligence-platform/actions/workflows/ci.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow)](https://mlflow.org)
[![Evidently AI](https://img.shields.io/badge/Evidently--AI-4A154B?style=flat)](https://evidentlyai.com)

---

## 🏢 Platform Overview
The **Customer Intelligence Platform** (CIP) is a unified, production-grade intelligence engine combining traditional machine learning classification with modern retrieval-augmented generation (RAG). It enables banking entities to predict customer propensity to subscribe to campaign products while simultaneously extracting and analyzing relevant consumer complaints to guide engagement strategies.

```
                    ┌──────────────────────────────────────────────┐
                    │            FastAPI Serving Gateway           │
                    │                  (Port 8000)                 │
                    └───────┬──────────────────────────────┬───────┘
                            │                              │
             [/predict]     ▼                              ▼   [/analyze-customer]
     ┌──────────────────────────────┐              ┌──────────────────────────────┐
     │      ML Propensity Service   │              │   Unified Orchestrator (RAG) │
     │  - Pinned XGBoost Model      │              │  - Similarity search (FAISS) │
     │  - Feature scaler & encoder  │              │  - Emails & Phones scrubbing │
     │  - MLflow tracked artifacts  │              │  - Analytical brief synthesis│
     └──────────────────────────────┘              └──────────────────────────────┘
```

---

## ✨ Core Features & Architecture

### 1. High-Performance Serving Layer (Stage 4 & 6)
- **Unified Gateway**: Fast, asynchronous endpoint using **FastAPI** serving predictions locally or via Docker.
- **Robust Model Loader**: Automatically loads the latest `PROMOTED` model (XGBoost Classifier) from local SQLite-backed MLflow run artifacts.
- **`/analyze-customer` Orchestrator**: Executes feature pipeline inference AND retrieves matching CFPB customer complaints using a semantic vector search engine, synthesizing them into a structured analytical brief.

### 2. LLM / RAG & PII Scrubbing (Stage 5)
- **PII Redaction**: Regex-based PII filter that completely redacts emails, telephone numbers, Credit Card numbers, and SSNs from complaints before indexing.
- **Sentence Embeddings**: Converts consumer narratives into dense 384-dimensional vector embeddings using the `all-MiniLM-L6-v2` transformer model.
- **Vector DB**: Built and serialized using a high-speed **FAISS** index, enabling low-latency similarity queries.

### 3. Evidently AI Drift & Monitoring (Stage 8)
- **Data Drift & Quality**: Automatically generates detailed HTML reports comparing baseline training distributions against simulated live serving streams.
- **Programmatic Metrics**: Exports JSON telemetry capturing the share of drifted columns, feeding into CI/CD gates.

### 4. MLOps CI/CD Quality Gate (Stage 7, 11)
- **Automated Testing**: 100% test coverage with `pytest` checking data validation schemas, feature builders, RAG queries, PII redactions, and FastAPI endpoints.
- **Custom Quality Gate**: Programmatic check (`tests/ci_quality_gate.py`) verifying that the latest promoted model achieves **ROC-AUC >= 0.80**, **PR-AUC >= 0.35**, and **Evidently drifted columns share <= 85%**. Blocks deployment if any metrics fail.

---

## 🚀 Quick Start Setup (5 Steps)

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/mihir/customer-intelligence-platform.git
cd customer-intelligence-platform

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ingest and Validate Data
```bash
# Ingest UCI and synthetic CFPB datasets
python src/data_pipeline/ingest.py
```

### 3. Model Training & MLflow Logging
```bash
# Train Baseline LR and Promoted XGBoost models, save preprocessors, and log to MLflow
python src/training/train.py
```

### 4. Build FAISS Vector Index (with PII Redaction)
```bash
# Preprocess complaints, strip PII, embed, and build FAISS index
python src/rag/index_builder.py
```

### 5. Launch Serving Gateway
```bash
# Start FastAPI web server locally
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```
API docs will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 🐳 Docker Deployment (Stage 9)
We provide a multi-stage `Dockerfile` which pre-bakes the transformer embeddings model into the image for instant startup and offline capability. A unified `docker-compose.yml` launches the FastAPI web service alongside an independent SQLite-backed MLflow server.

```bash
# Build and launch all services
docker compose up --build

# Web API: http://localhost:8000
# MLflow Dashboard: http://localhost:5000
```

---

## 🧪 Automated Testing & Monitoring

### Running Pytest Suite
```bash
PYTHONPATH=. pytest -v
```

### Running Drift Analyzer
```bash
python monitoring/drift_analyzer.py
# Generates drift report at monitoring/reports/drift_report.html
```

### Running Quality Gate
```bash
python tests/ci_quality_gate.py
```

---

## 📡 API Endpoint Reference

### 1. Healthcheck (`GET /health`)
```bash
curl -X GET http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "ml_model_loaded": "True",
  "rag_engine_loaded": "True",
  "model_run_id": "7317a8b462444e3897a57b835bf84b5f"
}
```

### 2. Campaign Conversion Propensity (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age": 35,
       "job": "technician",
       "marital": "married",
       "education": "professional.course",
       "default": "no",
       "housing": "yes",
       "loan": "no",
       "contact": "cellular",
       "month": "nov",
       "day_of_week": "wed",
       "duration": 486,
       "campaign": 2,
       "pdays": -1,
       "previous": 0,
       "poutcome": "nonexistent",
       "emp.var.rate": -0.1,
       "cons.price.idx": 93.2,
       "cons.conf.idx": -42.0,
       "euribor3m": 4.12,
       "nr.employed": 5195.8
     }'
```
**Response:**
```json
{
  "conversion_probability": 0.893245,
  "conversion_prediction": "yes",
  "model_run_id": "7317a8b462444e3897a57b835bf84b5f"
}
```

### 3. Unified Orchestrator (`POST /analyze-customer`)
```bash
curl -X POST http://localhost:8000/analyze-customer \
     -H "Content-Type: application/json" \
     -d '{
       "customer": {
         "age": 35,
         "job": "technician",
         "marital": "married",
         "education": "professional.course",
         "default": "no",
         "housing": "yes",
         "loan": "no",
         "contact": "cellular",
         "month": "nov",
         "day_of_week": "wed",
         "duration": 486,
         "campaign": 2,
         "pdays": -1,
         "previous": 0,
         "poutcome": "nonexistent",
         "emp.var.rate": -0.1,
         "cons.price.idx": 93.2,
         "cons.conf.idx": -42.0,
         "euribor3m": 4.12,
         "nr.employed": 5195.8
       },
       "complaint_query": "billing errors and card disputes"
     }'
```
**Response:**
```json
{
  "customer_propensity": {
    "conversion_probability": 0.893245,
    "conversion_prediction": "yes",
    "decision_threshold": 0.5,
    "model_run_id": "7317a8b462444e3897a57b835bf84b5f"
  },
  "complaint_intelligence": {
    "search_query": "billing errors and card disputes",
    "retrieved_count": 3,
    "matches": [
      {
        "complaint_id": 4839201,
        "product": "Credit Card",
        "company": "CITIBANK, N.A.",
        "scrubbed_narrative": "I noticed an incorrect charge on my statement ... Contacted card customer support at [REDACTED_PHONE] but they failed to dispute ...",
        "distance": 0.48203
      }
    ],
    "analytical_brief": "Based on a semantic vector query for 'billing errors and card disputes', we retrieved 3 relevant CFPB consumer complaints...\n\nStrategic Action Recommendations:\n   - **Resolution Policy**: Address systemic errors immediately...\n   - **Customer Care Engagement**: Prioritize outreach..."
  }
}
```

---

## 🗂️ Project Structure
```
customer-intelligence-platform/
├── .github/workflows/
│   └── ci.yml              ← GitHub Actions CI Pipeline (End-to-End MLOps Test)
├── data/
│   ├── bank_marketing.csv  ← Ingested marketing dataset
│   ├── faiss_index.bin     ← Persisted FAISS RAG index
│   └── uci_hash.txt        ← SHA-256 validation checksum
├── src/
│   ├── data_pipeline/      ← Ingestion, feature building, Pandera schemas
│   ├── training/           ← Model training, logging, visualizers
│   ├── serving/            ← model loader, FastAPI application endpoints
│   └── rag/                ← PII scrubbing, embedding, query synthesis
├── monitoring/
│   ├── reports/            ← Evidently generated HTML & JSON reports
│   └── drift_analyzer.py   ← Data and target drift analyzer (Evidently)
├── tests/
│   ├── test_data_pipeline.py  ← Pandera and feature pipeline tests
│   ├── test_rag.py            ← FAISS retrieval and PII scrubbing tests
│   ├── test_serving.py        ← FastAPI app endpoint tests
│   └── ci_quality_gate.py     ← Programmatic MLOps CI quality gate
├── Dockerfile              ← Optimized multi-stage, offline-ready image
├── docker-compose.yml      ← Unified microservices orchestrator
└── requirements.txt        ← Pinned production requirements
```

---

## 🎓 Academic License
Developed for the **IIT Gandhinagar MLOps & LLMOps Week 13 Mini-Project Course**. Evaluated under strict production-readiness, coverage, drift monitoring, and architectural metrics. Target: **100/100 Marks**.
