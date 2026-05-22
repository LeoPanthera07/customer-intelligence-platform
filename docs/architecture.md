# Customer Intelligence Platform — Architecture

## System Architecture Diagram

```mermaid
flowchart TD
    subgraph INGESTION["📥 Data Ingestion"]
        UCI["UCI Bank Marketing CSV\n41,188 rows"]
        CFPB["CFPB Complaints CSV\n5,000 records"]
        INGEST["ingest.py\n+ SHA-256 hash"]
        VALID["validate.py\nPandera schema\n+ 5 business rules"]
        UCI --> INGEST --> VALID
        CFPB --> INGEST
    end

    subgraph ML_LANE["🤖 ML Lane"]
        FEAT["features.py\nLabel encoding\nAge binning\nContact features\nStandard scaling"]
        TRAIN["train.py\nBaseline: LogisticRegression\nImproved: XGBoostClassifier\nstratified 80/20 split"]
        EVAL["evaluate.py\nRelative promotion gate\nROC-AUC, PR-AUC, F1\nBusiness interpretation"]
        MLFLOW[("MLflow\nSQLite backend\nmlruns/ artifacts\nRun IDs + tags")]
        FEAT --> TRAIN --> EVAL --> MLFLOW
        VALID --> FEAT
    end

    subgraph RAG_LANE["🧠 RAG Lane"]
        SCRUB["PII Scrubber\nEmails, Phones\nSSNs, Credit Cards"]
        EMBED["SentenceTransformer\nall-MiniLM-L6-v2\n384-dim embeddings"]
        FAISS[("FAISS Index\nIndexFlatL2\n5,000 vectors")]
        QENG["query_engine.py\nSimilarity search\nRefusal threshold=1.5\nEvidence-sufficiency note\nTheme grouping"]
        CFPB --> SCRUB --> EMBED --> FAISS --> QENG
    end

    subgraph SERVING["⚡ FastAPI Serving (port 8000)"]
        LOADER["model_loader.py\nLoads PROMOTED run\nPreprocessors from MLflow"]
        HEALTH["GET /health\nstatus + model version\n+ index version"]
        PREDICT["POST /predict\nprobability + band\n+ model version"]
        BATCH["POST /batch-score\nbatch list → band counts"]
        ASK["POST /ask-complaints\nQ&A + evidence IDs\n+ sufficiency note\n+ refusal logic"]
        INTEL["POST /customer-intel\nML band + complaint themes\n+ cited IDs"]
        METS["GET /metrics\nlatency, counts\nprediction dist + RAG stats"]
        MLFLOW --> LOADER --> PREDICT
        FAISS --> ASK
        LOADER --> INTEL
        FAISS --> INTEL
    end

    subgraph MONITORING["📊 Monitoring"]
        DRIFT["drift_analyzer.py\nEvidently DataDriftPreset\n+ TargetDriftPreset\nHTML + JSON reports"]
        RAGMON["rag_monitor.py\nhit-rate, refusal-rate\navg distance, latency\ntoken count"]
        RAGEVAL["rag_eval.py\n10-question harness\npass/fail + evidence IDs"]
        GATE["ci_quality_gate.py\nROC-AUC ≥ 0.80\nPR-AUC ≥ 0.35\ndrift share ≤ 85%"]
    end

    subgraph CI["🔄 CI/CD (GitHub Actions)"]
        GHA[".github/workflows/ci.yml\n1. Data ingestion\n2. Model training\n3. RAG indexing\n4. pytest suite\n5. Drift analysis\n6. Quality gate"]
    end

    subgraph DOCKER["🐳 Docker"]
        DF["Dockerfile\nmulti-stage\nbaked transformer weights"]
        DC["docker-compose.yml\ncip-serving-api:8000\ncip-mlflow-server:5000"]
    end

    VALID --> DRIFT
    PREDICT --> METS
    ASK --> METS
    INTEL --> METS
    GHA --> GATE
    GHA --> RAGEVAL
    GHA --> RAGMON
```

---

## Data Flow Summary

| Stage | Input | Output |
|-------|-------|--------|
| Ingestion | Raw URLs (UCI + CFPB API) | CSVs + SHA-256 hashes |
| Validation | Raw CSV | Pandera schema pass/fail |
| Feature Engineering | Raw bank columns | 23 encoded + scaled features |
| Training | Feature matrix | MLflow run IDs + artifacts |
| Promotion Gate | Two run IDs | PROMOTED / BLOCKED status |
| RAG Indexing | CFPB narratives (PII-scrubbed) | FAISS index + metadata.pkl |
| Serving | HTTP JSON requests | Predictions, RAG answers |
| Monitoring | Reference vs. current data | Drift HTML/JSON reports |
| CI Quality Gate | MLflow metrics + drift JSON | Exit 0 (deploy) or Exit 1 (block) |

---

## Component Responsibilities

| Module | Responsibility |
|--------|---------------|
| `src/data_pipeline/ingest.py` | Download datasets, compute SHA-256 |
| `src/data_pipeline/validate.py` | Pandera schema + 5+ business rules |
| `src/data_pipeline/features.py` | Reusable train/serve feature functions |
| `src/training/train.py` | Train LR + XGBoost, log to MLflow |
| `src/training/evaluate.py` | Relative promotion gate |
| `src/serving/model_loader.py` | Load PROMOTED run artifacts |
| `src/serving/app.py` | FastAPI — all 7 endpoints |
| `src/rag/index_builder.py` | PII scrub, embed, build FAISS |
| `src/rag/query_engine.py` | Semantic search, refusal, synthesis |
| `src/rag/rag_eval.py` | 10-question pass/fail eval harness |
| `monitoring/drift_analyzer.py` | Evidently ML drift report |
| `monitoring/rag_monitor.py` | RAG retrieval quality metrics |
| `tests/ci_quality_gate.py` | Automated production deployment gate |
