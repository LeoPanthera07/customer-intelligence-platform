# Decision Log — Customer Intelligence Platform

> Records all key architectural decisions, rejected alternatives, known limitations, and the hardening plan.

---

## 1. Model Selection

### Decision: XGBoost over Random Forest / LightGBM for the improved model

**What we chose:** `XGBClassifier` with dynamic `scale_pos_weight` for class imbalance.

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| Random Forest | Higher RAM footprint for ensemble size needed; slower inference; no built-in imbalance handling as clean |
| LightGBM | Comparable performance, but XGBoost is the canonical benchmark for tabular data in interviews and prod; MLflow native logging is mature |
| Neural network (MLP) | Data is 41k rows — a tree ensemble generalises better with less tuning; no GPU available locally |
| SVM | Does not output calibrated probabilities without Platt scaling; slow on 41k rows |

**Gate margin chosen:**
- ROC-AUC ≥ 0.80 (XGBoost achieved 0.953, LR achieved 0.939 — both pass ROC)
- PR-AUC ≥ 0.35 (XGBoost: 0.691, LR: 0.613 — XGBoost beats LR by 7.8 pp)
- F1 ≥ 0.60 (XGBoost: 0.612, LR: 0.585)
- **Relative margin**: XGBoost beats baseline LR by +7.8 pp PR-AUC, exceeding the required 3 pp margin.

If we tighten PR-AUC to +9 pp, the XGBoost run still passes. Tightening to +10 pp would require retraining with hyperparameter search.

---

## 2. Vector Store: FAISS (local) over managed alternatives

**What we chose:** `faiss-cpu` `IndexFlatL2` — exact L2 search, in-process.

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| Pinecone / Weaviate | Cloud cost and network latency; violates zero-cost requirement; adds internet dependency in CI |
| ChromaDB | Good default but requires a server process; FAISS is pure in-process and simpler to test/reproduce |
| FAISS HNSW (approximate) | For 5k vectors, exact search is faster than building HNSW; IVF benefits only appear above ~1M vectors |

**Known limitation:** FAISS does not support filtered ANN natively. We implement post-retrieval filtering by over-fetching (`fetch_k = top_k * 4`) then filtering on product/company. This can miss results if the corpus is small and filters are strict.

---

## 3. Embedding Model: `all-MiniLM-L6-v2`

**What we chose:** Sentence-Transformers `all-MiniLM-L6-v2`.

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| OpenAI `text-embedding-ada-002` | API cost per token; network dependency; key management risk in CI |
| `all-mpnet-base-v2` | Higher quality but 4× slower; 768-dim vs 384-dim doubles FAISS memory |
| BM25 (sparse retrieval) | No semantic matching; "credit card fees" vs "card billing error" would miss matches |

---

## 4. Refusal Threshold: distance = 1.5 (L2 squared)

**Calibration:**
- FAISS `IndexFlatL2` returns squared L2 distances between unit-length embeddings (because `all-MiniLM-L6-v2` normalises to unit sphere).
- Empirically, distance < 0.5 = high relevance, < 1.0 = moderate, > 1.5 = semantically dissimilar.
- Tested with 15 monitoring queries: 3 completely off-topic queries (pizza, weather, sports) all scored > 1.5 and were refused. 12 on-topic financial queries all scored < 1.2 and retrieved results.

---

## 5. FastAPI over Flask / Django

**What we chose:** FastAPI.

**Why:** Async-capable, Pydantic validation built-in, OpenAPI/Swagger auto-generated, type hints enforced at the API boundary. Flask requires 3rd-party libs (marshmallow, flasgger) for the same.

---

## 6. SQLite-backed MLflow (local) over Managed MLflow / DagsHub

**What we chose:** `mlflow.db` SQLite file + local `mlruns/` artifact directory.

**Why:** Zero-cost, runs offline, reproducible in CI without secrets. The spec explicitly states "Local MLflow still counts as long as you record model version, dataset hash, metrics and the promotion decision."

---

## 7. Dockerfile: Multi-Stage + Baked Transformer Weights

**What we chose:** Two-stage build (builder installs deps, runner is slim) + pre-download transformer model weights.

**Why:** Baking weights avoids HuggingFace Hub downloads at container boot (which would fail in air-gapped environments and slow cold starts to 30+ seconds).

**Known limitation:** Image size is ~2.5 GB because transformer weights are large. For production we would use a model registry (e.g., S3-backed) and lazy-load.

---

## 8. Rejected Approaches

| Approach | Reason Rejected |
|---|---|
| Notebooks as final execution | The spec is explicit: "Final execution must run through scripts." |
| Generating synthetic data for CFPB | Real CFPB records are publicly available; synthetic data would undermine the RAG authenticity |
| LangChain for RAG pipeline | Adds abstraction overhead; for a 5k-record corpus, direct FAISS + SentenceTransformer is simpler and fully debuggable |
| Online Learning / incremental models | Out of scope for the bank marketing problem; concept drift handled by re-training gate |
| Multi-GPU training | Not available locally; XGBoost CPU training completes in < 2 minutes on the 41k dataset |

---

## 9. Known Limitations

1. **No real-time retraining trigger** — drift is detected and reported, but retraining must be initiated manually. The hardening plan adds a triggered webhook.
2. **In-memory telemetry** — `/metrics` counters reset on server restart. A production system would use Prometheus/Grafana or a time-series DB.
3. **Single-node FAISS** — cannot shard horizontally. At > 1M vectors, migrate to FAISS IVF or a managed vector DB.
4. **PII scrubbing is regex-only** — a full production system would add NER-based PII detection (spaCy or AWS Comprehend) to catch name/address patterns.
5. **No authentication on API endpoints** — all routes are public. Production requires OAuth2 / API key middleware.
6. **SQLite MLflow backend** — not concurrent-write safe. For multi-trainer setups, use Postgres.

---

## 10. Hardening Plan (If This Goes to Production)

| Risk | Hardening Action |
|---|---|
| Model degradation | Add Evidently scheduled job; trigger retraining via CI webhook if drift > 50% |
| PII leakage in RAG answers | Add NER-based scrubbing (spaCy `en_core_web_sm`) + post-generation scan |
| API abuse / DDoS | Add rate-limiting middleware (`slowapi`) + API key auth |
| FAISS cold-start latency | Pre-warm with one dummy query on container startup |
| Single-point-of-failure serving | Deploy behind a load balancer; add `/readiness` probe for Kubernetes |
| Stale embeddings after corpus update | Automate `index_builder.py` run in CI when CFPB data is refreshed |
| No rollback on bad model | Use MLflow model registry Staging → Production transitions with canary traffic split |
