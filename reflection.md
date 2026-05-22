# Reflection — Customer Intelligence Platform
### IIT Gandhinagar Week 13 Mini-Project

---

## 1. Why this model family and this threshold over alternatives?

We chose **XGBoost** (`XGBClassifier`) as the improved model over Random Forest and LightGBM because XGBoost provides the best balance of inference speed, calibrated probabilities, and native class-imbalance handling through `scale_pos_weight`. On 41k rows of tabular bank marketing data, gradient-boosted trees consistently outperform linear models and match neural networks without requiring GPU resources.

The promotion threshold of **PR-AUC ≥ 0.35** (relative: improved must beat baseline by ≥ 3 percentage points) was chosen because PR-AUC is the correct metric for imbalanced classification — only ~11% of customers in the UCI dataset subscribed, making ROC-AUC overly optimistic. The 3 pp relative margin forces a meaningful lift; a model that merely memorises the majority class cannot pass. XGBoost achieved a **+7.8 pp PR-AUC lift** over the logistic regression baseline (0.691 vs. 0.613), making the gating decision unambiguous.

---

## 2. What broke first when you tried to deploy, and what did you change?

The first deployment failure was an **`AttributeError: 'XGBClassifier' has no attribute 'n_classes_'`** when calling `predict_proba()` on an MLflow-deserialized XGBoost model. MLflow's `mlflow.xgboost.log_model()` serialises the Booster weights but does not always restore scikit-learn fit-time attributes like `n_classes_`.

**Fix:** In `model_loader.py` and `app.py`, we detect this condition at load time and patch the attribute:
```python
if not hasattr(model, "n_classes_"):
    model.n_classes_ = len(model.classes_) if hasattr(model, "classes_") else 2
```
And as a double-safety net in `_run_ml_inference()`, we fall back to raw Booster prediction via `model.get_booster()` if `predict_proba()` still raises.

The second deployment failure was the FastAPI test suite returning **503s** instead of 200s because the `TestClient` was instantiated at module import time, before the `startup` event fired. Fixed by wrapping the client in a `pytest.fixture(scope="module")` with a `with TestClient(app) as c: yield c` context manager, which guarantees the startup handler completes before any test runs.

---

## 3. Why your gate margin, and what fails if you tighten PR-AUC by another 2 points?

The margin is **+3 pp PR-AUC**, chosen because it represents a statistically meaningful improvement over the baseline on a ~8k-sample held-out test set, while being achievable with standard hyperparameter choices (no exotic tuning).

If we tighten the relative gate to **+9 pp** (3+6), the current XGBoost run still passes (it achieved +7.8 pp — barely under). At **+10 pp**, both models would fail without retraining. The consequence: the CI pipeline blocks deployment, the system falls back to the logistic regression baseline (which is still BLOCKED by the absolute PR-AUC ≥ 0.35 gate), and no model gets promoted — triggering a mandatory retraining run with a hyperparameter sweep.

This is the intended behaviour of a relative gate: it forces continuous improvement rather than accepting a "good enough once" model indefinitely.

---

## 4. Show one complaint answer your RAG got wrong or ungrounded. How did your eval catch it?

**Query:** `"pizza topping preferences and restaurant menu suggestions"` (Q10 in `rag_eval.py`)

The FAISS index still returns the _least-bad_ matches — in this case, complaints about food-delivery-related financial disputes (returned with a distance of ~1.6–1.8). Without a refusal gate, the engine would synthesize a misleading answer about banking disputes when the user asked about pizza.

**How it was caught:** The `ComplaintQueryEngine` checks the best match distance against a threshold of **1.5**. Since 1.6 > 1.5, the engine returns:
> *"REFUSED — The closest retrieved complaint has a similarity distance of 1.623, which exceeds the threshold of 1.5. No complaint in the corpus is sufficiently relevant to ground an answer."*

The RAG eval harness (`rag_eval.py`) marks Q10 as **PASS** only if `refused == True`. This was the deliberate adversarial test to validate the refusal logic.

**What slipped through (real failure):** Weak financial queries like *"financial services company bad experience"* retrieve results with distance ~1.1–1.3, below the threshold, and so the engine synthesises an answer. The answer cites valid complaint IDs but the complaint themes are generic and not necessarily relevant. The refusal threshold does not help here — this is a **retrieval precision** problem, not a retrieval recall problem. A production fix would add a second-stage re-ranking (cross-encoder) to filter out semantically weak matches even within the threshold.

---

## 5. If this went live to real customers tomorrow, name the one risk you did not fully close.

**PII leakage in RAG synthesis.** The regex-based PII scrubber in `index_builder.py` redacts emails, phone numbers, SSNs, and credit card numbers from indexed narratives. However, it does not redact:
- **Full names** ("My name is John Smith and Bank X...")
- **Street addresses** ("I live at 123 Main Street...")
- **Account numbers** that do not match standard patterns

If a complaint narrative contains these, they would be stored in `complaints_metadata.pkl` and could appear verbatim in the API response under `scrubbed_narrative`. A real customer or regulator viewing the API output could see another consumer's personal data, violating CFPB data use policies and potentially GDPR/CCPA.

**The gap:** NER-based PII detection (e.g., spaCy `en_core_web_sm` + `en_core_web_trf`) was not added due to time constraints and the local CPU inference cost.

---

## 6. What would a senior MLOps engineer criticize first in your repo?

The most likely first criticism: **in-memory telemetry that resets on every server restart.**

The `/metrics` endpoint accumulates request counts, latency totals, and prediction distribution in a Python `dict` (`_telemetry`). This means:
1. Every container restart or uvicorn reload loses all historical metrics.
2. Multi-worker deployments (`uvicorn --workers 4`) would have separate, non-aggregated counters per worker — the reported counts would be fractional and misleading.

A production-grade fix: push metrics to **Prometheus** (via `prometheus-fastapi-instrumentator`) with a Grafana dashboard, or at minimum write metrics to a SQLite/Redis store shared across restarts and workers.

The second criticism would likely be the **SQLite MLflow backend** — SQLite is not safe for concurrent writes from multiple training workers, and the `.db` file is committed to Git (a bad habit even with small size).

Third: the **absence of a `/readiness` and `/liveness` probe** pair. Kubernetes and cloud load balancers need both endpoints to distinguish between "container is starting" and "container is healthy but serving is broken". The current `/health` endpoint serves both purposes but does not distinguish between them.
