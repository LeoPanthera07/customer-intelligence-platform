"""
Serving API — FastAPI application serving:
  • GET  /health           — system health + version info
  • POST /predict          — ML campaign conversion propensity
  • POST /batch-score      — batch ML scoring for a list of customers
  • POST /ask-complaints   — RAG complaint Q&A with evidence IDs, refusal logic
  • POST /customer-intel   — unified: ML conversion band + complaint themes
  • POST /analyze-customer — alias for /customer-intel (backwards compat.)
  • GET  /metrics          — API telemetry: latency, counts, RAG stats
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_pipeline.features import build_features
from src.rag.query_engine import ComplaintQueryEngine
from src.serving.model_loader import load_promoted_model

# ── FastAPI App ──────────────────────────────────────────────────
app = FastAPI(
    title="Customer Intelligence Platform API",
    description=(
        "Unified ML Propensity Scoring + LLM/RAG Complaint Analytics Engine. "
        "Built for IIT Gandhinagar Week-13 Mini-Project."
    ),
    version="2.0.0",
)


# ── Pydantic Schemas ─────────────────────────────────────────────
class CustomerInput(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")

    model_config = {"populate_by_name": True}


class BatchScoreRequest(BaseModel):
    customers: List[CustomerInput]


class AskComplaintsRequest(BaseModel):
    question: str
    product: Optional[str] = None
    company: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)


class CustomerIntelRequest(BaseModel):
    customer: CustomerInput
    product_filter: Optional[str] = None
    issue_filter: Optional[str] = None
    company_filter: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=10)


class CustomerAnalysisRequest(BaseModel):
    """Backwards-compatible alias for /customer-intel."""
    customer: CustomerInput
    complaint_query: str


# ── Global State ─────────────────────────────────────────────────
MODEL: Any = None
ENCODERS: Dict[str, Any] = {}
SCALER: Any = None
MODEL_RUN_ID: str = ""
RAG_ENGINE: Optional[ComplaintQueryEngine] = None
INDEX_VERSION: str = "v1"

# In-memory telemetry counters
_telemetry: Dict[str, Any] = defaultdict(float)
_telemetry["request_count"] = 0
_telemetry["error_count"] = 0
_telemetry["prediction_yes"] = 0
_telemetry["prediction_no"] = 0
_telemetry["total_latency_ms"] = 0.0


def _record(latency_ms: float, y_pred: Optional[str] = None, error: bool = False) -> None:
    _telemetry["request_count"] += 1
    _telemetry["total_latency_ms"] += latency_ms
    if error:
        _telemetry["error_count"] += 1
    if y_pred == "yes":
        _telemetry["prediction_yes"] += 1
    elif y_pred == "no":
        _telemetry["prediction_no"] += 1


# ── Startup ───────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event() -> None:
    """Load model, preprocessors, and RAG vector store on startup."""
    global MODEL, ENCODERS, SCALER, MODEL_RUN_ID, RAG_ENGINE

    print("⏳ CIP Server starting up: Loading ML model and vector index ...")
    try:
        MODEL, ENCODERS, SCALER, MODEL_RUN_ID = load_promoted_model()
        print(f"✅ Promoted model loaded (Run ID: {MODEL_RUN_ID})")
    except Exception as exc:
        print(f"❌ WARNING: Failed to load ML model — /predict will return 503: {exc}")

    try:
        RAG_ENGINE = ComplaintQueryEngine()
        print("✅ FAISS vector index loaded.")
    except Exception as exc:
        print(f"❌ WARNING: Failed to load RAG engine — RAG endpoints will return 503: {exc}")


# ── Internal helpers ─────────────────────────────────────────────
def _build_customer_dict(c: CustomerInput) -> Dict[str, Any]:
    return {
        "age": c.age, "job": c.job, "marital": c.marital, "education": c.education,
        "default": c.default, "housing": c.housing, "loan": c.loan, "contact": c.contact,
        "month": c.month, "day_of_week": c.day_of_week, "duration": c.duration,
        "campaign": c.campaign, "pdays": c.pdays, "previous": c.previous,
        "poutcome": c.poutcome,
        "emp.var.rate": c.emp_var_rate, "cons.price.idx": c.cons_price_idx,
        "cons.conf.idx": c.cons_conf_idx, "euribor3m": c.euribor3m,
        "nr.employed": c.nr_employed,
    }


def _run_ml_inference(customer_in: CustomerInput) -> Tuple[float, str]:
    """Run feature pipeline + model inference. Returns (probability, 'yes'/'no')."""
    if MODEL is None or SCALER is None or not ENCODERS:
        raise HTTPException(
            status_code=503,
            detail="ML model is not loaded. Run training pipeline first.",
        )
    df_raw = pd.DataFrame([_build_customer_dict(customer_in)])
    df_proc, _, _ = build_features(df_raw, encoders=ENCODERS, scaler=SCALER)

    if isinstance(MODEL, xgb.Booster):
        y_prob = float(MODEL.predict(xgb.DMatrix(df_proc))[0])
    else:
        try:
            if hasattr(MODEL, "classes_") and not hasattr(MODEL, "n_classes_"):
                MODEL.n_classes_ = len(MODEL.classes_)
            y_prob = float(MODEL.predict_proba(df_proc)[0, 1])
        except Exception:
            if hasattr(MODEL, "get_booster"):
                y_prob = float(MODEL.get_booster().predict(xgb.DMatrix(df_proc))[0])
            else:
                raise

    return y_prob, ("yes" if y_prob >= 0.5 else "no")


def _conversion_band(prob: float) -> str:
    """Map probability to labelled conversion band."""
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


# ── GET /health ───────────────────────────────────────────────────
@app.get("/health", summary="System health and version info")
def health() -> Dict[str, Any]:
    """Returns the platform health, active model version, and vector index version."""
    status = "healthy" if (MODEL is not None and RAG_ENGINE is not None) else "degraded"
    return {
        "status": status,
        "ml_model_loaded": MODEL is not None,
        "rag_engine_loaded": RAG_ENGINE is not None,
        "model_version": MODEL_RUN_ID or "not_loaded",
        "vector_index_version": INDEX_VERSION,
        "api_version": "2.0.0",
    }


# ── POST /predict ─────────────────────────────────────────────────
@app.post("/predict", summary="ML conversion propensity for a single customer")
def predict(customer: CustomerInput) -> Dict[str, Any]:
    """Returns probability, binary prediction, conversion band, and model version."""
    t0 = time.perf_counter()
    try:
        y_prob, y_pred = _run_ml_inference(customer)
        latency = (time.perf_counter() - t0) * 1000
        _record(latency, y_pred)
        return {
            "conversion_probability": round(y_prob, 6),
            "conversion_prediction": y_pred,
            "conversion_band": _conversion_band(y_prob),
            "decision_threshold": 0.5,
            "model_version": MODEL_RUN_ID,
        }
    except HTTPException:
        _record((time.perf_counter() - t0) * 1000, error=True)
        raise


# ── POST /batch-score ─────────────────────────────────────────────
@app.post("/batch-score", summary="Batch ML scoring for a list of customers")
def batch_score(request: BatchScoreRequest) -> Dict[str, Any]:
    """Score multiple customers at once. Returns per-customer results and band counts."""
    if not request.customers:
        raise HTTPException(status_code=422, detail="'customers' list must not be empty.")

    t0 = time.perf_counter()
    results = []
    band_counts: Dict[str, int] = {"High": 0, "Medium": 0, "Low": 0}

    for i, customer in enumerate(request.customers):
        try:
            y_prob, y_pred = _run_ml_inference(customer)
            band = _conversion_band(y_prob)
            band_counts[band] += 1
            results.append({
                "index": i,
                "conversion_probability": round(y_prob, 6),
                "conversion_prediction": y_pred,
                "conversion_band": band,
            })
            _record(0.0, y_pred)
        except HTTPException as exc:
            results.append({"index": i, "error": exc.detail})
            _record(0.0, error=True)

    latency = (time.perf_counter() - t0) * 1000
    return {
        "scored_count": len(results),
        "model_version": MODEL_RUN_ID,
        "band_distribution": band_counts,
        "latency_ms": round(latency, 2),
        "results": results,
    }


# ── POST /ask-complaints ──────────────────────────────────────────
@app.post("/ask-complaints", summary="RAG complaint Q&A with evidence IDs and refusal logic")
def ask_complaints(request: AskComplaintsRequest) -> Dict[str, Any]:
    """Answer a complaint intelligence question grounded in the CFPB corpus.

    Returns retrieved evidence IDs, a one-line evidence-sufficiency note,
    and a synthesised analytical brief. Refuses when no complaint crosses
    the similarity threshold.
    """
    if RAG_ENGINE is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialised.")

    response = RAG_ENGINE.query(
        request.question,
        top_k=request.top_k,
        product_filter=request.product,
        company_filter=request.company,
    )

    evidence_ids = [r["complaint_id"] for r in response["results"]]

    return {
        "question": request.question,
        "answer": response["synthesis"],
        "evidence_ids": evidence_ids,
        "evidence_sufficiency": response["evidence_sufficiency"],
        "refused": response["refused"],
        "retrieved_complaints": response["results"],
        "retrieval_stats": response["retrieval_stats"],
        "prompt_version": "template-v1",
    }


# ── POST /customer-intel ──────────────────────────────────────────
@app.post("/customer-intel", summary="Unified: ML conversion band + complaint themes")
def customer_intel(request: CustomerIntelRequest) -> Dict[str, Any]:
    """Given a customer profile and optional complaint filters, returns:
      - ML conversion band (High / Medium / Low) with probability
      - Top complaint themes for the segment, with cited evidence IDs
    """
    if RAG_ENGINE is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialised.")

    t0 = time.perf_counter()

    # 1. ML propensity
    y_prob, y_pred = _run_ml_inference(request.customer)
    band = _conversion_band(y_prob)

    # 2. Build a contextual query from the customer's profile + filters
    query_parts = [f"{request.customer.job} customer"]
    if request.product_filter:
        query_parts.append(request.product_filter)
    if request.issue_filter:
        query_parts.append(request.issue_filter)
    query_text = " ".join(query_parts) if len(query_parts) > 1 else "banking financial dispute"

    rag_response = RAG_ENGINE.query(
        query_text,
        top_k=request.top_k,
        product_filter=request.product_filter,
        company_filter=request.company_filter,
    )

    # 3. Build theme grouping
    themes = RAG_ENGINE.get_themes(rag_response["results"]) if not rag_response["refused"] else {}
    theme_summary = [
        {"theme": theme, "cited_ids": ids} for theme, ids in themes.items()
    ]

    latency = (time.perf_counter() - t0) * 1000
    _record(latency, y_pred)

    return {
        "ml_scoring": {
            "conversion_probability": round(y_prob, 6),
            "conversion_band": band,
            "conversion_prediction": y_pred,
            "decision_threshold": 0.5,
            "model_version": MODEL_RUN_ID,
        },
        "complaint_intelligence": {
            "query_used": query_text,
            "top_complaint_themes": theme_summary,
            "evidence_sufficiency": rag_response["evidence_sufficiency"],
            "refused": rag_response["refused"],
            "retrieved_count": len(rag_response["results"]),
        },
        "latency_ms": round(latency, 2),
    }


# ── POST /analyze-customer ────────────────────────────────────────
@app.post("/analyze-customer", summary="Backwards-compatible alias for /customer-intel")
def analyze_customer(request: CustomerAnalysisRequest) -> Dict[str, Any]:
    """Combines ML propensity scoring with RAG complaint intelligence (v1 format)."""
    if RAG_ENGINE is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialised.")

    y_prob, y_pred = _run_ml_inference(request.customer)
    rag_response = RAG_ENGINE.query(request.complaint_query, top_k=3)

    return {
        "customer_propensity": {
            "conversion_probability": round(y_prob, 6),
            "conversion_prediction": y_pred,
            "conversion_band": _conversion_band(y_prob),
            "decision_threshold": 0.5,
            "model_version": MODEL_RUN_ID,
        },
        "complaint_intelligence": {
            "search_query": request.complaint_query,
            "retrieved_count": len(rag_response["results"]),
            "evidence_ids": [r["complaint_id"] for r in rag_response["results"]],
            "evidence_sufficiency": rag_response["evidence_sufficiency"],
            "refused": rag_response["refused"],
            "matches": rag_response["results"],
            "analytical_brief": rag_response["synthesis"],
        },
    }


# ── GET /metrics ──────────────────────────────────────────────────
@app.get("/metrics", summary="API telemetry: latency, counts, RAG stats")
def metrics() -> Dict[str, Any]:
    """Returns running API telemetry including request counts, latency percentiles,
    prediction distribution, and RAG retrieval statistics."""
    total = int(_telemetry["request_count"])
    avg_latency = (
        _telemetry["total_latency_ms"] / total if total > 0 else 0.0
    )
    pred_total = int(_telemetry["prediction_yes"]) + int(_telemetry["prediction_no"])
    yes_pct = (
        round(_telemetry["prediction_yes"] / pred_total * 100, 2) if pred_total > 0 else 0.0
    )

    rag_stats = RAG_ENGINE.monitoring_stats() if RAG_ENGINE else {}

    return {
        "request_count": total,
        "error_count": int(_telemetry["error_count"]),
        "avg_latency_ms": round(avg_latency, 2),
        "prediction_distribution": {
            "yes_count": int(_telemetry["prediction_yes"]),
            "no_count": int(_telemetry["prediction_no"]),
            "yes_percentage": yes_pct,
        },
        "rag_stats": rag_stats,
        "model_version": MODEL_RUN_ID,
        "vector_index_version": INDEX_VERSION,
    }
