"""
FastAPI Server Tests — validates health checks, prediction inference,
batch scoring, RAG ask-complaints (including refusal), customer-intel,
and the unified analyze-customer endpoint using FastAPI TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from src.serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Sample payloads ──────────────────────────────────────────────
SAMPLE_CUSTOMER = {
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
    "nr.employed": 5195.8,
}


# ── /health ──────────────────────────────────────────────────────
def test_health_endpoint(client):
    """Health endpoint returns status, model version, and vector index version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ml_model_loaded" in data
    assert "rag_engine_loaded" in data
    assert "model_version" in data
    assert "vector_index_version" in data


# ── /predict ─────────────────────────────────────────────────────
def test_predict_endpoint(client):
    """POST /predict returns probability, prediction, band, and model version."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "conversion_probability" in data
    assert "conversion_prediction" in data
    assert "conversion_band" in data
    assert "model_version" in data
    assert isinstance(data["conversion_probability"], float)
    assert data["conversion_prediction"] in ["yes", "no"]
    assert data["conversion_band"] in ["High", "Medium", "Low"]


def test_predict_invalid_payload(client):
    """POST /predict with invalid payload returns 422."""
    response = client.post("/predict", json={"age": "not_a_number"})
    assert response.status_code == 422


# ── /batch-score ─────────────────────────────────────────────────
def test_batch_score_endpoint(client):
    """POST /batch-score returns per-customer results and band distribution."""
    payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    response = client.post("/batch-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "scored_count" in data
    assert "band_distribution" in data
    assert "results" in data
    assert data["scored_count"] == 2
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 2
    # Each result must have a conversion band
    for r in data["results"]:
        assert "conversion_band" in r
        assert r["conversion_band"] in ["High", "Medium", "Low"]


def test_batch_score_empty_list(client):
    """POST /batch-score with empty list returns 422."""
    response = client.post("/batch-score", json={"customers": []})
    assert response.status_code == 422


# ── /ask-complaints ──────────────────────────────────────────────
def test_ask_complaints_on_topic(client):
    """POST /ask-complaints returns evidence IDs and sufficiency note for on-topic query."""
    payload = {
        "question": "billing errors and credit card dispute charges",
        "top_k": 3,
    }
    response = client.post("/ask-complaints", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "evidence_ids" in data
    assert "evidence_sufficiency" in data
    assert "refused" in data
    assert "prompt_version" in data
    # An on-topic query should NOT be refused
    assert data["refused"] is False
    assert isinstance(data["evidence_ids"], list)
    assert len(data["evidence_ids"]) > 0


def test_ask_complaints_refusal(client):
    """POST /ask-complaints refuses when query is completely off-topic."""
    payload = {
        "question": "pizza topping and restaurant dessert menu preferences",
        "top_k": 3,
    }
    response = client.post("/ask-complaints", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "refused" in data
    assert "evidence_sufficiency" in data
    # Off-topic query MUST be refused
    assert data["refused"] is True
    assert data["evidence_ids"] == []
    assert "REFUSED" in data["evidence_sufficiency"]


# ── /customer-intel ───────────────────────────────────────────────
def test_customer_intel_endpoint(client):
    """POST /customer-intel returns conversion band + complaint themes."""
    payload = {
        "customer": SAMPLE_CUSTOMER,
        "product_filter": "Credit card",
        "top_k": 3,
    }
    response = client.post("/customer-intel", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "ml_scoring" in data
    ml = data["ml_scoring"]
    assert "conversion_band" in ml
    assert ml["conversion_band"] in ["High", "Medium", "Low"]
    assert "conversion_probability" in ml

    assert "complaint_intelligence" in data
    ci = data["complaint_intelligence"]
    assert "top_complaint_themes" in ci
    assert "evidence_sufficiency" in ci
    assert "refused" in ci


# ── /analyze-customer (backwards-compat alias) ───────────────────
def test_analyze_customer_endpoint(client):
    """POST /analyze-customer returns integrated results (v1 format)."""
    payload = {
        "customer": SAMPLE_CUSTOMER,
        "complaint_query": "credit reporting errors and billing disputes",
    }
    response = client.post("/analyze-customer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "customer_propensity" in data
    assert "complaint_intelligence" in data
    propensity = data["customer_propensity"]
    assert "conversion_probability" in propensity
    assert "conversion_band" in propensity
    complaints = data["complaint_intelligence"]
    assert "evidence_ids" in complaints
    assert "evidence_sufficiency" in complaints
    assert isinstance(complaints["evidence_ids"], list)


# ── /metrics ──────────────────────────────────────────────────────
def test_metrics_endpoint(client):
    """GET /metrics returns telemetry counters after requests have been made."""
    # Make one request first to populate counters
    client.post("/predict", json=SAMPLE_CUSTOMER)
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "request_count" in data
    assert "error_count" in data
    assert "avg_latency_ms" in data
    assert "prediction_distribution" in data
    assert "rag_stats" in data
    assert data["request_count"] >= 1
