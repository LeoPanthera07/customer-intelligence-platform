"""
RAG Engine Tests — validates PII scrubbing (emails, phones, SSNs, CCs),
vector similarity search, refusal logic, evidence-sufficiency notes,
and theme grouping.
"""

import pytest
from src.rag.index_builder import scrub_pii
from src.rag.query_engine import ComplaintQueryEngine, SIMILARITY_REFUSAL_THRESHOLD


# ── PII Scrubbing ────────────────────────────────────────────────
def test_pii_scrubbing():
    """Verify that all PII types are completely redacted from narratives."""
    raw_text = (
        "Hello, my email is john.doe@example.com and you can call me at "
        "1-800-555-0199 or 202-555-0143. My SSN is 123-45-6789. "
        "Also my card is 4111 1111 1111 1111."
    )
    scrubbed = scrub_pii(raw_text)

    assert "john.doe@example.com" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed

    assert "1-800-555-0199" not in scrubbed
    assert "202-555-0143" not in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed

    assert "123-45-6789" not in scrubbed
    assert "[REDACTED_SSN]" in scrubbed

    assert "4111 1111 1111 1111" not in scrubbed
    assert "[REDACTED_CC]" in scrubbed


# ── RAG Query Engine ─────────────────────────────────────────────
def test_rag_query_returns_structured_response():
    """FAISS query returns all required keys in the response dict."""
    try:
        engine = ComplaintQueryEngine()
        response = engine.query("mortgage loan processing delay", top_k=2)

        assert "query" in response
        assert "results" in response
        assert "synthesis" in response
        assert "evidence_sufficiency" in response
        assert "refused" in response
        assert "retrieval_stats" in response

    except FileNotFoundError:
        pytest.skip("FAISS index or metadata not present. Run index_builder.py first.")


def test_rag_query_result_fields():
    """Each retrieved result record has required metadata fields."""
    try:
        engine = ComplaintQueryEngine()
        response = engine.query("credit card billing dispute", top_k=2)

        if not response["refused"]:
            assert len(response["results"]) > 0
            first = response["results"][0]
            assert "complaint_id" in first
            assert "product" in first
            assert "scrubbed_narrative" in first
            assert "company" in first
            assert "distance" in first

    except FileNotFoundError:
        pytest.skip("FAISS index or metadata not present. Run index_builder.py first.")


def test_rag_on_topic_not_refused():
    """On-topic financial query must NOT trigger refusal."""
    try:
        engine = ComplaintQueryEngine()
        response = engine.query("debt collection harassment repeated calls", top_k=3)
        # A clear on-topic query must never be refused
        assert response["refused"] is False
        assert len(response["results"]) > 0

    except FileNotFoundError:
        pytest.skip("FAISS index or metadata not present. Run index_builder.py first.")


def test_rag_off_topic_refused():
    """Completely off-topic query must be refused with a sufficiency note."""
    try:
        engine = ComplaintQueryEngine()
        response = engine.query(
            "pizza topping preferences and restaurant dessert menu", top_k=3
        )
        assert response["refused"] is True
        assert response["results"] == []
        assert "REFUSED" in response["evidence_sufficiency"]
        assert response["synthesis"] is None

    except FileNotFoundError:
        pytest.skip("FAISS index or metadata not present. Run index_builder.py first.")


def test_rag_theme_grouping():
    """get_themes returns a non-empty dict with complaint IDs when results exist."""
    try:
        engine = ComplaintQueryEngine()
        response = engine.query("student loan repayment plan calculation error", top_k=3)

        if not response["refused"] and response["results"]:
            themes = engine.get_themes(response["results"])
            assert isinstance(themes, dict)
            assert len(themes) > 0
            for theme_key, ids in themes.items():
                assert isinstance(ids, list)
                assert len(ids) > 0

    except FileNotFoundError:
        pytest.skip("FAISS index or metadata not present. Run index_builder.py first.")


def test_rag_refusal_threshold_value():
    """The refusal threshold constant is defined and within a sensible range."""
    assert isinstance(SIMILARITY_REFUSAL_THRESHOLD, float)
    assert 0.5 <= SIMILARITY_REFUSAL_THRESHOLD <= 3.0
