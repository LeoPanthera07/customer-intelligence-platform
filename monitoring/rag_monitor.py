"""
RAG Monitoring — measures retrieval hit-rate, empty-retrieval count,
average top-k similarity score, refusal rate, token count and latency
for a sample of representative queries against the live RAG engine.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.rag.query_engine import ComplaintQueryEngine

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"

# ── Monitoring Query Set ─────────────────────────────────────────
# A representative mix of on-topic and off-topic queries to measure
# the engine's real-world retrieval health.
MONITORING_QUERIES: List[Dict[str, Any]] = [
    {"query": "credit card billing dispute and incorrect charges", "product_hint": None},
    {"query": "mortgage closing cost delay and escrow error", "product_hint": "Mortgage"},
    {"query": "debt collector calling too many times per day", "product_hint": "Debt collection"},
    {"query": "credit report error not fixed after dispute", "product_hint": "Credit reporting"},
    {"query": "bank account frozen without explanation", "product_hint": "Checking or savings account"},
    {"query": "student loan income driven repayment calculation wrong", "product_hint": "Student loan"},
    {"query": "auto loan interest rate changed after signing", "product_hint": "Vehicle loan"},
    {"query": "wire transfer lost money not returned", "product_hint": "Money transfer"},
    {"query": "personal loan denied and no reason given", "product_hint": "personal loan"},
    # Deliberately weak queries — should still retrieve something
    {"query": "financial services company bad experience", "product_hint": None},
    {"query": "bank fee refund request ignored", "product_hint": None},
    {"query": "loan payment not applied correctly", "product_hint": None},
    # True off-topic — should trigger refusal
    {"query": "weather forecast for tomorrow morning", "product_hint": None},
    {"query": "sports team roster and match results", "product_hint": None},
    {"query": "recipe for chocolate cake with frosting", "product_hint": None},
]


def _approx_token_count(text: str) -> int:
    """Rough token estimate: ~4 characters per token (GPT-style)."""
    return max(1, len(text) // 4)


def run_rag_monitoring(verbose: bool = True) -> Dict[str, Any]:
    """Run the monitoring query set and compute RAG quality metrics."""
    print("─" * 60)
    print("🔍 RAG Monitoring — Retrieval Quality Metrics")
    print("─" * 60)

    engine = ComplaintQueryEngine()

    total_queries = len(MONITORING_QUERIES)
    hit_count = 0          # queries that returned ≥1 result
    refused_count = 0      # queries that triggered refusal
    total_latency_ms = 0.0
    total_tokens = 0
    distance_scores: List[float] = []

    per_query_log = []

    for item in MONITORING_QUERIES:
        q = item["query"]
        t0 = time.perf_counter()
        response = engine.query(q, top_k=3, product_filter=item.get("product_hint"))
        latency_ms = (time.perf_counter() - t0) * 1000

        total_latency_ms += latency_ms
        refused = response["refused"]
        num_results = len(response["results"])

        # Approximate token count of the synthesised answer
        answer_text = response.get("synthesis") or response.get("evidence_sufficiency", "")
        token_count = _approx_token_count(q + answer_text)
        total_tokens += token_count

        if refused:
            refused_count += 1
        else:
            hit_count += 1
            best_dist = response["retrieval_stats"].get("best_distance", 0.0)
            distance_scores.append(best_dist)

        entry = {
            "query": q,
            "retrieved": num_results,
            "refused": refused,
            "best_distance": response["retrieval_stats"].get("best_distance"),
            "latency_ms": round(latency_ms, 2),
            "approx_tokens": token_count,
        }
        per_query_log.append(entry)

        if verbose:
            status = "🚫 REFUSED" if refused else f"✅ {num_results} result(s)"
            dist = response["retrieval_stats"].get("best_distance", "—")
            dist_str = f"{dist:.3f}" if isinstance(dist, float) else "—"
            print(f"  [{status}] dist={dist_str:>7} | {q[:60]}")

    # Aggregate metrics
    hit_rate = hit_count / total_queries
    refusal_rate = refused_count / total_queries
    avg_latency = total_latency_ms / total_queries
    avg_top_k_dist = (
        sum(distance_scores) / len(distance_scores) if distance_scores else 0.0
    )

    report = {
        "total_queries": total_queries,
        "retrieval_hit_count": hit_count,
        "empty_retrieval_count": refused_count,   # synonymous with refused in this engine
        "retrieval_hit_rate": round(hit_rate, 4),
        "refusal_rate": round(refusal_rate, 4),
        "avg_top_k_distance": round(avg_top_k_dist, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "total_approx_tokens": total_tokens,
        "per_query_log": per_query_log,
    }

    print("\n" + "═" * 60)
    print("📊 RAG MONITORING SUMMARY")
    print("═" * 60)
    print(f"  Total queries       : {total_queries}")
    print(f"  Retrieval hit-rate  : {hit_rate:.1%}  ({hit_count}/{total_queries} returned results)")
    print(f"  Refusal rate        : {refusal_rate:.1%}  ({refused_count}/{total_queries} refused)")
    print(f"  Avg top-k distance  : {avg_top_k_dist:.4f}  (lower = more relevant)")
    print(f"  Avg latency         : {avg_latency:.2f} ms")
    print(f"  Total tokens (est.) : {total_tokens}")
    print("═" * 60)

    # Save JSON report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "rag_monitoring_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ RAG monitoring report saved → {json_path}")

    return report


if __name__ == "__main__":
    run_rag_monitoring(verbose=True)
