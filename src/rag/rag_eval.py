"""
RAG Evaluation Harness — runs 10 structured Q&A tests against the FAISS
complaint corpus and produces a pass/fail report with expected evidence IDs,
actual retrieval results, and overall quality metrics.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.rag.query_engine import ComplaintQueryEngine

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"

# ── Evaluation Question Bank ─────────────────────────────────────
# Each test case contains:
#   - question      : the natural-language query
#   - product_hint  : optional product keyword to narrow results
#   - expect_product: at least one result must belong to this product category
#   - expect_refused: True if we expect a refusal (deliberately off-topic query)
#   - notes         : human-readable intent of the test
EVAL_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "Q01",
        "question": "billing errors and incorrect charges on my credit card statement",
        "product_hint": "Credit card",
        "expect_product": "Credit card or prepaid card",
        "expect_refused": False,
        "notes": "Core credit-card billing dispute — should retrieve strong matches.",
    },
    {
        "id": "Q02",
        "question": "mortgage loan processing delays and closing cost issues",
        "product_hint": "Mortgage",
        "expect_product": "Mortgage",
        "expect_refused": False,
        "notes": "Mortgage origination complaint — common CFPB category.",
    },
    {
        "id": "Q03",
        "question": "debt collectors harassing me with repeated calls",
        "product_hint": "Debt collection",
        "expect_product": "Debt collection",
        "expect_refused": False,
        "notes": "Debt-collection harassment — FDCPA-related complaints.",
    },
    {
        "id": "Q04",
        "question": "inaccurate information on my credit report from Equifax",
        "product_hint": "Credit reporting",
        "expect_product": "Credit reporting, credit repair services, or other personal consumer reports",
        "expect_refused": False,
        "notes": "Credit reporting error — one of the top CFPB complaint categories.",
    },
    {
        "id": "Q05",
        "question": "savings account fees charged without notice and denied refund",
        "product_hint": "Checking or savings account",
        "expect_product": "Checking or savings account",
        "expect_refused": False,
        "notes": "Savings account fee dispute — should retrieve checking/savings complaints.",
    },
    {
        "id": "Q06",
        "question": "student loan servicer providing wrong payoff information",
        "product_hint": "Student loan",
        "expect_product": "Student loan",
        "expect_refused": False,
        "notes": "Student loan servicing — important post-COVID complaint category.",
    },
    {
        "id": "Q07",
        "question": "personal loan payoff process problem and prepayment penalty",
        "product_hint": "personal loan",
        "expect_product": "Payday loan, title loan, or personal loan",
        "expect_refused": False,
        "notes": "Personal loan payoff — verify retrieval across loan product types.",
    },
    {
        "id": "Q08",
        "question": "vehicle loan denied and repo without proper notice",
        "product_hint": "Vehicle loan",
        "expect_product": "Vehicle loan or lease",
        "expect_refused": False,
        "notes": "Auto loan repossession — should retrieve vehicle loan complaints.",
    },
    {
        "id": "Q09",
        "question": "money transfer failed and funds not returned by wire service",
        "product_hint": "Money transfer",
        "expect_product": "Money transfer, virtual currency, or money service",
        "expect_refused": False,
        "notes": "Wire/money-transfer failure — fintech-adjacent complaint type.",
    },
    {
        "id": "Q10",
        "question": "pizza topping preferences and restaurant menu suggestions",
        "product_hint": None,
        "expect_product": None,
        "expect_refused": True,
        "notes": "REFUSAL TEST — completely off-topic; engine must refuse this query.",
    },
]


def run_rag_eval(top_k: int = 3, verbose: bool = True) -> Dict[str, Any]:
    """Run all 10 evaluation questions and return a structured report."""
    print("═" * 60)
    print("🔬 RAG EVALUATION HARNESS — 10-Question Test Suite")
    print("═" * 60)

    engine = ComplaintQueryEngine()
    results = []
    passed = 0
    failed = 0

    for q in EVAL_QUESTIONS:
        t0 = time.perf_counter()
        response = engine.query(
            q["question"],
            top_k=top_k,
            product_filter=q.get("product_hint"),
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        refused = response["refused"]
        retrieved_products = list({r["product"] for r in response["results"]})
        evidence_ids = [r["complaint_id"] for r in response["results"]]

        # ── Pass/Fail logic ──────────────────────────────────────
        if q["expect_refused"]:
            # Pass if the engine correctly refused
            test_pass = refused
            fail_reason = "Expected a refusal but got results." if not test_pass else None
        else:
            if refused:
                test_pass = False
                fail_reason = "Engine refused but valid results were expected."
            elif q["expect_product"] and not any(
                q["expect_product"].lower() in p.lower() for p in retrieved_products
            ):
                test_pass = False
                fail_reason = (
                    f"Expected product '{q['expect_product']}' not found in "
                    f"retrieved products: {retrieved_products}"
                )
            else:
                test_pass = True
                fail_reason = None

        status = "PASS ✅" if test_pass else "FAIL ❌"
        if test_pass:
            passed += 1
        else:
            failed += 1

        result_entry = {
            "id": q["id"],
            "question": q["question"],
            "expected_product": q["expect_product"],
            "expected_refused": q["expect_refused"],
            "retrieved_products": retrieved_products,
            "evidence_ids": evidence_ids,
            "evidence_sufficiency": response["evidence_sufficiency"],
            "refused": refused,
            "latency_ms": round(latency_ms, 2),
            "pass": test_pass,
            "fail_reason": fail_reason,
            "notes": q["notes"],
        }
        results.append(result_entry)

        if verbose:
            print(f"\n[{q['id']}] {status}")
            print(f"  Q: {q['question'][:80]}...")
            print(f"  Evidence IDs: {evidence_ids}")
            print(f"  Sufficiency : {response['evidence_sufficiency']}")
            if fail_reason:
                print(f"  ⚠ Reason   : {fail_reason}")

    # Summary
    pass_rate = passed / len(EVAL_QUESTIONS) * 100
    report = {
        "total_questions": len(EVAL_QUESTIONS),
        "passed": passed,
        "failed": failed,
        "pass_rate_pct": round(pass_rate, 1),
        "avg_latency_ms": round(
            sum(r["latency_ms"] for r in results) / len(results), 2
        ),
        "results": results,
    }

    print("\n" + "═" * 60)
    print(f"📊 EVAL SUMMARY: {passed}/{len(EVAL_QUESTIONS)} passed ({pass_rate:.1f}%)")
    print(f"   Average latency: {report['avg_latency_ms']} ms")
    print("═" * 60)

    # Save JSON report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "rag_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ RAG eval report saved → {report_path}")

    return report


if __name__ == "__main__":
    run_rag_eval(top_k=3, verbose=True)
