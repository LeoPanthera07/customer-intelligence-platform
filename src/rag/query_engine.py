"""
RAG Query Engine — retrieves relevant CFPB complaints using FAISS vector search,
enforces a similarity-distance refusal threshold, and generates a structured
analytical summary with evidence IDs, complaint themes, and a one-line
evidence-sufficiency note.
"""

import sys
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "complaints_metadata.pkl"

# ── Refusal Threshold ────────────────────────────────────────────
# FAISS returns L2 squared distances. Empirically, distances > 1.5 mean
# the retrieved complaints are semantically too dissimilar from the query.
SIMILARITY_REFUSAL_THRESHOLD: float = 1.5


class ComplaintQueryEngine:
    """Query engine for customer complaints using FAISS vector retrieval.

    Includes refusal logic: if the best matching complaint distance exceeds
    SIMILARITY_REFUSAL_THRESHOLD, the engine refuses to synthesise an answer
    and returns an evidence-sufficiency note explaining why.
    """

    def __init__(self) -> None:
        # Load embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load FAISS index
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run 'python src/rag/index_builder.py' first."
            )
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        # Load metadata mapping
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Complaints metadata not found at {METADATA_PATH}. "
                "Run 'python src/rag/index_builder.py' first."
            )
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        # Telemetry counters (in-memory; reset on restart)
        self._request_count: int = 0
        self._refused_count: int = 0
        self._total_latency_ms: float = 0.0
        self._total_distance: float = 0.0

    # ── Public API ───────────────────────────────────────────────
    def query(
        self,
        query_text: str,
        top_k: int = 3,
        product_filter: Optional[str] = None,
        company_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform semantic search and synthesise an analytical summary response.

        Parameters
        ----------
        query_text     : Free-text question or topic to search for.
        top_k          : Number of candidate results to retrieve.
        product_filter : Optional product name substring to filter results.
        company_filter : Optional company name substring to filter results.

        Returns
        -------
        dict with keys:
          query, results, synthesis, evidence_sufficiency, refused, retrieval_stats
        """
        t0 = time.perf_counter()
        self._request_count += 1

        # 1. Embed query
        query_vector = (
            self.model.encode([query_text], convert_to_numpy=True)
            .astype("float32")
        )
        query_vector = np.ascontiguousarray(query_vector)

        # 2. Search FAISS index — fetch extra candidates for post-filter
        fetch_k = max(top_k * 4, 20)  # over-fetch to allow filtering
        distances, indices = self.index.search(query_vector, fetch_k)

        # 3. Gather matching metadata records (apply optional filters)
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            record = self.metadata[idx].copy()
            record["distance"] = float(dist)

            if product_filter and product_filter.lower() not in record.get("product", "").lower():
                continue
            if company_filter and company_filter.lower() not in record.get("company", "").lower():
                continue

            candidates.append(record)
            if len(candidates) == top_k:
                break

        # 4. Refusal logic — if best match is too distant, refuse
        best_distance = candidates[0]["distance"] if candidates else float("inf")
        refused = best_distance > SIMILARITY_REFUSAL_THRESHOLD

        latency_ms = (time.perf_counter() - t0) * 1000
        self._total_latency_ms += latency_ms
        if candidates:
            self._total_distance += sum(r["distance"] for r in candidates) / len(candidates)

        if refused:
            self._refused_count += 1
            return {
                "query": query_text,
                "results": [],
                "synthesis": None,
                "evidence_sufficiency": (
                    f"REFUSED — The closest retrieved complaint has a similarity "
                    f"distance of {best_distance:.3f}, which exceeds the threshold "
                    f"of {SIMILARITY_REFUSAL_THRESHOLD}. No complaint in the corpus "
                    "is sufficiently relevant to ground an answer. Please try a "
                    "more specific financial product or issue keyword."
                ),
                "refused": True,
                "retrieval_stats": {
                    "top_k_requested": top_k,
                    "retrieved": 0,
                    "best_distance": best_distance,
                    "threshold": SIMILARITY_REFUSAL_THRESHOLD,
                    "latency_ms": round(latency_ms, 2),
                },
            }

        # 5. Synthesise analytical RAG summary
        synthesis = self._synthesize_response(query_text, candidates)
        avg_distance = sum(r["distance"] for r in candidates) / len(candidates)
        sufficiency = self._sufficiency_note(candidates, avg_distance)

        return {
            "query": query_text,
            "results": candidates,
            "synthesis": synthesis,
            "evidence_sufficiency": sufficiency,
            "refused": False,
            "retrieval_stats": {
                "top_k_requested": top_k,
                "retrieved": len(candidates),
                "best_distance": best_distance,
                "avg_distance": round(avg_distance, 4),
                "threshold": SIMILARITY_REFUSAL_THRESHOLD,
                "latency_ms": round(latency_ms, 2),
            },
        }

    def get_themes(self, results: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Group retrieved complaints by product/issue theme with cited IDs."""
        themes: Dict[str, List[int]] = {}
        for r in results:
            key = f"{r.get('product', 'Unknown')} — {r.get('issue', 'Unknown')}"
            themes.setdefault(key, []).append(r["complaint_id"])
        return themes

    def monitoring_stats(self) -> Dict[str, Any]:
        """Return running telemetry metrics for the RAG engine."""
        empty_count = self._refused_count
        avg_latency = (
            self._total_latency_ms / self._request_count
            if self._request_count > 0 else 0.0
        )
        avg_score = (
            self._total_distance / self._request_count
            if self._request_count > 0 else 0.0
        )
        refusal_rate = (
            self._refused_count / self._request_count
            if self._request_count > 0 else 0.0
        )
        return {
            "total_requests": self._request_count,
            "empty_retrieval_count": empty_count,
            "refusal_rate": round(refusal_rate, 4),
            "avg_top_k_distance": round(avg_score, 4),
            "avg_latency_ms": round(avg_latency, 2),
        }

    # ── Private Helpers ──────────────────────────────────────────
    def _sufficiency_note(self, results: List[Dict[str, Any]], avg_dist: float) -> str:
        """Produce a one-line evidence-sufficiency note grounded in retrieval scores."""
        n = len(results)
        if avg_dist < 0.5:
            quality = "high"
        elif avg_dist < 1.0:
            quality = "moderate"
        else:
            quality = "low"
        ids = ", ".join(str(r["complaint_id"]) for r in results)
        return (
            f"Evidence sufficiency: {quality} "
            f"(avg distance {avg_dist:.3f}, {n} complaint(s) retrieved — "
            f"IDs: {ids})."
        )

    def _synthesize_response(self, query_text: str, results: List[Dict[str, Any]]) -> str:
        """Synthesise a structured business intelligence brief from retrieved complaints."""
        if not results:
            return "No matching complaints were found in the database."

        # Theme grouping by product
        themes = self.get_themes(results)
        products = list(set(r["product"] for r in results))
        companies = list(set(r["company"] for r in results))

        bullet_points = []
        for i, r in enumerate(results, 1):
            short = (
                r["scrubbed_narrative"][:220] + "…"
                if len(r["scrubbed_narrative"]) > 220
                else r["scrubbed_narrative"]
            )
            bullet_points.append(
                f"   {i}. [ID:{r['complaint_id']}] {r['product']} @ {r['company']} "
                f"({r.get('state','?')}): \"{short}\""
            )

        theme_lines = [
            f"   • {theme}: cited IDs {ids}" for theme, ids in themes.items()
        ]

        synthesis = (
            f"Query: '{query_text}'\n\n"
            f"Retrieved {len(results)} CFPB complaint(s) spanning "
            f"{', '.join(products)}.\n\n"
            f"Key Corporate Entities: {', '.join(companies)}.\n\n"
            f"Complaint Excerpts (with evidence IDs):\n"
            + "\n".join(bullet_points)
            + "\n\nTop Complaint Themes:\n"
            + "\n".join(theme_lines)
            + "\n\nStrategic Recommendations:\n"
            "   • Resolution Policy: Address systemic errors immediately — "
            "recurring patterns across companies indicate regulatory risk.\n"
            "   • Customer Engagement: Proactively contact affected accounts "
            "with a clear resolution timeline and fee-review offer."
        )
        return synthesis


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/rag/query_engine.py '<query_string>'")
        sys.exit(1)

    query_str = sys.argv[1]
    print(f"\n🔍 Querying RAG Database for: '{query_str}' ...")

    try:
        engine = ComplaintQueryEngine()
        response = engine.query(query_str, top_k=3)

        print("\n" + "═" * 60)
        if response["refused"]:
            print("🚫 RETRIEVAL REFUSED")
            print(response["evidence_sufficiency"])
        else:
            print("💡 RETRIEVED COMPLAINT INTELLIGENCE BRIEF")
            print("═" * 60)
            print(response["synthesis"])
            print("\n" + response["evidence_sufficiency"])
        print("═" * 60)
        print(f"\nRetrieval stats: {response['retrieval_stats']}")

    except Exception as exc:
        print(f"ERROR: {exc}")
