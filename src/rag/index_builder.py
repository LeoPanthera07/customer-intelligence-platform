"""
RAG Index Builder — reads CFPB complaints, performs strict PII scrubbing,
embeds narratives using SentenceTransformers, and builds a FAISS vector index.
"""

import os
import re
import pickle
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CFPB_CSV_PATH = DATA_DIR / "cfpb_complaints.csv"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PATH = DATA_DIR / "complaints_metadata.pkl"


# ── Rule 8: PII Scrubbing Checklist ──────────────────────────────
def scrub_pii(text: str) -> str:
    """Scan and redact personally identifiable information (PII) from narratives.

    Specifically targets:
    1. Social Security Numbers (SSNs) / Tax IDs: XXX-XX-XXXX or 9 digits
    2. Phone numbers: (XXX) XXX-XXXX, XXX-XXX-XXXX, or XXXXXXXXXX
    3. Email addresses: user@domain.com
    """
    if not isinstance(text, str):
        return ""
        
    # Redact Emails
    email_regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    text = re.sub(email_regex, "[REDACTED_EMAIL]", text)
    
    # Redact Phone Numbers
    phone_regex = r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"
    text = re.sub(phone_regex, "[REDACTED_PHONE]", text)
    
    # Redact SSNs (pattern: 3 digits - 2 digits - 4 digits or raw 9 digits where appropriate)
    ssn_regex = r"\b\d{3}-\d{2}-\d{4}\b"
    text = re.sub(ssn_regex, "[REDACTED_SSN]", text)
    
    # Redact common credit card numbers (15-16 digits)
    cc_regex = r"\b(?:\d[ -]*?){13,16}\b"
    text = re.sub(cc_regex, "[REDACTED_CC]", text)
    
    return text


def build_faiss_index() -> None:
    print("─" * 60)
    print("RAG Pipeline — Building FAISS vector index …")
    
    if not CFPB_CSV_PATH.exists():
        print(f"ERROR: CFPB complaints CSV not found at {CFPB_CSV_PATH}. Run ingest.py first.")
        sys.exit(1)
        
    # 1. Load complaints
    df = pd.read_csv(CFPB_CSV_PATH)
    # Ensure narratives exist and are not empty
    df = df[df["consumer_complaint_narrative"].notna() & (df["consumer_complaint_narrative"].str.strip() != "")]
    
    if len(df) == 0:
        print("ERROR: No complaints with valid narratives found.")
        sys.exit(1)
        
    print(f"Found {len(df):,} complaints with valid narratives. Processing PII scrubbing …")
    
    # 2. Apply PII scrubbing
    df["scrubbed_narrative"] = df["consumer_complaint_narrative"].apply(scrub_pii)
    
    # 3. Load embedding model
    print("Loading SentenceTransformer model ('all-MiniLM-L6-v2') …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 4. Generate embeddings
    print("Generating embeddings for all narratives (this may take a minute) …")
    narratives = df["scrubbed_narrative"].tolist()
    embeddings = model.encode(narratives, show_progress_bar=True, batch_size=64, convert_to_numpy=True)
    
    # Convert embeddings to float32 (required by FAISS)
    embeddings = np.ascontiguousarray(embeddings.astype("float32"))
    dimension = embeddings.shape[1]
    
    # 5. Build FAISS index (IndexFlatL2 is exact L2 search)
    print(f"Building FAISS IndexFlatL2 with dimension {dimension} …")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # 6. Save FAISS index and metadata
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    
    # Store clean metadata mapping index row to complaint fields
    metadata = df[[
        "complaint_id", "date_received", "product", "issue", "company", "state", "scrubbed_narrative"
    ]].to_dict(orient="records")
    
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
        
    print("─" * 60)
    print("✅ RAG Indexing complete.")
    print(f"   FAISS Index → {FAISS_INDEX_PATH}")
    print(f"   Metadata    → {METADATA_PATH}")
    print(f"   Indexed     : {index.ntotal:,} complaints")


if __name__ == "__main__":
    build_faiss_index()
