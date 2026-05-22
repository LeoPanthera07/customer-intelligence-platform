"""
Data Ingestion — downloads the UCI Bank Marketing and CFPB Consumer
Complaint datasets, saves them to data/, and writes SHA-256 hash
sidecar files for data versioning.

Usage:
    python src/data_pipeline/ingest.py               # full download
    python src/data_pipeline/ingest.py --sample 100   # CI-friendly subset
"""

import argparse
import csv
import hashlib
import io
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import requests

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

UCI_CSV_PATH = DATA_DIR / "bank_marketing.csv"
UCI_HASH_PATH = DATA_DIR / "uci_hash.txt"

CFPB_CSV_PATH = DATA_DIR / "cfpb_complaints.csv"
CFPB_HASH_PATH = DATA_DIR / "cfpb_hash.txt"

# ── Source URLs ──────────────────────────────────────────────────
# UCI Bank Marketing (zip containing bank-additional-full.csv)
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00222/bank-additional.zip"
)

# CFPB Consumer Complaint Database — SOCRATA Open Data API
# We request records that have a consumer_complaint_narrative (needed for RAG).
CFPB_API_URL = (
    "https://data.consumerfinance.gov/resource/s6ew-h6mp.json"
)


# ── Helpers ──────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download_uci(sample_n: int | None = None) -> None:
    """Download UCI Bank Marketing dataset from its public zip URL.

    The zip contains ``bank-additional/bank-additional-full.csv``
    which is a semicolon-separated CSV with 41,188 rows.
    """
    print("─" * 60)
    print("UCI Bank Marketing — downloading …")
    start = time.time()

    resp = requests.get(UCI_ZIP_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Find the full CSV inside the zip
        csv_name = [
            n for n in zf.namelist()
            if n.endswith("bank-additional-full.csv")
        ]
        if not csv_name:
            # Fallback: take any .csv
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_name:
            sys.exit("ERROR: no CSV found inside UCI zip archive.")

        raw = zf.read(csv_name[0]).decode("utf-8")

    # The file uses ';' as separator and has quoted strings.
    # Convert to standard comma-separated CSV for consistency.
    reader = csv.reader(io.StringIO(raw), delimiter=";")
    rows = list(reader)
    header, data_rows = rows[0], rows[1:]

    if sample_n and sample_n < len(data_rows):
        random.seed(42)
        data_rows = random.sample(data_rows, sample_n)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(UCI_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)

    elapsed = time.time() - start
    digest = sha256_file(UCI_CSV_PATH)
    UCI_HASH_PATH.write_text(digest + "\n", encoding="utf-8")

    print(f"  File   : {UCI_CSV_PATH}")
    print(f"  Rows   : {len(data_rows):,}")
    print(f"  SHA-256: {digest[:16]}…")
    print(f"  Time   : {elapsed:.1f}s")


def download_cfpb(sample_n: int = 5000) -> None:
    """Download CFPB Consumer Complaint records via the SOCRATA API.

    Only records with a non-empty consumer_complaint_narrative are
    fetched because the RAG pipeline needs complaint text.

    Falls back to generating a realistic synthetic dataset if the
    API is unreachable or returns insufficient data.
    """
    print("─" * 60)
    print("CFPB Consumer Complaints — downloading …")
    start = time.time()

    records: list[dict] = []
    try:
        # SOCRATA API supports $limit and $offset for pagination.
        # We fetch in batches of 1000 to stay under rate limits.
        batch_size = 1000
        offset = 0
        while len(records) < sample_n:
            params = {
                "$limit": batch_size,
                "$offset": offset,
                "$where": "consumer_complaint_narrative IS NOT NULL",
                "$order": "date_received DESC",
            }
            resp = requests.get(CFPB_API_URL, params=params, timeout=60)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            records.extend(batch)
            offset += batch_size

        records = records[:sample_n]

        if len(records) < 100:
            raise ValueError(
                f"API returned only {len(records)} records — too few."
            )

    except Exception as exc:
        print(f"  ⚠  API unavailable ({exc}). Generating synthetic data …")
        records = _generate_synthetic_cfpb(sample_n)

    # Normalise column names to a consistent set.
    columns = [
        "complaint_id", "date_received", "product", "sub_product",
        "issue", "sub_issue", "consumer_complaint_narrative",
        "company", "state", "zip_code", "submitted_via",
        "company_response_to_consumer",
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CFPB_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            # Map SOCRATA field names to our column names
            row = {
                "complaint_id": rec.get("complaint_id", ""),
                "date_received": rec.get("date_received", ""),
                "product": rec.get("product", ""),
                "sub_product": rec.get("sub_product", ""),
                "issue": rec.get("issue", ""),
                "sub_issue": rec.get("sub_issue", ""),
                "consumer_complaint_narrative": rec.get(
                    "consumer_complaint_narrative", ""
                ),
                "company": rec.get("company", ""),
                "state": rec.get("state", ""),
                "zip_code": rec.get("zip_code", ""),
                "submitted_via": rec.get("submitted_via", ""),
                "company_response_to_consumer": rec.get(
                    "company_response_to_consumer", ""
                ),
            }
            writer.writerow(row)

    elapsed = time.time() - start
    digest = sha256_file(CFPB_CSV_PATH)
    CFPB_HASH_PATH.write_text(digest + "\n", encoding="utf-8")

    print(f"  File   : {CFPB_CSV_PATH}")
    print(f"  Rows   : {len(records):,}")
    print(f"  SHA-256: {digest[:16]}…")
    print(f"  Time   : {elapsed:.1f}s")


# ── Synthetic fallback ───────────────────────────────────────────
def _generate_synthetic_cfpb(n: int) -> list[dict]:
    """Generate *n* realistic-looking CFPB complaint records.

    Used as a fallback when the CFPB API is unreachable.
    The narratives are realistic templates — good enough for
    building and testing the RAG pipeline.
    """
    random.seed(42)

    products = [
        "Credit reporting, credit repair services, or other personal consumer reports",
        "Debt collection",
        "Mortgage",
        "Credit card or prepaid card",
        "Checking or savings account",
        "Student loan",
        "Vehicle loan or lease",
        "Money transfer, virtual currency, or money service",
        "Payday loan, title loan, or personal loan",
    ]

    issues_by_product = {
        "Credit reporting, credit repair services, or other personal consumer reports": [
            "Incorrect information on your report",
            "Improper use of your report",
            "Problem with a credit reporting company's investigation into an existing problem",
            "Unable to get your credit report or credit score",
            "Credit monitoring or identity theft protection services",
        ],
        "Debt collection": [
            "Attempts to collect debt not owed",
            "Written notification about debt",
            "False statements or representation",
            "Took or threatened to take negative or legal action",
            "Communication tactics",
        ],
        "Mortgage": [
            "Trouble during payment process",
            "Applying for a mortgage or refinancing an existing mortgage",
            "Struggling to pay mortgage",
            "Closing on a mortgage",
            "Problem with a company's investigation into an existing problem",
        ],
        "Credit card or prepaid card": [
            "Problem with a purchase shown on your statement",
            "Getting a credit card",
            "Fees or interest",
            "Other features, terms, or problems",
            "Closing your account",
        ],
        "Checking or savings account": [
            "Managing an account",
            "Problem caused by your funds being low",
            "Opening an account",
            "Closing an account",
            "Problem with a lender or other company charging your account",
        ],
        "Student loan": [
            "Dealing with your lender or servicer",
            "Struggling to repay your loan",
            "Getting a loan",
            "Problem with a credit reporting company's investigation into an existing problem",
        ],
        "Vehicle loan or lease": [
            "Managing the loan or lease",
            "Problems at the end of the loan or lease",
            "Getting a loan or lease",
            "Struggling to pay your loan",
        ],
        "Money transfer, virtual currency, or money service": [
            "Domestic (US) money transfer",
            "Fraud or scam",
            "Other transaction problem",
            "Money was not available when promised",
        ],
        "Payday loan, title loan, or personal loan": [
            "Charged fees or interest I didn't expect",
            "Problem with the payoff process at the end of the loan",
            "Getting the loan",
            "Struggling to pay your loan",
        ],
    }

    narrative_templates = [
        "I contacted {company} regarding my {product} account on {date}. "
        "The issue was related to {issue}. Despite multiple attempts to resolve "
        "the problem, the company failed to address my concern adequately. "
        "I submitted documentation proving my case but received no response "
        "for over 30 days. This has caused significant financial hardship.",

        "On {date}, I noticed an error on my {product} account with {company}. "
        "Specifically, {issue}. I have called their customer service line "
        "three times and each time was told it would be resolved within 5-7 "
        "business days. It has now been over two months with no resolution. "
        "I am requesting immediate investigation and correction.",

        "I am filing this complaint against {company} for their handling of "
        "my {product}. The problem relates to {issue}. I have been a customer "
        "for over five years and have never experienced such poor service. "
        "The company has not provided clear information about fees charged "
        "to my account and has been unresponsive to my written inquiries.",

        "My experience with {company} regarding {product} has been extremely "
        "frustrating. {issue} — I first reported this problem on {date} and "
        "have followed up multiple times. The company initially acknowledged "
        "the error but has since stopped responding to my communications. "
        "I believe this violates my consumer rights.",

        "{company} has failed to properly manage my {product} account. "
        "The core issue is {issue}. I have documentation including account "
        "statements, correspondence, and call records showing repeated "
        "failures by the company. I reported this on {date} and demand "
        "that the company correct the error and compensate for damages.",

        "I am writing to formally complain about {company} and their "
        "{product} services. {issue} has been an ongoing problem since "
        "{date}. I have attempted to resolve this through their internal "
        "dispute process but have been met with delays and contradictory "
        "information. I need regulatory intervention to resolve this matter.",

        "After reviewing my account with {company} for {product}, I "
        "discovered that {issue}. This was first identified on {date}. "
        "I have sent certified letters and made numerous phone calls. "
        "The company has acknowledged receipt of my dispute but has not "
        "provided a substantive response or corrected the problem.",

        "I have a {product} with {company} and have encountered a serious "
        "problem: {issue}. Since {date}, I have been trying to get this "
        "resolved but the company keeps giving me the runaround. Each "
        "representative I speak to gives different information. I need "
        "this resolved immediately as it is affecting my financial health.",
    ]

    companies = [
        "EQUIFAX, INC.", "TRANSUNION INTERMEDIATE HOLDINGS, INC.",
        "EXPERIAN INFORMATION SOLUTIONS INC.", "BANK OF AMERICA",
        "JPMORGAN CHASE & CO.", "WELLS FARGO & COMPANY",
        "CITIBANK, N.A.", "CAPITAL ONE FINANCIAL CORPORATION",
        "SYNCHRONY FINANCIAL", "DISCOVER BANK",
        "U.S. BANCORP", "PNC BANK, N.A.",
        "NAVIENT SOLUTIONS, LLC.", "ALLY FINANCIAL INC.",
    ]

    states = [
        "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
        "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
    ]

    records = []
    for i in range(n):
        product = random.choice(products)
        issue = random.choice(issues_by_product[product])
        company = random.choice(companies)
        # Generate a date between 2020-01-01 and 2024-12-31
        year = random.randint(2020, 2024)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date_str = f"{year}-{month:02d}-{day:02d}T00:00:00.000"

        narrative = random.choice(narrative_templates).format(
            company=company,
            product=product.lower(),
            issue=issue.lower(),
            date=f"{year}-{month:02d}-{day:02d}",
        )

        records.append({
            "complaint_id": str(7000000 + i),
            "date_received": date_str,
            "product": product,
            "sub_product": "",
            "issue": issue,
            "sub_issue": "",
            "consumer_complaint_narrative": narrative,
            "company": company,
            "state": random.choice(states),
            "zip_code": f"{random.randint(10000, 99999)}",
            "submitted_via": random.choice(["Web", "Phone", "Referral"]),
            "company_response_to_consumer": random.choice([
                "Closed with explanation",
                "Closed with monetary relief",
                "Closed with non-monetary relief",
                "Closed without relief",
                "In progress",
            ]),
        })

    return records


# ── CLI ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download UCI + CFPB datasets to data/."
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Limit UCI rows to N (useful for CI). CFPB always fetches 5000.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    download_uci(sample_n=args.sample)
    download_cfpb(sample_n=5000)

    print("─" * 60)
    print("✅ Ingestion complete.")
    print(f"   UCI  → {UCI_CSV_PATH}")
    print(f"   CFPB → {CFPB_CSV_PATH}")


if __name__ == "__main__":
    main()
