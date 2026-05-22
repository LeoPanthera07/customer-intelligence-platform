# Data Directory

**⚠️ No raw data is committed to this repository.**

All data files are downloaded at runtime by `src/data_pipeline/ingest.py`.

## Data Sources

| Dataset | Source | URL |
|---------|--------|-----|
| UCI Bank Marketing | UCI Machine Learning Repository | https://archive.ics.uci.edu/ml/datasets/Bank+Marketing |
| CFPB Consumer Complaints | Consumer Financial Protection Bureau | https://www.consumerfinance.gov/data-research/consumer-complaints/ |

## Files Generated After Ingestion

| File | Description |
|------|-------------|
| `bank_marketing.csv` | UCI Bank Marketing dataset (full) |
| `cfpb_complaints.csv` | CFPB complaint sample (5,000–10,000 records) |
| `uci_hash.txt` | SHA-256 hash of `bank_marketing.csv` for versioning |
| `cfpb_hash.txt` | SHA-256 hash of `cfpb_complaints.csv` for versioning |
| `faiss.index` | FAISS vector index built from complaint narratives |
| `chunk_map.json` | Mapping of chunk IDs to metadata and text |
