# Data directory

This folder stores local data files that are **not** committed to Git.

Datasets used:

- UCI Bank Marketing dataset  
  - Source: https://archive.ics.uci.edu/ml/datasets/bank+marketing
  - Usage: Campaign conversion prediction (term-deposit subscription `y`)

- CFPB Consumer Complaint Database (sample only)  
  - Source: https://www.consumerfinance.gov/data-research/consumer-complaints/
  - Usage: Complaint intelligence and RAG over complaint narratives

Per project rules, only small samples and hash sidecar files are stored here; full raw data should never be committed.
