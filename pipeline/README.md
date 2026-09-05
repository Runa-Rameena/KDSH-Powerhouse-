# Pathway RAG & Narrative Evidence Verification Pipeline

This directory contains the primary deterministic Claim-Verification & Temporal Evidence pipeline for the Kharagpur Data Science Hackathon (KDSH).

---

## Overview

The pipeline ingests raw 19th-century literature (*The Count of Monte Cristo*, *In Search of the Castaways*), streams and chunks the novels, indexes narrative vectors, extracts atomic claims from backstories, retrieves top evidence chunks, performs sentence-level entity-aware polarity scoring, tracks temporal chapter progression, and generates mathematically grounded consistency decisions.

### Pipeline Stages
1. **`step1_ingest_pathway.py`**: Pathway-powered streaming table ingestion of novel texts and dataset CSVs (with cross-platform fallback).
2. **`step2_chunk_pathway.py`**: Pathway streaming sliding-window chunker preserving narrative continuity.
3. **`step3_index_embeddings.py`**: Dense vector indexing via SBERT / TF-IDF with FAISS / Scikit-Learn backends.
4. **`step4_extract_claims.py`**: Atomic sentence extraction and semantic categorization (early life, beliefs, fears, motivations).
5. **`step5_retrieve_evidence.py`**: Top-$k$ narrative retrieval with cosine similarity and in-memory caching.
6. **`step6_evaluate_claims.py`**: Sentence-level entity-aware polarity scoring to detect true contradictions vs. stylistic negation.
7. **`step7_temporal_analysis.py`**: Chapter-order timeline tracking to detect late narrative developments.
8. **`step8_aggregate_decision.py`**: Calibrated evidence aggregation, threshold scoring, and stratified validation reporting.
9. **`run_all.py`**: Orchestrator executing steps 1–8 in isolation and outputting final verified predictions.

---

## Pathway Streaming Engine

- **Ubuntu / Linux**: Pathway executes natively via `pw.Table` streaming pipelines (`pip install pathway`).
- **Windows**: The engine automatically enables its cross-platform sliding-window fallback, ensuring zero runtime crashes regardless of environment.

---

## How to Run

```bash
# Run end-to-end pipeline
python -m pipeline.run_all
```

---

## Outputs & Artifacts

- **Pipeline Results**: `pipeline/results.csv`
- **Root Results**: `results.csv` (competition format) and `results_pipeline.csv`
- **Validation Evaluation**: `artifacts/final/train_evaluation.json`
- **Schema**: `id,predicted_label,rationale` (1 = Consistent, 0 = Contradict)
