# Natural Language Inference (NLI) Verification Pipeline

This directory contains the NLI-based narrative consistency pipeline for the Kharagpur Data Science Hackathon (KDSH).

---

## Overview

The NLI pipeline evaluates narrative consistency by treating novel excerpts as premises and extracted backstory claims as hypotheses. It classifies each pair into Entailment, Contradiction, or Neutral using transformer NLI (or calibrated entity-aware heuristic inference when running offline).

### Pipeline Stages
1. **`step1_load_data.py`**: Ingests train and test datasets, normalizing text fields into structured JSON artifacts.
2. **`step2_claim_extraction.py`**: Splits backstory narratives into discrete atomic claims.
3. **`step3_retrieve_evidence.py`**: Chunks full novel texts and retrieves top-k relevant excerpts per claim.
4. **`step4_nli_inference.py`**: Executes batched NLI inference across (premise, hypothesis) pairs.
5. **`step5_aggregate_decision.py`**: Aggregates chunk-level labels to row decisions and generates formatted results.
6. **`run_pipeline.py`**: End-to-end runner orchestrating steps 1 through 5.

---

## How to Run

```bash
# Run end-to-end NLI pipeline
python -m pipeline_nli.run_pipeline
```

---

## Outputs & Artifacts

- **Pipeline Results**: `pipeline_nli/results.csv`
- **Root Results**: `results_nli.csv`
- **Metrics (if train split evaluated)**: `artifacts/pipeline_nli/metrics/metrics.json`
- **Schema**: `id,predicted_label,rationale` (1 = Consistent, 0 = Contradict)
