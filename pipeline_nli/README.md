# Pipeline 2: Natural Language Inference (NLI) Verification Pipeline

This directory contains the **NLI Premise-Hypothesis Consistency Pipeline**. It evaluates narrative consistency by formulating novel excerpts as premises and extracted backstory claims as hypotheses, leveraging transformer-based Natural Language Inference (NLI) to classify relationships into **Entailment**, **Contradiction**, or **Neutral**.

---

## Core Concept: NLI Premise-Hypothesis Modeling

Natural Language Inference models formalize verification as:
- **Premise ($P$)**: Retrived excerpt from the unabridged novel text.
- **Hypothesis ($H$)**: Atomic claim extracted from a character's backstory.
- **Verdict**:
  - **Entailment ($\rightarrow$)**: The novel text confirms the claim ($1 = \text{Consistent}$).
  - **Contradiction ($\bot$)**: The novel text directly conflicts with the claim ($0 = \text{Contradict}$).
  - **Neutral ($\sim$)**: The novel text does not specify or conflict with the claim.

---

## Pipeline Stage Breakdown

```
Input Data (train.csv / test.csv)
       │
       ▼
1. step1_load_data.py ───────► Parse and normalize backstories into JSON artifacts
       │
       ▼
2. step2_claim_extraction.py ─► Split backstories into discrete atomic claims
       │
       ▼
3. step3_retrieve_evidence.py ─► Extract top-k novel excerpts matching claims
       │
       ▼
4. step4_nli_inference.py ────► Run batched NLI inference over (Premise, Hypothesis) pairs
       │
       ▼
5. step5_aggregate_decision.py ─► Aggregate claim decisions to row verdicts & output CSV
       │
       ▼
   run_pipeline.py ──────────► Master Pipeline Orchestrator
```

### Component Details

1. **`step1_load_data.py`**: Normalizes train/test CSVs, standardizing identifier keys (`id`, `story_id`, `backstory`).
2. **`step2_claim_extraction.py`**: Uses sentence boundary disambiguation to isolate single-fact claims from multi-sentence character backstories.
3. **`step3_retrieve_evidence.py`**: Performs lexical & vector retrieval to pull top matching novel passages per claim.
4. **`step4_nli_inference.py`**: Computes NLI probabilities over `(Premise, Hypothesis)` pairs using sentence-transformers / cross-encoder NLI models (with entity-aware rule fallback for offline runtime).
5. **`step5_aggregate_decision.py`**: Combines claim-level NLI decisions using a contradiction-priority rule: if *any* claim is contradicted by strong evidence, the backstory is marked as `0` (Contradict); otherwise `1` (Consistent).
6. **`run_pipeline.py`**: Standalone module runner that executes steps 1–5 in sequence.

---

## How to Run

```bash
# Execute end-to-end NLI pipeline
python3 -m pipeline_nli.run_pipeline
```

---

## Outputs & Artifacts

- **Local Results**: `pipeline_nli/results.csv`
- **Root Results**: `results_nli.csv`
- **Metrics Report** *(when evaluated on ground truth)*: `artifacts/pipeline_nli/metrics/metrics.json`
- **Intermediate Results**: `pipeline_nli/nli_results.jsonl`
- **Output Schema**:
  ```csv
  id,predicted_label,rationale
  46,1,"46_claim_1:SUPPORT, 46_claim_2:SUPPORT"
  137,0,"137_claim_1:CONTRADICT, 137_claim_2:NEUTRAL"
  ```
  - `predicted_label`: `1` (Consistent / Support), `0` (Contradict)
  - `rationale`: Detailed mapping of claim IDs and their respective NLI verdicts.
