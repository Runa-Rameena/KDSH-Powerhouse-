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

## Complete File Inventory & Purpose Breakdown

| File Name | Purpose & Functionality |
| :--- | :--- |
| **`step0_config.py`** | NLI pipeline configuration (model names, device selection, batch sizes, confidence thresholds). |
| **`step1_load_data.py`** | Data loader module normalizing train/test CSVs into structured JSON inputs (`id`, `story_id`, `backstory`). |
| **`step2_claim_extraction.py`** | Sentence boundary parsing module isolating atomic claims from backstory paragraphs. |
| **`step3_retrieve_evidence.py`** | Passage retriever searching unabridged novel text to extract candidate premises for each claim. |
| **`step4_nli_inference.py`** | Batched NLI inference module executing classification over `(Premise, Hypothesis)` pairs. |
| **`step5_aggregate_decision.py`** | Decision aggregator combining claim-level NLI labels into final backstory predictions and rationales. |
| **`models.py`** | Transformer NLI model wrapper and rule-based heuristic fallback inference engine. |
| **`run_nli_smoke.py`** | Smoke test script verifying NLI pipeline components and environment execution. |
| **`run_pipeline.py`** | Master pipeline orchestrator executing steps 1–5 in sequence and generating CSV outputs. |
| **`results.csv`** | Subfolder-isolated output CSV containing Pipeline 2 predictions (`id,predicted_label,rationale`). |

---

## Pipeline Stage Execution Diagram

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

---

## How to Run

```bash
# Execute end-to-end NLI pipeline
python3 -m pipeline_nli.run_pipeline

# Execute quick smoke test
python3 -m pipeline_nli.run_nli_smoke
```

---

## Performance & Accuracy Benchmark (Pipeline 2)

| Metric | Score | Detail |
| :--- | :---: | :--- |
| **Accuracy** | **48.7%** | Conservative entailment classification decisions |
| **F1-Score** | **0.468** | Balanced harmonic precision-recall metric |
| **Precision** | **69.2%** | High precision when detecting direct contradictions |
| **Recall** | **35.3%** | Strict entailment filter trade-off |

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
