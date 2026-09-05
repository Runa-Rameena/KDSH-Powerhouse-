# Pipeline 1: Pathway Streaming RAG & Narrative Evidence Verification Engine

This directory contains the primary **Pathway Streaming RAG & Temporal Evidence Pipeline** for narrative consistency verification. It ingests 19th-century literature (*The Count of Monte Cristo*, *In Search of the Castaways*), streams narrative text, indexes semantic vectors, extracts character backstory claims, retrieves top evidence excerpts, performs sentence-level entity-aware polarity scoring, and calculates chronological progression to detect true contradictions vs. stylistic negation.

---

## Technical Highlights

- **Pathway Engine Integration**: Uses `pathway` (`pw.Table`) for real-time streaming ingestion and sliding-window document chunking.
- **Entity-Aware Polarity Analysis**: Distinguishes between genuine factual contradictions and narrative stylistic negation (e.g., figurative phrasing, metaphor).
- **Temporal Timeline Tracking**: Tracks chapter-order narrative progression across book lengths to weight later character developments appropriately.
- **Calibrated Aggregation**: Mathematical threshold scoring balancing support score vs. contradiction ratio.

---

## Complete File Inventory & Purpose Breakdown

| File Name | Purpose & Functionality |
| :--- | :--- |
| **`step0_config.py`** | Central configuration file defining data paths, vector dimensions, retrieval top-$k$, and threshold constants ($\tau$). |
| **`step1_ingest_pathway.py`** | Ingests novel texts (`Books/*.txt`) and input CSVs using Pathway streaming tables (`pw.Table`) with cross-platform fallback. |
| **`step2_chunk_pathway.py`** | Pathway-powered sliding-window text chunker preserving paragraph and chapter boundaries. |
| **`step3_index_embeddings.py`** | Encodes text chunks into dense vector embeddings using Sentence-Transformers / TF-IDF with FAISS vector indexing. |
| **`step4_extract_claims.py`** | Parses character backstory narratives into testable atomic claim statements. |
| **`step5_retrieve_evidence.py`** | Queries vector indices for each claim to retrieve top-$k$ relevant novel passages. |
| **`step6_evaluate_claims.py`** | Sentence-level entity-aware polarity scoring module detecting direct negation vs stylistic phrasing. |
| **`step7_temporal_analysis.py`** | Maps retrieved chunks to chapter positions and computes late-narrative progression weights. |
| **`step8_aggregate_decision.py`** | Mathematical decision aggregator computing support/contradiction scores and outputting validation metrics (`train_evaluation.json`). |
| **`llm_provider.py`** | Unified LLM interface supporting external API (OpenAI/Gemini) calls and deterministic entity-aware offline reasoning. |
| **`metrics_genai_disagreements.py`** | Diagnostic evaluation utility analyzing prediction disagreements between different pipelines. |
| **`run_all.py`** | Master pipeline orchestrator executing steps 1–8 sequentially and generating output CSVs. |
| **`results.csv`** | Subfolder-isolated output CSV containing Pipeline 1 predictions (`id,predicted_label,rationale`). |

---

## Pipeline Stage Execution Diagram

```
Raw Novels + Backstories 
       │
       ▼
1. step1_ingest_pathway.py ─────► Stream novel texts via Pathway pw.Table
       │
       ▼
2. step2_chunk_pathway.py  ─────► Sliding-window windowed chunking
       │
       ▼
3. step3_index_embeddings.py ───► Dense Vector Indexing (Sentence-Transformers / TF-IDF)
       │
       ▼
4. step4_extract_claims.py ────► Extract atomic claims from character backstories
       │
       ▼
5. step5_retrieve_evidence.py ─► Top-k Cosine Similarity Evidence Retrieval
       │
       ▼
6. step6_evaluate_claims.py ───► Sentence-level Polarity & Entity Matching
       │
       ▼
7. step7_temporal_analysis.py ─► Chapter Timeline & Late-Narrative Weighting
       │
       ▼
8. step8_aggregate_decision.py ─► Threshold Decision Aggregation & Metric Scoring
       │
       ▼
   run_all.py ─────────────────► Master Orchestrator
```

---

## Pathway Streaming Engine Details

- **Linux / Ubuntu**: Executes natively using Pathway's rust-backed streaming runtime (`pip install pathway`).
- **Cross-Platform Fallback**: Includes an automated in-memory streaming fallback to guarantee cross-platform execution stability on Windows or non-x86 environments.

---

## How to Run

```bash
# Execute end-to-end Pathway RAG pipeline
python3 -m pipeline.run_all
```

---

## Outputs & Artifacts

- **Local Results**: `pipeline/results.csv`
- **Root Results**: `results_pipeline.csv`
- **Evaluation Report**: `artifacts/final/train_evaluation.json`
- **Output Schema**:
  ```csv
  id,predicted_label,rationale
  46,1,"DECISION=SUPPORT | rule=supported | support_count=3 | max_confidence=0.85"
  137,0,"DECISION=CONTRADICT | rule=C>tau_and_dominates | C=2.40 | S=0.30 | tau=1.00"
  ```

---

## Performance & Accuracy Benchmark

- **Validation Accuracy**: **62.5%**
- **Recall**: **80.0%**
- **Precision**: **66.7%**
- **F1-Score**: **72.7%**
