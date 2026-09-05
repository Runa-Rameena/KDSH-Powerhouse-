# Pipeline 1: Pathway Streaming RAG & Narrative Evidence Verification Engine

This directory contains the primary **Pathway Streaming RAG & Temporal Evidence Pipeline** for narrative consistency verification. It ingests 19th-century literature (*The Count of Monte Cristo*, *In Search of the Castaways*), streams narrative text, indexes semantic vectors, extracts character backstory claims, retrieves top evidence excerpts, performs sentence-level entity-aware polarity scoring, and calculates chronological progression to detect true contradictions vs. stylistic negation.

---

## Technical Highlights

- **Pathway Engine Integration**: Uses `pathway` (`pw.Table`) for real-time streaming ingestion and sliding-window document chunking.
- **Entity-Aware Polarity Analysis**: Distinguishes between genuine factual contradictions and narrative stylistic negation (e.g., figurative phrasing, metaphor).
- **Temporal Timeline Tracking**: Tracks chapter-order narrative progression across book lengths to weight later character developments appropriately.
- **Calibrated Aggregation**: Mathematical threshold scoring balancing support score vs. contradiction ratio.

---

## Pipeline Architecture & Execution Stages

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

### Detailed Stage Breakdown

1. **`step1_ingest_pathway.py`**: Reads novel texts (`Books/*.txt`) and normalizes input data into structured tables using Pathway streaming tables.
2. **`step2_chunk_pathway.py`**: Applies sliding-window chunking (preserving paragraph boundaries and chapter headings) to maintain narrative context.
3. **`step3_index_embeddings.py`**: Encodes novel chunks into high-dimensional vector embeddings using SBERT / TF-IDF with FAISS vector indexing.
4. **`step4_extract_claims.py`**: Parses complex backstory narratives into atomic, testable claim statements (categorized by early life, beliefs, fears, motivations).
5. **`step5_retrieve_evidence.py`**: Queries the vector index for each claim to extract the top-$k$ most relevant book excerpts.
6. **`step6_evaluate_claims.py`**: Performs deep polarity scoring on retrieved excerpts against claims, detecting direct negation, entity alignment, and context match.
7. **`step7_temporal_analysis.py`**: Map chunk positions to chapter progression, applying position multipliers for late-narrative reveals.
8. **`step8_aggregate_decision.py`**: Computes overall support vs contradiction scores, applies decision rules, and generates decision rationales.
9. **`run_all.py`**: End-to-end master orchestrator that executes steps 1–8 sequentially and collects final predictions.

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
  - `predicted_label`: `1` (Consistent / Support), `0` (Contradict)
  - `rationale`: Explainable audit trace containing decision rule, confidence scores, and support ratios.

---

## Performance & Accuracy Benchmark

- **Validation Accuracy**: **62.5%**
- **Recall**: **80.0%**
- **Precision**: **66.7%**
- **F1-Score**: **72.7%**
