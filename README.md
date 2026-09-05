# KDSH-Powerhouse: Narrative Consistency & Character Backstory Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pathway](https://img.shields.io/badge/Pathway-Streaming%20RAG-green.svg)](https://pathway.com/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20NLI-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade, multi-pipeline narrative consistency verification system developed for the **Kharagpur Data Science Hackathon (KDSH)**. 

The system validates character backstories against massive literary texts (*The Count of Monte Cristo* and *In Search of the Castaways*) using streaming RAG, Natural Language Inference, and GenAI hallucination guardrails.

---

## Repository Structure & Pipelines

This repository provides **three independent, end-to-end verification pipelines**, each generating its own dedicated `results.csv`:

```
KDSH-Powerhouse-/
├── Books/                                 # Complete unabridged novel texts (.txt)
├── pipeline/                              # [Pipeline 1] Pathway Streaming RAG & Heuristic Engine
│   ├── README.md                          # Detailed documentation for Pipeline 1
│   ├── run_all.py                         # End-to-end orchestrator for Steps 1-8
│   ├── step0_config.py to step8_*.py      # Ingestion, Chunking, Retrieval, Scoring, Decision
│   └── results.csv                        # Dedicated results for Pipeline 1
├── pipeline_nli/                          # [Pipeline 2] Natural Language Inference (NLI) Pipeline
│   ├── README.md                          # Detailed documentation for Pipeline 2
│   ├── run_pipeline.py                    # End-to-end orchestrator for Steps 1-5
│   ├── models.py                          # Transformer & calibrated heuristic NLI inference
│   └── results.csv                        # Dedicated results for Pipeline 2
├── pipelinegenai/                         # [Pipeline 3] GenAI Reasoner & Hallucination Guard
│   ├── README.md                          # Detailed documentation for Pipeline 3
│   ├── run_pipeline.py                    # End-to-end standalone runner
│   ├── models.py                          # Pydantic & dataclass validation schemas
│   └── results.csv                        # Dedicated results for Pipeline 3
├── artifacts/                             # Intermediate cached tables, chunks, and embeddings
├── requirements.txt                       # Unified Python dependencies
├── train.csv                              # Ground-truth labeled character backstories
├── test.csv                               # Competition test split for evaluation
├── results.csv                            # Primary Hackathon Submission (Pipeline 1)
├── results_pipeline.csv                   # Exact alias of Pipeline 1 predictions
├── results_nli.csv                        # Exact alias of Pipeline 2 (NLI) predictions
└── results_genai.csv                      # Exact alias of Pipeline 3 (GenAI) predictions
```

---

## Comparison of Results Files

All result files strictly follow the competition schema: `id,predicted_label,rationale` (where `1` = Consistent / Support, `0` = Contradict).

| Results File | Pipeline | Core Engine | Key Strengths |
| :--- | :--- | :--- | :--- |
| **`results.csv`** (Root) | `pipeline/` | Pathway Streaming + TF-IDF/Dense Vector Retrieval + Temporal Progression | Top competition accuracy (~75%) and F1 (~80%); grounded in novel chapter chronology. |
| **`results_pipeline.csv`** | `pipeline/` | Alias of primary Pathway pipeline | Reproducible copy of `results.csv`. |
| **`results_nli.csv`** | `pipeline_nli/` | Premise-Hypothesis NLI Transformers | Deep semantic entailment / contradiction reasoning per claim-excerpt pair. |
| **`results_genai.csv`** | `pipelinegenai/` | Structured JSON LLM + Hallucination Guard | Strictly typed schema enforcement and lexical anchor verification. |

---

## Quickstart & Execution

### Running on Ubuntu / Linux (Recommended for Native Pathway)

Pathway distributes native streaming binary wheels for Linux. To run natively on Ubuntu:

```bash
# 1. Clone the repository
git clone https://github.com/Runa-Rameena/KDSH-Powerhouse-.git
cd KDSH-Powerhouse-

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies and Pathway
pip install -r requirements.txt
pip install pathway

# 4. Run any of the pipelines:
python3 -m pipeline.run_all          # Generates results.csv & pipeline/results.csv
python3 -m pipeline_nli.run_pipeline # Generates results_nli.csv & pipeline_nli/results.csv
python3 -m pipelinegenai.run_pipeline # Generates results_genai.csv & pipelinegenai/results.csv
```

### Running on Windows

The repository includes a built-in cross-platform sliding-window fallback for Windows, allowing immediate execution without Docker:

```powershell
# Run primary pipeline
python -m pipeline.run_all

# Run NLI pipeline
python -m pipeline_nli.run_pipeline

# Run GenAI pipeline
python -m pipelinegenai.run_pipeline
```

---

## Verification & Validation Metrics

Running `pipeline.run_all` evaluates against a 20% stratified holdout of `train.csv` and outputs full evaluation metrics to `artifacts/final/train_evaluation.json`:

```json
{
  "split": "validation (20% stratified holdout)",
  "metrics": {
    "accuracy": 0.75,
    "precision": 0.80,
    "recall": 0.80,
    "f1": 0.80,
    "confusion_matrix": [[4, 2], [2, 8]]
  }
}
```

---

## Sub-Pipeline Documentation

For in-depth stage breakdowns, refer to the respective documentation:
- [Pipeline 1: Pathway RAG Engine README](pipeline/README.md)
- [Pipeline 2: NLI Transformer Pipeline README](pipeline_nli/README.md)
- [Pipeline 3: GenAI Reasoner & Guardrails README](pipelinegenai/README.md)