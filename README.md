# Narrative Consistency & Character Backstory Verification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pathway](https://img.shields.io/badge/Pathway-Streaming%20RAG-green.svg)](https://pathway.com/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20NLI-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade, multi-pipeline narrative consistency verification system designed to validate character backstories against massive 19th-century literary texts (*The Count of Monte Cristo* and *In Search of the Castaways*).

The project implements **three independent, end-to-end verification pipelines**:
1. **Pathway Streaming RAG & Temporal Evidence Engine**
2. **Natural Language Inference (NLI) Transformer Pipeline**
3. **GenAI Reasoner & Hallucination Guard Pipeline**

---

## Directory Structure & Pipelines

```
KDSH-Powerhouse-/
├── Books/                                 # Complete unabridged novel texts (.txt)
├── pipeline/                              # [Pipeline 1] Pathway Streaming RAG & Heuristic Engine
│   ├── README.md                          # Detailed documentation for Pipeline 1
│   ├── run_all.py                         # End-to-end orchestrator for Steps 1-8
│   ├── step0_config.py to step8_*.py      # Ingestion, Chunking, Retrieval, Scoring, Decision
│   └── results.csv                        # Subfolder output for Pipeline 1
├── pipeline_nli/                          # [Pipeline 2] Natural Language Inference (NLI) Pipeline
│   ├── README.md                          # Detailed documentation for Pipeline 2
│   ├── run_pipeline.py                    # End-to-end orchestrator for Steps 1-5
│   ├── models.py                          # Transformer NLI inference engine
│   └── results.csv                        # Subfolder output for Pipeline 2
├── pipelinegenai/                         # [Pipeline 3] GenAI Reasoner & Hallucination Guard
│   ├── README.md                          # Detailed documentation for Pipeline 3
│   ├── run_pipeline.py                    # End-to-end standalone runner
│   ├── models.py                          # Pydantic & dataclass validation schemas
│   └── results.csv                        # Subfolder output for Pipeline 3
├── artifacts/                             # Intermediate cached tables, chunks, and embeddings
├── requirements.txt                       # Python dependencies
├── train.csv                              # Ground-truth labeled character backstories
├── test.csv                               # Test split for evaluation
├── results_pipeline.csv                   # Root output for Pipeline 1 (Pathway Streaming RAG)
├── results_nli.csv                        # Root output for Pipeline 2 (NLI Transformers)
└── results_genai.csv                      # Root output for Pipeline 3 (GenAI Reasoner)
```

---

## Detailed Pipeline Architecture Comparison

All pipeline outputs strictly adhere to the verification schema: `id,predicted_label,rationale` (where `1` = Consistent / Support, `0` = Contradict).

| Pipeline | Execution Module | Core Technology | Primary Strengths | Detailed Docs |
| :--- | :--- | :--- | :--- | :--- |
| **Pipeline 1** | `python3 -m pipeline.run_all` | **Pathway Streaming Tables (`pw.Table`)** + Vector Indexing + Polarity Scoring + Temporal Progression | Top verification accuracy (62.5% validation accuracy, 72.7% F1), high recall (80%), and timeline tracking. | [Pipeline 1 README](pipeline/README.md) |
| **Pipeline 2** | `python3 -m pipeline_nli.run_pipeline` | **NLI Transformers** (Premise vs Hypothesis Entailment/Contradiction) | Deep semantic claim-excerpt entailment modeling per backstory claim. | [Pipeline 2 README](pipeline_nli/README.md) |
| **Pipeline 3** | `python3 -m pipelinegenai.run_pipeline` | **Structured JSON LLM** + Pydantic Schema + Hallucination Guardrails | Typed schema enforcement, lexical anchor verification, and cited quote validation. | [Pipeline 3 README](pipelinegenai/README.md) |

---

## Quickstart & Execution Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Runa-Rameena/KDSH-Powerhouse-.git
cd KDSH-Powerhouse-

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install unified dependencies and Pathway engine
pip install -r requirements.txt
pip install pathway
```

### 2. Running Pipelines

You can run any or all pipelines independently:

```bash
# Run Pipeline 1 (Pathway Streaming RAG Engine)
python3 -m pipeline.run_all          # Generates results_pipeline.csv & pipeline/results.csv

# Run Pipeline 2 (NLI Transformer Engine)
python3 -m pipeline_nli.run_pipeline # Generates results_nli.csv & pipeline_nli/results.csv

# Run Pipeline 3 (GenAI Reasoner Engine)
python3 -m pipelinegenai.run_pipeline # Generates results_genai.csv & pipelinegenai/results.csv
```

---

## Performance & Accuracy Benchmark

Evaluating Pipeline 1 against the stratified validation holdout (`artifacts/final/train_evaluation.json`):

```json
{
  "split": "validation (20% stratified holdout)",
  "metrics": {
    "accuracy": 0.625,
    "precision": 0.667,
    "recall": 0.800,
    "f1": 0.727,
    "confusion_matrix": [[2, 4], [2, 8]]
  }
}
```

---

## Sub-Pipeline Detailed Documentation

For full architectural breakdowns, step-by-step module flows, and schema definitions, see:
- 📖 [Pipeline 1: Pathway RAG Engine Documentation](pipeline/README.md)
- 📖 [Pipeline 2: Natural Language Inference Documentation](pipeline_nli/README.md)
- 📖 [Pipeline 3: GenAI Reasoner & Guardrails Documentation](pipelinegenai/README.md)