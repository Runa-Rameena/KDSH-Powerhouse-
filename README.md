# Narrative Consistency & Character Backstory Verification System

> **Competition Note**: This repository represents an official entry and attempt for the **Kharagpur Data Science Hackathon 2026 (KDSH 2026)**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pathway](https://img.shields.io/badge/Pathway-Streaming%20RAG-green.svg)](https://pathway.com/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20NLI-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade, multi-pipeline narrative consistency verification system designed to validate character backstories against massive 19th-century literary texts (*The Count of Monte Cristo* and *In Search of the Castaways*).

The project implements **three independent, end-to-end verification pipelines**:
1. **Pathway Streaming RAG & Temporal Evidence Engine** (`pipeline/`)
2. **Natural Language Inference (NLI) Transformer Pipeline** (`pipeline_nli/`)
3. **GenAI Reasoner & Hallucination Guard Pipeline** (`pipelinegenai/`)

---

## Master File & Directory Inventory

Below is a complete index detailing the purpose and responsibility of every file and directory in this repository:

### Root Repository Files

| File Path | Description & Purpose |
| :--- | :--- |
| **`README.md`** | Primary system documentation detailing repository structure, file inventory, setup, pipeline usage, and performance benchmarks for KDSH 2026. |
| **`train.csv`** | Ground-truth training dataset containing 140 character backstories with book identifiers (`story_id`) and binary consistency labels (`label`). |
| **`test.csv`** | Evaluation test dataset containing 60 character backstories for generating competition predictions. |
| **`requirements.txt`** | Complete Python dependency manifest (Pathway, Sentence-Transformers, FAISS, Pandas, PyTorch, Scikit-Learn). |
| **`results_pipeline.csv`** | Root prediction output produced by **Pipeline 1** (Pathway Streaming RAG Engine). |
| **`results_nli.csv`** | Root prediction output produced by **Pipeline 2** (Natural Language Inference Transformer). |
| **`results_genai.csv`** | Root prediction output produced by **Pipeline 3** (GenAI Reasoner & Guardrails Engine). |
| **`.gitignore`** | Git ignore rules excluding virtual environments, temporary cache files, and binary bytecode. |

---

### Data Directories

| Directory | Purpose & Contents |
| :--- | :--- |
| **`Books/`** | Contains the complete unabridged raw novel text files used as narrative ground truth: <br> • `The Count of Monte Cristo.txt` <br> • `In Search of the Castaways.txt` |
| **`artifacts/`** | Central cache and intermediate data store: <br> • `artifacts/chunks/`: Windowed narrative text chunks. <br> • `artifacts/indexing/`: TF-IDF matrices & SBERT dense vector embeddings. <br> • `artifacts/retrieval/`: Top-$k$ evidence retrieval JSON files per backstory ID. <br> • `artifacts/evidence_scoring/`: Sentence-level polarity scores. <br> • `artifacts/temporal_analysis/`: Timeline chapter progression flags. <br> • `artifacts/final/train_evaluation.json`: Stratified validation benchmark evaluation report. |

---

### Pipeline Directories & Module Purpose Breakdown

#### 1. Pathway Streaming RAG Pipeline (`pipeline/`)
*For full technical details, see [pipeline/README.md](pipeline/README.md).*

| File Path | Purpose |
| :--- | :--- |
| `pipeline/README.md` | Detailed architectural and execution documentation for Pipeline 1. |
| `pipeline/step0_config.py` | Configuration constants, vector dimensions, and contradiction threshold parameters ($\tau$). |
| `pipeline/step1_ingest_pathway.py` | Ingests novels and backstories using Pathway streaming tables (`pw.Table`) with cross-platform fallback. |
| `pipeline/step2_chunk_pathway.py` | Pathway-powered sliding-window document text chunking. |
| `pipeline/step3_index_embeddings.py` | Dense SBERT / TF-IDF vector indexer using FAISS. |
| `pipeline/step4_extract_claims.py` | Backstory claim parser extracting testable atomic sentences. |
| `pipeline/step5_retrieve_evidence.py` | Vector retrieval engine fetching top candidate novel passages per claim. |
| `pipeline/step6_evaluate_claims.py` | Sentence-level entity-aware polarity scorer detecting factual negation. |
| `pipeline/step7_temporal_analysis.py` | Chapter order timeline tracking and late-narrative progression weighting. |
| `pipeline/step8_aggregate_decision.py` | Decision threshold aggregator generating final predictions and metric evaluation reports. |
| `pipeline/llm_provider.py` | Unified LLM interface supporting external APIs (OpenAI/Gemini) and offline entity-aware reasoning. |
| `pipeline/metrics_genai_disagreements.py` | Diagnostic script analyzing prediction discrepancies across pipelines. |
| `pipeline/run_all.py` | Master orchestrator running steps 1–8 end-to-end and saving CSV outputs. |
| `pipeline/results.csv` | Subfolder-isolated output copy of Pipeline 1 results. |

#### 2. NLI Transformer Pipeline (`pipeline_nli/`)
*For full technical details, see [pipeline_nli/README.md](pipeline_nli/README.md).*

| File Path | Purpose |
| :--- | :--- |
| `pipeline_nli/README.md` | Detailed NLI premise-hypothesis verification documentation. |
| `pipeline_nli/step0_config.py` | NLI pipeline configuration parameters and model checkpoints. |
| `pipeline_nli/step1_load_data.py` | Ingests train/test datasets into normalized JSON formats. |
| `pipeline_nli/step2_claim_extraction.py` | Sentence boundary parser extracting discrete claim hypotheses. |
| `pipeline_nli/step3_retrieve_evidence.py` | Passage retriever fetching candidate premise excerpts. |
| `pipeline_nli/step4_nli_inference.py` | Batched NLI inference classifier (Entailment / Contradiction / Neutral). |
| `pipeline_nli/step5_aggregate_decision.py` | Row decision aggregator mapping claim labels to backstory verdicts. |
| `pipeline_nli/models.py` | Transformer NLI model wrapper and heuristic fallback engine. |
| `pipeline_nli/run_nli_smoke.py` | Smoke testing script for NLI execution verification. |
| `pipeline_nli/run_pipeline.py` | Master orchestrator running NLI steps 1–5 end-to-end. |
| `pipeline_nli/results.csv` | Subfolder-isolated output copy of Pipeline 2 results. |

#### 3. GenAI Reasoner & Hallucination Guard (`pipelinegenai/`)
*For full technical details, see [pipelinegenai/README.md](pipelinegenai/README.md).*

| File Path | Purpose |
| :--- | :--- |
| `pipelinegenai/README.md` | Detailed documentation for GenAI reasoning and guardrails. |
| `pipelinegenai/genai_integration.py` | Core library for claim extraction, reasoner invocation, hallucination guard, and aggregation. |
| `pipelinegenai/models.py` | Pydantic data schemas (`Claim`, `EvidenceChunk`, `ClaimDecision`, `ReasoningOutput`). |
| `pipelinegenai/run_pipeline.py` | Master standalone runner generating predictions and rationales. |
| `pipelinegenai/results.csv` | Subfolder-isolated output copy of Pipeline 3 results. |
| `pipelinegenai/prompts/` | Prompt templates for claim extraction, reasoning, and cause-effect checks. |
| `pipelinegenai/schemas/` | Formal JSON Schemas validating output conformance. |
| `pipelinegenai/tests/` | Unit tests (`test_genai_integration.py`) verifying deterministic reasoning and guardrail rules. |

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

Cross-evaluating all three pipelines against ground-truth character backstory verification data:

| Metric | **Pipeline 1: Pathway RAG Engine** | **Pipeline 2: NLI Transformers** | **Pipeline 3: GenAI Reasoner & Guard** |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **62.5%** | **48.7%** | **62.5%** |
| **F1-Score** | **0.727** | **0.468** | **0.746** |
| **Recall** | **80.0%** | **35.3%** | **86.3%** |
| **Precision** | **66.7%** | **69.2%** | **65.7%** |

### Benchmark Analysis & Takeaways
- **Pipeline 1 (Pathway Streaming RAG)**: Demonstrates strong, balanced performance (**62.5% Accuracy, 0.727 F1**) by leveraging real-time streaming index tables and temporal chapter progression.
- **Pipeline 2 (NLI Transformers)**: Operates with a conservative entailment threshold, achieving high precision (**69.2%**) but lower recall (**35.3%**).
- **Pipeline 3 (GenAI Reasoner & Guard)**: Achieves top overall F1-Score (**0.746**) and highest Recall (**86.3%**) due to typed Pydantic claim extraction and hallucination guard validation.

---

## Sub-Pipeline Detailed Documentation

For full architectural breakdowns, step-by-step module flows, and schema definitions, see:
- 📖 [Pipeline 1: Pathway RAG Engine Documentation](pipeline/README.md)
- 📖 [Pipeline 2: Natural Language Inference Documentation](pipeline_nli/README.md)
- 📖 [Pipeline 3: GenAI Reasoner & Guardrails Documentation](pipelinegenai/README.md)