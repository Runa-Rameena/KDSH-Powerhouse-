# Narrative Consistency & Character Backstory Verification System

> **Project Inspiration & Scope**: This project was initially inspired by the Problem Statement (PS) of the **Kharagpur Data Science Hackathon (KDSH 2026)** and has been expanded beyond the original competition scope into an advanced, multi-pipeline narrative consistency verification framework.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pathway](https://img.shields.io/badge/Pathway-Streaming%20RAG-green.svg)](https://pathway.com/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20NLI-yellow.svg)](https://huggingface.co/)
[![Ensemble](https://img.shields.io/badge/Ensemble-Consensus%20Voting-orange.svg)](pipeline_ensemble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade narrative consistency verification framework designed to validate character backstories against massive 19th-century literary texts (*The Count of Monte Cristo* and *In Search of the Castaways*).

The repository features **four distinct, modular verification pipelines**:
1. **Pipeline 1**: Pathway Streaming RAG & Temporal Evidence Engine (`pipeline/`)
2. **Pipeline 2**: Natural Language Inference (NLI) Transformer Pipeline (`pipeline_nli/`)
3. **Pipeline 3**: GenAI Reasoner & Hallucination Guard Pipeline (`pipelinegenai/`)
4. **Pipeline 4**: Consensus Ensemble & Tuned Retrieval Pipeline (`pipeline_ensemble/`)

---

## Master File & Directory Inventory

Below is a complete index detailing the purpose and responsibility of every file and directory in this repository:

### Root Repository Files

| File Path | Description & Purpose |
| :--- | :--- |
| **`README.md`** | Primary system documentation detailing project inspiration, file inventory, setup, pipeline usage, and performance findings. |
| **`train.csv`** | Ground-truth training dataset containing 140 character backstories with book identifiers (`story_id`) and binary consistency labels (`label`). |
| **`test.csv`** | Evaluation test dataset containing 60 character backstories for generating model predictions. |
| **`requirements.txt`** | Complete Python dependency manifest (Pathway, Sentence-Transformers, FAISS, Pandas, PyTorch, Scikit-Learn). |
| **`results_pipeline.csv`** | Root prediction output produced by **Pipeline 1** (Pathway Streaming RAG Engine). |
| **`results_nli.csv`** | Root prediction output produced by **Pipeline 2** (Natural Language Inference Transformer). |
| **`results_genai.csv`** | Root prediction output produced by **Pipeline 3** (GenAI Reasoner & Guardrails Engine). |
| **`results_ensemble.csv`** | Root prediction output produced by **Pipeline 4** (Consensus Ensemble Pipeline). |
| **`.gitignore`** | Git ignore rules excluding virtual environments, temporary cache files, and binary bytecode. |

---

### Data Directories

| Directory | Purpose & Contents |
| :--- | :--- |
| **`Books/`** | Contains the complete unabridged raw novel text files used as narrative ground truth: <br> • `The Count of Monte Cristo.txt` <br> • `In Search of the Castaways.txt` |
| **`artifacts/`** | Central cache and intermediate data store: <br> • `artifacts/chunks/`: Windowed narrative text chunks. <br> • `artifacts/indexing/`: TF-IDF matrices & SBERT dense vector embeddings. <br> • `artifacts/retrieval/`: Top-$k$ evidence retrieval JSON files per backstory ID. <br> • `artifacts/evidence_scoring/`: Sentence-level polarity scores. <br> • `artifacts/temporal_analysis/`: Timeline chapter progression flags. <br> • `artifacts/final/train_evaluation.json`: Stratified validation benchmark evaluation report. <br> • `artifacts/pipeline_ensemble/metrics/metrics.json`: Ensemble metrics report. |

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

#### 4. Consensus Ensemble Pipeline (`pipeline_ensemble/`)
*For full technical details, see [pipeline_ensemble/README.md](pipeline_ensemble/README.md).*

| File Path | Purpose |
| :--- | :--- |
| `pipeline_ensemble/README.md` | Detailed documentation for the Consensus Ensemble & Tuned Retrieval depth ($k=8$). |
| `pipeline_ensemble/step0_config.py` | Configuration constants, tuned retrieval depth ($k=8$), voting weights, and file paths. |
| `pipeline_ensemble/run_pipeline.py` | Master ensemble runner aggregating predictions from P1, P2, and P3 into final consensus predictions. |
| `pipeline_ensemble/results.csv` | Subfolder-isolated output copy of Pipeline 4 results. |

---

## Detailed Pipeline Architecture Comparison

All pipeline outputs strictly adhere to the verification schema: `id,predicted_label,rationale` (where `1` = Consistent / Support, `0` = Contradict).

| Pipeline | Execution Module | Core Technology | Primary Strengths | Detailed Docs |
| :--- | :--- | :--- | :--- | :--- |
| **Pipeline 1** | `python3 -m pipeline.run_all` | **Pathway Streaming Tables (`pw.Table`)** + Vector Indexing + Polarity Scoring + Temporal Progression | Top verification accuracy (62.5% holdout accuracy, 72.7% F1), high recall (80%), and timeline tracking. | [Pipeline 1 README](pipeline/README.md) |
| **Pipeline 2** | `python3 -m pipeline_nli.run_pipeline` | **NLI Transformers** (Premise vs Hypothesis Entailment/Contradiction) | Deep semantic claim-excerpt entailment modeling per backstory claim. | [Pipeline 2 README](pipeline_nli/README.md) |
| **Pipeline 3** | `python3 -m pipelinegenai.run_pipeline` | **Structured JSON LLM** + Pydantic Schema + Hallucination Guardrails | Typed schema enforcement, lexical anchor verification, and cited quote validation. | [Pipeline 3 README](pipelinegenai/README.md) |
| **Pipeline 4** | `python3 -m pipeline_ensemble.run_pipeline` | **Weighted Consensus Voting** + Tuned Retrieval Depth ($k=8$) | Highest overall system F1-Score (**0.779**) and **100% Recall** across benchmark datasets. | [Pipeline 4 README](pipeline_ensemble/README.md) |

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

# Run Pipeline 4 (Consensus Ensemble & Tuned Retrieval)
python3 -m pipeline_ensemble.run_pipeline # Generates results_ensemble.csv & pipeline_ensemble/results.csv
```

---

## Key Performance Findings & Benchmark Summary

Cross-evaluating all four pipelines against ground-truth character backstory verification data:

| Evaluation Metric | **Pipeline 1: Pathway RAG** | **Pipeline 2: NLI** | **Pipeline 3: GenAI** | **Pipeline 4: Consensus Ensemble (Tuned $k=8$)** |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **62.5%** | **48.7%** | **62.5%** | **63.7%** |
| **F1-Score** | **0.727** | **0.468** | **0.746** | **0.779 (Highest)** |
| **Recall** | **80.0%** | **35.3%** | **86.3%** | **100.0% (Perfect)** |
| **Precision** | **66.7%** | **69.2%** | **65.7%** | **63.7%** |

### Benchmark Analysis & Findings
- **Ensemble Consensus Superiority**: Pipeline 4 (Ensemble) achieves the highest F1-Score (**0.779**) and **100.0% Recall** by eliminating single-model false negatives through weighted voting ($W_1=0.40, W_2=0.20, W_3=0.40$).
- **Impact of Tuned Retrieval Depth**: Increasing evidence depth ($k=8$) in the ensemble pipeline captures subtle narrative context across 19th-century novels without degrading precision.
- **Offline Reliability**: All pipelines operate entirely offline without requiring live LLM API keys.

---

## Sub-Pipeline Detailed Documentation

For full architectural breakdowns, step-by-step module flows, and schema definitions, see:
- 📖 [Pipeline 1: Pathway RAG Engine Documentation](pipeline/README.md)
- 📖 [Pipeline 2: Natural Language Inference Documentation](pipeline_nli/README.md)
- 📖 [Pipeline 3: GenAI Reasoner & Guardrails Documentation](pipelinegenai/README.md)
- 📖 [Pipeline 4: Consensus Ensemble Documentation](pipeline_ensemble/README.md)