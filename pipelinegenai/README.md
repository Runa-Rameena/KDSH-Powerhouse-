# Pipeline 3: GenAI Reasoner & Hallucination Guard Pipeline

This directory contains the **GenAI Reasoning & Hallucination Guard Pipeline**. It combines structured LLM reasoning prompts, Pydantic schema validation, temperature=0 determinism, and lexical anchor guardrails to evaluate backstory claims against novel evidence while guarding against model hallucination.

---

## Key System Design & Architectural Features

- **Structured Prompts**: Prompt templates in `prompts/` enforce deterministic, structured JSON outputs.
- **Pydantic Validation**: `models.py` defines schemas for atomic claims, reasoner verdicts, and cause-effect relationships.
- **Hallucination Guard**: Post-processes reasoner decisions to check whether quoted evidence actually exists in retrieved novel chunks and matches similarity thresholds.
- **Lexical Anchor Verification**: Checks key entities (names, places, dates) before finalizing a contradiction verdict.

---

## Complete File Inventory & Purpose Breakdown

| File Name / Subdirectory | Purpose & Functionality |
| :--- | :--- |
| **`genai_integration.py`** | Core reasoning & guardrail library (`extract_claims_genai`, `reason_claim_vs_evidence_genai`, `hallucination_guard`, `aggregate_claim_decisions`). |
| **`models.py`** | Pydantic data schemas defining structures for `Claim`, `EvidenceChunk`, `ClaimDecision`, and `ReasoningOutput`. |
| **`run_pipeline.py`** | Master pipeline runner ingesting `test.csv`, retrieving novel chunks, evaluating claims, and exporting CSV outputs. |
| **`results.csv`** | Subfolder-isolated output CSV containing Pipeline 3 predictions (`id,predicted_label,rationale`). |
| **`prompts/claim_extraction.txt`** | Prompt template instructing LLM to extract atomic testable claims from character backstories. |
| **`prompts/reasoner.txt`** | Prompt template guiding LLM to reason over claims vs retrieved novel evidence chunks. |
| **`prompts/cause_effect.txt`** | Prompt template verifying causal consistency and chronological narrative events. |
| **`schemas/extracted_claims.json`** | Formal JSON Schema validating claim extraction responses. |
| **`schemas/claim_decision.json`** | Formal JSON Schema validating claim reasoner output structure. |
| **`schemas/cause_effect_check.json`** | Formal JSON Schema validating cause-effect verification responses. |
| **`tests/test_genai_integration.py`** | Unit test suite verifying deterministic reasoning and guardrail validation rules. |

---

## System Flow & Component Overview

```
Input Backstory & Novel Evidence
              │
              ▼
    extract_claims_genai() ────────► Extract atomic claims via GenAI prompts
              │
              ▼
reason_claim_vs_evidence_genai() ──► Evaluate Claim vs Evidence chunks
              │
              ▼
     hallucination_guard() ────────► Verify quoted text against raw chunks & similarity
              │
              ▼
  aggregate_claim_decisions() ─────► Aggregate into row verdict & rationale
              │
              ▼
       run_pipeline.py ────────────► Master Standalone Runner
```

---

## How to Run

```bash
# Run end-to-end GenAI pipeline
python3 -m pipelinegenai.run_pipeline

# Run unit tests for GenAI modules
python3 -m unittest pipelinegenai/tests/test_genai_integration.py
```

---

## Outputs & Artifacts

- **Local Results**: `pipelinegenai/results.csv`
- **Root Results**: `results_genai.csv`
- **Output Schema**:
  ```csv
  id,predicted_label,rationale
  46,1,"GenAI-v1 | decisions={'SUPPORT': 2, 'CONTRADICT': 0} | validated=True"
  137,0,"GenAI-v1 | decisions={'SUPPORT': 0, 'CONTRADICT': 1} | validated=True"
  ```
