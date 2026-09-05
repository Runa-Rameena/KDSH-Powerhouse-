# Pipeline 3: GenAI Reasoner & Hallucination Guard Pipeline

This directory contains the **GenAI Reasoning & Hallucination Guard Pipeline**. It combines structured LLM reasoning prompts, Pydantic schema validation, temperature=0 determinism, and lexical anchor guardrails to evaluate backstory claims against novel evidence while guarding against model hallucination.

---

## Key System Design & Architectural Features

- **Structured Prompts**: Prompt templates in `prompts/` enforce deterministic, structured JSON outputs.
- **Pydantic Validation**: `models.py` defines schemas for atomic claims, reasoner verdicts, and cause-effect relationships.
- **Hallucination Guard**: Post-processes reasoner decisions to check whether quoted evidence actually exists in retrieved novel chunks.
- **Lexical Anchor Verification**: Checks key entities (names, places, dates) before finalizing a contradiction verdict.

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
     hallucination_guard() ────────► Verify quoted text against raw chunks
              │
              ▼
  aggregate_claim_decisions() ─────► Aggregate into row verdict & rationale
              │
              ▼
       run_pipeline.py ────────────► Standalone Module Runner
```

### Module Structure

1. **`models.py`**: Contains Pydantic dataclasses and models (`Claim`, `EvidenceChunk`, `ClaimDecision`, `ReasoningOutput`).
2. **`genai_integration.py`**:
   - `extract_claims_genai()`: Extracts testable claim objects from raw backstory text.
   - `reason_claim_vs_evidence_genai()`: Prompts LLM reasoner to evaluate claim against evidence.
   - `hallucination_guard()`: Validates that cited evidence snippets exist in original source text.
   - `aggregate_claim_decisions()`: Combines individual claim decisions into a final row label (`1` or `0`).
3. **`prompts/`**: Formatted prompt templates for extraction and reasoning.
4. **`schemas/`**: Formal JSON schemas for output validation.
5. **`run_pipeline.py`**: End-to-end runner reading `test.csv` and writing outputs.
6. **`tests/test_genai_integration.py`**: Unit tests verifying deterministic reasoning and guardrail compliance.

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
  - `predicted_label`: `1` (Consistent / Support), `0` (Contradict)
  - `rationale`: GenAI model version, decision counts summary, and hallucination validation state.
