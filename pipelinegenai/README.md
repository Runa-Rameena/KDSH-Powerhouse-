# GenAI Reasoning & Hallucination Guard Pipeline

This directory contains the GenAI and LLM-based narrative verification module for the Kharagpur Data Science Hackathon (KDSH).

---

## Overview

The GenAI pipeline utilizes structured JSON prompts, strict Pydantic/JSON schemas, temperature=0 determinism, and hallucination guardrails to verify character backstory claims against retrieved novel evidence.

### Components
1. **`models.py`**: Pydantic/dataclass schemas for atomic claims, reasoner verdicts, and cause-effect checks.
2. **`genai_integration.py`**: Core functions for claim extraction, reasoner invocation, hallucination guarding, and decision aggregation.
3. **`prompts/`**: Prompt templates for claim extraction and claim-vs-evidence reasoning.
4. **`schemas/`**: Formal JSON schemas ensuring output conformance.
5. **`run_pipeline.py`**: End-to-end standalone runner generating verified predictions.
6. **`tests/test_genai_integration.py`**: Unit tests verifying deterministic reasoning and guardrail behavior.

---

## How to Run

```bash
# Run end-to-end GenAI pipeline
python -m pipelinegenai.run_pipeline

# Run unit tests
python -m unittest pipelinegenai/tests/test_genai_integration.py
```

---

## Outputs & Artifacts

- **Pipeline Results**: `pipelinegenai/results.csv`
- **Root Results**: `results_genai.csv`
- **Schema**: `id,predicted_label,rationale` (1 = Consistent, 0 = Contradict)
