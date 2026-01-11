## GenAI Integration: prompts, schemas, and tests

This folder contains minimal, auditable GenAI integration pieces for the evidence-based verification pipeline.

Files:
- `prompts/claim_extraction.txt` - System + user prompt for claim extraction (JSON-only, temp=0)
- `prompts/reasoner.txt` - Claim-vs-evidence reasoner prompt (JSON-only, temp=0)
- `prompts/cause_effect.txt` - Cause-effect contradiction check prompt (JSON-only, temp=0)
- `schemas/*.json` - JSON Schemas to validate outputs
- `models.py` - Pydantic models for Claim, ClaimDecision, CauseEffectCheck
- `genai_integration.py` - Minimal functions: extract_claims_genai, reason_claim_vs_evidence_genai, hallucination_guard, aggregate_claim_decisions
- `tests/test_genai_integration.py` - pytest tests with a mock llm

Running tests:
- Ensure `pytest` is installed: `pip install pytest pydantic`
- Run: `pytest pipeline/tests/test_genai_integration.py -q`

Usage notes:
- Provide a provider-specific `llm_call(system, user, temperature=0)` function by implementing `pipeline/llm_provider.py`.
- Keep temperature=0 and fixed `model_version` to ensure determinism.
- Use the JSON schemas (e.g., via `jsonschema` or Pydantic) to validate outputs and reject non-conforming responses.

Feature flags / toggles
- To enable GenAI claim extraction in Step 4 (runs in parallel, writes `claims_genai_<row_id>.json` and `disagreements_<row_id>.json`): set environment variable `GENAI_ENABLE_CLAIM_EXTRACTION=true`.
- To enable GenAI reasoner in Step 6 (runs in parallel, writes/updates `disagreements_<row_id>.json`): set environment variable `GENAI_ENABLE_REASONER=true`.
- Control mode via `GENAI_MODE`: `parallel` (default) or `primary`. In `parallel` mode outputs are logged and disagreements recorded; in `primary` mode, validated GenAI outputs can be used as primary decisions (use with caution and after evaluation).
- Disagreements from Step 4 (claim extraction) and Step 6 (reasoner) are merged into `artifacts/backstory_claims/disagreements_<row_id>.json` when present. Reasoner entries are tagged with `"source": "reasoner"` and `"model_version"` so you can filter/aggregate them easily. Step 6 will merge into existing disagreement files rather than overwrite them.
- Example (bash):
  export GENAI_ENABLE_CLAIM_EXTRACTION=true
  export GENAI_ENABLE_REASONER=true
  export GENAI_MODE=parallel

Notes:
- By default heuristics remain the final authority for pipeline outputs; this behavior is safe for gradual rollout and evaluation.

Quick analysis script ✅
- A small, standalone script is provided to compute GenAI vs heuristic disagreement statistics: `pipeline/metrics_genai_disagreements.py`.
- Run it like:

```bash
python3 pipeline/metrics_genai_disagreements.py --artifacts artifacts/backstory_claims
```

- The script prints counts and percentages: total files scanned, reasoner disagreement entries, counts per GenAI label and per heuristic overall label, agreement rate (GenAI vs heuristic where comparable), GenAI insufficient fraction, and validation stats (if present).
- Interpretation tips:
  - "Agreement rate" shows how often GenAI label equals the heuristic-derived label (simple aggregation of per-hit heuristic labels). Use it to gauge alignment; low agreement suggests manual review is warranted.
  - "GenAI insufficient/invalid" fraction shows cases where GenAI did not produce a decisive support/contradict label (e.g., `INSUFFICIENT`) — a high fraction may indicate poor prompt coverage or low evidence density.
  - This script is for analysis only; it does not change pipeline outputs or authority.

