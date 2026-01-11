import json
from pipeline.genai_integration import extract_claims_genai, reason_claim_vs_evidence_genai, hallucination_guard, deterministic_sha256

# Mock LLM that echoes a fixed response for testing
def mock_llm_claim_extractor(system, user, temperature=0):
    # Return a deterministic claim list irrespective of input for unit test
    return json.dumps({
        "claims": [
            {"text":"John lost his sight in an accident.", "entities":["John"], "anchor_span": [0, 30], "type": "EVENT"}
        ]
    })

def mock_llm_reasoner(system, user, temperature=0):
    # Return SUPPORT with chunk id chunk-1
    return json.dumps({"label": "SUPPORT", "evidence_ids": ["chunk-1"], "rationale": "text explicitly says John lost sight"})

def test_extract_claims_and_idempotence():
    backstory = "John, after the accident, lost his sight and adapted to living in darkness."
    res = extract_claims_genai(backstory, backstory_id="bs-1", llm_call=mock_llm_claim_extractor, model_version="test-v1")
    assert "claims" in res
    claim = res["claims"][0]
    assert claim["entities"] == ["John"]
    # deterministic id check
    expected = deterministic_sha256(claim["text"].lower(), "bs-1", "test-v1")
    assert claim["claim_id"] == expected

def test_reason_and_hallucination_guard():
    claim = {"claim_id":"cid-1","text":"John lost his sight.","entities":["John"]}
    evidence_chunks = [{"chunk_id":"chunk-1","text":"John lost his sight in the accident.","similarity_score":0.8}]
    res = reason_claim_vs_evidence_genai(claim, evidence_chunks, llm_call=mock_llm_reasoner, model_version="test-v1")
    decision = res["decision"]
    assert decision["label"] == "SUPPORT"
    # Guard should validate since chunk contains entity and similarity high
    guarded = hallucination_guard(claim, decision, evidence_chunks)
    assert guarded["decision"]["validated"] is True
    assert guarded["flags"] == []
