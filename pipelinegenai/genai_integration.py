import hashlib
import json
import re
from typing import List, Dict, Any, Optional
from .models import Claim, ClaimDecision

# Helper: deterministic id
def deterministic_sha256(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.strip().encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()

# NOTE: llm_call is a callable provided by the integrating code. It should accept (system_prompt, user_prompt, temperature)

# 1) Claim extraction: returns list of Claim dicts + metadata
def extract_claims_genai(backstory_text: str, backstory_id: str, llm_call, model_version: str = "genai-v1") -> Dict[str, Any]:
    system = "You are a JSON-only claim extractor. Return {\"claims\":[...]} following schema; do NOT add extra text; temperature=0."
    user = f'Backstory_text: """{backstory_text}"""'
    raw = llm_call(system, user, temperature=0)
    data = json.loads(raw)
    claims_out = []
    for c in data.get("claims", []):
        claim = Claim(**c)
        if not claim.claim_id:
            claim.claim_id = deterministic_sha256(claim.text.lower(), backstory_id, model_version)
        claim_dict = claim.dict()
        claim_dict.update({"model_version": model_version, "source_backstory_id": backstory_id})
        claims_out.append(claim_dict)
    return {"claims": claims_out, "raw": raw, "model_version": model_version}

# 2) Reasoner: claim vs evidence
def reason_claim_vs_evidence_genai(claim: Dict[str,Any], evidence_chunks: List[Dict[str,Any]], llm_call=None, model_version: str = "genai-v1") -> Dict[str,Any]:
    system = "JSON-only reasoner. Output {label,evidence_ids,rationale}. Use ONLY provided chunks. temperature=0."
    evidence_json = json.dumps(evidence_chunks)
    user = f"Claim: {json.dumps(claim)}\nEvidence_chunks: {evidence_json}"
    raw = llm_call(system, user, temperature=0)
    parsed = json.loads(raw)
    decision = ClaimDecision(
        claim_id=claim["claim_id"],
        label=parsed["label"],
        evidence_ids=parsed.get("evidence_ids", []),
        rationale=parsed.get("rationale", "")[:300],
    )
    return {"decision": decision.dict(), "raw": raw, "model_version": model_version}

# 3) Hallucination guard
def hallucination_guard(claim: Dict[str,Any], decision: Dict[str,Any], evidence_chunks: List[Dict[str,Any]], support_threshold: float = 0.65, min_similarity: float = 0.3) -> Dict[str,Any]:
    failures = []
    chunk_map = {c["chunk_id"]: c for c in evidence_chunks}
    # evidence id validation
    for cid in decision.get("evidence_ids", []):
        if cid not in chunk_map:
            failures.append("evidence_id_missing:"+cid)
    # lexical anchor check
    entities = claim.get("entities", [])
    cited_texts = " ".join(chunk_map[cid]["text"] for cid in decision.get("evidence_ids",[]) if cid in chunk_map)
    if entities:
        found = any(re.search(r"\b" + re.escape(e).lower() + r"\b", cited_texts.lower()) for e in entities)
        if not found:
            failures.append("lexical_anchor_missing")
    # similarity check
    similarities = [chunk_map[cid]["similarity_score"] for cid in decision.get("evidence_ids",[]) if cid in chunk_map and "similarity_score" in chunk_map[cid]]
    max_sim = max(similarities) if similarities else 0.0
    if decision["label"] == "SUPPORT" and max_sim < support_threshold:
        failures.append("low_similarity_for_support")
    if max_sim < min_similarity:
        failures.append("overall_low_similarity")
    # apply overrides
    if failures:
        decision["validated"] = False
        decision.setdefault("validation_failures", []).extend(failures)
        # conservative fallback
        decision["label"] = "INSUFFICIENT"
    else:
        decision["validated"] = True
    return {"decision": decision, "flags": failures}

# 4) Aggregation
def aggregate_claim_decisions(decisions: List[Dict[str,Any]]) -> Dict[str,Any]:
    has_contradict = any(d.get("label") == "CONTRADICT" for d in decisions)
    final_label = 0 if has_contradict else 1
    return {"final_label": final_label, "counts": {"CONTRADICT": sum(1 for d in decisions if d.get("label")=="CONTRADICT"), "SUPPORT": sum(1 for d in decisions if d.get("label")=="SUPPORT"), "INSUFFICIENT": sum(1 for d in decisions if d.get("label")=="INSUFFICIENT")}}
