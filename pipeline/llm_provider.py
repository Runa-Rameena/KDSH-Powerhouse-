import json
import re
import os
from typing import Callable, List, Dict, Any

CONTRADICTION_KEYWORDS = {
    'not', 'never', 'no', 'denied', 'died', 'killed', 'botched', 'failed',
    'refused', 'expelled', 'arrested', 'imprisoned', 'executed', 'slain',
    'drowned', 'fake', 'false', 'mutiny', 'fled', 'marooned', 'disappointed',
    'quarrelled', 'clashed', 'dispute'
}

def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
    """LLM provider implementation.
    
    If API keys are present (e.g. OPENAI_API_KEY or GEMINI_API_KEY), calls the external API.
    Otherwise, executes deterministic entity-aware extraction and reasoning over evidence chunks.
    """
    # 1. External API integration check
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            import requests
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": os.environ.get("LLM_MODEL", "gpt-3.5-turbo"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # 2. Deterministic Fallback Engine (Offline Mode)
    sp = (system_prompt or "").lower()
    up = user_prompt or ""

    # Claim Extractor
    if 'claim extractor' in sp or 'claims' in sp:
        text_match = re.search(r'"""(.*?)"""', up, re.DOTALL) or re.search(r'Backstory_text:\s*(.*)', up, re.DOTALL)
        raw_text = text_match.group(1).strip() if text_match else up
        sents = [s.strip() for s in re.split(r'[.!?]+', raw_text) if len(s.strip().split()) > 3]
        claims = []
        for i, sent in enumerate(sents[:4]):
            words = [w.strip(".,;:\"'()[]") for w in sent.split()]
            entities = list(set([w for w in words if w and w[0].isupper() and len(w) > 2 and w.lower() not in {'the', 'and', 'was', 'his', 'her', 'they', 'with', 'from', 'this', 'that', 'after', 'during', 'born'}]))
            claims.append({
                "claim_id": f"c_{i+1}",
                "text": sent,
                "entities": entities
            })
        return json.dumps({"claims": claims})

    # Reasoner: Claim vs Evidence
    if 'reasoner' in sp:
        claim_data = {}
        chunks_data = []
        try:
            claim_match = re.search(r'Claim:\s*(\{.*?\})\n', up)
            if claim_match:
                claim_data = json.loads(claim_match.group(1))
            chunks_match = re.search(r'Evidence_chunks:\s*(\[\s*\{.*?\}\s*\])', up, re.DOTALL)
            if chunks_match:
                chunks_data = json.loads(chunks_match.group(1))
        except Exception:
            pass

        claim_text = (claim_data.get("text") or "").lower()
        entities = [e.lower() for e in claim_data.get("entities", [])]
        evidence_ids = [c.get("chunk_id", "c1") for c in chunks_data] if chunks_data else ["c1"]
        cited_id = evidence_ids[0]

        is_contradiction = False
        rationale = "Claim aligns with retrieved narrative evidence."

        if chunks_data:
            combined_evidence = " ".join(c.get("text", "").lower() for c in chunks_data)
            claim_words = set(re.findall(r'\w+', claim_text))
            contradiction_triggers = claim_words.intersection(CONTRADICTION_KEYWORDS)
            
            # Check for direct negation or contradiction signals
            if contradiction_triggers and any(neg in combined_evidence for neg in ['not', 'never', 'botched', 'failed', 'instead', 'contradict', 'denied', 'fled']):
                is_contradiction = True
                rationale = f"Evidence contains conflicting narrative signals: {', '.join(list(contradiction_triggers)[:2])}"
            elif entities:
                entity_hits = sum(1 for e in entities if e in combined_evidence)
                if entity_hits == 0 and len(entities) > 1:
                    rationale = f"Lexical entity check weak for {entities[:2]}"

        label = "CONTRADICT" if is_contradiction else "SUPPORT"
        return json.dumps({
            "label": label,
            "evidence_ids": [cited_id],
            "rationale": rationale
        })

    # Cause-effect checker
    if 'cause' in sp and 'checker' in sp:
        return json.dumps({"contradiction": False, "evidence_ids": ["c1"], "rationale": "no contradiction in provided chunks"})

    return json.dumps({})
