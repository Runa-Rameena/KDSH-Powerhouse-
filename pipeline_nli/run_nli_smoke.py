"""
Small smoke runner: run NLI on one claim and top-k pathway-retrieved chunks.
Produces artifacts/pipeline_nli/nli_outputs/smoke_nli_results.jsonl

This script will attempt to load a HuggingFace NLI model; if model loading or inference
fails (e.g., no network/auth), it falls back to a deterministic heuristic NLI function
based on token overlap and simple negation detection. This keeps the end-to-end smoke
flow deterministic and traceable.
"""
from pathlib import Path
import json
import re
from step0_config import NLI_MODEL
from models import load_nli_model, run_nli_batch

# Choose claim and pathway retrieval file
CLAIM_FILE = Path.cwd() / "artifacts" / "pipeline_nli" / "claims" / "1.json"
PATHWAY_RESULT = Path.cwd() / "artifacts" / "pathway_retrieval" / "query_At_twelve,_Jacques_Paganel_fell_.json"

with open(CLAIM_FILE, "r", encoding="utf-8") as f:
    claims = json.load(f)
claim = claims[0]
claim_id = claim["claim_id"]
claim_text = claim["claim_text"]

with open(PATHWAY_RESULT, "r", encoding="utf-8") as f:
    retrieved = json.load(f)

pairs = []
for r in retrieved[:3]:
    pairs.append((claim_id, r["chunk_id"], r["text"], claim_text, r.get("score", 0.0)))


def heuristic_nli(premise: str, hypothesis: str):
    """Deterministic heuristic NLI:
    - If premise contains a negation word and shares tokens with hypothesis -> CONTRADICTION
    - If token overlap (content words) >= 3 -> ENTAILMENT
    - If overlap == 0 -> NEUTRAL
    - Else -> NEUTRAL
    Returns: (label, confidence)
    """
    negation_tokens = {"not", "n't", "no", "never", "none", "without", "nobody", "neither"}

    def tokens(s):
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return [t for t in s.split() if len(t) > 1]

    p_tokens = set(tokens(premise))
    h_tokens = set(tokens(hypothesis))
    overlap = len(p_tokens & h_tokens)
    if overlap == 0:
        return "NEUTRAL", 1.0
    if any(neg in p_tokens for neg in negation_tokens) and overlap >= 1:
        return "CONTRADICTION", 0.9
    if overlap >= 3:
        return "ENTAILMENT", 0.9
    return "NEUTRAL", float(overlap) / (len(h_tokens) + 1)


# Try to use a real NLI model; fallback to heuristic on any failure
use_fallback = False
try:
    tokenizer, model, device = load_nli_model(NLI_MODEL)
    premises = [p[2] for p in pairs]
    hypotheses = [p[3] for p in pairs]
    try:
        results = run_nli_batch(tokenizer, model, device, premises, hypotheses)
    except Exception:
        use_fallback = True
except Exception:
    use_fallback = True

out_path = Path.cwd() / "artifacts" / "pipeline_nli" / "nli_outputs" / "smoke_nli_results.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    if use_fallback:
        for (claim_id, chunk_id, premise, hypothesis, similarity) in pairs:
            lbl, conf = heuristic_nli(premise, hypothesis)
            entry = {"claim_id": claim_id, "chunk_id": chunk_id, "nli_label": lbl, "confidence_score": conf, "similarity_score": similarity}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    else:
        for (claim_id, chunk_id, _, _, similarity), (label, confidence) in zip(pairs, results):
            if label.startswith("ENTAIL"):
                lbl = "ENTAILMENT"
            elif label.startswith("CONTRA"):
                lbl = "CONTRADICTION"
            elif label.startswith("NEUT"):
                lbl = "NEUTRAL"
            else:
                lbl = label
            entry = {"claim_id": claim_id, "chunk_id": chunk_id, "nli_label": lbl, "confidence_score": confidence, "similarity_score": similarity}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Wrote {out_path} (fallback_used={use_fallback})")
