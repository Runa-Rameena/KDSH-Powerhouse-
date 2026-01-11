"""
STEP 3 & 4: Deterministic chunk-level verification and aggregation to claim-level results.

- Reads `artifacts/pathway_genai/retrieved_claims.json`
- For each (claim, chunk) pair computes overlap, negation, and similarity based rules
- Writes chunk-level decisions to `artifacts/pathway_genai/claim_chunk_verdicts.jsonl`
- Aggregates per-claim decisions into `artifacts/pathway_genai/results.csv`

Run: python3 pipeline_pathway_genai/step3_verify_and_aggregate.py
"""
from pathlib import Path
import json
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "pathway_genai"
RETRIEVED = ART / "retrieved_claims.json"
CHUNK_VERDICTS = ART / "claim_chunk_verdicts.jsonl"
RESULTS_CSV = ART / "results.csv"

NEG_TOKENS = {"not", "n't", "no", "never", "none", "without", "nobody", "neither"}
TOKEN_RE = re.compile(r"[^a-z0-9]+")

# thresholds (deterministic)
CONTRADICT_SIM_THRESHOLD = 0.35
SUPPORT_SIM_THRESHOLD = 0.40


def tokens(s: str):
    s = s.lower()
    s = TOKEN_RE.sub(" ", s)
    toks = [t for t in s.split() if len(t) > 1]
    return toks


# small stopword list to heuristically detect key tokens (nouns/entities)
STOPWORDS = {"the", "and", "or", "a", "an", "in", "on", "of", "for", "to", "by", "with", "as", "at", "from", "into", "about", "after", "before", "over", "under", "between", "among", "but", "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those", "he", "she", "they", "we", "you", "it", "its", "his", "her", "their", "them", "who", "which"}

def chunk_level_verdict(claim_text: str, chunk_text: str, similarity: float):
    c_toks = set(tokens(chunk_text))
    h_toks = set(tokens(claim_text))
    overlap = len(c_toks & h_toks)
    negation = any(tok in c_toks for tok in NEG_TOKENS)

    # prepare key claim tokens (heuristic nouns/entities): longer tokens not in stopwords
    key_tokens = [t for t in h_toks if len(t) > 2 and t not in STOPWORDS]
    matched_tokens = [t for t in key_tokens if t in c_toks]

    # Contradiction: only if negation detected AND similarity >= threshold
    if negation and similarity >= CONTRADICT_SIM_THRESHOLD:
        return "CONTRADICTS", "", overlap, negation, []

    # Strong support: similarity >= SUPPORT_SIM_THRESHOLD AND at least 2 matching key tokens
    if similarity >= SUPPORT_SIM_THRESHOLD and len(matched_tokens) >= 2:
        return "SUPPORTS", "strong", overlap, negation, matched_tokens

    # Support (non-strong): similarity above support threshold
    if similarity >= SUPPORT_SIM_THRESHOLD:
        return "SUPPORTS", "", overlap, negation, matched_tokens

    return "NEUTRAL", "", overlap, negation, []


def run():
    if not RETRIEVED.exists():
        raise FileNotFoundError(f"Retrieved file not found: {RETRIEVED}")
    with open(RETRIEVED, "r", encoding="utf-8") as f:
        claims = json.load(f)

    # chunk-level verdicts
    with open(CHUNK_VERDICTS, "w", encoding="utf-8") as vf:
        for cl in claims:
            claim_id = cl.get("claim_id")
            row_id = cl.get("row_id")
            claim_text = cl.get("claim_text", "")
            retrieved = cl.get("retrieved", [])
            for r in retrieved:
                chunk_id = r.get("chunk_id")
                chunk_text = r.get("text", "")
                sim = float(r.get("score", 0.0))
                label, strength, overlap, neg, matched = chunk_level_verdict(claim_text, chunk_text, sim)
                entry = {
                    "claim_id": claim_id,
                    "row_id": row_id,
                    "chunk_id": chunk_id,
                    "n_sent": overlap,
                    "negation": neg,
                    "similarity": sim,
                    "chunk_label": label,
                    "support_strength": strength,
                    "matched_tokens": matched,
                }
                vf.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Aggregation
    # For each claim accumulate counts and max similarity
    agg = {}
    with open(CHUNK_VERDICTS, "r", encoding="utf-8") as vf:
        for line in vf:
            e = json.loads(line)
            cid = e["claim_id"]
            rec = agg.setdefault(cid, {"row_id": e["row_id"], "claim_text": "", "support_count": 0, "contradict_count": 0, "strong_support_present": False, "max_sim": 0.0})
            # claim_text needs to be filled; find from original claims list map
            rec["max_sim"] = max(rec["max_sim"], float(e.get("similarity", 0.0)))
            if e["chunk_label"] == "CONTRADICTS":
                rec["contradict_count"] += 1
            elif e["chunk_label"] == "SUPPORTS":
                rec["support_count"] += 1
                if e.get("support_strength") == "strong":
                    rec["strong_support_present"] = True

    # fill claim_text from original claims
    claim_text_map = {c["claim_id"]: c["claim_text"] for c in claims}
    for cid, rec in agg.items():
        rec["claim_text"] = claim_text_map.get(cid, "")

    # Determine final label and write CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["row_id", "claim_id", "claim_text", "final_label", "support_count", "contradict_count", "max_similarity"])
        writer.writeheader()
        for cid, rec in sorted(agg.items()):
            if rec["contradict_count"] > 0:
                final = "CONTRADICTS"
            elif rec["strong_support_present"]:
                final = "SUPPORTS"
            else:
                final = "INSUFFICIENT"
            writer.writerow({"row_id": rec["row_id"], "claim_id": cid, "claim_text": rec["claim_text"], "final_label": final, "support_count": rec["support_count"], "contradict_count": rec["contradict_count"], "max_similarity": rec["max_sim"]})

    print(f"Wrote {CHUNK_VERDICTS} and {RESULTS_CSV}")


if __name__ == "__main__":
    run()
