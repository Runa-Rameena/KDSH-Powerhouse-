"""
STEP 2: Deterministic claim extraction from backstory text.
Uses simple sentence splitting by punctuation.
Saves per-row claims to artifacts/pipeline_nli/claims/<row_id>.json
"""
import json
import re
from pathlib import Path
from step0_config import INPUT_ROWS_DIR, CLAIMS_DIR

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MIN_WORDS = 2


def extract_sentences(text: str):
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]
    return parts


def row_to_claims(row_json_path: Path):
    with open(row_json_path, "r", encoding="utf-8") as f:
        row = json.load(f)
    text = row.get("backstory", "") or row.get("content", "")
    sents = extract_sentences(text)
    claims = []
    for i, s in enumerate(sents, start=1):
        if len(s.split()) < MIN_WORDS:
            continue
        claim = {
            "claim_id": f"{row['id']}_claim_{i}",
            "claim_text": s,
            "row_id": row["id"],
        }
        claims.append(claim)
    out_path = CLAIMS_DIR / f"{row['id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(claims, f, ensure_ascii=False, indent=2)
    return claims


if __name__ == "__main__":
    print("Extracting claims for each input row...")
    for p in INPUT_ROWS_DIR.glob("*.json"):
        row_to_claims(p)
    print("Claims extraction finished.")
