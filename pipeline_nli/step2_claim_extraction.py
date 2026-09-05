import json
import re
from pathlib import Path

try:
    from .step0_config import INPUT_ROWS_DIR, CLAIMS_DIR
except (ImportError, ValueError):
    from pipeline_nli.step0_config import INPUT_ROWS_DIR, CLAIMS_DIR

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MIN_WORDS = 2

def extract_sentences(text: str):
    if not text:
        return []
    return [p.strip() for p in SENT_SPLIT_RE.split(text) if p.strip()]

def row_to_claims(row_json_path: Path):
    with open(row_json_path, "r", encoding="utf-8") as f:
        row = json.load(f)
    text = row.get("backstory", "") or row.get("content", "")
    sents = extract_sentences(text)
    claims = []
    for i, s in enumerate(sents, start=1):
        if len(s.split()) < MIN_WORDS:
            continue
        claims.append({
            "claim_id": f"{row['id']}_claim_{i}",
            "claim_text": s,
            "row_id": str(row["id"]),
        })
    out_path = CLAIMS_DIR / f"{row['id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(claims, f, ensure_ascii=False, indent=2)
    return claims

def run():
    for p in INPUT_ROWS_DIR.glob("*.json"):
        row_to_claims(p)

if __name__ == "__main__":
    run()
