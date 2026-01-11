"""
STEP 2: Deterministic retrieval using Pathway-produced chunks CSV as data source.
- Reads `artifacts/pathway_genai/pw_chunks_stream.csv` produced by Pathway
- Extracts deterministic atomic claims from `train.csv` (falls back to `test.csv`) using sentence splitting
- Uses TF-IDF + cosine similarity to retrieve top-K chunks per claim
- Saves results to `artifacts/pathway_genai/retrieved_claims.json`

Run: python3 pipeline_pathway_genai/step2_retrieve.py
"""
from pathlib import Path
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "pathway_genai"
PW_CHUNKS_CSV = ARTIFACTS / "pw_chunks_stream.csv"
RETRIEVED_OUT = ARTIFACTS / "retrieved_claims.json"

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MIN_WORDS = 2
TOP_K = 5


def extract_claims_from_rows(df: pd.DataFrame):
    claims = []
    for _, row in df.iterrows():
        row_id = str(row.get("id", ""))
        text = (row.get("backstory") or row.get("content") or "")
        text = str(text).strip()
        if not text:
            continue
        sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
        for i, s in enumerate(sents, start=1):
            if len(s.split()) < MIN_WORDS:
                continue
            claims.append({"claim_id": f"{row_id}_claim_{i}", "claim_text": s, "row_id": row_id})
    return claims


def load_chunks():
    if not PW_CHUNKS_CSV.exists():
        raise FileNotFoundError(f"Pathway chunks CSV not found at {PW_CHUNKS_CSV}")
    df = pd.read_csv(PW_CHUNKS_CSV)
    # Expect columns: chunk_id, book_title, chunk_index, text
    df = df.fillna("")
    chunks = []
    for _, r in df.iterrows():
        chunks.append({"chunk_id": r["chunk_id"], "book_title": r.get("book_title", ""), "chunk_index": int(r.get("chunk_index", 0)), "text": r.get("text", "")})
    return chunks


def run_retrieval():
    # Load claims from train.csv or test.csv
    train_csv = ROOT / "train.csv"
    test_csv = ROOT / "test.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
    elif test_csv.exists():
        df = pd.read_csv(test_csv)
    else:
        raise FileNotFoundError("No train.csv or test.csv found at project root")
    claims = extract_claims_from_rows(df)
    if not claims:
        print("No claims extracted from rows")
        return

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_chunks = vectorizer.fit_transform(texts)

    results = []
    for cl in claims:
        q = cl["claim_text"]
        q_tf = vectorizer.transform([q])
        sims = cosine_similarity(tfidf_chunks, q_tf).reshape(-1)
        idxs = list(range(len(sims)))
        idxs.sort(key=lambda i: (-float(sims[i]), i))
        top = []
        for i in idxs[:TOP_K]:
            top.append({"chunk_id": chunks[i]["chunk_id"], "book_title": chunks[i]["book_title"], "score": float(sims[i]), "text": chunks[i]["text"]})
        results.append({"claim_id": cl["claim_id"], "row_id": cl["row_id"], "claim_text": cl["claim_text"], "retrieved": top})

    with open(RETRIEVED_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote retrieval results to {RETRIEVED_OUT} (claims: {len(results)}, chunks: {len(chunks)})")


if __name__ == "__main__":
    run_retrieval()
