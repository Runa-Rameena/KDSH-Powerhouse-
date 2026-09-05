import json
from pathlib import Path

try:
    from .step0_config import BOOKS_DIR, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, CHUNKS_DIR, RETRIEVAL_DIR, CLAIMS_DIR, INPUT_ROWS_DIR, TOP_K
except (ImportError, ValueError):
    from pipeline_nli.step0_config import BOOKS_DIR, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, CHUNKS_DIR, RETRIEVAL_DIR, CLAIMS_DIR, INPUT_ROWS_DIR, TOP_K

BOOK_CHUNKS_CACHE = {}

def chunk_text(text: str, size: int, overlap: int):
    words = text.split()
    chunks = []
    if size <= 0:
        return chunks
    i = 0
    idx = 0
    while i < len(words):
        chunk_words = words[i : i + size]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append((idx, chunk_text))
            idx += 1
        i += size - overlap
    return chunks

def build_chunks_for_books():
    for book_file in BOOKS_DIR.glob("*.txt"):
        book_name = book_file.stem
        out_path = CHUNKS_DIR / f"{book_name}_chunks.json"
        if out_path.exists():
            continue
        with open(book_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        out = [{
            "chunk_id": f"{book_name}_chunk_{idx}",
            "book_name": book_name,
            "text": chunk_text_str,
        } for idx, chunk_text_str in chunks]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

def get_book_chunks(book_name: str):
    if book_name in BOOK_CHUNKS_CACHE:
        return BOOK_CHUNKS_CACHE[book_name]
    chunks_path = CHUNKS_DIR / f"{book_name}_chunks.json"
    if not chunks_path.exists():
        for cp in CHUNKS_DIR.glob("*_chunks.json"):
            stem_norm = cp.name.lower().replace(" ", "").replace("_chunks.json", "")
            if stem_norm == book_name.lower().replace(" ", ""):
                chunks_path = cp
                break
    if not chunks_path.exists():
        return []
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    for c in chunks:
        c["_tokens"] = set(c["text"].lower().split())
    BOOK_CHUNKS_CACHE[book_name] = chunks
    return chunks

def retrieve_top_k_for_claim(claim_text: str, book_name: str, top_k: int = TOP_K):
    chunks = get_book_chunks(book_name)
    if not chunks:
        return []
    claim_tokens = set(claim_text.lower().split())
    scored = []
    for c in chunks:
        overlap = len(claim_tokens & c["_tokens"])
        scored.append((overlap, c["chunk_id"], c["text"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [{"chunk_id": chunk_id, "text": text, "similarity_score": float(score)} for score, chunk_id, text in scored[:top_k]]

def run():
    build_chunks_for_books()
    row_cache = {}
    for p in INPUT_ROWS_DIR.glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            row_cache[p.stem] = json.load(f)

    for claim_file in CLAIMS_DIR.glob("*.json"):
        with open(claim_file, "r", encoding="utf-8") as f:
            claims = json.load(f)
        for cl in claims:
            row_id = cl["row_id"]
            row = row_cache.get(str(row_id))
            if not row:
                continue
            book_name = row.get("book_name") or row.get("book") or row.get("story_id") or ""
            if not book_name:
                continue
            retrieved = retrieve_top_k_for_claim(cl["claim_text"], book_name, TOP_K)
            out_path = RETRIEVAL_DIR / f"{cl['claim_id']}.json"
            with open(out_path, "w", encoding="utf-8") as of:
                json.dump(retrieved, of, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run()
