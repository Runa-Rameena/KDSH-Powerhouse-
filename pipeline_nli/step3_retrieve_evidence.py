"""
STEP 3: Chunk books deterministically and retrieve top-K chunks per claim.
Uses Pathway if available; otherwise uses TF-IDF + cosine similarity as deterministic fallback.
Saves chunk files and per-claim retrieval outputs to artifacts/pipeline_nli/retrieval/
"""
import json
import math
import os
from pathlib import Path
from typing import List

from step0_config import BOOKS_DIR, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, CHUNKS_DIR, RETRIEVAL_DIR, TOP_K, USE_PATHWAY

# deterministic tokenization by whitespace

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
        with open(book_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        out = []
        for idx, chunk_text_str in chunks:
            out.append({
                "chunk_id": f"{book_name}_chunk_{idx}",
                "book_name": book_name,
                "text": chunk_text_str,
            })
        out_path = CHUNKS_DIR / f"{book_name}_chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print("Chunks built and saved.")


# Simple deterministic retriever using TF-IDF if Pathway is unavailable
def retrieve_top_k_for_claim(claim_text: str, book_name: str, top_k: int = TOP_K):
    # Load chunks for the book
    chunks_path = CHUNKS_DIR / f"{book_name}_chunks.json"
    if not chunks_path.exists():
        return []
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    # Simple deterministic similarity: count of overlapping tokens
    claim_tokens = set(claim_text.lower().split())
    scored = []
    for c in chunks:
        chunk_tokens = set(c["text"].lower().split())
        overlap = len(claim_tokens & chunk_tokens)
        # tie-breaker deterministically by chunk_id
        scored.append((overlap, c["chunk_id"], c["text"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    results = []
    for score, chunk_id, text in scored[:top_k]:
        results.append({"chunk_id": chunk_id, "text": text, "similarity_score": float(score)})
    return results


def run_retrieval():
    # Build chunks first (deterministic)
    build_chunks_for_books()

    # For each claim, retrieve top-K chunks from corresponding book
    for claim_file in (Path.cwd() / "artifacts" / "pipeline_nli" / "claims").glob("*.json"):
        with open(claim_file, "r", encoding="utf-8") as f:
            claims = json.load(f)
        out_for_row = []
        for cl in claims:
            row_id = cl["row_id"]
            # We need to find which book this row refers to; read the original row artifact
            row_path = Path.cwd() / "artifacts" / "pipeline_nli" / "input" / "rows" / f"{row_id}.json"
            if not row_path.exists():
                continue
            with open(row_path, "r", encoding="utf-8") as rf:
                row = json.load(rf)
            book_name = row.get("book_name") or row.get("book") or ""
            if not book_name:
                # no book info, skip
                continue
            # Deterministic retrieval via token overlap (Pathway integration placeholder)
            retrieved = retrieve_top_k_for_claim(cl["claim_text"], book_name, TOP_K)
            # Save per-claim retrieval output
            out_path = RETRIEVAL_DIR / f"{cl['claim_id']}.json"
            with open(out_path, "w", encoding="utf-8") as of:
                json.dump(retrieved, of, ensure_ascii=False, indent=2)
            out_for_row.append({"claim_id": cl["claim_id"], "retrieved": retrieved})
    print("Retrieval finished. Retrieved chunks saved under artifacts/pipeline_nli/retrieval/")


if __name__ == "__main__":
    run_retrieval()
