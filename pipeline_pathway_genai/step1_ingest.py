import json
from pathlib import Path
from step0_config import BOOKS_DIR, CHUNKS_FILE, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS
try:
    import pathway as pw
except Exception as e:
    raise


def chunk_text(text, size, overlap):
    words = text.split()
    out = []
    i = 0
    idx = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size]).strip()
        if chunk:
            out.append((idx, chunk))
            idx += 1
        i += size - overlap
    return out


def hash_vec(s, dim=32):
    import hashlib
    v = []
    for i in range(dim):
        h = int(hashlib.md5((s + str(i)).encode()).hexdigest()[:8], 16)
        v.append(((h % 1000) / 1000.0))
    return v


def main():
    all_chunks = []
    for txt in sorted(Path(BOOKS_DIR).glob("*.txt")):
        book = txt.stem
        text = txt.read_text(encoding="utf-8", errors="ignore")
        for idx, c in chunk_text(text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS):
            chunk_id = f"{book}_chunk_{idx}"
            all_chunks.append({"chunk_id": chunk_id, "book_title": book, "chunk_index": idx, "text": c})
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    # Create a Pathway table from the chunks and persist it using Pathway IO
    try:
        import pandas as pd
        df = pd.DataFrame(all_chunks)
        # Use chunk_id as the row id
        table = pw.debug.table_from_pandas(df, id_from=["chunk_id"]) if hasattr(pw.debug, 'table_from_pandas') else pw.debug.table_from_markdown('')
        out_csv = Path(CHUNKS_FILE).parent / "pw_chunks_stream.csv"
        pw.io.csv.write(table, str(out_csv), name="pathway_chunks_write")
        # Execute the Pathway graph to perform the write
        pw.run()
        print(f"Pathway table written to {out_csv}")
    except Exception as e:
        print(f"Pathway table creation or write failed: {e}")


if __name__ == "__main__":
    main()
