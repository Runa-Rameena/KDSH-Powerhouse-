import logging
from pathlib import Path
import pandas as pd

try:
    from .step0_config import INGESTION_DIR, CHUNKING_DIR, CHUNK_CHARS, CHUNK_OVERLAP
except (ImportError, ValueError):
    from pipeline.step0_config import INGESTION_DIR, CHUNKING_DIR, CHUNK_CHARS, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except Exception:
    pw = None
    PATHWAY_AVAILABLE = False

def _chunk_text(text: str, window: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP):
    if not text:
        return []
    start = 0
    step = max(1, window - overlap)
    L = len(text)
    order = 0
    chunks = []
    while start < L:
        end = min(start + window, L)
        chunk_text = text[start:end]
        chunks.append({'chunk_id': f"chunk_{order}", 'text': chunk_text, 'start_pos': int(start), 'end_pos': int(end), 'order': int(order)})
        if end >= L:
            break
        start += step
        order += 1
    return chunks

def run():
    novels_path = INGESTION_DIR / 'novels_table.json'
    assert novels_path.exists(), f"{novels_path} missing; run step 1 first"
    novels_df = pd.read_json(novels_path, orient='records')

    logging.info('Step 2: Chunking novels with sliding window')
    rows = []
    for _, r in novels_df.iterrows():
        sid = str(r['story_id'])
        text = r.get('text', '') or ''
        for item in _chunk_text(text):
            rows.append({
                'story_id': sid,
                'chunk_id': f"{sid}_{item['chunk_id']}",
                'text': item['text'],
                'start_pos': item['start_pos'],
                'end_pos': item['end_pos'],
                'order': item['order']
            })

    chunks_df = pd.DataFrame(rows)
    chunks_df.sort_values(['story_id', 'order'], inplace=True)
    chunks_df.reset_index(drop=True, inplace=True)

    if PATHWAY_AVAILABLE and pw is not None:
        try:
            t = pw.debug.table_from_pandas(chunks_df)
            pw.debug.compute_and_print(t, include_id=False)
            pw.run()
        except Exception as e:
            logging.warning('Pathway runtime note: %s', e)

    out_path = CHUNKING_DIR / 'chunks.csv'
    chunks_df.to_csv(out_path, index=False, encoding='utf-8')
    logging.info('Saved chunks to %s (%d rows)', out_path, len(chunks_df))

if __name__ == '__main__':
    run()
