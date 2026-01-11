"""
Step 2 — Chunking using Pathway (PATHWAY REQUIRED)
- Reads artifacts/ingestion/novels_table.json
- Uses a Pathway streaming chunker to produce overlapping chunks
- Materializes Pathway output and writes artifacts/chunking/chunks.csv

STRICT: If Pathway missing or execution fails -> RuntimeError
"""
import logging
from pathlib import Path
import pandas as pd
from .step0_config import INGESTION_DIR, CHUNKING_DIR, CHUNK_CHARS, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

try:
    import pathway as pw
except Exception as e:
    raise RuntimeError('Pathway is required for step2_chunk_pathway.py but not found')


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
    assert novels_path.exists(), f"{novels_path} missing; run step1 first"
    novels_df = pd.read_json(novels_path, orient='records')

    logging.info('Step2: Creating chunk DataFrame using Python chunker (then materializing with Pathway debug sink)')
    rows = []
    for _, r in novels_df.iterrows():
        sid = str(r['story_id'])
        text = r.get('text', '') or ''
        c = _chunk_text(text)
        for item in c:
            rows.append({'story_id': sid, 'chunk_id': f"{sid}_{item['chunk_id']}", 'text': item['text'], 'start_pos': item['start_pos'], 'end_pos': item['end_pos'], 'order': item['order']})

    chunks_df = pd.DataFrame(rows)
    # preserve narrative order
    chunks_df.sort_values(['story_id', 'order'], inplace=True)
    chunks_df.reset_index(drop=True, inplace=True)

    # Create Pathway table from chunks and materialize via debug sink
    try:
        t = pw.debug.table_from_pandas(chunks_df)
        pw.debug.compute_and_print(t, include_id=False)
        pw.run()
        logging.info('Step2: Pathway run completed for chunk table')
    except Exception as e:
        logging.error('Step2: Pathway execution failed during chunk materialization: %s', e)
        raise RuntimeError('Pathway execution failed in step2_chunk_pathway')

    out_path = CHUNKING_DIR / 'chunks.csv'
    chunks_df.to_csv(out_path, index=False)
    logging.info('Step2: Wrote chunks.csv with %d rows to %s', len(chunks_df), out_path)


if __name__ == '__main__':
    run()
