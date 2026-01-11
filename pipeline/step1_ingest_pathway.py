"""
Step 1 — Data ingestion (Pathway REQUIRED)
- Loads train/test CSVs
- Loads novels from books/<story_id>.txt
- Builds a Pathway table and executes it
- Writes artifacts/ingestion/novels_table.json and copies of train/test
FAILS HARD if Pathway is missing or Pathway execution fails
"""
import sys
import logging
from pathlib import Path
import pandas as pd
from .step0_config import TRAIN_CSV, TEST_CSV, BOOKS_DIR, INGESTION_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

try:
    import pathway as pw
except Exception as e:
    raise RuntimeError('Pathway is required for step1_ingest_pathway.py but not found')


def _find_novel_file(name: str):
    # try direct match, then case-insensitive stem lookup, then compacted-space match
    cand = BOOKS_DIR / f"{name}.txt"
    if cand.exists():
        return cand
    # look through all txt files
    files = list(BOOKS_DIR.glob('**/*.txt'))
    name_key = name.strip().lower()
    stems = {p.stem.lower(): p for p in files}
    if name_key in stems:
        return stems[name_key]
    name_compact = name_key.replace(' ', '').replace('-', '')
    for p in files:
        if p.stem.lower().replace(' ', '').replace('-', '') == name_compact:
            return p
    return None


def _normalize_df(df, is_train=False):
    # map available columns to canonical: id, story_id, backstory, label (for train)
    out = pd.DataFrame()
    # id is mandatory for per-row predictions
    if 'id' in df.columns:
        out['id'] = df['id'].astype(str)
    else:
        raise RuntimeError('Input CSV missing id column; each row must have an id')

    if 'story_id' in df.columns:
        out['story_id'] = df['story_id'].astype(str)
    elif 'book_name' in df.columns:
        out['story_id'] = df['book_name'].astype(str)
    else:
        raise RuntimeError('Input CSV missing story identifier column (expected story_id or book_name)')

    if 'backstory' in df.columns:
        out['backstory'] = df['backstory'].astype(str).fillna('')
    else:
        # prefer content then caption
        if 'content' in df.columns:
            out['backstory'] = df['content'].astype(str).fillna('')
        elif 'caption' in df.columns:
            out['backstory'] = df['caption'].astype(str).fillna('')
        else:
            out['backstory'] = ''

    if is_train:
        if 'label' in df.columns:
            # map to 1 = consistent, 0 = contradict
            def map_label(v):
                if pd.isna(v):
                    return ''
                s = str(v).strip().lower()
                if s in ('consistent', 'true', '1', 'yes'):
                    return 1
                if s in ('contradict', 'contradicts', 'false', '0', 'no'):
                    return 0
                return ''
            out['label'] = df['label'].apply(map_label)
        else:
            out['label'] = ''
    return out


def run():
    logging.info('Step1: Loading train/test CSVs (with flexible column mapping)')
    train_raw = pd.read_csv(TRAIN_CSV)
    test_raw = pd.read_csv(TEST_CSV)

    train = _normalize_df(train_raw, is_train=True)
    test = _normalize_df(test_raw, is_train=False)

    # Save canonicalized CSVs
    (INGESTION_DIR / 'train_loaded.csv').write_text(train.to_csv(index=False))
    (INGESTION_DIR / 'test_loaded.csv').write_text(test.to_csv(index=False))

    # Build novels table by story_id (use mapping to files in BOOKS_DIR)
    story_ids = pd.concat([train['story_id'], test['story_id']]).dropna().unique()
    records = []
    for sid in story_ids:
        sid_str = str(sid)
        p = _find_novel_file(sid_str)
        if p is None:
            logging.warning(f"Novel for story_id={sid_str} not found under {BOOKS_DIR}; writing empty text")
            text = ''
        else:
            text = p.read_text(encoding='utf-8')
        records.append({'story_id': sid_str, 'text': text})

    novels_df = pd.DataFrame(records)

    # Create Pathway table and materialize using only the allowed debug APIs
    logging.info('Step1: Creating Pathway table from novels and executing pipeline (Pathway required)')
    try:
        t = pw.debug.table_from_pandas(novels_df)
        # print / debug the table to create a sink for materialization
        pw.debug.compute_and_print(t, include_id=False)
        # execute the Pathway runtime (no args)
        pw.run()
        logging.info('Step1: Pathway run completed and materialized via debug sink')
    except Exception as e:
        logging.error('Step1: Pathway execution failed: %s', e)
        raise RuntimeError('Pathway execution failed in step1_ingest_pathway')

    # Materialize to disk (novels_table.json)
    out_path = INGESTION_DIR / 'novels_table.json'
    novels_df.to_json(out_path, orient='records', force_ascii=False)
    logging.info('Step1: Wrote novels_table.json to %s', out_path)


if __name__ == '__main__':
    run()
