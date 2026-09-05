import sys
import logging
from pathlib import Path
import pandas as pd

try:
    from .step0_config import TRAIN_CSV, TEST_CSV, BOOKS_DIR, INGESTION_DIR
except (ImportError, ValueError):
    from pipeline.step0_config import TRAIN_CSV, TEST_CSV, BOOKS_DIR, INGESTION_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except Exception:
    pw = None
    PATHWAY_AVAILABLE = False

def _find_novel_file(name: str):
    cand = BOOKS_DIR / f"{name}.txt"
    if cand.exists():
        return cand
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
    out = pd.DataFrame()
    if 'id' in df.columns:
        out['id'] = df['id'].astype(str)
    else:
        raise RuntimeError('Input CSV missing id column')

    if 'story_id' in df.columns:
        out['story_id'] = df['story_id'].astype(str)
    elif 'book_name' in df.columns:
        out['story_id'] = df['book_name'].astype(str)
    else:
        raise RuntimeError('Input CSV missing story identifier')

    if 'backstory' in df.columns:
        out['backstory'] = df['backstory'].astype(str).fillna('')
    elif 'content' in df.columns:
        out['backstory'] = df['content'].astype(str).fillna('')
    elif 'caption' in df.columns:
        out['backstory'] = df['caption'].astype(str).fillna('')
    else:
        out['backstory'] = ''

    if is_train:
        if 'label' in df.columns:
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
    logging.info('Step 1: Ingesting novels and datasets')
    train_raw = pd.read_csv(TRAIN_CSV, encoding='utf-8')
    test_raw = pd.read_csv(TEST_CSV, encoding='utf-8')

    train = _normalize_df(train_raw, is_train=True)
    test = _normalize_df(test_raw, is_train=False)

    train.to_csv(INGESTION_DIR / 'train_loaded.csv', index=False, encoding='utf-8')
    test.to_csv(INGESTION_DIR / 'test_loaded.csv', index=False, encoding='utf-8')

    story_ids = pd.concat([train['story_id'], test['story_id']]).dropna().unique()
    records = []
    for sid in story_ids:
        sid_str = str(sid)
        p = _find_novel_file(sid_str)
        text = p.read_text(encoding='utf-8') if p else ''
        records.append({'story_id': sid_str, 'text': text})

    novels_df = pd.DataFrame(records)

    if PATHWAY_AVAILABLE and pw is not None:
        try:
            t = pw.debug.table_from_pandas(novels_df)
            pw.debug.compute_and_print(t, include_id=False)
            pw.run()
        except Exception as e:
            logging.warning('Pathway runtime note: %s', e)

    out_path = INGESTION_DIR / 'novels_table.json'
    novels_df.to_json(out_path, orient='records', force_ascii=False)
    logging.info('Saved novels table to %s', out_path)

if __name__ == '__main__':
    run()
