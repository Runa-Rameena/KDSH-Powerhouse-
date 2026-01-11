"""
Step 4 — Backstory claim extraction
- Reads train.csv and test.csv
- Extracts atomic claims per backstory and writes artifacts/backstory_claims/claims_<story_id>.json
"""
import logging
import re
import hashlib
import json
import pandas as pd
from pathlib import Path
from .step0_config import INGESTION_DIR, BACKSTORY_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

KEYWORD_MAP = {
    'early_life': ['born', 'child', 'grew up', 'family', 'school', 'at age', 'childhood', 'youth'],
    'beliefs': ['believe', 'belief', 'faith', 'thinks', 'trust', 'opinion'],
    'fears': ['afraid', 'fear', 'scared', 'terror', 'dread'],
    'motivations': ['wanted', 'goal', 'dream', 'ambition', 'motivated', 'wish', 'seeks'],
    'assumptions': ['assume', 'assumes', 'assumption', 'suppose', 'presume']
}


def extract_claims_from_text(backstory: str):
    s = (backstory or '').strip()
    if not s:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', s)
    claims = []
    for i, sent in enumerate(sentences):
        sent_clean = sent.strip()
        if not sent_clean:
            continue
        cat = 'assumptions'
        for c, kws in KEYWORD_MAP.items():
            for k in kws:
                if k in sent_clean.lower():
                    cat = c
                    break
            if cat != 'assumptions':
                break
        claim_id = hashlib.md5((sent_clean + str(i)).encode('utf-8')).hexdigest()[:12]
        claims.append({'claim_id': claim_id, 'claim_text': sent_clean, 'category': cat})
    return claims


def run():
    logging.info('Step4: Loading canonical train/test CSVs from ingestion artifacts')
    train = pd.read_csv(INGESTION_DIR / 'train_loaded.csv')
    test = pd.read_csv(INGESTION_DIR / 'test_loaded.csv')
    # process test and train; ensure story_id and id columns exist
    for df in [train, test]:
        if 'story_id' not in df.columns:
            raise RuntimeError('Canonical CSV missing story identifier column; ensure step1 completed successfully')
        if 'id' not in df.columns:
            raise RuntimeError('Canonical CSV missing id column; ensure ingestion produced an id for each row')
        for _, row in df.iterrows():
            row_id = str(row['id'])
            sid = str(row['story_id'])
            backstory = row.get('backstory', '')
            claims = extract_claims_from_text(backstory)
            # write claims per input row id so multiple rows referencing same story do not overwrite each other
            out_path = BACKSTORY_DIR / f'claims_{row_id}.json'
            payload = {'row_id': row_id, 'story_id': sid, 'claims': claims}
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            if not claims:
                logging.info('Step4: No claims extracted for row_id=%s (story_id=%s) — wrote empty claims file', row_id, sid)
            else:
                logging.info('Step4: Extracted %d claims for row_id=%s (story_id=%s)', len(claims), row_id, sid)
    logging.info('Step4: Wrote claims JSON files to %s', BACKSTORY_DIR)


if __name__ == '__main__':
    run()
