import logging
import re
import hashlib
import json
import pandas as pd
from pathlib import Path

try:
    from .step0_config import INGESTION_DIR, BACKSTORY_DIR
except (ImportError, ValueError):
    from pipeline.step0_config import INGESTION_DIR, BACKSTORY_DIR

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
    logging.info('Step 4: Extracting atomic backstory claims')
    train = pd.read_csv(INGESTION_DIR / 'train_loaded.csv', encoding='utf-8')
    test = pd.read_csv(INGESTION_DIR / 'test_loaded.csv', encoding='utf-8')

    for df in [train, test]:
        for _, row in df.iterrows():
            row_id = str(row['id'])
            sid = str(row['story_id'])
            backstory = row.get('backstory', '')
            claims = extract_claims_from_text(backstory)

            out_path = BACKSTORY_DIR / f'claims_{row_id}.json'
            payload = {'row_id': row_id, 'story_id': sid, 'claims': claims}
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

if __name__ == '__main__':
    run()
