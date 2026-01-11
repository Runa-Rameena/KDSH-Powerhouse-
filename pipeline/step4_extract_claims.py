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

            # --- Optional GenAI extraction (runs in parallel when enabled) ---
            try:
                from .step0_config import GENAI_ENABLE_CLAIM_EXTRACTION, GENAI_MODEL_VERSION
                if GENAI_ENABLE_CLAIM_EXTRACTION:
                    import pipelinegenai.genai_integration as genai_integration
                    from . import llm_provider
                    ga_res = genai_integration.extract_claims_genai(backstory, backstory_id=row_id, llm_call=llm_provider.llm_call, model_version=GENAI_MODEL_VERSION)
                    genai_out_path = BACKSTORY_DIR / f'claims_genai_{row_id}.json'
                    genai_out_path.write_text(json.dumps(ga_res, indent=2, ensure_ascii=False))
                    # simple disagreement logging: compare sets of claim texts
                    heuristic_texts = set(c.get('claim_text') for c in claims)
                    genai_texts = set(c.get('text') for c in ga_res.get('claims', []))
                    disagreement = heuristic_texts != genai_texts
                    disag = {'row_id': row_id, 'story_id': sid, 'heuristic_count': len(heuristic_texts), 'genai_count': len(genai_texts), 'disagreement': disagreement, 'heuristic_texts': list(heuristic_texts), 'genai_texts': list(genai_texts)}
                    (BACKSTORY_DIR / f'disagreements_{row_id}.json').write_text(json.dumps(disag, indent=2, ensure_ascii=False))
                    logging.info('Step4: Wrote GenAI claims to %s and disagreement=%s', genai_out_path, disagreement)
            except Exception as e:
                logging.warning('Step4: GenAI claim extraction skipped/failed for row_id=%s: %s', row_id, str(e))
    logging.info('Step4: Wrote claims JSON files to %s', BACKSTORY_DIR)


if __name__ == '__main__':
    run()
