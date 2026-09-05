import re
import json
import logging
from pathlib import Path
import pandas as pd

try:
    from .step0_config import RETRIEVAL_DIR, EVIDENCE_DIR, CHUNKING_DIR, BACKSTORY_DIR, INGESTION_DIR
except (ImportError, ValueError):
    from pipeline.step0_config import RETRIEVAL_DIR, EVIDENCE_DIR, CHUNKING_DIR, BACKSTORY_DIR, INGESTION_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

STOPWORDS = {
    'the', 'and', 'or', 'a', 'an', 'in', 'on', 'of', 'for', 'to', 'by', 'with',
    'as', 'at', 'from', 'into', 'about', 'after', 'before', 'over', 'under',
    'between', 'among', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'that', 'this', 'these', 'those', 'he', 'she', 'they', 'we', 'you', 'it',
    'its', 'his', 'her', 'their', 'them', 'who', 'which', 'had', 'have', 'has'
}
NEGATION_WORDS = {'not', "n't", 'never', 'no', 'none', 'neither', 'without', 'nowhere'}
CONTRAST_WORDS = {'however', 'contrary', 'contradicts', 'falsely', 'untrue', 'denied', 'refused'}

def extract_key_tokens(text: str):
    words = re.findall(r'[a-z0-9]+', (text or '').lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]

def evaluate_pair(claim_text: str, chunk_text: str, sim: float, category: str):
    claim_keys = set(extract_key_tokens(claim_text))
    if not claim_keys:
        return 'NEUTRAL', 0.05

    sentences = re.split(r'(?<=[.!?])\s+', chunk_text or '')
    contradiction_found = False
    support_found = False
    max_sent_sim = 0.0

    for sent in sentences:
        sent_words = set(re.findall(r'[a-z0-9]+', sent.lower()))
        matched = claim_keys & sent_words
        overlap = len(matched)
        if overlap == 0:
            continue

        overlap_ratio = overlap / len(claim_keys)
        if overlap_ratio > max_sent_sim:
            max_sent_sim = overlap_ratio

        has_neg = any(neg in sent_words for neg in NEGATION_WORDS)
        has_contrast = any(c in sent_words for c in CONTRAST_WORDS)

        if (overlap >= 2 or overlap_ratio >= 0.35) and (has_neg or has_contrast):
            contradiction_found = True
            break
        elif overlap >= 2 or overlap_ratio >= 0.35:
            support_found = True

    if contradiction_found:
        eval_label = 'CONTRADICTS'
        confidence = min(1.0, 0.45 + max_sent_sim * 0.5)
    elif support_found or sim >= 0.20 or max_sent_sim >= 0.30:
        eval_label = 'SUPPORTS'
        confidence = min(1.0, 0.35 + max_sent_sim * 0.5)
    else:
        eval_label = 'NEUTRAL'
        confidence = max(0.02, sim * 0.5)

    if category == 'beliefs' and eval_label == 'SUPPORTS' and sim < 0.20:
        confidence *= 0.8

    return eval_label, confidence

def run():
    logging.info('Step 6: Evaluating claim-evidence pairs')
    try:
        train_df = pd.read_csv(INGESTION_DIR / 'train_loaded.csv', encoding='utf-8')
        test_df = pd.read_csv(INGESTION_DIR / 'test_loaded.csv', encoding='utf-8')
        valid_row_ids = set(train_df['id'].astype(str).tolist() + test_df['id'].astype(str).tolist())
    except Exception:
        valid_row_ids = None

    for f in RETRIEVAL_DIR.glob('retrieved_*.json'):
        row_id = f.stem.split('_', 1)[1]
        if valid_row_ids is not None and row_id not in valid_row_ids:
            continue

        retrieved = json.loads(f.read_text(encoding='utf-8'))
        rows = []
        claims_file = BACKSTORY_DIR / f'claims_{row_id}.json'
        payload = json.loads(claims_file.read_text(encoding='utf-8')) if claims_file.exists() else {}
        if isinstance(payload, list):
            payload0 = payload[0] if len(payload) > 0 and isinstance(payload[0], dict) else {}
        else:
            payload0 = payload or {}

        story_id = payload0.get('story_id')
        claims_list = payload0.get('claims', [])

        for claim_id, hits in retrieved.items():
            category = 'assumptions'
            claim_text = ''
            for c in claims_list:
                if c['claim_id'] == claim_id:
                    category = c.get('category', 'assumptions')
                    claim_text = c.get('claim_text', '')
                    break

            for h in hits:
                lbl, conf = evaluate_pair(claim_text, h.get('text', ''), float(h.get('similarity', 0.0)), category)
                rows.append({
                    'row_id': row_id,
                    'story_id': story_id,
                    'claim_id': claim_id,
                    'chunk_id': h['chunk_id'],
                    'similarity': h.get('similarity', 0.0),
                    'evaluation': lbl,
                    'confidence': conf,
                    'start_pos': h.get('start_pos'),
                    'end_pos': h.get('end_pos')
                })

        df = pd.DataFrame(rows)
        out_path = EVIDENCE_DIR / f'scores_{row_id}.csv'
        df.to_csv(out_path, index=False, encoding='utf-8')

if __name__ == '__main__':
    run()
