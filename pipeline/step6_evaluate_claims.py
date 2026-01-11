"""
Step 6 — Claim evaluation
- Reads retrieved_<story_id>.json and evaluates each claim->chunk pair
- Labels SUPPORTS / CONTRADICTS / NEUTRAL and assigns confidence
- Writes artifacts/evidence_scoring/scores_<story_id>.csv
"""
import json
import logging
from pathlib import Path
import pandas as pd
from .step0_config import RETRIEVAL_DIR, EVIDENCE_DIR, CHUNKING_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

NEGATION_WORDS = ['not', "n't", 'never', 'no', 'without']
CONTRAST_WORDS = ['however', 'but', 'although', 'later', 'contrary', 'yet']


def evaluate_pair(claim_text: str, chunk_text: str, sim: float, category: str):
    """Heuristic evaluation of a claim–chunk pair.
    Strengthened contradiction detection and confidence adjustments to
    produce more discriminative per-row evidence scores.
    """
    eval_label = 'NEUTRAL'
    # baseline confidence biased slightly toward sim
    confidence = min(1.0, sim + 0.05)
    ct = (chunk_text or '').lower()

    # Adjust thresholds for smaller similarity scales (fallback token overlap produces small sims)
    # Slightly amplify small sims so they can trigger heuristics when appropriate
    adj_sim = sim
    if sim < 0.06:
        adj_sim = sim * 4.0
    elif sim < 0.12:
        adj_sim = sim * 2.0

    # Strong support for relatively high similarity on this scale
    # Lowered threshold slightly (0.24 -> 0.20), now conservatively reduced to 0.18 to surface more borderline supports
    if adj_sim >= 0.18:
        eval_label = 'SUPPORTS'
        # Apply a modest boost to support confidences (set to +0.15 above sim)
        confidence = min(1.0, sim + 0.15)

    # Contradiction heuristics tuned for the observed sim scale
    if any(n in ct for n in NEGATION_WORDS) and adj_sim >= 0.12:
        eval_label = 'CONTRADICTS'
        confidence = min(1.0, max(confidence, sim + 0.35))
    if any(c in ct for c in CONTRAST_WORDS) and adj_sim >= 0.14:
        eval_label = 'CONTRADICTS'
        confidence = min(1.0, max(confidence, sim + 0.30))

    # If the claim belongs to beliefs, be conservative about support
    if category == 'beliefs' and eval_label == 'SUPPORTS' and adj_sim < 0.32:
        confidence *= 0.7

    # If nothing strong, lower confidence to reflect NEUTRAL/weak evidence but not too tiny
    if eval_label == 'NEUTRAL':
        confidence = max(0.02, confidence * 0.6)

    return eval_label, confidence


def run():
    # Load valid row ids from canonical ingestion outputs to avoid processing legacy story-level retrievals
    try:
        train_df = pd.read_csv(INGESTION_DIR / 'train_loaded.csv')
        test_df = pd.read_csv(INGESTION_DIR / 'test_loaded.csv')
        valid_row_ids = set(train_df['id'].astype(str).tolist() + test_df['id'].astype(str).tolist())
    except Exception:
        valid_row_ids = None

    for f in RETRIEVAL_DIR.glob('retrieved_*.json'):
        row_id = f.stem.split('_', 1)[1]
        # Skip retrievals that do not correspond to any known input row (legacy story-level artifacts)
        if valid_row_ids is not None and row_id not in valid_row_ids:
            logging.info('Step6: Skipping retrieval %s — not a valid row id', f.name)
            continue

        retrieved = json.loads(f.read_text())
        rows = []
        # Attempt to read the claims payload (contains story_id and claims list)
        claims_file = (Path('artifacts') / 'backstory_claims' / f'claims_{row_id}.json')
        payload = json.loads(claims_file.read_text()) if claims_file.exists() else {}
        # Handle payload being dict or list (robustness): if list, use first element if dict
        if isinstance(payload, list):
            payload0 = payload[0] if len(payload) > 0 and isinstance(payload[0], dict) else {}
        else:
            payload0 = payload or {}
        story_id = payload0.get('story_id') if payload0 else None
        claims_list = payload0.get('claims', []) if payload0 else []

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
                rows.append({'row_id': row_id, 'story_id': story_id, 'claim_id': claim_id, 'chunk_id': h['chunk_id'], 'similarity': h.get('similarity', 0.0), 'evaluation': lbl, 'confidence': conf, 'start_pos': h.get('start_pos'), 'end_pos': h.get('end_pos')})
        df = pd.DataFrame(rows)
        out_path = EVIDENCE_DIR / f'scores_{row_id}.csv'
        df.to_csv(out_path, index=False)
        logging.info('Step6: Wrote evidence scores to %s', out_path)


if __name__ == '__main__':
    run()
