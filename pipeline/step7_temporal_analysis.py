"""
Step 7 — Temporal consistency analysis
- Reads evidence scoring CSVs and chunks.csv; flags late contradictions
- Writes artifacts/temporal_analysis/temporal_flags_<story_id>.json
"""
import json
import logging
import pandas as pd
from pathlib import Path
from .step0_config import EVIDENCE_DIR, TEMPORAL_DIR, CHUNKING_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def run():
    chunks_df = pd.read_csv(CHUNKING_DIR / 'chunks.csv')
    # build mapping from chunk_id to midpoint fraction for each story
    story_lengths = {}
    for sid, g in chunks_df.groupby('story_id'):
        # approximate story length by last chunk end_pos
        story_lengths[sid] = int(g['end_pos'].max()) if not g.empty else 0

    for f in EVIDENCE_DIR.glob('scores_*.csv'):
        sid = f.stem.split('_', 1)[1]
        df = pd.read_csv(f)
        flags = {'contradictions_early': 0, 'contradictions_mid': 0, 'contradictions_late': 0, 'contradiction_positions': []}
        story_len = story_lengths.get(sid, 1)
        for _, r in df.iterrows():
            if r['evaluation'] == 'CONTRADICTS':
                pos = int((r['start_pos'] + r['end_pos'])/2)
                frac = pos / max(1, story_len)
                if frac < 1/3:
                    flags['contradictions_early'] += 1
                    flags['contradiction_positions'].append('early')
                elif frac < 2/3:
                    flags['contradictions_mid'] += 1
                    flags['contradiction_positions'].append('mid')
                else:
                    flags['contradictions_late'] += 1
                    flags['contradiction_positions'].append('late')
        flags['story_length_chars'] = story_len
        out_path = TEMPORAL_DIR / f'temporal_flags_{sid}.json'
        out_path.write_text(json.dumps(flags, indent=2))
        logging.info('Step7: Wrote temporal flags to %s', out_path)


if __name__ == '__main__':
    run()
