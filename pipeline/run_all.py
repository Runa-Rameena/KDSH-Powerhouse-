"""
Run All — orchestrates execution of all steps in order
- Executes each step as an independent process (ensures isolation)
- Stops on any failure
- Produces final results.csv at project root
"""
import subprocess
import os
import sys
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
PIPELINE_DIR = Path(__file__).parent
MODULES = [
    'pipeline.step1_ingest_pathway',
    'pipeline.step2_chunk_pathway',
    'pipeline.step3_index_embeddings',
    'pipeline.step4_extract_claims',
    'pipeline.step5_retrieve_evidence',
    'pipeline.step6_evaluate_claims',
    'pipeline.step7_temporal_analysis',
    'pipeline.step8_aggregate_decision'
]

# Enforce execution as module to avoid fragile relative imports or environment differences.
if (__package__ is None or __package__ == '') and os.environ.get('PIPELINE_RUN_AS_MODULE') != '1':
    logging.info('Re-launching as module: python -m pipeline.run_all')
    env = os.environ.copy()
    env['PIPELINE_RUN_AS_MODULE'] = '1'
    res = subprocess.run([sys.executable, '-m', 'pipeline.run_all'], env=env)
    sys.exit(res.returncode)


def run_step(module_name):
    logging.info('Running module %s', module_name)
    res = subprocess.run([sys.executable, '-m', module_name], capture_output=True, text=True)
    if res.returncode != 0:
        logging.error('Module %s failed. stdout:\n%s\n stderr:\n%s', module_name, res.stdout, res.stderr)
        raise SystemExit(f"Module {module_name} failed")
    logging.info('Finished %s', module_name)


def gather_results():
    # Collect per-row decisions from configured final artifacts directory
    final_dir = None
    try:
        from pipeline.step0_config import FINAL_DIR as CAND
        final_dir = Path(CAND)
    except Exception:
        pass
    if final_dir is None or not Path(final_dir).exists():
        cand = Path('KDSH') / 'artifacts' / 'final'
        if cand.exists():
            final_dir = cand
    if final_dir is None or not Path(final_dir).exists():
        final_dir = Path('artifacts') / 'final'

    logging.info('Gathering final decisions from %s', final_dir)
    import pandas as pd
    # Validate against test set: number of rows, ids preserved, no duplicates, and test has no labels
    try:
        from pipeline.step0_config import INGESTION_DIR
        test_df = pd.read_csv(INGESTION_DIR / 'test_loaded.csv')
        test_ids = set(test_df['id'].astype(str).tolist())
        # Ensure test file does not contain labels (labels must only come from train set)
        if 'label' in test_df.columns:
            raise SystemExit('Validation failed: test_loaded.csv contains a label column — labels must only be in train.csv')
    except Exception as e:
        raise SystemExit(f'Could not validate test ingestion: {e}')

    files = sorted(list(Path(final_dir).glob('decision_*.json')))
    logging.info('Found %d decision files in %s', len(files), final_dir)

    if not files:
        logging.error('No decision_*.json files found in %s — cannot produce results.csv', final_dir)
        raise SystemExit(1)

    rows = []
    extra_files = []
    for f in files:
        try:
            d = json.loads(f.read_text())
            rid = str(d.get('id') or d.get('row_id') or d.get('story_id'))
            if rid not in test_ids:
                extra_files.append((f.name, rid))
                logging.warning('Skipping legacy or unexpected decision file %s (id=%s) — not in test set', f, rid)
                continue
            logging.info('Loading decision file %s (id=%s)', f, rid)
            rows.append({'id': rid, 'predicted_label': d['predicted_label'], 'rationale': d.get('rationale', '')})
        except Exception as e:
            logging.warning('Could not read %s: %s', f, e)

    # deterministic order by id
    rows = sorted(rows, key=lambda r: str(r['id']))

    ids_in_rows = [str(r['id']) for r in rows]
    if len(ids_in_rows) != len(set(ids_in_rows)):
        raise SystemExit('Validation failed: duplicate ids present in decision files')

    if set(ids_in_rows) != test_ids:
        missing = test_ids - set(ids_in_rows)
        extra = set(ids_in_rows) - test_ids
        raise SystemExit(f'Validation failed: decision ids do not match test ids. Missing: {len(missing)}, Extra: {len(extra)}')

    out_path = final_dir.parent.parent / 'results.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logging.info('Wrote results.csv to %s (rows=%d)', out_path, len(rows))


if __name__ == '__main__':
    for m in MODULES:
        run_step(m)
    gather_results()
    logging.info('All steps completed successfully')
