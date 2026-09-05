import subprocess
import os
import sys
from pathlib import Path
import logging
import json
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent

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

if (__package__ is None or __package__ == '') and os.environ.get('PIPELINE_RUN_AS_MODULE') != '1':
    env = os.environ.copy()
    env['PIPELINE_RUN_AS_MODULE'] = '1'
    res = subprocess.run([sys.executable, '-m', 'pipeline.run_all'], env=env, cwd=str(ROOT_DIR))
    sys.exit(res.returncode)

def run_step(module_name):
    logging.info('Running %s', module_name)
    res = subprocess.run([sys.executable, '-m', module_name], cwd=str(ROOT_DIR), capture_output=True, text=True)
    if res.returncode != 0:
        logging.error('Module %s failed.\nstdout: %s\nstderr: %s', module_name, res.stdout, res.stderr)
        raise SystemExit(f"Module {module_name} failed")

def gather_results():
    final_dir = ROOT_DIR / 'artifacts' / 'final'
    ingestion_dir = ROOT_DIR / 'artifacts' / 'ingestion'

    test_df = pd.read_csv(ingestion_dir / 'test_loaded.csv', encoding='utf-8')
    test_ids = set(test_df['id'].astype(str).tolist())
    if 'label' in test_df.columns:
        raise SystemExit('Validation error: test_loaded.csv cannot contain label column.')

    files = sorted(list(final_dir.glob('decision_*.json')))
    if not files:
        raise SystemExit(f'No decision files found in {final_dir}')

    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
            rid = str(d.get('id') or d.get('row_id') or d.get('story_id'))
            if rid in test_ids:
                rows.append({
                    'id': rid,
                    'predicted_label': d['predicted_label'],
                    'rationale': d.get('rationale', '')
                })
        except Exception as e:
            logging.warning('Could not parse %s: %s', f, e)

    rows = sorted(rows, key=lambda r: str(r['id']))
    ids_in_rows = [str(r['id']) for r in rows]
    if len(ids_in_rows) != len(set(ids_in_rows)):
        raise SystemExit('Validation failed: duplicate IDs in decisions.')

    if set(ids_in_rows) != test_ids:
        missing = test_ids - set(ids_in_rows)
        extra = set(ids_in_rows) - test_ids
        raise SystemExit(f'ID mismatch: missing {len(missing)}, extra {len(extra)}')

    out_df = pd.DataFrame(rows)
    pipeline_alias = ROOT_DIR / 'results_pipeline.csv'
    pipeline_csv = PIPELINE_DIR / 'results.csv'

    out_df.to_csv(pipeline_alias, index=False, encoding='utf-8')
    out_df.to_csv(pipeline_csv, index=False, encoding='utf-8')
    logging.info('Wrote results to %s and %s (%d rows)', pipeline_alias, pipeline_csv, len(rows))

if __name__ == '__main__':
    for m in MODULES:
        run_step(m)
    gather_results()
    logging.info('Pipeline completed successfully.')
