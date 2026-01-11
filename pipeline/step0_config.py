"""
Step 0 — Central config for the Kharagpur DS Hackathon pipeline
- centralizes paths, seeds, and constants
- ensures artifact folders exist
"""
from pathlib import Path
import os
import json

# Project root (assumed to be repo root). If a KDSH folder exists, use it as data root
ROOT = Path('.').resolve()
DATA_ROOT = ROOT / 'KDSH' if (ROOT / 'KDSH').exists() else ROOT
# Books dir: prefer KDSH/Books, else books/
BOOKS_DIR = DATA_ROOT / 'Books' if (DATA_ROOT / 'Books').exists() else DATA_ROOT / 'books'
TRAIN_CSV = DATA_ROOT / 'train.csv'
TEST_CSV = DATA_ROOT / 'test.csv'
PIPELINE_DIR = ROOT / 'pipeline'

# Artifacts root (placed inside DATA_ROOT)
ARTIFACTS = DATA_ROOT / 'artifacts'
INGESTION_DIR = ARTIFACTS / 'ingestion'
CHUNKING_DIR = ARTIFACTS / 'chunking'
INDEXING_DIR = ARTIFACTS / 'indexing'
BACKSTORY_DIR = ARTIFACTS / 'backstory_claims'
RETRIEVAL_DIR = ARTIFACTS / 'retrieval'
EVIDENCE_DIR = ARTIFACTS / 'evidence_scoring'
TEMPORAL_DIR = ARTIFACTS / 'temporal_analysis'
FINAL_DIR = ARTIFACTS / 'final'

_all_dirs = [INGESTION_DIR, CHUNKING_DIR, INDEXING_DIR, BACKSTORY_DIR, RETRIEVAL_DIR, EVIDENCE_DIR, TEMPORAL_DIR, FINAL_DIR]
for d in _all_dirs:
    d.mkdir(parents=True, exist_ok=True)

# Determinism
RANDOM_SEED = 42

# Chunking / indexing parameters (can be overridden)
CHUNK_CHARS = 4000
CHUNK_OVERLAP = 500
TOP_K = 5

# Scoring & decision thresholds
# Weights applied when aggregating evidence
CONTRADICT_WEIGHT = 2.0
SUPPORT_WEIGHT = 4.0  # amplify supports so positive evidence matters
NEUTRAL_WEIGHT = -0.0025  # very small penalty for neutral evidence

# Decision thresholds
CONTRADICTION_TAU = 2.0
CONTRADICT_RATIO = 1.5
MIN_SUPPORT_THRESHOLD = 0.05  # lower threshold to allow positives
EPS = 1e-6

# Export helper
def save_json(path, obj):
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def load_json(path):
    import json
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8'))
