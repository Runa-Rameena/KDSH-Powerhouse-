from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / 'KDSH' if (ROOT / 'KDSH').exists() else ROOT
BOOKS_DIR = DATA_ROOT / 'Books' if (DATA_ROOT / 'Books').exists() else DATA_ROOT / 'books'
TRAIN_CSV = DATA_ROOT / 'train.csv'
TEST_CSV = DATA_ROOT / 'test.csv'
PIPELINE_DIR = ROOT / 'pipeline'

ARTIFACTS = DATA_ROOT / 'artifacts'
INGESTION_DIR = ARTIFACTS / 'ingestion'
CHUNKING_DIR = ARTIFACTS / 'chunking'
INDEXING_DIR = ARTIFACTS / 'indexing'
BACKSTORY_DIR = ARTIFACTS / 'backstory_claims'
RETRIEVAL_DIR = ARTIFACTS / 'retrieval'
EVIDENCE_DIR = ARTIFACTS / 'evidence_scoring'
TEMPORAL_DIR = ARTIFACTS / 'temporal_analysis'
FINAL_DIR = ARTIFACTS / 'final'

for d in [INGESTION_DIR, CHUNKING_DIR, INDEXING_DIR, BACKSTORY_DIR, RETRIEVAL_DIR, EVIDENCE_DIR, TEMPORAL_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
CHUNK_CHARS = 4000
CHUNK_OVERLAP = 500
TOP_K = 5

CONTRADICT_WEIGHT = 2.0
SUPPORT_WEIGHT = 4.0
NEUTRAL_WEIGHT = -0.0025

CONTRADICTION_TAU = 2.0
CONTRADICT_RATIO = 1.8
MIN_SUPPORT_THRESHOLD = 0.05
EPS = 1e-6

GENAI_ENABLE_CLAIM_EXTRACTION = os.environ.get('GENAI_ENABLE_CLAIM_EXTRACTION', 'false').lower() == 'true'
GENAI_ENABLE_REASONER = os.environ.get('GENAI_ENABLE_REASONER', 'false').lower() == 'true'
GENAI_MODE = os.environ.get('GENAI_MODE', 'parallel')
GENAI_MODEL_VERSION = os.environ.get('GENAI_MODEL_VERSION', 'genai-v1')
GENAI_SUPPORT_SIMILARITY_THRESHOLD = float(os.environ.get('GENAI_SUPPORT_SIMILARITY_THRESHOLD', '0.65'))

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
