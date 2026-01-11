"""
Configuration for pipeline_nli.
"""
from pathlib import Path
import multiprocessing

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT
ARTIFACT_DIR = ROOT / "artifacts" / "pipeline_nli"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
BOOKS_DIR = ROOT / "Books"

# Chunking params
CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 50
TOP_K = 5

# NLI model
NLI_MODEL = "joeddav/distilbert-base-uncased-mnli"
BATCH_SIZE = 16

# Determinism & resources
NUM_WORKERS = min(8, multiprocessing.cpu_count())
SEED = 42

# File paths
INPUT_ROWS_DIR = ARTIFACT_DIR / "input" / "rows"
INPUT_ROWS_DIR.mkdir(parents=True, exist_ok=True)
CLAIMS_DIR = ARTIFACT_DIR / "claims"
CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR = ARTIFACT_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
RETRIEVAL_DIR = ARTIFACT_DIR / "retrieval"
RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
NLI_DIR = ARTIFACT_DIR / "nli_outputs"
NLI_DIR.mkdir(parents=True, exist_ok=True)
METRICS_FILE = ARTIFACT_DIR / "metrics" / "metrics.json"

# Pathway settings placeholder
USE_PATHWAY = True
PATHWAY_INDEX_DIR = ARTIFACT_DIR / "pathway_index"
PATHWAY_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Misc
VERBOSE = True
