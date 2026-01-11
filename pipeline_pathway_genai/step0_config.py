from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = PROJECT_ROOT / "Books"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "pathway_genai"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_FILE = ARTIFACTS_DIR / "chunks.json"

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 50
