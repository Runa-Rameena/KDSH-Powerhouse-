from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT
ARTIFACT_DIR = ROOT / "artifacts" / "pipeline_ensemble"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"

# Tuned retrieval depth and consensus parameters for Ensemble Pipeline
TUNED_TOP_K = 8
VOTE_WEIGHTS = {
    "pipeline1": 0.40,  # Pathway Streaming RAG Weight
    "pipeline2": 0.20,  # NLI Transformer Weight
    "pipeline3": 0.40   # GenAI Reasoner & Guard Weight
}

RESULTS_PIPELINE1 = ROOT / "results_pipeline.csv"
RESULTS_PIPELINE2 = ROOT / "results_nli.csv"
RESULTS_PIPELINE3 = ROOT / "results_genai.csv"

ENSEMBLE_LOCAL_RESULTS = ROOT / "pipeline_ensemble" / "results.csv"
ENSEMBLE_ROOT_RESULTS = ROOT / "results_ensemble.csv"
METRICS_FILE = ARTIFACT_DIR / "metrics" / "metrics.json"
METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
