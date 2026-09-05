import sys
from pathlib import Path

NLI_DIR = Path(__file__).resolve().parent
ROOT_DIR = NLI_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(NLI_DIR) not in sys.path:
    sys.path.insert(0, str(NLI_DIR))

try:
    from . import step1_load_data
    from . import step2_claim_extraction
    from . import step3_retrieve_evidence
    from . import step4_nli_inference
    from . import step5_aggregate_decision
except (ImportError, ValueError):
    import step1_load_data
    import step2_claim_extraction
    import step3_retrieve_evidence
    import step4_nli_inference
    import step5_aggregate_decision

def run_all():
    print("Running Pipeline NLI...")
    step1_load_data.run()
    step2_claim_extraction.run()
    step3_retrieve_evidence.run()
    step4_nli_inference.run()
    step5_aggregate_decision.run()
    print("Pipeline NLI complete.")

if __name__ == "__main__":
    run_all()
