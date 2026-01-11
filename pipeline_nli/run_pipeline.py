"""
Entry point to run the pipeline end-to-end. Each step can be invoked separately.
"""
import argparse
import subprocess
import sys
from pathlib import Path

STEPS = [
    ("step1", "step1_load_data.py"),
    ("step2", "step2_claim_extraction.py"),
    ("step3", "step3_retrieve_evidence.py"),
    ("step4", "step4_nli_inference.py"),
    ("step5", "step5_aggregate_decision.py"),
]


def run_step(step_script: str):
    print(f"Running {step_script}...")
    subprocess.check_call([sys.executable, str(Path(__file__).parent / step_script)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="*", help="Steps to run (e.g. step1 step2). If omitted, runs all steps in order.")
    args = parser.parse_args()
    steps_to_run = args.steps if args.steps else [s[0] for s in STEPS]
    for step_name, script in STEPS:
        if step_name in steps_to_run:
            run_step(script)
    print("Pipeline run complete.")
