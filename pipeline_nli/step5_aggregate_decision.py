import json
import csv
from collections import defaultdict
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from .step0_config import NLI_DIR, CLAIMS_DIR, ARTIFACT_DIR, ROOT, TRAIN_CSV, TEST_CSV
except (ImportError, ValueError):
    from pipeline_nli.step0_config import NLI_DIR, CLAIMS_DIR, ARTIFACT_DIR, ROOT, TRAIN_CSV, TEST_CSV

def load_nli_results(path: Path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def aggregate_claim(results_for_claim):
    labels = [r["nli_label"] for r in results_for_claim]
    if any(l == "CONTRADICTION" for l in labels):
        return "CONTRADICT"
    if any(l == "ENTAILMENT" for l in labels):
        return "SUPPORT"
    return "NEUTRAL"

def run():
    nli_path = NLI_DIR / "nli_results.jsonl"
    if not nli_path.exists():
        return

    results = load_nli_results(nli_path)
    by_claim = defaultdict(list)
    for r in results:
        by_claim[r["claim_id"]].append(r)

    claim_decisions = {cid: aggregate_claim(recs) for cid, recs in by_claim.items()}

    row_to_claims = defaultdict(list)
    for cid, dec in claim_decisions.items():
        row_id = cid.split("_claim_")[0]
        row_to_claims[row_id].append({"claim_id": cid, "decision": dec})

    all_row_predictions = {}
    for row_id, cds in row_to_claims.items():
        final_label = 0 if any(c["decision"] == "CONTRADICT" for c in cds) else 1
        rationale = ", ".join([f"{c['claim_id']}:{c['decision']}" for c in cds])
        all_row_predictions[str(row_id)] = {
            "id": str(row_id),
            "predicted_label": final_label,
            "rationale": rationale
        }

    test_ids = []
    if TEST_CSV.exists():
        test_df = pd.read_csv(TEST_CSV, encoding='utf-8')
        test_ids = [str(x) for x in test_df['id'].tolist()]

    test_rows = []
    for tid in test_ids:
        if tid in all_row_predictions:
            test_rows.append(all_row_predictions[tid])
        else:
            test_rows.append({"id": tid, "predicted_label": 1, "rationale": "default_consistent"})

    test_rows = sorted(test_rows, key=lambda r: str(r["id"]))

    nli_pipeline_csv = Path(__file__).resolve().parent / "results.csv"
    root_nli_csv = ROOT / "results_nli.csv"

    for target in [nli_pipeline_csv, root_nli_csv]:
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "predicted_label", "rationale"])
            writer.writeheader()
            for r in test_rows:
                writer.writerow(r)

    if TRAIN_CSV.exists():
        train_df = pd.read_csv(TRAIN_CSV, encoding='utf-8')
        def parse_label(val):
            if pd.isnull(val):
                return None
            s = str(val).strip().lower()
            if s in ("1", "consistent", "true", "yes", "support"):
                return 1
            if s in ("0", "contradict", "contradicts", "false", "no"):
                return 0
            return None

        gt = {str(row["id"]): parse_label(row.get("label")) for _, row in train_df.iterrows()}
        y_true, y_pred = [], []
        for rid, pred_dict in all_row_predictions.items():
            if rid in gt and gt[rid] is not None:
                y_true.append(gt[rid])
                y_pred.append(pred_dict["predicted_label"])

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
            metrics_path = ARTIFACT_DIR / "metrics" / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as mf:
                json.dump(metrics, mf, indent=2)

if __name__ == "__main__":
    run()
