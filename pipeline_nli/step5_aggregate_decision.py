"""
STEP 5: Aggregate chunk-level NLI results to claim-level and then to final row-level labels.
Also compute metrics if ground-truth labels are available (train.csv).
"""
import json
import csv
from collections import defaultdict
from pathlib import Path
from step0_config import NLI_DIR, CLAIMS_DIR, ARTIFACT_DIR

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_nli_results(path: Path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def aggregate_claim(results_for_claim):
    labels = [r["nli_label"] for r in results_for_claim]
    if any(l == "CONTRADICTION" for l in labels):
        return "CONTRADICT"
    if any(l == "ENTAILMENT" for l in labels):
        return "SUPPORT"
    return "NEUTRAL"


def aggregate_all():
    nli_path = NLI_DIR / "nli_results.jsonl"
    if not nli_path.exists():
        print(f"NLI results not found at {nli_path}")
        return
    results = load_nli_results(nli_path)
    by_claim = defaultdict(list)
    for r in results:
        by_claim[r["claim_id"]].append(r)
    claim_decisions = {}
    for claim_id, recs in by_claim.items():
        claim_decisions[claim_id] = aggregate_claim(recs)
    # Aggregate to row-level
    row_to_claims = defaultdict(list)
    for cid, dec in claim_decisions.items():
        row_id = cid.split("_claim_")[0]
        row_to_claims[row_id].append({"claim_id": cid, "decision": dec})
    final_rows = []
    for row_id, cds in row_to_claims.items():
        if any(c["decision"] == "CONTRADICT" for c in cds):
            final_label = 0
        else:
            final_label = 1
        rationale = ", ".join([f"{c['claim_id']}:{c['decision']}" for c in cds])
        final_rows.append({"id": row_id, "predicted_label": final_label, "rationale": rationale})
    # Save results CSV
    out_csv = ARTIFACT_DIR.parent / "results_nli.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "predicted_label", "rationale"])
        writer.writeheader()
        for r in final_rows:
            writer.writerow(r)
    print(f"Saved final predictions to {out_csv}")

    # If ground-truth labels exist, compute metrics
    # Try to load train.csv
    train_csv = ARTIFACT_DIR.parent / "train.csv"
    if train_csv.exists():
        import pandas as pd
        df = pd.read_csv(train_csv)
        gt = {}
        for _, row in df.iterrows():
            gt[str(row["id"])] = int(row["label"]) if "label" in row and not pd.isnull(row["label"]) else None
        y_true = []
        y_pred = []
        for r in final_rows:
            if r["id"] in gt and gt[r["id"]] is not None:
                y_true.append(gt[r["id"]])
                y_pred.append(r["predicted_label"])
        if y_true:
            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
            metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
            # Save metrics
            metrics_path = ARTIFACT_DIR / "metrics" / "metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as mf:
                json.dump(metrics, mf, indent=2)
            print("Metrics:", metrics)
        else:
            print("No labeled rows found in train to compute metrics.")
    else:
        print("train.csv not found; skipping evaluation.")


if __name__ == "__main__":
    aggregate_all()
