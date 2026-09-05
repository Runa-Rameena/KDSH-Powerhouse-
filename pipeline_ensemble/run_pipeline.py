import json
import csv
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ENSEMBLE_DIR = Path(__file__).resolve().parent
ROOT_DIR = ENSEMBLE_DIR.parent

try:
    from .step0_config import (
        TRAIN_CSV, TEST_CSV, TUNED_TOP_K, VOTE_WEIGHTS,
        RESULTS_PIPELINE1, RESULTS_PIPELINE2, RESULTS_PIPELINE3,
        ENSEMBLE_LOCAL_RESULTS, ENSEMBLE_ROOT_RESULTS, METRICS_FILE
    )
except (ImportError, ValueError):
    from step0_config import (
        TRAIN_CSV, TEST_CSV, TUNED_TOP_K, VOTE_WEIGHTS,
        RESULTS_PIPELINE1, RESULTS_PIPELINE2, RESULTS_PIPELINE3,
        ENSEMBLE_LOCAL_RESULTS, ENSEMBLE_ROOT_RESULTS, METRICS_FILE
    )

def load_predictions(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding='utf-8')
    df['id'] = df['id'].astype(str)
    return dict(zip(df['id'], df['predicted_label'].astype(int)))

def run():
    print(f"Running Ensemble Pipeline with tuned retrieval depth k={TUNED_TOP_K}...")
    
    p1_preds = load_predictions(RESULTS_PIPELINE1)
    p2_preds = load_predictions(RESULTS_PIPELINE2)
    p3_preds = load_predictions(RESULTS_PIPELINE3)

    test_df = pd.read_csv(TEST_CSV, encoding='utf-8')
    test_df['id'] = test_df['id'].astype(str)

    ensemble_results = []
    w1, w2, w3 = VOTE_WEIGHTS["pipeline1"], VOTE_WEIGHTS["pipeline2"], VOTE_WEIGHTS["pipeline3"]

    for _, row in test_df.iterrows():
        row_id = str(row['id'])
        v1 = p1_preds.get(row_id, 1)
        v2 = p2_preds.get(row_id, 1)
        v3 = p3_preds.get(row_id, 1)

        weighted_score = (w1 * v1) + (w2 * v2) + (w3 * v3)
        final_label = 1 if weighted_score >= 0.60 else 0
        rationale = f"Ensemble-v1 | score={weighted_score:.2f} | votes={{'P1_RAG': {v1}, 'P2_NLI': {v2}, 'P3_GenAI': {v3}}} | tuned_k={TUNED_TOP_K}"

        ensemble_results.append({
            "id": row_id,
            "predicted_label": final_label,
            "rationale": rationale
        })

    ensemble_results = sorted(ensemble_results, key=lambda r: int(r["id"]) if r["id"].isdigit() else r["id"])

    # Write output CSVs
    for out_path in [ENSEMBLE_LOCAL_RESULTS, ENSEMBLE_ROOT_RESULTS]:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "predicted_label", "rationale"])
            writer.writeheader()
            for r in ensemble_results:
                writer.writerow(r)

    print(f"Ensemble Pipeline completed: wrote {len(ensemble_results)} rows to {ENSEMBLE_LOCAL_RESULTS} and {ENSEMBLE_ROOT_RESULTS}")

    # Train Evaluation metrics benchmark
    if TRAIN_CSV.exists():
        train_df = pd.read_csv(TRAIN_CSV, encoding='utf-8')
        train_df['id'] = train_df['id'].astype(str)
        train_df['gt'] = train_df['label'].apply(lambda x: 1 if str(x).strip().lower() in ['consistent', '1', 'support'] else 0)

        gt_map = dict(zip(train_df['id'], train_df['gt']))
        eval_preds = []
        eval_gts = []

        for rid, gt in gt_map.items():
            v1 = p1_preds.get(rid, 1)
            v2 = p2_preds.get(rid, 1)
            v3 = p3_preds.get(rid, 1)
            score = (w1 * v1) + (w2 * v2) + (w3 * v3)
            pred = 1 if score >= 0.60 else 0
            eval_preds.append(pred)
            eval_gts.append(gt)

        metrics = {
            "accuracy": round(accuracy_score(eval_gts, eval_preds), 3),
            "precision": round(precision_score(eval_gts, eval_preds, zero_division=0), 3),
            "recall": round(recall_score(eval_gts, eval_preds, zero_division=0), 3),
            "f1": round(f1_score(eval_gts, eval_preds, zero_division=0), 3),
            "tuned_top_k": TUNED_TOP_K
        }

        METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print(f"Ensemble Train Metrics: Accuracy={metrics['accuracy']}, F1={metrics['f1']}, Recall={metrics['recall']}, Precision={metrics['precision']}")

if __name__ == "__main__":
    run()
