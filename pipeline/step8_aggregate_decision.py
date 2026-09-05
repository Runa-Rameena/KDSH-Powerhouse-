import json
import logging
import pandas as pd
from pathlib import Path

try:
    from .step0_config import EVIDENCE_DIR, TEMPORAL_DIR, FINAL_DIR, INGESTION_DIR, BACKSTORY_DIR
    from .step0_config import CONTRADICT_WEIGHT, SUPPORT_WEIGHT, NEUTRAL_WEIGHT, CONTRADICTION_TAU, CONTRADICT_RATIO, MIN_SUPPORT_THRESHOLD, EPS
except (ImportError, ValueError):
    from pipeline.step0_config import EVIDENCE_DIR, TEMPORAL_DIR, FINAL_DIR, INGESTION_DIR, BACKSTORY_DIR
    from pipeline.step0_config import CONTRADICT_WEIGHT, SUPPORT_WEIGHT, NEUTRAL_WEIGHT, CONTRADICTION_TAU, CONTRADICT_RATIO, MIN_SUPPORT_THRESHOLD, EPS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def evaluate_predictions(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

def run():
    logging.info('Step 8: Aggregating evidence and making final verdicts')
    summaries = {}
    for f in EVIDENCE_DIR.glob('scores_*.csv'):
        row_id = f.stem.split('_', 1)[1]
        df = pd.read_csv(f, encoding='utf-8')
        max_confidence = float(df['confidence'].max()) if 'confidence' in df.columns and not df.empty else 0.0

        story_id = None
        if 'story_id' in df.columns:
            story_ids = df['story_id'].dropna().unique()
            story_id = str(story_ids[0]) if len(story_ids) > 0 else None
        else:
            claims_file = BACKSTORY_DIR / f'claims_{row_id}.json'
            if claims_file.exists():
                story_id = json.loads(claims_file.read_text(encoding='utf-8')).get('story_id')

        flags = {'contradictions_late': 0, 'story_length_chars': 1}
        if story_id and (TEMPORAL_DIR / f'temporal_flags_{story_id}.json').exists():
            flags = json.loads((TEMPORAL_DIR / f'temporal_flags_{story_id}.json').read_text(encoding='utf-8'))

        score_contradict = 0.0
        score_support = 0.0
        support_count = 0
        contradict_count = 0

        for _, r in df.iterrows():
            if r['evaluation'] == 'CONTRADICTS':
                pos = (r['start_pos'] + r['end_pos']) / 2
                story_len = flags.get('story_length_chars') or 0
                temporal_multiplier = (1.0 + (pos / float(story_len))) if story_len > 1 else 1.0
                score_contradict += CONTRADICT_WEIGHT * float(r['confidence']) * temporal_multiplier
                contradict_count += 1
            elif r['evaluation'] == 'SUPPORTS':
                score_support += SUPPORT_WEIGHT * float(r['confidence'])
                support_count += 1
            else:
                score_support += NEUTRAL_WEIGHT * float(r['confidence'])

        S_clipped = max(score_support, MIN_SUPPORT_THRESHOLD)
        support_was_clipped = score_support < MIN_SUPPORT_THRESHOLD
        score_support = S_clipped

        decision = 1
        rule = 'consistent_default'
        rule_details = {}

        if contradict_count > 0 and score_contradict > CONTRADICTION_TAU and score_contradict > score_support:
            decision = 0
            rule = 'C>tau_and_dominates'
            rule_details = {'C': score_contradict, 'S': score_support, 'tau': CONTRADICTION_TAU}
        elif contradict_count > support_count and score_contradict > 1.5:
            decision = 0
            rule = 'contradictions_dominate'
            rule_details = {'contradict_count': contradict_count, 'support_count': support_count, 'C': score_contradict}
        elif contradict_count > 0 and (score_contradict / (score_support + EPS)) > CONTRADICT_RATIO:
            decision = 0
            rule = 'ratio'
            rule_details = {'C': score_contradict, 'S': score_support, 'ratio': score_contradict / (score_support + EPS), 'thresh': CONTRADICT_RATIO}
        elif support_count > 0:
            decision = 1
            rule = 'supported'
            rule_details = {'support_count': support_count, 'max_confidence': max_confidence, 'S': score_support}
        else:
            decision = 1
            rule = 'consistent_default'
            rule_details = {'C': score_contradict, 'S': score_support}

        late_c = flags.get('contradictions_late', 0)
        decision_str = 'SUPPORT' if decision == 1 else 'CONTRADICT'
        parts = [f"DECISION={decision_str}", f"rule={rule}"]
        for k in ['C', 'S', 'ratio', 'thresh', 'tau', 'support_count', 'max_confidence']:
            if k in rule_details:
                val = rule_details[k]
                parts.append(f"{k}={val:.2f}" if isinstance(val, float) else f"{k}={val}")
        if late_c > 0:
            parts.append(f"late_contradictions={int(late_c)}")
        rationale = " | ".join(parts)

        out = {'id': row_id, 'predicted_label': int(decision), 'rationale': rationale}
        out_path = FINAL_DIR / f'decision_{row_id}.json'
        out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')

        summaries[row_id] = {
            'C': float(score_contradict),
            'S': float(score_support),
            'clipped': bool(support_was_clipped),
            'support_count': int(support_count),
            'max_confidence': float(max_confidence),
            'ratio': float(score_contradict / (score_support + EPS)),
            'reason': rule,
            'rationale': rationale,
            'story_id': story_id
        }

    try:
        from sklearn.model_selection import train_test_split
        train_df = pd.read_csv(INGESTION_DIR / 'train_loaded.csv', encoding='utf-8')
        train_df['id'] = train_df['id'].astype(str)
        train_df = train_df.set_index('id')

        preds = {}
        for f in FINAL_DIR.glob('decision_*.json'):
            d = json.loads(f.read_text(encoding='utf-8'))
            rid = str(d.get('id'))
            if rid in train_df.index:
                preds[rid] = int(d.get('predicted_label'))

        if preds:
            ids_for_eval = sorted(preds.keys(), key=lambda x: int(x) if x.isdigit() else x)
            y_true_full = [int(train_df.loc[k, 'label']) for k in ids_for_eval]
            y_pred_full = [preds[k] for k in ids_for_eval]

            if len(y_true_full) >= 10:
                _, X_val_ids, _, y_val = train_test_split(ids_for_eval, y_true_full, test_size=0.2, random_state=42, stratify=y_true_full)
                y_val_pred = [preds[k] for k in X_val_ids]
                metrics = evaluate_predictions(y_val, y_val_pred)
                report = {'split': 'validation (20% stratified holdout)', 'metrics': metrics}
            else:
                metrics = evaluate_predictions(y_true_full, y_pred_full)
                report = {'split': 'train (all)', 'metrics': metrics}

            out_eval = FINAL_DIR / 'train_evaluation.json'
            out_eval.write_text(json.dumps(report, indent=2), encoding='utf-8')
            logging.info('Evaluation results: Accuracy=%.3f, F1=%.3f', metrics['accuracy'], metrics['f1'])
    except Exception as e:
        logging.warning('Train evaluation skipped: %s', e)

if __name__ == '__main__':
    run()
