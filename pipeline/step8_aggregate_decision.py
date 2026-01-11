"""
Step 8 — Aggregation and final decision
- Reads evidence scores and temporal flags
- Aggregates verdict and writes artifacts/final/decision_<story_id>.json
"""
import json
import logging
import pandas as pd
from pathlib import Path
from .step0_config import EVIDENCE_DIR, TEMPORAL_DIR, FINAL_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def evaluate_predictions(y_true, y_pred):
    """Compute accuracy, precision, recall, f1 and confusion matrix. Returns a dict."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}


def run():
    # Load thresholds and weights from config
    try:
        from .step0_config import CONTRADICT_WEIGHT, SUPPORT_WEIGHT, NEUTRAL_WEIGHT, CONTRADICTION_TAU, CONTRADICT_RATIO, MIN_SUPPORT_THRESHOLD, EPS
    except Exception:
        # fallback defaults (should not happen)
        CONTRADICT_WEIGHT = 3.0
        SUPPORT_WEIGHT = 1.0
        NEUTRAL_WEIGHT = -0.05
        CONTRADICTION_TAU = 1.5
        CONTRADICT_RATIO = 1.2
        MIN_SUPPORT_THRESHOLD = 0.5
        EPS = 1e-6

    # Process each scores file (now per input row id)
    # We'll collect per-row aggregates so we can run diagnostics later
    summaries = {}
    SUPPORT_CONFIDENCE_MIN = 0.10  # fallback min single-chunk confidence to allow support when no explicit SUPPORTS labels
    for f in EVIDENCE_DIR.glob('scores_*.csv'):
        row_id = f.stem.split('_', 1)[1]
        df = pd.read_csv(f)
        max_confidence = float(df['confidence'].max()) if 'confidence' in df.columns else 0.0
        # story-level temporal flags are per story_id — read story_id from the scores file if available
        story_id = None
        if 'story_id' in df.columns:
            story_ids = df['story_id'].dropna().unique()
            if len(story_ids) > 1:
                logging.warning('Step8: Multiple story_id values found in %s — using the first', f)
            story_id = str(story_ids[0]) if len(story_ids) > 0 else None
        else:
            # fallback: try to read claims file
            claims_file = (Path('artifacts') / 'backstory_claims' / f'claims_{row_id}.json')
            if claims_file.exists():
                story_id = json.loads(claims_file.read_text()).get('story_id')

        flags = {'contradictions_late': 0, 'story_length_chars': 1}
        if story_id and (TEMPORAL_DIR / f'temporal_flags_{story_id}.json').exists():
            flags = json.loads((TEMPORAL_DIR / f'temporal_flags_{story_id}.json').read_text())

        score_contradict = 0.0
        score_support = 0.0
        support_count = 0
        for _, r in df.iterrows():
            # penalize or reward evidence using configurable weights
            if r['evaluation'] == 'CONTRADICTS':
                pos = (r['start_pos'] + r['end_pos']) / 2
                story_len = flags.get('story_length_chars') or 0
                # If we don't have a sensible story length, avoid exploding multipliers
                if story_len and story_len > 1:
                    frac = pos / float(story_len)
                    temporal_multiplier = 1.0 + frac
                else:
                    temporal_multiplier = 1.0
                score_contradict += CONTRADICT_WEIGHT * float(r['confidence']) * temporal_multiplier
            elif r['evaluation'] == 'SUPPORTS':
                score_support += SUPPORT_WEIGHT * float(r['confidence'])
                support_count += 1
            else:
                # NEUTRAL evidence penalized slightly to avoid collapsing to positive class
                score_support += NEUTRAL_WEIGHT * float(r['confidence'])

        # decision logic using multiple checks
        reason = ''
        decision = 1
        # avoid ratio explosion by clipping support to a sensible floor (MIN_SUPPORT_THRESHOLD)
        S_clipped = max(score_support, MIN_SUPPORT_THRESHOLD)
        support_was_clipped = score_support < MIN_SUPPORT_THRESHOLD
        # enforce the small positive floor on the support score so the threshold check and ratio use the clipped value
        score_support = S_clipped

        if score_contradict > CONTRADICTION_TAU:
            decision = 0
            reason = f'C>{CONTRADICTION_TAU:.2f} (C={score_contradict:.2f})'
        elif (score_contradict / (score_support + EPS)) > CONTRADICT_RATIO:
            decision = 0
            reason = f'ratio>{CONTRADICT_RATIO:.2f} (C/S={score_contradict / (score_support + EPS):.2f}, S_clipped={score_support:.2f})'
        elif score_support < MIN_SUPPORT_THRESHOLD:
            # This should not trigger for clipped rows, kept for safety
            decision = 0
            reason = f'S<min_support ({score_support:.2f} < {MIN_SUPPORT_THRESHOLD:.2f})'
        else:
            # Require at least one explicit SUPPORTS chunk to predict support, or a single high-confidence chunk
            if support_count > 0 or max_confidence >= SUPPORT_CONFIDENCE_MIN:
                decision = 1
                reason = 'support'
            else:
                # No explicit SUPPORTS evidence — treat as non-support (likely neutral)
                decision = 0
                reason = 'no_support_evidence (S_clipped)'

        # emphasize late contradictions
        late_c = flags.get('contradictions_late', 0)
        rationale = f"{reason} | C={score_contradict:.2f}, S={score_support:.2f}"
        if late_c > 0:
            rationale += f" | {late_c} late contradiction(s)"

        out = {'id': row_id, 'predicted_label': int(decision), 'rationale': rationale}
        out_path = FINAL_DIR / f'decision_{row_id}.json'
        out_path.write_text(json.dumps(out, indent=2))
        if out_path.exists():
            logging.info('Step8: Wrote decision for %s to %s (reason=%s)', row_id, out_path, reason)
        else:
            logging.error('Step8: Failed to write decision for %s to %s', row_id, out_path)

        # store summary stats for diagnostics later (record original S_clipped flag)
        summaries[row_id] = {
            'C': float(score_contradict),
            'S': float(score_support),
            'S_clipped': float(S_clipped),
            'clipped': bool(support_was_clipped),
            'support_count': int(support_count),
            'max_confidence': float(max_confidence),
            'ratio': float(score_contradict / (score_support + EPS)),
            'reason': reason,
            'rationale': rationale,
            'story_id': story_id
        }

    # After producing decisions for all rows, evaluate on TRAIN split (using a stratified 80/20 validation holdout)
    try:
        from sklearn.model_selection import train_test_split
        from .step0_config import INGESTION_DIR
        train_df = pd.read_csv(INGESTION_DIR / 'train_loaded.csv')
        train_df['id'] = train_df['id'].astype(str)
        train_df = train_df.set_index('id')
        # collect predictions for train ids
        preds = {}
        for f in FINAL_DIR.glob('decision_*.json'):
            d = json.loads(f.read_text())
            rid = str(d.get('id'))
            if rid in train_df.index:
                preds[rid] = int(d.get('predicted_label'))
        if len(preds) == 0:
            raise RuntimeError('No predictions found for any train rows — cannot compute metrics')

        # Fail loudly if any train rows are missing predictions
        missing = set(train_df.index.tolist()) - set(preds.keys())
        if missing:
            raise RuntimeError(f'Step8 validation failed: Missing predictions for {len(missing)} train rows (example ids: {list(missing)[:5]})')

        # Mandatory sanity check: ensure predictions are not collapsed to a single class
        unique_preds = set(preds.values())
        if len(unique_preds) < 2:
            # Write diagnostic file to help debugging before failing hard
            ids_for_eval = sorted(preds.keys(), key=lambda x: int(x) if x.isdigit() else x)
            diag_rows = []
            for rid in ids_for_eval:
                true_label = int(train_df.loc[rid, 'label']) if rid in train_df.index else None
                s = summaries.get(rid, {'C': 0.0, 'S': 0.0, 'ratio': 0.0, 'reason': '', 'rationale': '', 'clipped': False, 'S_clipped': 0.0})
                diag_rows.append({'id': rid, 'label': true_label, 'predicted': preds[rid], 'C': s['C'], 'S': s['S'], 'S_clipped': s.get('S_clipped', s['S']), 'clipped': bool(s.get('clipped', False)), 'S_clipped': s.get('S_clipped', s['S']), 'clipped': bool(s.get('clipped', False)), 'ratio': s['ratio'], 'reason': s['reason'], 'rationale': s['rationale']})
            diag_df = pd.DataFrame(diag_rows)
            support_zero = int((diag_df['S'] == 0.0).sum())
            contradiction_zero = int((diag_df['C'] == 0.0).sum())
            stats = {}
            for col in ['S', 'C', 'ratio']:
                vals = diag_df[col].fillna(0.0)
                stats[col] = {
                    'count': int(vals.count()),
                    'mean': float(vals.mean()),
                    'std': float(vals.std()),
                    'min': float(vals.min()),
                    '25%': float(vals.quantile(0.25)),
                    '50%': float(vals.median()),
                    '75%': float(vals.quantile(0.75)),
                    'max': float(vals.max())
                }
            # Top false positives if any
            false_positives = diag_df[(diag_df['label'] == 0) & (diag_df['predicted'] == 1)].copy()
            fp_examples = []
            if not false_positives.empty:
                false_positives['rank_score'] = false_positives['ratio']
                top_fp = false_positives.sort_values('rank_score', ascending=False).head(5)
                for _, r in top_fp.iterrows():
                    rid = str(r['id'])
                    scores_file = EVIDENCE_DIR / f'scores_{rid}.csv'
                    top_evidence = []
                    if scores_file.exists():
                        try:
                            sf = pd.read_csv(scores_file)
                            sf = sf.sort_values('confidence', ascending=False).head(3)
                            for _, e in sf.iterrows():
                                text = e.get('text') if 'text' in e.index else ''
                                top_evidence.append({'evaluation': e.get('evaluation'), 'confidence': float(e.get('confidence') or 0.0), 'snippet': (text[:200] + '...') if text else ''})
                        except Exception:
                            logging.warning('Step8: Failed to read evidence file for %s', rid)
                    fp_examples.append({'id': rid, 'label': int(r['label']), 'predicted': int(r['predicted']), 'C': float(r['C']), 'S': float(r['S']), 'ratio': float(r['ratio']), 'rationale': r['rationale'], 'top_evidence': top_evidence})

            # count how many train rows had their support clipped to the floor
            support_clipped_count = int(diag_df.get('clipped', pd.Series(False)).sum()) if 'clipped' in diag_df.columns else 0

            report = {
                'error': f'all train predictions collapsed to a single class ({unique_preds})',
                'diagnostics': {
                    'support_zero_count': support_zero,
                    'support_clipped_count': support_clipped_count,
                    'contradiction_zero_count': contradiction_zero,
                    'distributions': stats,
                    'false_positive_examples': fp_examples
                }
            }
            out_eval = FINAL_DIR / 'train_evaluation.json'
            out_eval.write_text(json.dumps(report, indent=2))
            logging.info('Step8: Wrote diagnostic train evaluation report to %s before failing', out_eval)
            raise RuntimeError(f'Step8 validation failed: all train predictions collapsed to a single class ({unique_preds}). Failing loudly as required.')
        y_true_full = [int(train_df.loc[k, 'label']) for k in ids_for_eval]
        y_pred_full = [preds[k] for k in ids_for_eval]
        if len(y_true_full) >= 10:
            X_train_ids, X_val_ids, y_train, y_val = train_test_split(ids_for_eval, y_true_full, test_size=0.2, random_state=42, stratify=y_true_full)
            y_val_pred = [preds[k] for k in X_val_ids]
            metrics = evaluate_predictions(y_val, y_val_pred)
            report = {'split': 'validation (20% stratified holdout from train)', 'metrics': metrics}
        else:
            metrics = evaluate_predictions(y_true_full, y_pred_full)
            report = {'split': 'train (all, too few rows for stratified holdout)', 'metrics': metrics}
        out_eval = FINAL_DIR / 'train_evaluation.json'
        out_eval.write_text(json.dumps(report, indent=2))
        logging.info('Step8: Wrote train evaluation report to %s', out_eval)
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        logging.warning('Step8: Train evaluation skipped or failed: %s', e)

    # After producing decisions for all rows, evaluate on TRAIN split (using a stratified 80/20 validation holdout)
    try:
        from sklearn.model_selection import train_test_split
        from .step0_config import INGESTION_DIR
        train_df = pd.read_csv(INGESTION_DIR / 'train_loaded.csv')
        train_df['id'] = train_df['id'].astype(str)
        train_df = train_df.set_index('id')
        # collect predictions for train ids
        preds = {}
        for f in FINAL_DIR.glob('decision_*.json'):
            d = json.loads(f.read_text())
            rid = str(d.get('id'))
            if rid in train_df.index:
                preds[rid] = int(d.get('predicted_label'))
        if len(preds) == 0:
            raise RuntimeError('No predictions found for any train rows — cannot compute metrics')
        missing = set(train_df.index.tolist()) - set(preds.keys())
        if missing:
            logging.warning('Step8: Missing predictions for %d train rows (examples: %s)', len(missing), list(missing)[:5])

        # Mandatory sanity check: ensure predictions are not collapsed to a single class
        unique_preds = set(preds.values())
        if len(unique_preds) < 2:
            raise RuntimeError(f'Step8 validation failed: all train predictions collapsed to a single class ({unique_preds}). Failing loudly as required.')

        # build a diagnostics DataFrame joining ground truth, preds and per-row summaries
        ids_for_eval = sorted(preds.keys(), key=lambda x: int(x) if x.isdigit() else x)
        diag_rows = []
        for rid in ids_for_eval:
            true_label = int(train_df.loc[rid, 'label']) if rid in train_df.index else None
            s = summaries.get(rid, {'C': 0.0, 'S': 0.0, 'ratio': 0.0, 'reason': '', 'rationale': ''})
            diag_rows.append({'id': rid, 'label': true_label, 'predicted': preds[rid], 'C': s['C'], 'S': s['S'], 'ratio': s['ratio'], 'reason': s['reason'], 'rationale': s['rationale']})
        diag_df = pd.DataFrame(diag_rows)

        # high-level counts to explain label collapse
        support_zero = int((diag_df['S'] == 0.0).sum())
        contradiction_zero = int((diag_df['C'] == 0.0).sum())
        logging.info('Step8 diagnostics: %d train rows have S==0.0; %d train rows have C==0.0', support_zero, contradiction_zero)

        # distributions
        stats = {}
        for col in ['S', 'C', 'ratio']:
            vals = diag_df[col].fillna(0.0)
            stats[col] = {
                'count': int(vals.count()),
                'mean': float(vals.mean()),
                'std': float(vals.std()),
                'min': float(vals.min()),
                '25%': float(vals.quantile(0.25)),
                '50%': float(vals.median()),
                '75%': float(vals.quantile(0.75)),
                'max': float(vals.max())
            }
        logging.info('Step8 diagnostics stats: %s', {k: {'mean': stats[k]['mean'], 'median': stats[k]['50%']} for k in stats})

        # Find top 5 potentially problematic rows: label=0 but predicted=1
        false_positives = diag_df[(diag_df['label'] == 0) & (diag_df['predicted'] == 1)].copy()
        if not false_positives.empty:
            # rank by ratio descending (strongest mismatch)
            false_positives['rank_score'] = false_positives['ratio']
            top_fp = false_positives.sort_values('rank_score', ascending=False).head(5)
            fp_examples = []
            for _, r in top_fp.iterrows():
                rid = str(r['id'])
                scores_file = EVIDENCE_DIR / f'scores_{rid}.csv'
                top_evidence = []
                if scores_file.exists():
                    try:
                        sf = pd.read_csv(scores_file)
                        sf = sf.sort_values('confidence', ascending=False).head(3)
                        for _, e in sf.iterrows():
                            text = e.get('text') if 'text' in e.index else ''
                            top_evidence.append({'evaluation': e.get('evaluation'), 'confidence': float(e.get('confidence') or 0.0), 'snippet': (text[:200] + '...') if text else ''})
                    except Exception:
                        logging.warning('Step8: Failed to read evidence file for %s', rid)
                fp_examples.append({'id': rid, 'label': int(r['label']), 'predicted': int(r['predicted']), 'C': float(r['C']), 'S': float(r['S']), 'ratio': float(r['ratio']), 'rationale': r['rationale'], 'top_evidence': top_evidence})
            logging.info('Step8: Top false-positive examples (label=0 predicted=1): %s', [e['id'] for e in fp_examples])
        else:
            fp_examples = []
            logging.info('Step8: No label=0/predicted=1 examples found in train set')

        # compute metrics (stratified holdout when enough rows)
        y_true_full = [int(train_df.loc[k, 'label']) for k in ids_for_eval]
        y_pred_full = [preds[k] for k in ids_for_eval]
        if len(y_true_full) >= 10:
            X_train_ids, X_val_ids, y_train, y_val = train_test_split(ids_for_eval, y_true_full, test_size=0.2, random_state=42, stratify=y_true_full)
            y_val_pred = [preds[k] for k in X_val_ids]
            metrics = evaluate_predictions(y_val, y_val_pred)
            report = {'split': 'validation (20% stratified holdout from train)', 'metrics': metrics}
        else:
            metrics = evaluate_predictions(y_true_full, y_pred_full)
            report = {'split': 'train (all, too few rows for stratified holdout)', 'metrics': metrics}

        # attach diagnostics to the report
        support_clipped_count = int(diag_df.get('clipped', pd.Series(False)).sum()) if 'clipped' in diag_df.columns else 0
        report['diagnostics'] = {
            'support_zero_count': support_zero,
            'support_clipped_count': support_clipped_count,
            'contradiction_zero_count': contradiction_zero,
            'distributions': stats,
            'false_positive_examples': fp_examples
        }

        out_eval = FINAL_DIR / 'train_evaluation.json'
        out_eval.write_text(json.dumps(report, indent=2))
        logging.info('Step8: Wrote train evaluation report to %s', out_eval)
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        logging.warning('Step8: Train evaluation skipped or failed: %s', e)


if __name__ == '__main__':
    run()
