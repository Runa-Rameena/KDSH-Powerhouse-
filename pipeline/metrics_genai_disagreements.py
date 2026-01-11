"""Compute simple GenAI-vs-heuristic disagreement statistics.

Usage:
  python3 pipeline/metrics_genai_disagreements.py --artifacts artifacts/backstory_claims

Outputs a human-readable report with counts and percentages:
 - total files scanned
 - total reasoner disagreement entries
 - counts per GenAI label
 - counts per heuristic overall label
 - agreement rate between GenAI label and heuristic label (where comparable)
 - insufficient/invalid GenAI fraction and validated fraction (if available)

The script is intentionally dependency-free and robust to older Step 4 "extraction-only"
disagreement files (which contain `heuristic_texts` / `genai_texts`) and the Step 6 merged
format (`{ "row_id": ..., "disagreements": [ ... ] }`).
"""

import argparse
import json
from pathlib import Path
from collections import Counter


def overall_heuristic_label(heuristic_hits):
    """Derive an overall heuristic label from per-hit heuristic labels.
    Priority: CONTRADICTS > SUPPORTS > NEUTRAL
    Returns None if no heuristic labels present.
    """
    labels = [h.get('heuristic_label') for h in (heuristic_hits or []) if h.get('heuristic_label')]
    labels = [l.upper() for l in labels]
    if any(l == 'CONTRADICTS' for l in labels):
        return 'CONTRADICTS'
    if any(l == 'SUPPORTS' for l in labels):
        return 'SUPPORTS'
    if labels:
        return 'NEUTRAL'
    return None


def scan(disagreements_dir: Path):
    files = list(disagreements_dir.glob('disagreements_*.json'))
    total_files = len(files)
    extraction_only_files = 0

    reasoner_entries = 0
    genai_labels = Counter()
    heuristic_overalls = Counter()
    agreements = 0
    comparisons = 0
    insufficient = 0
    validated_true = 0
    validated_total = 0
    malformed = 0

    for f in files:
        try:
            payload = json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            malformed += 1
            continue

        # Old Step4 extraction disagreement format
        if isinstance(payload, dict) and ('heuristic_texts' in payload or 'genai_texts' in payload) and 'disagreements' not in payload:
            extraction_only_files += 1
            continue

        entries = []
        if isinstance(payload, dict) and isinstance(payload.get('disagreements'), list):
            entries = payload.get('disagreements')
        elif isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict) and payload:
            # single entry older format — treat as single entry
            entries = [payload]

        for e in entries:
            # Prefer structured GenAI decision under 'genai_decision' or 'genai' or 'decision'
            genai = e.get('genai_decision') or e.get('genai') or e.get('decision')
            if not isinstance(genai, dict):
                # If the disagreement entry is from claim extraction, it may not have a GenAI decision
                continue
            reasoner_entries += 1
            label = (genai.get('label') or '').upper()
            genai_labels[label] += 1

            # validated flag if present
            if 'validated' in genai:
                validated_total += 1
                if genai.get('validated'):
                    validated_true += 1

            if label in ('INSUFFICIENT', 'UNKNOWN', 'UNSURE', ''):
                insufficient += 1

            # heuristic summary
            h_over = overall_heuristic_label(e.get('heuristic_hits'))
            if h_over:
                heuristic_overalls[h_over] += 1

            # compare when both present
            if label and h_over:
                comparisons += 1
                if label == h_over:
                    agreements += 1

    return {
        'total_files': total_files,
        'extraction_only_files': extraction_only_files,
        'malformed_files': malformed,
        'reasoner_entries': reasoner_entries,
        'genai_labels': genai_labels,
        'heuristic_overalls': heuristic_overalls,
        'comparisons': comparisons,
        'agreements': agreements,
        'insufficient': insufficient,
        'validated_true': validated_true,
        'validated_total': validated_total,
    }


def print_report(stats, dir_path: Path):
    print('\nGenAI vs Heuristic Disagreement Report')
    print('Directory: %s' % dir_path)
    print('-' * 60)
    print('Files scanned: %d' % stats['total_files'])
    print(' - extraction-only files (Step4): %d' % stats['extraction_only_files'])
    print(' - malformed/unreadable files: %d' % stats['malformed_files'])
    print('\nReasoner disagreement entries: %d' % stats['reasoner_entries'])

    def pct(n, denom):
        return f"{(100.0 * n / denom):.1f}%" if denom else 'n/a'

    print('\nGenAI labels:')
    for label, cnt in stats['genai_labels'].most_common():
        print('  - %-15s %6d (%s of reasoner entries)' % (label or '<empty>', cnt, pct(cnt, stats['reasoner_entries'])))

    print('\nHeuristic overall labels:')
    for label, cnt in stats['heuristic_overalls'].most_common():
        print('  - %-15s %6d (%s of reasoner entries)' % (label, cnt, pct(cnt, stats['reasoner_entries'])))

    print('\nComparisons (entries with both GenAI label and heuristic overall): %d' % stats['comparisons'])
    print('Agreement count: %d (%s)' % (stats['agreements'], pct(stats['agreements'], stats['comparisons'])))

    print('\nGenAI insufficient/invalid labels: %d (%s of reasoner entries)' % (stats['insufficient'], pct(stats['insufficient'], stats['reasoner_entries'])))
    if stats['validated_total']:
        print('GenAI validated: %d / %d (%.1f%%)' % (stats['validated_true'], stats['validated_total'], 100.0 * stats['validated_true'] / stats['validated_total']))

    print('\nNote: Heuristics remain the authority; this report is for analysis only.')
    print('-' * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', default='artifacts/backstory_claims', help='Path to backstory_claims artifacts directory')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    dir_path = Path(args.artifacts)
    if not dir_path.exists():
        print('Error: artifacts directory not found:', dir_path)
        return

    stats = scan(dir_path)
    print_report(stats, dir_path)


if __name__ == '__main__':
    main()
