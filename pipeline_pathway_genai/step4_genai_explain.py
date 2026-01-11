"""
Step 4: GenAI explanation-only layer (explain why evidence is insufficient)

Behavior:
- Reads `artifacts/pathway_genai/results.csv` and selects claims with final_label == 'INSUFFICIENT'
- For each such claim, get top-K retrieved chunk IDs from `artifacts/pathway_genai/retrieved_claims.json`
- Call the LLM provider (`pipeline.llm_provider.llm_call`) with a JSON-only explainer prompt; request that the model cite evidence IDs only
- Validate LLM output: ensure cited IDs are subset of provided top IDs; if not, mark `hallucination_flag` and use deterministic fallback explanation
- Write outputs to `artifacts/pathway_genai/explanations.json` (list of objects)

Run: python3 pipeline_pathway_genai/step4_genai_explain.py
"""
from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
# ensure project root on sys.path so we can import pipeline.llm_provider
sys.path.insert(0, str(ROOT))
from pipeline.llm_provider import llm_call
ART = ROOT / "artifacts" / "pathway_genai"
RETRIEVED = ART / "retrieved_claims.json"
RESULTS_CSV = ART / "results.csv"
EXPLANATIONS = ART / "explanations.json"
TOP_K = 3

SYSTEM_PROMPT = "You are a JSON-only explainer. Given a claim and a list of retrieved chunk IDs, explain concisely why the evidence is insufficient to support or contradict the claim. Output JSON exactly in the format: {\"explanation\": str, \"cited_evidence_ids\": [list_of_ids]}. Do NOT change the final label or suggest a new label. temperature=0."


def load_results(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def run():
    if not RESULTS_CSV.exists() or not RETRIEVED.exists():
        raise FileNotFoundError('Missing required artifacts: results or retrieved_claims')

    rows = load_results(RESULTS_CSV)
    with open(RETRIEVED, 'r', encoding='utf-8') as f:
        retrieved = json.load(f)
    retrieved_map = {c['claim_id']: c for c in retrieved}

    out = []
    for r in rows:
        if r['final_label'] != 'INSUFFICIENT':
            continue
        cid = r['claim_id']
        claim_text = r['claim_text']
        rec = retrieved_map.get(cid, {})
        retrieved_list = rec.get('retrieved', [])[:TOP_K]
        top_ids = [x['chunk_id'] for x in retrieved_list]

        user_prompt = json.dumps({"claim_id": cid, "claim_text": claim_text, "top_chunk_ids": top_ids})
        try:
            raw = llm_call(SYSTEM_PROMPT, user_prompt, temperature=0)
            parsed = json.loads(raw)
        except Exception:
            # deterministic fallback
            parsed = {"explanation": f"No clear evidence in top chunks {top_ids}", "cited_evidence_ids": top_ids}
            raw = json.dumps(parsed)

        # validate cited ids
        cited = parsed.get('cited_evidence_ids', []) if isinstance(parsed.get('cited_evidence_ids', []), list) else []
        hallucination_flag = False
        if any(cid not in top_ids for cid in cited):
            hallucination_flag = True
            # fallback explanation that cites only top_ids
            parsed = {"explanation": f"No clear evidence in top chunks {top_ids}", "cited_evidence_ids": top_ids}

        out_entry = {
            "claim_id": cid,
            "claim_text": claim_text,
            "top_chunk_ids": top_ids,
            "explanation": parsed.get('explanation', '')[:800],
            "cited_evidence_ids": parsed.get('cited_evidence_ids', []),
            "hallucination_flag": hallucination_flag,
            "raw_llm_output": raw[:2000],
        }
        out.append(out_entry)

    with open(EXPLANATIONS, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {EXPLANATIONS} with {len(out)} entries")


if __name__ == '__main__':
    run()
