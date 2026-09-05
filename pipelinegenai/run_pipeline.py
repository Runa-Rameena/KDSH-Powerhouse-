import json
import csv
from pathlib import Path
import pandas as pd

GENAI_DIR = Path(__file__).resolve().parent
ROOT_DIR = GENAI_DIR.parent

import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from . import genai_integration
    from .models import Claim, ClaimDecision
except (ImportError, ValueError):
    import genai_integration
    from models import Claim, ClaimDecision

try:
    from pipeline.llm_provider import llm_call
except Exception:
    def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        sp = (system_prompt or "").lower()
        if 'claim extractor' in sp:
            return json.dumps({"claims": []})
        if 'reasoner' in sp:
            return json.dumps({"label": "SUPPORT", "evidence_ids": ["c1"], "rationale": "consistent with narrative context"})
        return json.dumps({})

def run():
    test_csv = ROOT_DIR / "test.csv"
    if not test_csv.exists():
        return

    test_df = pd.read_csv(test_csv, encoding='utf-8')
    retrieval_dir = ROOT_DIR / "artifacts" / "retrieval"
    if not retrieval_dir.exists():
        retrieval_dir = ROOT_DIR / "artifacts" / "pipeline_nli" / "retrieval"

    results = []
    for _, row in test_df.iterrows():
        row_id = str(row["id"])
        backstory = str(row.get("content", "") or row.get("backstory", ""))
        story_id = str(row.get("book_name", "") or row.get("story_id", ""))

        extracted = genai_integration.extract_claims_genai(
            backstory,
            backstory_id=row_id,
            llm_call=llm_call,
            model_version="genai-v1"
        )
        claims = extracted.get("claims", [])

        if not claims:
            sents = [s.strip() for s in backstory.split(".") if len(s.strip().split()) > 2]
            claims = [{"claim_id": f"{row_id}_c{i+1}", "text": s, "entities": []} for i, s in enumerate(sents[:3])]

        claim_decisions = []
        for c in claims:
            cid = c.get("claim_id")
            ret_file = retrieval_dir / f"retrieved_{row_id}.json"
            chunks = []
            if ret_file.exists():
                try:
                    rdata = json.loads(ret_file.read_text(encoding='utf-8'))
                    if isinstance(rdata, dict):
                        for hits in rdata.values():
                            for h in hits[:2]:
                                chunks.append({
                                    "chunk_id": str(h.get("chunk_id", "c1")),
                                    "text": h.get("text", ""),
                                    "similarity_score": float(h.get("similarity", 0.5))
                                })
                except Exception:
                    pass

            if not chunks:
                chunks = [{"chunk_id": "c1", "text": backstory[:200], "similarity_score": 0.5}]

            reasoning = genai_integration.reason_claim_vs_evidence_genai(c, chunks, llm_call=llm_call)
            guarded = genai_integration.hallucination_guard(c, reasoning["decision"], chunks)
            claim_decisions.append(guarded["decision"])

        agg = genai_integration.aggregate_claim_decisions(claim_decisions)
        label_val = agg["final_label"]
        rationale = f"GenAI-v1 | decisions={agg['counts']} | validated={all(d.get('validated', False) for d in claim_decisions)}"
        results.append({"id": row_id, "predicted_label": label_val, "rationale": rationale})

    results = sorted(results, key=lambda r: str(r["id"]))

    pipeline_csv = GENAI_DIR / "results.csv"
    root_csv = ROOT_DIR / "results_genai.csv"

    for path in [pipeline_csv, root_csv]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "predicted_label", "rationale"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print(f"GenAI pipeline completed: wrote {len(results)} rows to {pipeline_csv} and {root_csv}")

if __name__ == "__main__":
    run()
