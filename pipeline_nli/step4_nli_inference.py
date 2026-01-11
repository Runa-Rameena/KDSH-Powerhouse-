"""
STEP 4: Run NLI inference on (claim, chunk) pairs and save per-pair outputs.
"""
import json
from pathlib import Path
from step0_config import NLI_DIR, RETRIEVAL_DIR, CLAIMS_DIR, NLI_MODEL, BATCH_SIZE
from models import load_nli_model, run_nli_batch


def collect_pairs():
    pairs = []  # (claim_id, chunk_id, premise, hypothesis)
    for retrieval_file in RETRIEVAL_DIR.glob("*.json"):
        claim_id = retrieval_file.stem
        with open(retrieval_file, "r", encoding="utf-8") as f:
            retrieved = json.load(f)
        # load claim text
        claim_json = CLAIMS_DIR / f"{claim_id.split('_claim_')[0]}.json"
        claim_text = None
        if claim_json.exists():
            with open(claim_json, "r", encoding="utf-8") as cf:
                claims = json.load(cf)
            for c in claims:
                if c["claim_id"] == claim_id:
                    claim_text = c["claim_text"]
                    break
        if claim_text is None:
            continue
        for r in retrieved:
            pairs.append((claim_id, r["chunk_id"], r["text"], claim_text, r.get("similarity_score", 0.0)))
    return pairs


def run_nli():
    tokenizer, model, device = load_nli_model(NLI_MODEL)
    pairs = collect_pairs()
    out_entries = []
    # process in batches for determinism
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        premises = [p[2] for p in batch]
        hypotheses = [p[3] for p in batch]
        results = run_nli_batch(tokenizer, model, device, premises, hypotheses)
        for (claim_id, chunk_id, _, _, similarity), (label, confidence) in zip(batch, results):
            # normalize label to ENT/CON/NTR
            lbl = label
            if lbl.startswith("ENTAIL"):
                lbl = "ENTAILMENT"
            elif lbl.startswith("CONTRA"):
                lbl = "CONTRADICTION"
            elif lbl.startswith("NEUT"):
                lbl = "NEUTRAL"
            entry = {
                "claim_id": claim_id,
                "chunk_id": chunk_id,
                "nli_label": lbl,
                "confidence_score": confidence,
                "similarity_score": similarity,
            }
            out_entries.append(entry)
    # save all to a single file
    out_path = NLI_DIR / "nli_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for e in out_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"NLI inference finished - saved {len(out_entries)} records to {out_path}")


if __name__ == "__main__":
    run_nli()
