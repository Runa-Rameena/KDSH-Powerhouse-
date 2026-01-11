"""
Step 5 — Claim -> Narrative retrieval
- Reads claims JSON and chunk embeddings and chunks.csv
- Retrieves top-k narrative chunks per claim, with similarity scores
- Writes artifacts/retrieval/retrieved_<story_id>.json
"""
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .step0_config import RETRIEVAL_DIR, CHUNKING_DIR, INDEXING_DIR, BACKSTORY_DIR, TOP_K

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False


def load_embeddings():
    em_path = INDEXING_DIR / 'embeddings.npy'
    assert em_path.exists(), 'embeddings.npy missing; run step3 first'
    return np.load(str(em_path))


def build_index(embeddings):
    if FAISS_AVAILABLE:
        import faiss
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add((embeddings).astype('float32'))
        return ('faiss', index)
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=TOP_K, metric='cosine')
        nn.fit(embeddings)
        return ('sklearn', nn)


def get_claim_emb(claim_text, model=None, dim=None):
    if SBERT_AVAILABLE and model is not None:
        emb = model.encode([claim_text], convert_to_numpy=True)[0]
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb
    else:
        return np.zeros((dim,), dtype=float)


def run():
    embeddings = load_embeddings()
    chunks_df = pd.read_csv(CHUNKING_DIR / 'chunks.csv')
    idx_meta = json.loads((INDEXING_DIR / 'index_metadata.json').read_text())

    model = None
    if SBERT_AVAILABLE:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            model = None

    index = build_index(embeddings)

    import hashlib
    # For every claims file (supports per-row payloads and legacy per-story lists)
    for f in BACKSTORY_DIR.glob('claims_*.json'):
        raw = json.loads(f.read_text())

        # Normalize to a list of claim dicts and determine the target row_ids to write
        if isinstance(raw, dict):
            sid = raw.get('story_id')
            claims = raw.get('claims', []) or []
            row_ids = [str(raw.get('row_id'))] if raw.get('row_id') else []
        elif isinstance(raw, list):
            # legacy file: list of claim dicts for a story (no row ids)
            sid = f.stem.split('_', 1)[1]
            claims = raw
            row_ids = []
            try:
                train_df = pd.read_csv(INGESTION_DIR / 'train_loaded.csv')
                test_df = pd.read_csv(INGESTION_DIR / 'test_loaded.csv')
                row_ids = train_df[train_df['story_id'].astype(str) == sid]['id'].astype(str).tolist()
                row_ids += test_df[test_df['story_id'].astype(str) == sid]['id'].astype(str).tolist()
                # deduplicate and keep stable ordering
                row_ids = sorted(set(row_ids), key=lambda x: int(x) if x.isdigit() else x)
            except Exception as e:
                logging.warning('Step5: Could not expand story-level claims for %s: %s', sid, e)
        else:
            logging.warning('Step5: Unrecognized claims format in %s — skipping', f)
            continue

        if not row_ids:
            logging.info('Step5: No target row ids found for story_id=%s — skipping file %s', sid, f)
            continue

        # Normalize each claim to a dict with claim_id and claim_text
        norm_claims = []
        for i, c in enumerate(claims):
            if isinstance(c, dict):
                claim_text = c.get('claim_text', '')
                claim_id = c.get('claim_id') or hashlib.md5((claim_text + str(i)).encode('utf-8')).hexdigest()[:12]
                category = c.get('category', 'assumptions')
            else:
                claim_text = str(c)
                claim_id = hashlib.md5((claim_text + str(i)).encode('utf-8')).hexdigest()[:12]
                category = 'assumptions'
            norm_claims.append({'claim_id': claim_id, 'claim_text': claim_text, 'category': category})

        # For each row referencing this story (or explicit row_id), perform retrieval and write per-row output
        for row_id in row_ids:
            retrieved = {}
            for c in norm_claims:
                claim_id = c['claim_id']
                claim_text = c['claim_text']
                # If a sentence-transformer model exists, use embedding search; otherwise fall back to cheap token-overlap similarity
                hits = []
                if model is not None:
                    emb = get_claim_emb(claim_text, model=model, dim=embeddings.shape[1])
                    if index[0] == 'faiss':
                        q = np.array([emb]).astype('float32')
                        D, I = index[1].search(q, TOP_K)
                        for sim, idx in zip(D[0].tolist(), I[0].tolist()):
                            row = chunks_df.iloc[idx]
                            hits.append({'chunk_idx': int(idx), 'chunk_id': row['chunk_id'], 'start_pos': int(row['start_pos']), 'end_pos': int(row['end_pos']), 'similarity': float(sim), 'text': row['text']})
                    else:
                        # sklearn returns distances
                        from sklearn.metrics.pairwise import cosine_distances
                        distances = cosine_distances([emb], embeddings)[0]
                        # get top-k smallest distances
                        idxs = distances.argsort()[:TOP_K]
                        for idx in idxs:
                            sim = max(0.0, 1 - float(distances[int(idx)]))
                            row = chunks_df.iloc[int(idx)]
                            hits.append({'chunk_idx': int(idx), 'chunk_id': row['chunk_id'], 'start_pos': int(row['start_pos']), 'end_pos': int(row['end_pos']), 'similarity': sim, 'text': row['text']})
                else:
                    # Simple token-overlap (Jaccard) fallback when embeddings/model not available
                    claim_tokens = set(str(claim_text or '').lower().split())
                    sims = []
                    for idx, row in chunks_df.iterrows():
                        chunk_tokens = set(str(row['text']).lower().split())
                        union = claim_tokens | chunk_tokens
                        inter = claim_tokens & chunk_tokens
                        sim = 0.0 if not union else (len(inter) / len(union))
                        sims.append((sim, int(idx)))
                    # select top-K by token-overlap
                    top = sorted(sims, key=lambda x: x[0], reverse=True)[:TOP_K]
                    for sim, idx in top:
                        row = chunks_df.iloc[int(idx)]
                        hits.append({'chunk_idx': int(idx), 'chunk_id': row['chunk_id'], 'start_pos': int(row['start_pos']), 'end_pos': int(row['end_pos']), 'similarity': float(sim), 'text': row['text']})
                retrieved[claim_id] = hits
            out_path = RETRIEVAL_DIR / f'retrieved_{row_id}.json'
            out_path.write_text(json.dumps(retrieved, indent=2, ensure_ascii=False))
            logging.info('Step5: Wrote retrieval for row_id=%s (story_id=%s) to %s', row_id, sid, out_path)


if __name__ == '__main__':
    run()
