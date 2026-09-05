import logging
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .step0_config import RETRIEVAL_DIR, CHUNKING_DIR, INDEXING_DIR, BACKSTORY_DIR, INGESTION_DIR, TOP_K
except (ImportError, ValueError):
    from pipeline.step0_config import RETRIEVAL_DIR, CHUNKING_DIR, INDEXING_DIR, BACKSTORY_DIR, INGESTION_DIR, TOP_K

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
    assert em_path.exists(), 'embeddings.npy missing; run step 3 first'
    return np.load(str(em_path))

def build_index(embeddings):
    if FAISS_AVAILABLE:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings.astype('float32'))
        return ('faiss', index)
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=TOP_K, metric='cosine')
        nn.fit(embeddings)
        return ('sklearn', nn)

def get_claim_emb(claim_text, model=None, dim=None):
    if SBERT_AVAILABLE and model is not None:
        emb = model.encode([claim_text], convert_to_numpy=True)[0]
        return emb / (np.linalg.norm(emb) + 1e-12)
    return np.zeros((dim,), dtype=float)

def run():
    logging.info('Step 5: Retrieving evidence chunks')
    embeddings = load_embeddings()
    chunks_df = pd.read_csv(CHUNKING_DIR / 'chunks.csv', encoding='utf-8')

    model = None
    if SBERT_AVAILABLE:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            model = None

    index = build_index(embeddings)
    chunk_tokens_cache = [set(str(t or '').lower().split()) for t in chunks_df['text']]

    for f in sorted(BACKSTORY_DIR.glob('claims_*.json')):
        row_id_stem = f.stem.split('_', 1)[1]
        if not row_id_stem.isdigit() and not row_id_stem.isalnum():
            continue
        raw = json.loads(f.read_text(encoding='utf-8'))
        if not isinstance(raw, dict) or 'row_id' not in raw:
            continue

        row_id = str(raw['row_id'])
        sid = raw.get('story_id')
        claims = raw.get('claims', [])

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

        retrieved = {}
        for c in norm_claims:
            claim_id = c['claim_id']
            claim_text = c['claim_text']
            hits = []
            if model is not None:
                emb = get_claim_emb(claim_text, model=model, dim=embeddings.shape[1])
                if index[0] == 'faiss':
                    D, I = index[1].search(np.array([emb]).astype('float32'), TOP_K)
                    for sim, idx in zip(D[0].tolist(), I[0].tolist()):
                        row = chunks_df.iloc[idx]
                        hits.append({'chunk_idx': int(idx), 'chunk_id': row['chunk_id'], 'start_pos': int(row['start_pos']), 'end_pos': int(row['end_pos']), 'similarity': float(sim), 'text': row['text']})
                else:
                    from sklearn.metrics.pairwise import cosine_distances
                    distances = cosine_distances([emb], embeddings)[0]
                    idxs = distances.argsort()[:TOP_K]
                    for idx in idxs:
                        sim = max(0.0, 1.0 - float(distances[int(idx)]))
                        row = chunks_df.iloc[int(idx)]
                        hits.append({'chunk_idx': int(idx), 'chunk_id': row['chunk_id'], 'start_pos': int(row['start_pos']), 'end_pos': int(row['end_pos']), 'similarity': sim, 'text': row['text']})
            else:
                claim_tokens = set(str(claim_text or '').lower().split())
                sims = []
                for idx, chunk_tokens in enumerate(chunk_tokens_cache):
                    union = claim_tokens | chunk_tokens
                    inter = claim_tokens & chunk_tokens
                    sim = 0.0 if not union else (len(inter) / len(union))
                    sims.append((sim, idx))
                top = sorted(sims, key=lambda x: x[0], reverse=True)[:TOP_K]
                for sim, idx in top:
                    row = chunks_df.iloc[int(idx)]
                    hits.append({'chunk_idx': int(idx), 'chunk_id': row['chunk_id'], 'start_pos': int(row['start_pos']), 'end_pos': int(row['end_pos']), 'similarity': float(sim), 'text': row['text']})
            retrieved[claim_id] = hits

        out_path = RETRIEVAL_DIR / f'retrieved_{row_id}.json'
        out_path.write_text(json.dumps(retrieved, indent=2, ensure_ascii=False), encoding='utf-8')

if __name__ == '__main__':
    run()
