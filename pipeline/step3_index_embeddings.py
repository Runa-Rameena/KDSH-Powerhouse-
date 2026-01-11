"""
Step 3 — Vector indexing
- Reads artifacts/chunking/chunks.csv
- Computes embeddings (SBERT preferred, TF-IDF fallback allowed)
- Normalizes vectors and builds FAISS or sklearn index
- Writes artifacts/indexing/embeddings.npy and index_metadata.json
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .step0_config import CHUNKING_DIR, INDEXING_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


def run(model_name: str = 'all-MiniLM-L6-v2'):
    chunks_path = CHUNKING_DIR / 'chunks.csv'
    assert chunks_path.exists(), 'chunks.csv missing; run step2 first'
    df = pd.read_csv(chunks_path)
    texts = df['text'].fillna('').tolist()

    embeddings = None
    method = 'tfidf'
    if SBERT_AVAILABLE:
        try:
            logging.info('Step3: Computing embeddings via SBERT')
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            method = 'sbert'
        except Exception as e:
            logging.warning('Step3: SBERT failed, falling back to TF-IDF: %s', e)
            embeddings = None

    if embeddings is None:
        logging.info('Step3: Using TF-IDF fallback embeddings (deterministic)')
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(max_features=2048)
        X = v.fit_transform(texts)
        embeddings = X.toarray().astype(float)
        method = 'tfidf'

    embeddings = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms

    np.save(INDEXING_DIR / 'embeddings.npy', embeddings)
    meta = {'method': method, 'shape': embeddings.shape}
    if FAISS_AVAILABLE:
        meta['index_backend'] = 'faiss'
    else:
        meta['index_backend'] = 'sklearn'

    import json
    with open(INDEXING_DIR / 'index_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    logging.info('Step3: Saved embeddings.npy and index_metadata.json')


if __name__ == '__main__':
    run()
