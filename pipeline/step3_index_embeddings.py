import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .step0_config import CHUNKING_DIR, INDEXING_DIR
except (ImportError, ValueError):
    from pipeline.step0_config import CHUNKING_DIR, INDEXING_DIR

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
    assert chunks_path.exists(), 'chunks.csv missing; run step 2 first'
    df = pd.read_csv(chunks_path, encoding='utf-8')
    texts = df['text'].fillna('').tolist()

    embeddings = None
    method = 'tfidf'
    if SBERT_AVAILABLE:
        try:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            method = 'sbert'
        except Exception as e:
            logging.warning('SBERT unavailable (%s); using TF-IDF fallback', e)
            embeddings = None

    if embeddings is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(max_features=2048)
        X = v.fit_transform(texts)
        embeddings = X.toarray().astype(float)
        method = 'tfidf'

    embeddings = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms

    np.save(INDEXING_DIR / 'embeddings.npy', embeddings)
    meta = {
        'method': method,
        'shape': embeddings.shape,
        'index_backend': 'faiss' if FAISS_AVAILABLE else 'sklearn'
    }

    with open(INDEXING_DIR / 'index_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    logging.info('Step 3: Saved embeddings (%s, shape=%s)', method, embeddings.shape)

if __name__ == '__main__':
    run()
