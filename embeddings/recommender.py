import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_PATH = os.path.join(BASE_DIR, "embeddings", "faiss.index")
META_PATH = os.path.join(BASE_DIR, "embeddings", "meta.pkl")

# Load index and metadata ONCE
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend(query, k=5):
    query_emb = model.encode([query])
    _, indices = index.search(np.array(query_emb).astype("float32"), k)

    results = []
    for i in indices[0]:
        item = metadata[i]

        results.append({
            # ✅ use existing key instead of "title"
            "title": item.get("search_text") or item.get("assessment_name", "Assessment"),
            "url": item.get("assessment_url", "")
        })

    return results

