import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INDEX_PATH = os.path.join(BASE_DIR, "embeddings", "faiss.index")
META_PATH = os.path.join(BASE_DIR, "embeddings", "meta.pkl")

# Load precomputed index (LOW MEMORY)
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# Use SMALL model (already cached in Render)
model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend(query, k=10):
    query_emb = model.encode([query])
    _, indices = index.search(np.array(query_emb), k)

    return [
        {
            "name": metadata[i].get("assessment_name", "Assessment"),
            "url": metadata[i].get("assessment_url", "")
        }
        for i in indices[0]
    ]
