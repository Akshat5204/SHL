import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load assets once
index = faiss.read_index("embeddings/faiss.index")

with open("embeddings/texts.pkl", "rb") as f:
    texts = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = []
    for idx in indices[0]:
        results.append(texts[idx])

    return results
