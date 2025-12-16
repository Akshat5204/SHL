import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("embeddings/faiss.index")

with open("embeddings/texts.pkl", "rb") as f:
    texts = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend(query, k=5):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)

    results = []
    for idx in I[0]:
        results.append(texts[idx])

    return results

if __name__ == "__main__":
    print(recommend("Java developer with teamwork skills"))
