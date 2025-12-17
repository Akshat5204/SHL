import os
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# ABSOLUTE PATH SETUP (CRITICAL)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
INDEX_PATH = os.path.join(EMB_DIR, "faiss.index")
META_PATH = os.path.join(EMB_DIR, "meta.pkl")

os.makedirs(EMB_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index():
    print("🔧 Building FAISS index...")

    df = pd.read_csv(DATA_PATH)

    if "search_text" not in df.columns:
        raise RuntimeError("search_text column missing in processed_data.csv")

    texts = df["search_text"].astype(str).tolist()

    embeddings = model.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    print("✅ FAISS index built successfully")

    return index, df.to_dict(orient="records")

# -----------------------------
# LOAD OR BUILD INDEX
# -----------------------------
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    print("✅ Loading existing FAISS index")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
else:
    index, metadata = build_index()

# -----------------------------
# RECOMMEND FUNCTION
# -----------------------------
def recommend(query, k=10):
    query_emb = model.encode([query])
    _, indices = index.search(np.array(query_emb), k)

    results = []
    for i in indices[0]:
        record = metadata[i]
        results.append({
            "name": record.get("assessment_name", "Assessment"),
            "url": record.get("assessment_url", "")
        })

    return results
