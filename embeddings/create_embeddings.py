import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "processed_data.csv")
EMB_DIR = os.path.join(BASE_DIR, "embeddings")

INDEX_PATH = os.path.join(EMB_DIR, "faiss.index")
META_PATH = os.path.join(EMB_DIR, "meta.pkl")

os.makedirs(EMB_DIR, exist_ok=True)

print("Loading processed data...")
df = pd.read_csv(DATA_PATH)

# 🔑 THIS COLUMN MAKES OUTPUT RICH (VERY IMPORTANT)
texts = df["search_text"].tolist()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

# 🔑 FULL METADATA (THIS FIXES YOUR UI)
metadata = []
for _, row in df.iterrows():
    metadata.append({
        "title": row["search_text"],
        "assessment_name": row.get("assessment_name", ""),
        "assessment_url": row.get("assessment_url", "")
    })

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("✅ Embeddings & metadata saved successfully")
