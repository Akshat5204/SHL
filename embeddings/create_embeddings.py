import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

data_path = "data/processed/processed_data.csv"

df = pd.read_csv(data_path)

texts = df["search_text"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "embeddings/faiss.index")

with open("embeddings/meta.pkl", "wb") as f:
    pickle.dump(df.to_dict(orient="records"), f)

print("✅ Structured embeddings saved")
