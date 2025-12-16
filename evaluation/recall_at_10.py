import pandas as pd
from embeddings.recommender import recommend

def recall_at_k(predicted, relevant, k=10):
    predicted = predicted[:k]
    hits = sum(1 for p in predicted if p in relevant)
    return hits / len(relevant) if relevant else 0.0

# Load labeled train data
df = pd.read_excel("data/raw/Gen_AI Dataset (1).xlsx")

recalls = []

for _, row in df.iterrows():
    query = row["query"]

    # Adjust column name if different
    relevant = str(row["relevant_assessments"]).split(",")

    preds = recommend(query, k=10)
    predicted_urls = [p["url"] for p in preds]

    r = recall_at_k(predicted_urls, relevant)
    recalls.append(r)

mean_recall = sum(recalls) / len(recalls)
print(f"Mean Recall@10: {mean_recall:.4f}")
