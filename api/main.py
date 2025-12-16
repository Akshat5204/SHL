from fastapi import FastAPI
from pydantic import BaseModel
from embeddings.recommend import recommend

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def get_recommendations(req: QueryRequest):
    results = recommend(req.query, k=5)
    return {"recommended_assessments": results}
