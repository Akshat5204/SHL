#!/usr/bin/env bash

echo "Starting SHL Recommendation API..."

# Build FAISS index first
python embeddings/create_embeddings.py

# Start FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 10000
