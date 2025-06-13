from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os

# === Configuration ===
PINECONE_API_KEY = "pcsk_65NWJU_JYvrkcTdXm1kmx3jhoQEwEBHjjmeFEdk1jP9PhFkR6YrpZAvQygmggnY6zvxVct"
INDEX_NAME = "audit-docs-free"
MODEL_NAME = "all-MiniLM-L6-v2"

# === Initialize ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)

app = FastAPI()

class MatchMetadata(BaseModel):
    doc_type: Optional[str]
    domain: Optional[str]
    last_updated: Optional[str]

class Match(BaseModel):
    id: str
    score: float
    metadata: MatchMetadata

class MatchResponse(BaseModel):
    matches: List[Match]

@app.get("/query", response_model=MatchResponse)
def query_pinecone(question: str, namespace: Optional[str] = "default", top_k: int = 3):
    vector = model.encode(question).tolist()
    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    matches = [
        Match(
            id=match["id"],
            score=match["score"],
            metadata=MatchMetadata(**match["metadata"])
        )
        for match in result["matches"]
    ]
    return MatchResponse(matches=matches)
