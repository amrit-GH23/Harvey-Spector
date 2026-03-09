"""
Harvey Spector — FastAPI Application
=====================================
GET  /          → Health check
POST /query     → Semantic search + legal summary
GET  /articles  → List all articles from COI.json
"""

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag import get_legal_advice

app = FastAPI(
    title="Harvey Spector",
    description="⚖️ Semantic search over the Indian Constitution with legally grounded summaries.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "COI.json")


# ── Models ───────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Your legal question")


class SourceInfo(BaseModel):
    article_no: str
    title: str
    part: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


# ── Endpoints ────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health():
    return {"status": "healthy", "service": "Harvey Spector"}


@app.post("/query", response_model=QueryResponse, tags=["Legal Query"])
async def query_constitution(req: QueryRequest):
    """Semantic search + LLM-grounded legal summary."""
    try:
        result = get_legal_advice(req.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/articles", tags=["Reference"])
async def list_articles():
    """List all articles from the Constitution data."""
    try:
        with open(os.path.normpath(DATA_PATH), "r", encoding="utf-8") as f:
            raw = json.load(f)
        articles = [
            {"article_no": a.get("ArtNo"), "title": a.get("Name"), "status": a.get("Status", "Active")}
            for a in raw[0]
        ]
        return {"total": len(articles), "articles": articles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
