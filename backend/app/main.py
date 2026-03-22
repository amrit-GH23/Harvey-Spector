"""
Jolly LLB — FastAPI Application
=================================
GET  /          → Health check
POST /query     → Semantic search + legal summary (Constitution + BNS + BNSS + BSA)
GET  /articles  → List all articles from COI.json
GET  /sections  → List all sections from BNS/BNSS/BSA
"""

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag import get_legal_advice

app = FastAPI(
    title="Jolly LLB",
    description="⚖️ AI-powered legal assistant for Indian Law — Constitution, BNS, BNSS & BSA.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")
COI_PATH = os.path.join(DATA_DIR, "COI.json")

LAW_FILES = {
    "bns": {"path": os.path.join(DATA_DIR, "BNS.json"), "label": "BNS (Bharatiya Nyaya Sanhita)"},
    "bnss": {"path": os.path.join(DATA_DIR, "BNSS.json"), "label": "BNSS (Bharatiya Nagarik Suraksha Sanhita)"},
    "bsa": {"path": os.path.join(DATA_DIR, "BSA.json"), "label": "BSA (Bharatiya Sakshya Adhiniyam)"},
}


# ── Models ───────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Your legal question")


class SourceInfo(BaseModel):
    article_no: str | None = None
    section_no: str | None = None
    title: str = ""
    part: str | None = None
    chapter: str | None = None
    source_type: str = "constitution"


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


# ── Endpoints ────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health():
    return {"status": "healthy", "service": "Jolly LLB", "version": "2.0.0"}


@app.post("/query", response_model=QueryResponse, tags=["Legal Query"])
async def query_law(req: QueryRequest):
    """Semantic search + LLM-grounded legal summary across all Indian law sources."""
    try:
        result = get_legal_advice(req.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/articles", tags=["Reference"])
async def list_articles():
    """List all articles from the Constitution data."""
    try:
        with open(os.path.normpath(COI_PATH), "r", encoding="utf-8") as f:
            raw = json.load(f)
        articles = [
            {"article_no": a.get("ArtNo"), "title": a.get("Name"), "status": a.get("Status", "Active")}
            for a in raw[0]
        ]
        return {"total": len(articles), "articles": articles}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="COI.json not found. Run convert_coi.py first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sections", tags=["Reference"])
async def list_sections(law: str | None = None):
    """
    List all sections from BNS, BNSS, and/or BSA.

    Query params:
      - law: optional filter — 'bns', 'bnss', or 'bsa' (omit for all)
    """
    try:
        result = {}

        laws_to_load = {law: LAW_FILES[law]} if law and law in LAW_FILES else LAW_FILES

        for law_code, info in laws_to_load.items():
            path = os.path.normpath(info["path"])
            if not os.path.exists(path):
                continue

            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            sections = [
                {
                    "section_no": s.get("SectionNo"),
                    "title": s.get("Name"),
                    "chapter": s.get("Chapter"),
                    "chapter_name": s.get("ChapterName"),
                }
                for s in raw[0]
            ]

            result[law_code] = {
                "label": info["label"],
                "total": len(sections),
                "sections": sections,
            }

        if not result:
            raise HTTPException(
                status_code=404,
                detail="No law data found. Run convert_laws.py and ingest_laws.py first.",
            )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
