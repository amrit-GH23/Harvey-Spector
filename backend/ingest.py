"""
Jolly LLB — Constitution of India Ingestion (Structural RAG)
=================================================================
Reads COI.json and produces TWO sets of documents:

  1. Parent documents — Full article text (stored in ChromaDB parents collection)
  2. Child chunks    — Smaller ~200-token pieces (stored in main collection)

Each child chunk carries metadata linking it back to its parent article,
enabling "Small-to-Big" parent-document retrieval at query time.
"""

import json
import os

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config import (
    EMBED_MODEL,
    CHROMA_COLLECTION,
    CHROMA_PARENT_COLLECTION,
    CHROMA_PERSIST_DIR,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "COI.json")


def _build_article_text(article: dict) -> str:
    """Flatten an article into: Article [No]: [Title]. [Content]"""
    art_no = article.get("ArtNo", "")
    name = article.get("Name", "")

    if article.get("Status") == "Omitted":
        return ""

    content_parts = []

    if "ArtDesc" in article:
        content_parts.append(article["ArtDesc"])

    if "Clauses" in article:
        for clause in article["Clauses"]:
            clause_no = clause.get("ClauseNo", "")
            clause_desc = clause.get("ClauseDesc", "")
            content_parts.append(f"({clause_no}) {clause_desc}")

            if "SubClauses" in clause:
                for sub in clause["SubClauses"]:
                    sub_no = sub.get("SubClauseNo", "")
                    sub_desc = sub.get("SubClauseDesc", "")
                    content_parts.append(f"({sub_no}) {sub_desc}")

            if "FollowUp" in clause:
                content_parts.append(clause["FollowUp"])

    if "Explanations" in article:
        for exp in article["Explanations"]:
            content_parts.append(
                f"Explanation {exp.get('ExplanationNo', '')}: {exp.get('Explanation', '')}"
            )

    content = " ".join(content_parts)
    return f"Article {art_no}: {name}. {content}"


def _get_part_for_article(art_no: str, parts_index: list) -> str:
    """Look up which Part an article belongs to."""
    for part in parts_index:
        if art_no in part.get("Articles", []):
            return f"Part {part['PartNo']} - {part['Name']}"
    return "Unknown"


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_documents() -> tuple[list[Document], list[Document]]:
    """
    Load COI.json and produce:
      - parent_docs: one Document per article (full text)
      - child_docs:  multiple smaller chunks per article
    """
    with open(os.path.normpath(DATA_PATH), "r", encoding="utf-8") as f:
        raw = json.load(f)

    articles_list = raw[0]
    parts_index = raw[1]

    parent_docs = []
    child_docs = []

    for article in articles_list:
        if article.get("Status") == "Omitted":
            print(f"  [skip] Omitted Article {article.get('ArtNo')}")
            continue

        text = _build_article_text(article)
        if not text:
            continue

        art_no = article.get("ArtNo", "")
        part = _get_part_for_article(art_no, parts_index)
        title = article.get("Name", "")
        parent_id = f"art_{art_no}"

        # ── Parent document (full article) ──────────────────────
        parent_doc = Document(
            page_content=text,
            metadata={
                "article_no": art_no,
                "part": part,
                "title": title,
                "parent_id": parent_id,
                "doc_type": "parent",
            },
        )
        parent_docs.append(parent_doc)

        # ── Child chunks ────────────────────────────────────────
        chunks = _chunk_text(text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            child_doc = Document(
                page_content=chunk,
                metadata={
                    "article_no": art_no,
                    "part": part,
                    "title": title,
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_index": idx,
                },
            )
            child_docs.append(child_doc)

    print(f"  [OK] Prepared {len(parent_docs)} parent docs, {len(child_docs)} child chunks.")
    return parent_docs, child_docs


def ingest():
    """Ingest articles into ChromaDB as parent + child documents."""
    print("\n== Ingestion (Structural RAG) =====================")

    persist_dir = os.path.normpath(CHROMA_PERSIST_DIR)
    os.makedirs(persist_dir, exist_ok=True)

    parent_docs, child_docs = load_documents()
    if not parent_docs:
        print("  [WARN] No documents to ingest.")
        return

    print(f"  Embedding with {EMBED_MODEL} (HuggingFace, local)...")
    print("  This may take a few minutes on first run (model download)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Store parent documents ──────────────────────────────
    print(f"  Ingesting {len(parent_docs)} parent documents...")
    Chroma.from_documents(
        documents=parent_docs,
        embedding=embeddings,
        collection_name=CHROMA_PARENT_COLLECTION,
        persist_directory=persist_dir,
    )
    print(f"  [OK] Parent collection '{CHROMA_PARENT_COLLECTION}' ready.")

    # ── Store child chunks ──────────────────────────────────
    print(f"  Ingesting {len(child_docs)} child chunks...")
    Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=persist_dir,
    )
    print(f"  [OK] Child collection '{CHROMA_COLLECTION}' ready.")

    print(f"\n  [OK] Ingestion complete: {len(parent_docs)} articles -> {len(child_docs)} chunks.")


if __name__ == "__main__":
    ingest()
