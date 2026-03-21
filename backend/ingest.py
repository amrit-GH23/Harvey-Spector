"""
Harvey Spector — Constitution of India Ingestion
=================================================
Reads COI.json, converts each Article into a LangChain Document
(format: "Article [No]: [Title]. [Content]") and stores in ChromaDB.
Skips if collection already has data.
"""

import json
import os

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config import OLLAMA_BASE_URL, EMBED_MODEL, CHROMA_COLLECTION, CHROMA_PERSIST_DIR

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
            content_parts.append(f"Explanation {exp.get('ExplanationNo', '')}: {exp.get('Explanation', '')}")

    content = " ".join(content_parts)
    return f"Article {art_no}: {name}. {content}"


def _get_part_for_article(art_no: str, parts_index: list) -> str:
    """Look up which Part an article belongs to."""
    for part in parts_index:
        if art_no in part.get("Articles", []):
            return f"Part {part['PartNo']} - {part['Name']}"
    return "Unknown"


def load_documents() -> list[Document]:
    """Load COI.json and convert to LangChain Documents."""
    with open(os.path.normpath(DATA_PATH), "r", encoding="utf-8") as f:
        raw = json.load(f)

    articles_list = raw[0]
    parts_index = raw[1]

    documents = []
    for article in articles_list:
        if article.get("Status") == "Omitted":
            print(f"  [skip] Omitted Article {article.get('ArtNo')}")
            continue

        text = _build_article_text(article)
        if not text:
            continue

        art_no = article.get("ArtNo", "")
        doc = Document(
            page_content=text,
            metadata={
                "article_no": art_no,
                "part": _get_part_for_article(art_no, parts_index),
                "title": article.get("Name", ""),
            },
        )
        documents.append(doc)

    print(f"  [OK] Prepared {len(documents)} documents.")
    return documents


def ingest():
    """Ingest articles into ChromaDB. Skips if already populated."""
    print("\n== Ingestion =====================================")

    persist_dir = os.path.normpath(CHROMA_PERSIST_DIR)
    os.makedirs(persist_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    # Idempotency check
    # try:
    #     collection = chroma_client.get_collection(CHROMA_COLLECTION)
    #     if collection.count() > 0:
    #         print(f"  [OK] Collection '{CHROMA_COLLECTION}' already has {collection.count()} docs. Skipping.")
    #         return
    # except Exception:
    #     pass

    documents = load_documents()
    if not documents:
        print("  [WARN] No documents to ingest.")
        return

    print(f"  Embedding with {EMBED_MODEL} via {OLLAMA_BASE_URL}...")
    print("  This may take a few minutes on first run...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=persist_dir,
    )
    print(f"  [OK] Ingested {len(documents)} articles into ChromaDB.")


if __name__ == "__main__":
    ingest()
