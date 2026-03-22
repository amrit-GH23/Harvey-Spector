"""
Jolly LLB — Criminal Law Ingestion (BNS / BNSS / BSA)
=======================================================
Reads BNS.json, BNSS.json, and BSA.json and produces parent + child documents
in separate ChromaDB collections for each law.

Uses the same structural RAG pattern as ingest.py (Constitution):
  1. Parent documents — Full section text
  2. Child chunks    — Smaller ~800-char pieces with overlap

Usage:
  python ingest_laws.py          # Ingest all 3 laws
  python ingest_laws.py bns      # Ingest only BNS
  python ingest_laws.py bnss bsa # Ingest BNSS and BSA
"""

import json
import os
import sys

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config import (
    EMBED_MODEL,
    CHROMA_BNS_COLLECTION,
    CHROMA_BNS_PARENT_COLLECTION,
    CHROMA_BNSS_COLLECTION,
    CHROMA_BNSS_PARENT_COLLECTION,
    CHROMA_BSA_COLLECTION,
    CHROMA_BSA_PARENT_COLLECTION,
    CHROMA_PERSIST_DIR,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)

# Law configurations: code → (data file, child collection, parent collection, label)
LAW_CONFIG = {
    "bns": {
        "file": "BNS.json",
        "child_collection": CHROMA_BNS_COLLECTION,
        "parent_collection": CHROMA_BNS_PARENT_COLLECTION,
        "label": "BNS (Bharatiya Nyaya Sanhita)",
        "source_type": "bns",
    },
    "bnss": {
        "file": "BNSS.json",
        "child_collection": CHROMA_BNSS_COLLECTION,
        "parent_collection": CHROMA_BNSS_PARENT_COLLECTION,
        "label": "BNSS (Bharatiya Nagarik Suraksha Sanhita)",
        "source_type": "bnss",
    },
    "bsa": {
        "file": "BSA.json",
        "child_collection": CHROMA_BSA_COLLECTION,
        "parent_collection": CHROMA_BSA_PARENT_COLLECTION,
        "label": "BSA (Bharatiya Sakshya Adhiniyam)",
        "source_type": "bsa",
    },
}


def _build_section_text(section: dict, source_type: str) -> str:
    """Flatten a section into: Section [No]: [Title]. [Description]"""
    sec_no = section.get("SectionNo", "")
    name = section.get("Name", "")
    desc = section.get("SectionDesc", "")

    if not desc:
        return ""

    return f"Section {sec_no}: {name}. {desc}"


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


def load_law_documents(
    data_path: str, source_type: str
) -> tuple[list[Document], list[Document]]:
    """
    Load a law JSON file and produce:
      - parent_docs: one Document per section (full text)
      - child_docs:  multiple smaller chunks per section
    """
    with open(os.path.normpath(data_path), "r", encoding="utf-8") as f:
        raw = json.load(f)

    sections_list = raw[0]
    chapters_index = raw[1]

    parent_docs = []
    child_docs = []

    for section in sections_list:
        text = _build_section_text(section, source_type)
        if not text:
            continue

        sec_no = section.get("SectionNo", "")
        chapter = section.get("Chapter", "")
        chapter_name = section.get("ChapterName", "")
        title = section.get("Name", "")
        parent_id = f"{source_type}_sec_{sec_no}"

        # ── Parent document (full section) ────────────────────
        parent_doc = Document(
            page_content=text,
            metadata={
                "section_no": sec_no,
                "chapter": chapter,
                "chapter_name": chapter_name,
                "title": title,
                "parent_id": parent_id,
                "doc_type": "parent",
                "source_type": source_type,
            },
        )
        parent_docs.append(parent_doc)

        # ── Child chunks ──────────────────────────────────────
        chunks = _chunk_text(text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            child_doc = Document(
                page_content=chunk,
                metadata={
                    "section_no": sec_no,
                    "chapter": chapter,
                    "chapter_name": chapter_name,
                    "title": title,
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_index": idx,
                    "source_type": source_type,
                },
            )
            child_docs.append(child_doc)

    print(f"  [OK] Prepared {len(parent_docs)} parent docs, {len(child_docs)} child chunks.")
    return parent_docs, child_docs


def ingest_law(law_code: str) -> None:
    """Ingest a single law into ChromaDB."""
    config = LAW_CONFIG[law_code]
    data_path = os.path.join(os.path.dirname(__file__), config["file"])

    if not os.path.exists(data_path):
        print(f"  [ERROR] {config['file']} not found. Run convert_laws.py first!")
        return

    print("\n" + "=" * 60)
    print(f"  Ingesting: {config['label']}")
    print("=" * 60)

    persist_dir = os.path.normpath(CHROMA_PERSIST_DIR)
    os.makedirs(persist_dir, exist_ok=True)

    parent_docs, child_docs = load_law_documents(data_path, config["source_type"])
    if not parent_docs:
        print(f"  [WARN] No documents to ingest for {law_code}.")
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
        collection_name=config["parent_collection"],
        persist_directory=persist_dir,
    )
    print(f"  [OK] Parent collection '{config['parent_collection']}' ready.")

    # ── Store child chunks ──────────────────────────────────
    print(f"  Ingesting {len(child_docs)} child chunks...")
    Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings,
        collection_name=config["child_collection"],
        persist_directory=persist_dir,
    )
    print(f"  [OK] Child collection '{config['child_collection']}' ready.")

    print(f"\n  [OK] {config['label']}: {len(parent_docs)} sections -> {len(child_docs)} chunks.")


def main():
    print("=" * 60)
    print("  Jolly LLB -- Criminal Law Ingestion")
    print("=" * 60)

    # Determine which laws to ingest
    args = [a.lower() for a in sys.argv[1:]]
    if args:
        laws_to_ingest = [a for a in args if a in LAW_CONFIG]
        if not laws_to_ingest:
            print(f"  [ERROR] Unknown law codes: {args}")
            print(f"  Valid codes: {list(LAW_CONFIG.keys())}")
            sys.exit(1)
    else:
        laws_to_ingest = list(LAW_CONFIG.keys())

    print(f"\n  Laws to ingest: {', '.join(l.upper() for l in laws_to_ingest)}")

    for law_code in laws_to_ingest:
        ingest_law(law_code)

    print("\n" + "=" * 60)
    print("[OK] Ingestion complete!")
    print("  Start the server: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
