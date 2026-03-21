"""
Harvey Spector — RAG Retrieval & Synthesis
==========================================
1. Similarity search → Top 3 articles from ChromaDB
2. Context injection → Formatted source strings
3. System prompt → Legal assistant with strict grounding
4. Output → { answer, sources }
"""

import os

import chromadb
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import OLLAMA_BASE_URL, LLM_MODEL, EMBED_MODEL, CHROMA_COLLECTION, CHROMA_PERSIST_DIR

SYSTEM_PROMPT = """You are a neutral Legal Assistant specialized in the Constitution of India.
Using ONLY the provided Context, answer the User Query.

Rules:
1. Start by explicitly stating which Article(s) apply.
2. Summarize the law in simple terms that a non-lawyer can understand.
3. Do not give personal opinions; stick to the text of the Constitution.
4. If the provided context does not contain the answer, state that the Constitution does not explicitly cover this specific detail.
5. Always reference articles by their number and title."""


def _get_vectorstore() -> Chroma:
    """Connect to local ChromaDB collection."""
    persist_dir = os.path.normpath(CHROMA_PERSIST_DIR)
    return Chroma(
        persist_directory=persist_dir,
        collection_name=CHROMA_COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL),
    )


def get_legal_advice(user_query: str) -> dict:
    """
    RAG workflow:
      1. Similarity search for top 3 articles
      2. Format context
      3. Query LLM with legal system prompt
      4. Return { answer, sources }
    """
    # Step 1 — Retrieve
    docs = _get_vectorstore().similarity_search(user_query, k=5)

    if not docs:
        return {"answer": "No relevant articles found.", "sources": []}

    # Step 2 — Format context
    context_parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        art_no = doc.metadata.get("article_no", "?")
        title = doc.metadata.get("title", "")
        part = doc.metadata.get("part", "")
        context_parts.append(f"Source {i}: Article {art_no} ({title}) [{part}]\n{doc.page_content}")
        sources.append({"article_no": art_no, "title": title, "part": part})

    context = "\n\n---\n\n".join(context_parts)

    # Step 3 — Query LLM
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\n---\n\nUser Query: {user_query}"),
    ])

    # Step 4 — Return
    return {"answer": response.content, "sources": sources}
