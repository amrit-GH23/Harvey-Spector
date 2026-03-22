"""
Jolly LLB — Multi-Collection Hybrid Search + Reranking Pipeline
=================================================================
Searches across ALL legal collections (Constitution, BNS, BNSS, BSA).

Optimized for speed:
  - bge-small-en-v1.5 embeddings (HuggingFace, 33MB, local)
  - FlashRank reranker (ONNX, ~4MB, ultra-fast)
  - BM25 index caching (built once per collection, reused)

Three-stage retrieval:
  1. Metadata-first filter  — Direct article/section lookup
  2. Hybrid search          — BM25 keyword + ChromaDB vector, fused via RRF
  3. FlashRank reranker     — Score candidates, keep top results
"""

import os
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

from app.config import (
    EMBED_MODEL,
    CHROMA_COLLECTION,
    CHROMA_PARENT_COLLECTION,
    CHROMA_BNS_COLLECTION,
    CHROMA_BNS_PARENT_COLLECTION,
    CHROMA_BNSS_COLLECTION,
    CHROMA_BNSS_PARENT_COLLECTION,
    CHROMA_BSA_COLLECTION,
    CHROMA_BSA_PARENT_COLLECTION,
    CHROMA_PERSIST_DIR,
    RERANKER_MODEL,
)

# ── Collection registry ────────────────────────────────────
COLLECTIONS = [
    {
        "child": CHROMA_COLLECTION,
        "parent": CHROMA_PARENT_COLLECTION,
        "source_type": "constitution",
        "label": "Indian Constitution",
        "id_field": "article_no",
        "parent_id_prefix": "art_",
    },
    {
        "child": CHROMA_BNS_COLLECTION,
        "parent": CHROMA_BNS_PARENT_COLLECTION,
        "source_type": "bns",
        "label": "BNS (Bharatiya Nyaya Sanhita)",
        "id_field": "section_no",
        "parent_id_prefix": "bns_sec_",
    },
    {
        "child": CHROMA_BNSS_COLLECTION,
        "parent": CHROMA_BNSS_PARENT_COLLECTION,
        "source_type": "bnss",
        "label": "BNSS (Bharatiya Nagarik Suraksha Sanhita)",
        "id_field": "section_no",
        "parent_id_prefix": "bnss_sec_",
    },
    {
        "child": CHROMA_BSA_COLLECTION,
        "parent": CHROMA_BSA_PARENT_COLLECTION,
        "source_type": "bsa",
        "label": "BSA (Bharatiya Sakshya Adhiniyam)",
        "id_field": "section_no",
        "parent_id_prefix": "bsa_sec_",
    },
]

# ── Lazy-loaded singletons ──────────────────────────────────
_reranker: Ranker | None = None
_embeddings: HuggingFaceEmbeddings | None = None
_stores: dict[str, Chroma] = {}
_bm25_cache: dict[str, tuple[BM25Okapi, list[str], list[dict]]] = {}


def _get_reranker() -> Ranker:
    """Load FlashRank reranker (cached after first call)."""
    global _reranker
    if _reranker is None:
        _reranker = Ranker(model_name=RERANKER_MODEL)
    return _reranker


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Get HuggingFace embeddings singleton (bge-small-en-v1.5)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_store(collection_name: str) -> Chroma:
    """Connect to a ChromaDB collection (cached after first call)."""
    if collection_name not in _stores:
        persist_dir = os.path.normpath(CHROMA_PERSIST_DIR)
        _stores[collection_name] = Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=_get_embeddings(),
        )
    return _stores[collection_name]


def _collection_exists(collection_name: str) -> bool:
    """Check if a ChromaDB collection actually has data."""
    try:
        store = _get_store(collection_name)
        count = store._collection.count()
        return count > 0
    except Exception:
        return False


def _get_bm25_index(collection_name: str) -> tuple[BM25Okapi, list[str], list[dict]] | None:
    """
    Get or build BM25 index for a collection (cached after first build).
    Returns (bm25_index, all_texts, all_metas) or None.
    """
    if collection_name in _bm25_cache:
        return _bm25_cache[collection_name]

    try:
        store = _get_store(collection_name)
        all_data = store.get(include=["documents", "metadatas"])
        all_texts = all_data["documents"]
        all_metas = all_data["metadatas"]

        if not all_texts:
            return None

        tokenized_corpus = [doc.lower().split() for doc in all_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        _bm25_cache[collection_name] = (bm25, all_texts, all_metas)
        return _bm25_cache[collection_name]
    except Exception:
        return None


# ── Helper: fetch parent documents by parent_id ─────────────
def _fetch_parents(parent_ids: list[str], parent_collection: str) -> list[Document]:
    """Retrieve full-text parent documents from a parent collection."""
    if not parent_ids:
        return []

    store = _get_store(parent_collection)
    results = store.get(where={"parent_id": {"$in": parent_ids}}, include=["documents", "metadatas"])

    docs = []
    seen = set()
    for text, meta in zip(results["documents"], results["metadatas"]):
        pid = meta.get("parent_id", "")
        if pid not in seen:
            seen.add(pid)
            docs.append(Document(page_content=text, metadata=meta))
    return docs


# ── Stage 1: Metadata-first filter ──────────────────────────
def _extract_article_number(query: str) -> str | None:
    match = re.search(r"(?:article|art\.?)\s*(\d+[A-Za-z]*)", query, re.IGNORECASE)
    return match.group(1) if match else None


def _extract_section_number(query: str) -> str | None:
    match = re.search(r"(?:section|sec\.?)\s*(\d+[A-Za-z]*)", query, re.IGNORECASE)
    return match.group(1) if match else None


def _metadata_filter(query: str) -> list[Document] | None:
    """If query references a specific article or section, fetch it directly."""
    art_no = _extract_article_number(query)
    sec_no = _extract_section_number(query)

    if art_no is None and sec_no is None:
        return None

    docs = []

    for coll in COLLECTIONS:
        if not _collection_exists(coll["parent"]):
            continue

        store = _get_store(coll["parent"])

        if art_no and coll["source_type"] == "constitution":
            results = store.get(where={"article_no": art_no}, include=["documents", "metadatas"])
            for text, meta in zip(results["documents"], results["metadatas"]):
                meta["source_type"] = coll["source_type"]
                docs.append(Document(page_content=text, metadata=meta))

        if sec_no and coll["source_type"] != "constitution":
            results = store.get(where={"section_no": sec_no}, include=["documents", "metadatas"])
            for text, meta in zip(results["documents"], results["metadatas"]):
                meta["source_type"] = coll["source_type"]
                docs.append(Document(page_content=text, metadata=meta))

    return docs if docs else None


# ── Stage 2: Hybrid search (BM25 + Vector + RRF) ────────────
def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def _hybrid_search_single(query: str, child_collection: str, source_type: str, top_k: int = 10) -> list[Document]:
    """Run BM25 + vector search on a single child collection with cached BM25."""
    if not _collection_exists(child_collection):
        return []

    child_store = _get_store(child_collection)

    # ── Vector search ──────────────────────────────────────
    try:
        vector_results = child_store.similarity_search(query, k=top_k)
    except Exception:
        vector_results = []

    # ── BM25 keyword search (CACHED) ───────────────────────
    bm25_results = []
    bm25_data = _get_bm25_index(child_collection)
    if bm25_data:
        bm25, all_texts, all_metas = bm25_data
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        bm25_ranked_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_k]

        bm25_results = [
            Document(page_content=all_texts[i], metadata=all_metas[i])
            for i in bm25_ranked_indices
        ]

    if not vector_results and not bm25_results:
        return []

    # ── RRF fusion ─────────────────────────────────────────
    def _doc_id(doc: Document) -> str:
        m = doc.metadata
        return f"{source_type}_{m.get('parent_id', '')}_{m.get('chunk_index', 0)}"

    vector_ids = [_doc_id(d) for d in vector_results]
    bm25_ids = [_doc_id(d) for d in bm25_results]
    fused_order = _reciprocal_rank_fusion([vector_ids, bm25_ids])

    doc_lookup: dict[str, Document] = {}
    for doc in vector_results + bm25_results:
        did = _doc_id(doc)
        if did not in doc_lookup:
            doc.metadata["source_type"] = source_type
            doc_lookup[did] = doc

    return [doc_lookup[did] for did in fused_order[:top_k] if did in doc_lookup]


def _multi_collection_hybrid_search(query: str, per_collection_k: int = 10) -> list[Document]:
    """Run hybrid search across ALL available collections."""
    all_candidates = []
    for coll in COLLECTIONS:
        candidates = _hybrid_search_single(query, coll["child"], coll["source_type"], top_k=per_collection_k)
        all_candidates.extend(candidates)
    return all_candidates


# ── Stage 3: FlashRank reranking ────────────────────────────
def _rerank(query: str, docs: list[Document], top_k: int = 5) -> list[Document]:
    """Score documents with FlashRank and return the top_k best matches."""
    if not docs:
        return []

    ranker = _get_reranker()

    # Build passages for FlashRank
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)

    # Map back to documents, sorted by FlashRank score
    reranked = []
    for result in results[:top_k]:
        idx = int(result["id"])
        reranked.append(docs[idx])

    return reranked


# ── Public API ──────────────────────────────────────────────
def hybrid_retrieve(query: str, final_k: int = 5) -> list[Document]:
    """
    Full multi-collection retrieval pipeline:
      1. Try metadata-first filter (direct article/section lookup)
      2. If no direct match -> hybrid search across ALL collections
      3. Rerank candidates with FlashRank
      4. Map winning child chunks -> parent (full) documents

    Returns up to `final_k` parent Documents from any law source.
    """
    # Stage 1: metadata filter (direct lookup)
    direct = _metadata_filter(query)
    if direct:
        if len(direct) > final_k:
            return _rerank(query, direct, top_k=final_k)
        return direct

    # Stage 2: hybrid search across all collections
    candidates = _multi_collection_hybrid_search(query, per_collection_k=10)

    if not candidates:
        return []

    # Stage 3: rerank all candidates together
    best_children = _rerank(query, candidates, top_k=final_k)

    # Stage 4: map children -> parent documents
    parents = []
    seen_parents = set()

    for doc in best_children:
        source_type = doc.metadata.get("source_type", "constitution")
        parent_id = doc.metadata.get("parent_id", "")

        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)

        for coll in COLLECTIONS:
            if coll["source_type"] == source_type:
                parent_docs = _fetch_parents([parent_id], coll["parent"])
                for pdoc in parent_docs:
                    pdoc.metadata["source_type"] = source_type
                parents.extend(parent_docs)
                break

    if parents:
        return parents[:final_k]

    # Fallback: return child chunks directly
    return best_children
