import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "Llama3.1:8B")

# Embedding: HuggingFace bge-small-en-v1.5 (fast, 33MB, runs locally without Ollama)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# ChromaDB collections — Constitution
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "indian_constitution")
CHROMA_PARENT_COLLECTION = os.getenv("CHROMA_PARENT_COLLECTION", "indian_constitution_parents")

# BNS (Bharatiya Nyaya Sanhita) — Substantive Criminal Law (replaces IPC)
CHROMA_BNS_COLLECTION = os.getenv("CHROMA_BNS_COLLECTION", "bns_sections")
CHROMA_BNS_PARENT_COLLECTION = os.getenv("CHROMA_BNS_PARENT_COLLECTION", "bns_sections_parents")

# BNSS (Bharatiya Nagarik Suraksha Sanhita) — Criminal Procedure (replaces CrPC)
CHROMA_BNSS_COLLECTION = os.getenv("CHROMA_BNSS_COLLECTION", "bnss_sections")
CHROMA_BNSS_PARENT_COLLECTION = os.getenv("CHROMA_BNSS_PARENT_COLLECTION", "bnss_sections_parents")

# BSA (Bharatiya Sakshya Adhiniyam) — Evidence Law (replaces IEA)
CHROMA_BSA_COLLECTION = os.getenv("CHROMA_BSA_COLLECTION", "bsa_sections")
CHROMA_BSA_PARENT_COLLECTION = os.getenv("CHROMA_BSA_PARENT_COLLECTION", "bsa_sections_parents")

# FlashRank reranker (ONNX-optimized, ~4MB nano model, blazing fast)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")

# Child chunk settings for hierarchical chunking (approximate token counts via char ratio)
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "800"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "200"))

# ChromaDB stores data locally in this folder (no server needed)
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_data")
