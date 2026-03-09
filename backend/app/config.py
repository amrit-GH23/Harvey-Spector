import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "Llama3.1:8B ")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Llama3.1:8B ")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "indian_constitution")

# ChromaDB stores data locally in this folder (no server needed)
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_data")
