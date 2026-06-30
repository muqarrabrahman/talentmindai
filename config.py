"""Central configuration for the Policy RAG pipeline (Phase 1)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT = Path(__file__).resolve().parent
RAW_POLICIES = ROOT / "data" / "raw" / "policies"
PROCESSED_TEXT = ROOT / "data" / "processed" / "policies_text"
PROCESSED_CHUNKS = ROOT / "data" / "processed" / "chunks"
MANIFEST_FILE = ROOT / "data" / "processed" / "manifest.json"

# --- Embeddings (local model, no API key needed) ---
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
# BGE v1.5 recommends this instruction be prepended to QUERIES only (not passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Chunking (token-based) ---
CHUNK_SIZE = 500     # target tokens per chunk
CHUNK_OVERLAP = 75   # tokens carried over between consecutive chunks

# --- Qdrant vector store ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "policies"

# --- Claude (used by the /chat endpoint) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CHAT_MODEL = "claude-opus-4-8"   # swap to "claude-haiku-4-5" for cheaper/faster demos
TOP_K = 5                         # how many chunks to retrieve per query

# --- API server ---
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8001"))   # 8000 was taken, so default to 8001
