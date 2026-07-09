"""Central configuration. Everything tunable lives in .env — no magic numbers in code."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = ROOT_DIR / "data" / "uploads"
STORAGE_DIR = ROOT_DIR / "storage"
INDEX_PATH = STORAGE_DIR / "kb.faiss"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"

# LLM (any OpenAI-compatible endpoint: Groq / DeepSeek / OpenRouter)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Embeddings + retrieval
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "3"))
# Cosine-similarity score below which we refuse to answer and hand off to a human.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

# Chunking
CHUNK_SIZE = 500      # characters
CHUNK_OVERLAP = 80    # characters
