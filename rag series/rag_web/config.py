import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
DB_DIR = BASE_DIR / "chroma_db"

UPLOAD_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = "llama3:latest"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "uploaded_documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120