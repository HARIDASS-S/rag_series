
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

LLM_MODEL = "llama3:latest"
EMBED_MODEL = "nomic-embed-text"

COLLECTION_NAME = "college_knowledge"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5
RETRIEVAL_CANDIDATES = 12
EMBED_BATCH_SIZE = 32