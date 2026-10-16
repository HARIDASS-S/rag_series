
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import LLM_MODEL, EMBED_MODEL
from rag_engine import answer_question, collection, ollama


app = FastAPI(
    title="College RAG Assistant API",
    description="Local document retrieval and grounded answers",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    history: list[dict[str, str]] = Field(default_factory=list)
    department: str | None = None
    year: str | None = None
    category: str | None = None


@app.get("/")
def home():
    return {
        "message": "College RAG Assistant API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    try:
        models = ollama.list()

        return {
            "status": "healthy",
            "indexed_chunks": collection.count(),
            "llm_model": LLM_MODEL,
            "embedding_model": EMBED_MODEL,
            "ollama_available": True,
            "installed_models": models.get("models", [])
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "indexed_chunks": collection.count(),
            "ollama_available": False,
            "error": str(exc)
        }


@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    result = answer_question(
        question=question,
        history=request.history,
        department=request.department,
        year=request.year,
        category=request.category
    )

    return {
        "question": question,
        "standalone_question": result["standalone_question"],
        "answer": result["answer"],
        "sources": result["sources"]
    }