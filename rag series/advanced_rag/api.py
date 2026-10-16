from fastapi import FastAPI
from pydantic import BaseModel

import chromadb

from ollama import embed
from ollama import chat


app = FastAPI(
    title="College RAG API"
)


client = chromadb.PersistentClient(
    path="chroma_db"
)


collection = client.get_collection(
    "college_rag"
)


class Question(BaseModel):

    question: str


@app.get("/")
def home():

    return {
        "message":
        "College RAG API is running"
    }


@app.get("/health")
def health():

    return {

        "status": "healthy",

        "documents":
        collection.count()

    }


@app.post("/ask")
def ask(
    request: Question
):

    question = request.question


    embedding = embed(

        model="nomic-embed-text",

        input=question

    )["embeddings"][0]


    results = collection.query(

        query_embeddings=[
            embedding
        ],

        n_results=5

    )


    context = "\n\n".join(

        results["documents"][0]

    )


    prompt = f"""
You are a college AI assistant.

Use ONLY the context.

Do not invent information.

If the information is missing,
say you don't know.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""


    response = chat(

        model="llama3:latest",

        messages=[
            {
                "role":
                "user",

                "content":
                prompt
            }
        ]

    )


    answer = response[
        "message"
    ][
        "content"
    ]


    sources = [

        metadata.get(
            "source",
            "unknown"
        )

        for metadata
        in results["metadatas"][0]

    ]


    return {

        "answer":
        answer,

        "sources":
        list(set(sources))

    }