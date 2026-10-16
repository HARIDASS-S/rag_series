from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ==========================================
# SETTINGS
# ==========================================
CHROMA_FOLDER = "chroma_db"
COLLECTION_NAME = "college_collection"
LLM_MODEL = "qwen3:4b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# ==========================================
# EMBEDDINGS
# ==========================================
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# ==========================================
# LOAD VECTOR DATABASE
# ==========================================
vector_db = Chroma(
    persist_directory=CHROMA_FOLDER,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

# ==========================================
# OLLAMA LLM
# ==========================================
llm = OllamaLLM(model=LLM_MODEL)

# ==========================================
# KEYWORD SCORE - MUST BE ON TOP
# ==========================================
def keyword_score(query, document):
    query_words = set(query.lower().split())
    document_words = set(document.lower().split())
    common = query_words & document_words
    return len(common) / max(len(query_words), 1)

# ==========================================
# RAG FUNCTION WITH HYBRID SCORE
# ==========================================
def ask_question(question, department_filter="All"):

    # 1. Get vector results WITH scores
    # Use similarity_search_with_score to get distance
    if department_filter!= "All":
        results = vector_db.similarity_search_with_score(
            question, k=10, filter={"department": department_filter}
        )
    else:
        results = vector_db.similarity_search_with_score(question, k=10)

    if not results:
        return "I could not find relevant information."

    # 2. Hybrid scoring
    scored_docs = []
    for doc, vector_distance in results:
        vector_score = 1 / (1 + vector_distance)
        kw_score = keyword_score(question, doc.page_content)
        final_score = (0.7 * vector_score) + (0.3 * kw_score)
        scored_docs.append((doc, final_score))

    # 3. Sort by final_score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, score in scored_docs[:4]]

    # 4. Create context
    context = "\n\n".join([d.page_content for d in top_docs])

    # 5. Prompt
    prompt = f"""
You are a College Information Assistant.
Answer the user's question using ONLY the context.

If not in context, say: "I don't have this information in the college documents."

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""
    answer = llm.invoke(prompt)

    sources = set(d.metadata.get("source", "Unknown") for d in top_docs)
    source_text = "\n\nSources:\n- " + "\n- ".join(sources)

    return answer + source_text