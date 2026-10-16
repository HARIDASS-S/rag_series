import os
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader
)


# ==========================================
# SETTINGS
# ==========================================

DATA_FOLDER = "data"
CHROMA_FOLDER = "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"


# ==========================================
# LOAD CSV FILES
# ==========================================

def load_csv(file_path):

    print("Reading CSV:", file_path)

    df = pd.read_csv(file_path)

    documents = []

    for index, row in df.iterrows():

        text = ""

        for column in df.columns:
            text += f"{column}: {row[column]}\n"

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "type": "csv"
                }
            )
        )

    return documents


# ==========================================
# LOAD EXCEL FILES
# ==========================================

def load_excel(file_path):

    print("Reading Excel:", file_path)

    df = pd.read_excel(file_path)

    documents = []

    for index, row in df.iterrows():

        text = ""

        for column in df.columns:
            text += f"{column}: {row[column]}\n"

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "type": "excel"
                }
            )
        )

    return documents


# ==========================================
# LOAD PDF
# ==========================================

def load_pdf(file_path):

    print("Reading PDF:", file_path)

    loader = PyMuPDFLoader(file_path)

    documents = loader.load()

    for document in documents:

        document.metadata["source"] = os.path.basename(file_path)
        document.metadata["type"] = "pdf"

    return documents


# ==========================================
# LOAD TXT
# ==========================================

def load_txt(file_path):

    print("Reading TXT:", file_path)

    loader = TextLoader(
        file_path,
        encoding="utf-8"
    )

    documents = loader.load()

    for document in documents:

        document.metadata["source"] = os.path.basename(file_path)
        document.metadata["type"] = "txt"

    return documents


# ==========================================
# LOAD DOCX
# ==========================================

def load_docx(file_path):

    print("Reading DOCX:", file_path)

    loader = Docx2txtLoader(file_path)

    documents = loader.load()

    for document in documents:

        document.metadata["source"] = os.path.basename(file_path)
        document.metadata["type"] = "docx"

    return documents


# ==========================================
# LOAD ALL DOCUMENTS
# ==========================================

def load_documents():

    all_documents = []

    if not os.path.exists(DATA_FOLDER):

        os.makedirs(DATA_FOLDER)

        print("Created data folder.")

    for filename in os.listdir(DATA_FOLDER):

        file_path = os.path.join(
            DATA_FOLDER,
            filename
        )

        extension = filename.lower().split(".")[-1]

        try:

            if extension == "csv":

                docs = load_csv(file_path)

            elif extension in ["xlsx", "xls"]:

                docs = load_excel(file_path)

            elif extension == "pdf":

                docs = load_pdf(file_path)

            elif extension == "txt":

                docs = load_txt(file_path)

            elif extension == "docx":

                docs = load_docx(file_path)

            else:

                print("Skipping:", filename)

                continue

            all_documents.extend(docs)

        except Exception as e:

            print(
                f"Error reading {filename}: {e}"
            )

    return all_documents


# ==========================================
# MAIN
# ==========================================

print("\n================================")
print("COLLEGE RAG - DOCUMENT INGESTION")
print("================================\n")


documents = load_documents()

print(
    f"\nTotal documents loaded: {len(documents)}"
)


# ==========================================
# TEXT CHUNKING
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=500,

    chunk_overlap=50

)

chunks = text_splitter.split_documents(
    documents
)

print(
    f"Total chunks created: {len(chunks)}"
)


# ==========================================
# OLLAMA EMBEDDINGS
# ==========================================

print("\nCreating Ollama embeddings...")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ==========================================
# CHROMA VECTOR DATABASE
# ==========================================

print("Creating ChromaDB...")

vector_db = Chroma.from_documents(

    documents=chunks,

    embedding=embeddings,

    persist_directory=CHROMA_FOLDER,

    collection_name="college_collection"

)


print("\n================================")
print("INGESTION COMPLETED")
print("================================")

print(
    f"Stored {len(chunks)} chunks in ChromaDB."
)