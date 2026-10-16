
import hashlib
from pathlib import Path

import chromadb
import pandas as pd
import fitz

from docx import Document
from ollama import Client

from config import (
    DATA_DIR,
    DB_DIR,
    OLLAMA_HOST,
    EMBED_MODEL,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_BATCH_SIZE,
)

ollama = Client(host=OLLAMA_HOST)

client = chromadb.PersistentClient(
    path=str(DB_DIR)
)


def extract_documents(path):
    """Return text units with source metadata."""

    ext = path.suffix.lower()
    source = path.name
    units = []

    # PDF: retain page numbers
    if ext == ".pdf":
        with fitz.open(path) as pdf:
            for page_no, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    units.append({
                        "text": text,
                        "page": page_no
                    })

    # TXT
    elif ext == ".txt":
        text = path.read_text(encoding="utf-8")
        units.append({"text": text})

    # DOCX
    elif ext == ".docx":
        doc = Document(path)

        text = "\n".join(
            p.text for p in doc.paragraphs if p.text.strip()
        )

        for table in doc.tables:
            for row in table.rows:
                text += "\n" + " | ".join(
                    cell.text for cell in row.cells
                )

        units.append({"text": text})

    # CSV and Excel: keep rows intact
    elif ext in {".csv", ".xlsx", ".xls"}:
        if ext == ".csv":
            df = pd.read_csv(path, dtype=str).fillna("")
        else:
            df = pd.read_excel(path, dtype=str).fillna("")

        for row_no, row in enumerate(
            df.to_dict(orient="records"), start=2
        ):
            text = "\n".join(
                f"{key}: {value}"
                for key, value in row.items()
                if str(value).strip()
            )

            if text.strip():
                units.append({
                    "text": text,
                    "row": row_no
                })

    return source, units


def split_text(text):
    """Simple character-based chunking with overlap."""

    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    if not any(DATA_DIR.iterdir()):
        print(f"Add files to: {DATA_DIR}")
        return

    # Rebuild the collection on each ingestion run.
    # Existing indexed content will be replaced.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    documents = []
    metadatas = []
    ids = []

    supported = {".pdf", ".txt", ".docx", ".csv", ".xlsx", ".xls"}

    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue

        source, units = extract_documents(path)

        for unit in units:
            for chunk_no, chunk in enumerate(
                split_text(unit["text"])
            ):
                metadata = {
                    "source": source,
                    "file_type": path.suffix.lower().lstrip("."),
                    "department": "general",
                    "year": "unknown",
                    "category": "general",
                }

                if "page" in unit:
                    metadata["page"] = unit["page"]

                if "row" in unit:
                    metadata["row"] = unit["row"]

                # Optional filename conventions:
                # AI_DS_2026_placement_report.pdf
                parts = path.stem.split("_")

                if len(parts) >= 3:
                    metadata["department"] = parts[0]
                    metadata["year"] = parts[1]

                if len(parts) >= 4:
                    metadata["category"] = parts[2].lower()

                stable_id = hashlib.sha256(
                    f"{source}|{unit.get('page', '')}|"
                    f"{unit.get('row', '')}|{chunk_no}".encode()
                ).hexdigest()

                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(stable_id)

    if not documents:
        print("No readable text found in the supported files.")
        return

    # Embed in batches to limit memory and request size.
    for start in range(0, len(documents), EMBED_BATCH_SIZE):
        end = start + EMBED_BATCH_SIZE

        batch_docs = documents[start:end]
        batch_embeddings = ollama.embed(
            model=EMBED_MODEL,
            input=batch_docs
        )["embeddings"]

        collection.add(
            ids=ids[start:end],
            documents=batch_docs,
            metadatas=metadatas[start:end],
            embeddings=batch_embeddings
        )

        print(f"Indexed {min(end, len(documents))}/{len(documents)} chunks")

    print("\nIngestion completed.")
    print("Total chunks:", collection.count())


if __name__ == "__main__":
    main()