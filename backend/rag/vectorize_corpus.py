import csv
import os
import shutil
import sys
from pathlib import Path

import chromadb
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


field_size_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_size_limit)
        break
    except OverflowError:
        field_size_limit //= 10


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DOCS_PATH = Path(os.getenv("DOCS_PATH", BASE_DIR / "corpus")).resolve()
DOCS_FILE = os.getenv("DOCS_FILE", "pddieu.csv")
DB_PERSIST_PATH = Path(os.getenv("TOPIC_DB_PATH", BASE_DIR / "chroma_db_law")).resolve()
ST_MODEL_PATH = os.getenv("ST_MODEL_PATH", "keepitreal/vietnamese-sbert")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").lower()
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "langchain")
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", "1024"))
ENCODE_BATCH_SIZE = int(os.getenv("ENCODE_BATCH_SIZE", "64"))


def row_text(row):
    title = (row.get("title") or "").strip()
    content = (row.get("content") or "").strip()
    if title and content:
        return f"{title}\n{content}"
    return title or content


def row_metadata(row, source_name, row_number):
    metadata = {
        "source": source_name,
        "row": row_number,
    }
    for key, value in row.items():
        if key == "content":
            continue
        value = str(value or "").strip()
        if value:
            metadata[key] = value
    return metadata


def csv_paths():
    path = DOCS_PATH / DOCS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    return [path]


def iter_csv_documents():
    for csv_path in csv_paths():
        print(f"Loading corpus file: {csv_path}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row_number, row in enumerate(reader, start=1):
                text = row_text(row)
                if not text:
                    continue
                doc_id = f"{csv_path.stem}:{row.get('id') or row_number}"
                yield doc_id, text, row_metadata(row, csv_path.name, row_number)


def resolve_device():
    if EMBEDDING_DEVICE != "auto":
        return EMBEDDING_DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


def flush_batch(collection, embedding_model, ids, documents, metadatas):
    if not ids:
        return 0
    embeddings = embedding_model.encode(
        documents,
        batch_size=ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(ids)


def main():
    print(f"Corpus path: {DOCS_PATH}")
    print(f"Corpus file: {DOCS_FILE}")
    print(f"Chroma DB path: {DB_PERSIST_PATH}")
    device = resolve_device()
    print(f"Embedding corpus with Vietnamese SBERT model: {ST_MODEL_PATH}")
    print(f"Embedding device: {device}")
    print(f"Chroma batch size: {CHROMA_BATCH_SIZE}")
    print(f"Encode batch size: {ENCODE_BATCH_SIZE}")

    if DB_PERSIST_PATH.exists():
        print(f"Resetting Chroma DB at {DB_PERSIST_PATH}")
        shutil.rmtree(DB_PERSIST_PATH)

    DB_PERSIST_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PERSIST_PATH))
    embedding_model = SentenceTransformer(ST_MODEL_PATH, device=device)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []
    total = 0

    for doc_id, text, metadata in iter_csv_documents():
        ids.append(doc_id)
        documents.append(text)
        metadatas.append(metadata)

        if len(ids) >= CHROMA_BATCH_SIZE:
            total += flush_batch(collection, embedding_model, ids, documents, metadatas)
            print(f"Inserted {total} documents")
            ids.clear()
            documents.clear()
            metadatas.clear()

    total += flush_batch(collection, embedding_model, ids, documents, metadatas)
    print(f"Done. Inserted {total} documents into {DB_PERSIST_PATH} collection {CHROMA_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
