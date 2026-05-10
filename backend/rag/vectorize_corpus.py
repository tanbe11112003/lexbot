import csv
import os
import re
import shutil
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def default_huggingface_cache_dir():
    virtual_env = os.getenv("VIRTUAL_ENV")
    if virtual_env:
        return Path(virtual_env) / ".cache" / "huggingface"

    for parent in [BASE_DIR, *BASE_DIR.parents]:
        venv_path = parent / ".venv"
        if venv_path.exists():
            return venv_path / ".cache" / "huggingface"

    return BASE_DIR.parent.parent / ".cache" / "huggingface"


HF_CACHE_DIR = Path(os.getenv("SAULAI_HF_CACHE_DIR") or os.getenv("HF_HOME") or default_huggingface_cache_dir())
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR / "sentence-transformers"))

import chromadb
import torch
from sentence_transformers import SentenceTransformer


field_size_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_size_limit)
        break
    except OverflowError:
        field_size_limit //= 10


DOCS_PATH = (BASE_DIR / "corpus").resolve()
DOCS_FILE = "pddieu.csv"
DB_PERSIST_PATH = (BASE_DIR / "chroma_db_law").resolve()
ST_MODEL_PATH = "keepitreal/vietnamese-sbert"
EMBEDDING_DEVICE = "auto"
CHROMA_COLLECTION_NAME = "langchain"
CHROMA_BATCH_SIZE = 1024
ENCODE_BATCH_SIZE = 64
ARTICLE_CHUNK_MAX_CHARS = 2000
ARTICLE_CHUNK_OVERLAP_CHARS = 150
DOCUMENT_TYPE_LABELS = {
    "LQ": "Luật",
    "NĐ": "Nghị định",
    "ND": "Nghị định",
    "TT": "Thông tư",
    "QĐ": "Quyết định",
    "QD": "Quyết định",
}
ISSUE_KEYWORDS = [
    "tiền lương",
    "thai sản",
    "hợp đồng",
    "bảo hiểm",
    "xử phạt",
    "bồi thường",
    "trợ cấp",
    "khiếu nại",
    "khởi kiện",
    "chấm dứt hợp đồng",
    "nghỉ việc",
    "thời hạn",
    "thẩm quyền",
    "hồ sơ",
]
ACTOR_KEYWORDS = [
    "người lao động",
    "người sử dụng lao động",
    "công ty",
    "doanh nghiệp",
    "cơ quan nhà nước",
    "cá nhân",
    "tổ chức",
    "người nộp thuế",
    "người dân",
]
PROCEDURE_TERMS = [
    "hồ sơ",
    "thời hạn",
    "thẩm quyền",
    "trình tự",
    "thủ tục",
    "nộp",
    "giải quyết",
    "khiếu nại",
    "khởi kiện",
    "cấp",
]


def row_text(row):
    title = (row.get("title") or "").strip()
    content = (row.get("content") or "").strip()
    if title and content:
        return f"{title}\n{content}"
    return title or content


def parse_title_metadata(title):
    title = (title or "").strip()
    metadata = {}
    if not title.startswith("Điều "):
        return metadata

    body = title[len("Điều "):]
    article_code, separator, article_heading = body.partition(". ")
    if not separator:
        return metadata

    metadata["article_code"] = article_code.strip()
    metadata["article_heading"] = article_heading.strip()

    code_parts = [part.strip() for part in re.split(r"[.]+", article_code) if part.strip()]
    document_type_code = next((part.upper() for part in code_parts if part.upper() in DOCUMENT_TYPE_LABELS), "")
    if document_type_code:
        metadata["document_type_code"] = document_type_code
        metadata["document_type"] = DOCUMENT_TYPE_LABELS[document_type_code]

    numeric_parts = [part for part in code_parts if part.isdigit()]
    if numeric_parts:
        metadata["article_number"] = numeric_parts[-1]

    return metadata


def matched_terms(text, terms):
    text = (text or "").lower()
    return [term for term in terms if term.lower() in text]


def enrich_text_metadata(row):
    title = (row.get("title") or "").strip()
    content = (row.get("content") or "").strip()
    text = f"{title}\n{content}"
    metadata = parse_title_metadata(title)

    issue_keywords = matched_terms(text, ISSUE_KEYWORDS)
    if issue_keywords:
        metadata["issue_keywords"] = "|".join(issue_keywords)

    actors = matched_terms(text, ACTOR_KEYWORDS)
    if actors:
        metadata["actors"] = "|".join(actors)

    procedure_terms = matched_terms(text, PROCEDURE_TERMS)
    if procedure_terms:
        metadata["procedure_terms"] = "|".join(procedure_terms)

    return metadata


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
    metadata.update(enrich_text_metadata(row))
    return metadata


def split_long_text(text, max_chars):
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split_at = max(text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind(", ", start, end))
            if split_at > start:
                end = split_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def chunk_article(text, max_chars=ARTICLE_CHUNK_MAX_CHARS, overlap_chars=ARTICLE_CHUNK_OVERLAP_CHARS):
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph_parts = split_long_text(paragraph, max_chars)
        for part in paragraph_parts:
            candidate = f"{current}\n{part}".strip() if current else part
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped_chunks = [chunks[0]]
    for index in range(1, len(chunks)):
        overlap = chunks[index - 1][-overlap_chars:].strip()
        chunk = f"{overlap}\n{chunks[index]}".strip() if overlap else chunks[index]
        overlapped_chunks.append(chunk)
    return overlapped_chunks


def iter_article_chunks(doc_id, text, metadata):
    chunks = chunk_article(text)
    total_chunks = len(chunks)
    for chunk_index, chunk_text in enumerate(chunks, start=1):
        chunk_metadata = dict(metadata)
        chunk_metadata["article_id"] = metadata.get("id", doc_id)
        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["chunk_total"] = total_chunks
        chunk_id = f"{doc_id}:chunk:{chunk_index}"
        yield chunk_id, chunk_text, chunk_metadata


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
                metadata = row_metadata(row, csv_path.name, row_number)
                yield from iter_article_chunks(doc_id, text, metadata)


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
    print(f"Hugging Face cache path: {HF_CACHE_DIR}")
    print(f"Chroma batch size: {CHROMA_BATCH_SIZE}")
    print(f"Encode batch size: {ENCODE_BATCH_SIZE}")
    print(f"Article chunk max chars: {ARTICLE_CHUNK_MAX_CHARS}")
    print(f"Article chunk overlap chars: {ARTICLE_CHUNK_OVERLAP_CHARS}")

    if DB_PERSIST_PATH.exists():
        print(f"Resetting Chroma DB at {DB_PERSIST_PATH}")
        shutil.rmtree(DB_PERSIST_PATH)

    DB_PERSIST_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PERSIST_PATH))
    embedding_model = SentenceTransformer(
        ST_MODEL_PATH,
        device=device,
        cache_folder=str(HF_CACHE_DIR / "sentence-transformers"),
    )
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
            print(f"Inserted {total} chunks")
            ids.clear()
            documents.clear()
            metadatas.clear()

    total += flush_batch(collection, embedding_model, ids, documents, metadatas)
    print(f"Done. Inserted {total} chunks into {DB_PERSIST_PATH} collection {CHROMA_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
