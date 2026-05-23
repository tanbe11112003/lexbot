import argparse
import unicodedata
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

from graph_rag.config import (
    EMBEDDING_MODEL_NAME,
    NEO4J_PASS,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_VECTOR_INDEX_NAME,
    ensure_dirs,
)


BATCH_SIZE = 128


def clean_parts(parts: Iterable[object]) -> list[str]:
    return [str(part).strip() for part in parts if part is not None and str(part).strip()]


def normalize_search_text(text: object) -> str:
    text = str(text or "").lower().replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.split())


def article_source_from_code(article_code: object) -> str | None:
    parts = str(article_code or "").split(".")
    if len(parts) >= 4:
        source = parts[2].strip().upper()
        return source or None
    return "OTHER" if str(article_code or "").strip() else None


def build_chunk_text(row):
    parts = [
        f"Luật: {row.get('law_name')}",
        f"Lĩnh vực: {row.get('field')}",
        row.get("chapter_title"),
        row.get("article_title"),
    ]
    if row.get("article_heading") and row.get("article_heading") != row.get("article_title"):
        parts.append(row.get("article_heading"))
    if row.get("clause_text"):
        parts.append(f"Khoản {row.get('clause_number')}: {row.get('clause_text')}")
    if row.get("point_text"):
        parts.append(f"Điểm {row.get('point_letter')}: {row.get('point_text')}")
    if row.get("article_preamble") and not row.get("clause_text") and not row.get("point_text"):
        parts.append(row.get("article_preamble"))
    return "\n".join(clean_parts(parts))


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-12)


def create_constraints(session):
    session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")


def create_lookup_indexes(session):
    session.run("CREATE INDEX chunk_law_id IF NOT EXISTS FOR (c:Chunk) ON (c.law_id)")
    session.run("CREATE INDEX chunk_law_article_number IF NOT EXISTS FOR (c:Chunk) ON (c.law_id, c.article_number)")
    session.run(
        "CREATE INDEX chunk_law_source_article_number IF NOT EXISTS FOR (c:Chunk) ON (c.law_id, c.article_source, c.article_number)"
    )


def create_vector_index(session, dimensions):
    session.run(
        f"""
        CREATE VECTOR INDEX {NEO4J_VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {int(dimensions)},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
    )


def fetch_leaf_units(session, law_id):
    rows = []

    article_preambles = session.run(
        """
        MATCH (l:Law {id:$law_id})-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_ARTICLE]->(a:Article)
        WHERE a.preamble IS NOT NULL AND trim(a.preamble) <> ''
        RETURN l.id AS law_id, l.name AS law_name, l.field AS field,
               ch.number AS chapter_number, ch.title AS chapter_title,
               a.uuid AS source_uuid, 'Article' AS source_label, 'article_preamble' AS chunk_kind,
               a.number AS article_number, a.code AS article_code,
               a.title AS article_title, a.heading AS article_heading,
               a.preamble AS article_preamble,
               null AS clause_number, null AS clause_text,
               null AS point_letter, null AS point_text
        ORDER BY ch.number, a.number
        """,
        {"law_id": law_id},
    ).data()
    rows.extend(article_preambles)

    points = session.run(
        """
        MATCH (l:Law {id:$law_id})-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_ARTICLE]->(a:Article)
              -[:HAS_CLAUSE]->(cl:Clause)-[:HAS_POINT]->(p:Point)
        RETURN l.id AS law_id, l.name AS law_name, l.field AS field,
               ch.number AS chapter_number, ch.title AS chapter_title,
               p.uuid AS source_uuid, 'Point' AS source_label, 'point' AS chunk_kind,
               a.number AS article_number, a.code AS article_code,
               a.title AS article_title, a.heading AS article_heading,
               a.preamble AS article_preamble,
               cl.number AS clause_number, cl.text AS clause_text,
               p.letter AS point_letter, p.text AS point_text
        ORDER BY ch.number, a.number, cl.number, p.letter
        """,
        {"law_id": law_id},
    ).data()
    rows.extend(points)

    leaf_clauses = session.run(
        """
        MATCH (l:Law {id:$law_id})-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_ARTICLE]->(a:Article)
              -[:HAS_CLAUSE]->(cl:Clause)
        WHERE NOT (cl)-[:HAS_POINT]->()
        RETURN l.id AS law_id, l.name AS law_name, l.field AS field,
               ch.number AS chapter_number, ch.title AS chapter_title,
               cl.uuid AS source_uuid, 'Clause' AS source_label, 'clause' AS chunk_kind,
               a.number AS article_number, a.code AS article_code,
               a.title AS article_title, a.heading AS article_heading,
               a.preamble AS article_preamble,
               cl.number AS clause_number, cl.text AS clause_text,
               null AS point_letter, null AS point_text
        ORDER BY ch.number, a.number, cl.number
        """,
        {"law_id": law_id},
    ).data()
    rows.extend(leaf_clauses)

    leaf_articles = session.run(
        """
        MATCH (l:Law {id:$law_id})-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_ARTICLE]->(a:Article)
        WHERE NOT (a)-[:HAS_CLAUSE]->() AND (a.preamble IS NULL OR trim(a.preamble) = '')
        RETURN l.id AS law_id, l.name AS law_name, l.field AS field,
               ch.number AS chapter_number, ch.title AS chapter_title,
               a.uuid AS source_uuid, 'Article' AS source_label, 'article' AS chunk_kind,
               a.number AS article_number, a.code AS article_code,
               a.title AS article_title, a.heading AS article_heading,
               a.preamble AS article_preamble,
               null AS clause_number, null AS clause_text,
               null AS point_letter, null AS point_text
        ORDER BY ch.number, a.number
        """,
        {"law_id": law_id},
    ).data()
    rows.extend(leaf_articles)

    chunks = []
    for row in rows:
        text = build_chunk_text(row)
        if not text:
            continue
        chunk_id = f"{row['source_uuid']}::chunk::{row['chunk_kind']}"
        chunks.append(
            {
                **row,
                "id": chunk_id,
                "text": text,
            }
        )
    return chunks


def write_chunks(session, chunks):
    if not chunks:
        return
    query = """
    UNWIND $chunks AS row
    MERGE (chunk:Chunk {id: row.id})
    SET chunk.law_id = row.law_id,
        chunk.law_name = row.law_name,
        chunk.field = row.field,
        chunk.source_uuid = row.source_uuid,
        chunk.source_label = row.source_label,
        chunk.chunk_kind = row.chunk_kind,
        chunk.chapter_number = row.chapter_number,
        chunk.chapter_title = row.chapter_title,
        chunk.article_number = row.article_number,
        chunk.article_code = row.article_code,
        chunk.article_source = row.article_source,
        chunk.article_title = row.article_title,
        chunk.article_heading = row.article_heading,
        chunk.article_preamble = row.article_preamble,
        chunk.clause_number = row.clause_number,
        chunk.clause_text = row.clause_text,
        chunk.point_letter = row.point_letter,
        chunk.point_text = row.point_text,
        chunk.text = row.text,
        chunk.search_text = row.search_text,
        chunk.embedding = row.embedding
    WITH row, chunk
    MATCH (source {uuid: row.source_uuid})
    WHERE row.source_label IN labels(source)
    MERGE (source)-[:HAS_CHUNK]->(chunk)
    """
    for offset in range(0, len(chunks), BATCH_SIZE):
        session.run(query, {"chunks": chunks[offset : offset + BATCH_SIZE]})


def main():
    load_dotenv()
    ensure_dirs()

    parser = argparse.ArgumentParser()
    parser.add_argument("--law-id", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dimensions = model.get_sentence_embedding_dimension()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as session:
        create_constraints(session)
        create_lookup_indexes(session)
        create_vector_index(session, dimensions)

        if args.law_id:
            laws = [args.law_id]
        else:
            laws = [row["id"] for row in session.run("MATCH (l:Law) RETURN l.id AS id ORDER BY l.id").data()]

        for law_id in laws:
            session.run("MATCH (chunk:Chunk {law_id:$law_id}) DETACH DELETE chunk", {"law_id": law_id})
            chunks = fetch_leaf_units(session, law_id)
            if not chunks:
                print(f"{law_id}: no legal chunks found")
                continue

            texts = [chunk["text"] for chunk in chunks]
            embeddings = model.encode(texts, convert_to_numpy=True, batch_size=args.batch_size).astype("float32")
            embeddings = normalize_embeddings(embeddings).astype("float32")
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding.tolist()
                chunk["search_text"] = normalize_search_text(chunk["text"])
                chunk["article_source"] = article_source_from_code(chunk.get("article_code"))

            write_chunks(session, chunks)
            print(f"{law_id}: indexed {len(chunks)} chunks in Neo4j vector index '{NEO4J_VECTOR_INDEX_NAME}'")

    driver.close()


if __name__ == "__main__":
    main()
