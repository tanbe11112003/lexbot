import argparse
import json

from dotenv import load_dotenv
from neo4j import GraphDatabase

from graph_rag.config import NEO4J_PASS, NEO4J_URI, NEO4J_USER, STRUCTURE_DIR


def run(tx, query, **params):
    tx.run(query, **params)


def create_constraints(tx):
    tx.run("CREATE CONSTRAINT law_id IF NOT EXISTS FOR (l:Law) REQUIRE l.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT chapter_uuid IF NOT EXISTS FOR (c:Chapter) REQUIRE c.uuid IS UNIQUE")
    tx.run("CREATE CONSTRAINT article_uuid IF NOT EXISTS FOR (a:Article) REQUIRE a.uuid IS UNIQUE")
    tx.run("CREATE CONSTRAINT clause_uuid IF NOT EXISTS FOR (c:Clause) REQUIRE c.uuid IS UNIQUE")
    tx.run("CREATE CONSTRAINT point_uuid IF NOT EXISTS FOR (p:Point) REQUIRE p.uuid IS UNIQUE")


def reset_graph(tx):
    tx.run("MATCH (n) DETACH DELETE n")


def import_law(session, law_id, structure):
    info = dict(structure.get("law_info", {}))
    info["id"] = law_id
    session.execute_write(
        run,
        """
        MERGE (l:Law {id:$law_id})
        SET l += $info
        """,
        law_id=law_id,
        info=info,
    )

    for chapter in structure.get("chapters", []):
        chapter_uuid = f"{law_id}::{chapter.get('chapter_id') or chapter.get('chapter_number')}"
        session.execute_write(
            run,
            """
            MATCH (l:Law {id:$law_id})
            MERGE (c:Chapter {uuid:$uuid})
            SET c.law_id=$law_id, c.number=$number, c.title=$title, c.source_id=$source_id
            MERGE (l)-[:HAS_CHAPTER]->(c)
            """,
            law_id=law_id,
            uuid=chapter_uuid,
            number=chapter.get("chapter_number"),
            title=chapter.get("chapter_title"),
            source_id=chapter.get("chapter_id"),
        )

        for article in chapter.get("articles", []):
            article_uuid = f"{law_id}::{article.get('article_id') or article.get('article_number')}"
            session.execute_write(
                run,
                """
                MATCH (c:Chapter {uuid:$chapter_uuid})
                MERGE (a:Article {uuid:$uuid})
                SET a.law_id=$law_id, a.number=$number, a.code=$code, a.title=$title, a.heading=$heading,
                    a.preamble=$preamble, a.source_id=$source_id
                MERGE (c)-[:HAS_ARTICLE]->(a)
                """,
                chapter_uuid=chapter_uuid,
                uuid=article_uuid,
                law_id=law_id,
                number=str(article.get("article_number")),
                code=article.get("article_code") or str(article.get("article_number")),
                title=article.get("article_title"),
                heading=article.get("article_heading"),
                preamble=article.get("preamble"),
                source_id=article.get("article_id"),
            )

            for clause in article.get("clauses", []):
                clause_uuid = f"{article_uuid}::clause::{clause.get('clause_number')}"
                session.execute_write(
                    run,
                    """
                    MATCH (a:Article {uuid:$article_uuid})
                    MERGE (cl:Clause {uuid:$uuid})
                    SET cl.number=$number, cl.text=$text
                    MERGE (a)-[:HAS_CLAUSE]->(cl)
                    """,
                    article_uuid=article_uuid,
                    uuid=clause_uuid,
                    number=clause.get("clause_number"),
                    text=clause.get("text"),
                )
                for point in clause.get("points", []):
                    point_uuid = f"{clause_uuid}::point::{point.get('point_letter')}"
                    session.execute_write(
                        run,
                        """
                        MATCH (cl:Clause {uuid:$clause_uuid})
                        MERGE (p:Point {uuid:$uuid})
                        SET p.letter=$letter, p.text=$text
                        MERGE (cl)-[:HAS_POINT]->(p)
                        """,
                        clause_uuid=clause_uuid,
                        uuid=point_uuid,
                        letter=point.get("point_letter"),
                        text=point.get("text"),
                    )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as session:
        if args.reset:
            session.execute_write(reset_graph)
        session.execute_write(create_constraints)

        for structure_path in sorted(STRUCTURE_DIR.glob("*_structure.json")):
            law_id = structure_path.name.replace("_structure.json", "")
            structure = json.loads(structure_path.read_text(encoding="utf-8"))
            import_law(session, law_id, structure)
            print(f"imported {law_id}")
    driver.close()


if __name__ == "__main__":
    main()
