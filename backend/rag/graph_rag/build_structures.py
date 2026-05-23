import csv
import json
import sys
from collections import defaultdict

from dotenv import load_dotenv

from graph_rag.config import (
    CORPUS_PATH,
    CRAWLER_DATA_DIR,
    FILTERED_CORPUS_PATH,
    LAW_SPECS,
    STRUCTURE_DIR,
    ensure_dirs,
)
from graph_rag.text_utils import parse_article_body, parse_article_title


field_size_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_size_limit)
        break
    except OverflowError:
        field_size_limit //= 10


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus():
    with CORPUS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def resolve_demuc_ids(spec, topics, demucs):
    topic = next((item for item in topics if item.get("Text") == spec["topic_text"]), None)
    allowed = set(spec["demuc_texts"])
    ids = {
        item["Value"]
        for item in demucs
        if item.get("Text") in allowed and (not topic or item.get("ChuDe") == topic.get("Value"))
    }
    return ids


def chapter_order_key(value):
    return str(value or "")


def build_structure(spec, rows, demuc_ids, demuc_names, chapter_names):
    chapters_map = defaultdict(list)
    for row in rows:
        chapters_map[row.get("chuong_id") or f"{spec['id']}-default"].append(row)

    chapters = []
    for chapter_index, (chapter_id, chapter_rows) in enumerate(
        sorted(chapters_map.items(), key=lambda item: chapter_order_key(item[0])),
        start=1,
    ):
        articles = []
        for article_index, row in enumerate(chapter_rows, start=1):
            article_number, article_code, heading = parse_article_title(row.get("title") or "")
            preamble, clauses = parse_article_body(row.get("content") or "")
            articles.append(
                {
                    "article_id": row.get("id"),
                    "article_number": article_number or str(article_index),
                    "article_code": article_code,
                    "article_title": row.get("title") or f"Điều {article_index}. {heading}",
                    "article_heading": heading,
                    "preamble": preamble,
                    "clauses": clauses,
                    "source": {
                        "demuc_id": row.get("demuc_id"),
                        "demuc_name": demuc_names.get(row.get("demuc_id")),
                        "chuong_id": row.get("chuong_id"),
                    },
                }
            )

        chapters.append(
            {
                "chapter_id": chapter_id,
                "chapter_number": chapter_index,
                "chapter_title": chapter_names.get(chapter_id) or f"Chương {chapter_index}",
                "articles": articles,
            }
        )

    return {
        "law_info": {
            "id": spec["id"],
            "name": spec["name"],
            "field": spec["field"],
            "mode": spec["mode"],
            "source": "backend/rag/corpus/pddieu.csv",
            "demuc_ids": sorted(demuc_ids),
            "demuc_names": [demuc_names.get(item, item) for item in sorted(demuc_ids)],
            "article_count": len(rows),
        },
        "chapters": chapters,
    }


def write_filtered_corpus(selected_rows):
    with FILTERED_CORPUS_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "title", "content", "demuc_id", "chuong_id", "law_id"])
        writer.writeheader()
        writer.writerows(selected_rows)


def main():
    load_dotenv()
    ensure_dirs()

    topics = load_json(CRAWLER_DATA_DIR / "chude.json")
    demucs = load_json(CRAWLER_DATA_DIR / "demuc.json")
    tree_nodes = load_json(CRAWLER_DATA_DIR / "treeNode.json")
    corpus_rows = load_corpus()

    demuc_names = {item["Value"]: item.get("Text") for item in demucs}
    chapter_names = {
        item["MAPC"]: item.get("TEN")
        for item in tree_nodes
        if str(item.get("TEN") or "").startswith(("Chương ", "Mục "))
    }

    all_selected = []
    stats = []
    for spec in LAW_SPECS:
        demuc_ids = resolve_demuc_ids(spec, topics, demucs)
        rows = [dict(row, law_id=spec["id"]) for row in corpus_rows if row.get("demuc_id") in demuc_ids]
        all_selected.extend(rows)

        structure = build_structure(spec, rows, demuc_ids, demuc_names, chapter_names)
        out_path = STRUCTURE_DIR / f"{spec['id']}_structure.json"
        out_path.write_text(json.dumps(structure, ensure_ascii=False, indent=2), encoding="utf-8")
        stats.append({"law_id": spec["id"], "demuc_ids": len(demuc_ids), "articles": len(rows), "path": str(out_path)})

    write_filtered_corpus(all_selected)
    print(json.dumps({"filtered_corpus": str(FILTERED_CORPUS_PATH), "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
