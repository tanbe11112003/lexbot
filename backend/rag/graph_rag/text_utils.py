import json
import re
import unicodedata
from typing import Any


ARTICLE_TITLE_RE = re.compile(r"^Điều\s+(.+?)\.\s+(.+)$", re.IGNORECASE | re.DOTALL)
ARTICLE_NUMBER_RE = re.compile(r"(?:^|\.)(\d+[a-zA-Z]?)$")
CLAUSE_RE = re.compile(r"(?m)^\s*(\d+)\.\s+")
POINT_RE = re.compile(r"(?m)^\s*([a-zđ])\)\s+")


def normalize_space(text: str) -> str:
    text = str(text or "").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify_vi(text: str) -> str:
    value = unicodedata.normalize("NFD", text or "")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d").replace("Đ", "D").lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "item"


def extract_json_object(text: str) -> Any:
    if not text:
        raise ValueError("Empty model output")
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


def parse_article_title(title: str):
    title = normalize_space(title)
    match = ARTICLE_TITLE_RE.match(title)
    if not match:
        return title, title, title
    article_code = match.group(1).strip()
    article_heading = match.group(2).strip()
    number_match = ARTICLE_NUMBER_RE.search(article_code)
    article_number = number_match.group(1) if number_match else article_code
    return article_number, article_code, article_heading


def _split_points(text: str):
    matches = list(POINT_RE.finditer(text))
    if not matches:
        return normalize_space(text), []

    clause_text = normalize_space(text[: matches[0].start()])
    points = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        point_text = normalize_space(text[match.start() : end])
        points.append({"point_letter": match.group(1), "text": point_text})
    return clause_text, points


def parse_article_body(content: str):
    body = normalize_space(content)
    if not body:
        return None, []

    matches = list(CLAUSE_RE.finditer(body))
    if not matches:
        return body, []

    preamble = normalize_space(body[: matches[0].start()]) or None
    clauses = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        block = body[match.start() : end]
        clause_text, points = _split_points(block)
        clauses.append(
            {
                "clause_number": int(match.group(1)),
                "text": clause_text,
                "points": points,
            }
        )
    return preamble, clauses
