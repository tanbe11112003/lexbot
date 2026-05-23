import json
import re
import unicodedata
from contextvars import ContextVar
from typing import Any

import numpy as np
import requests
from neo4j import GraphDatabase

from graph_rag.config import (
    EMBEDDING_MODEL_NAME,
    LLM_TIMEOUT,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_QA_MODEL,
    NEO4J_PASS,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_VECTOR_INDEX_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_QA_MODEL,
)
from graph_rag.text_utils import extract_json_object


INTERNAL_LEGAL_DOMAINS = {
    "law_hinh_su": "Hình sự",
    "law_dan_su": "Dân sự",
    "law_hon_nhan_gia_dinh": "Hôn nhân và gia đình",
    "law_an_ninh_mang": "An ninh mạng",
    "law_giao_duc": "Giáo dục",
}

SUPPORTED_CHAT_MODELS = {"gpt-5.5", "gemini-2.5-flash"}
CHAT_MODEL_OVERRIDE: ContextVar[str | None] = ContextVar("chat_model_override", default=None)


def normalize_search_text(text: object) -> str:
    text = str(text or "").lower().replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.split())


def clean_cypher(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:cypher)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


class GraphLawPipeline:
    """Neo4j Chunk Graph RAG runtime.

    Runtime flow mirrors the current build index:
    question -> query analysis -> Chunk vector/keyword retrieval in Neo4j
    -> graph context expansion -> answer composition.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        self.embedding_model = None
        self.search_text_laws = set()

    def _run_cypher(self, query: str, params: dict[str, Any] | None = None):
        with self.driver.session() as session:
            return session.run(query, params or {}).data()

    def _chat_completion(
        self,
        base_url: str,
        api_key: str,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> str:
        if not api_key:
            raise RuntimeError(f"{model} API key is missing")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": 0.9,
                "max_tokens": max_tokens,
            },
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _is_quota_exhausted(self, error: Exception) -> bool:
        response = getattr(error, "response", None)
        if response is None:
            return False

        status_code = getattr(response, "status_code", None)
        if status_code not in {429, 403}:
            return False

        payload: dict[str, Any] = {}
        try:
            data = response.json()
            if isinstance(data, dict):
                payload = data
        except Exception:
            payload = {}

        error_payload = payload.get("error") if isinstance(payload, dict) else {}
        if not isinstance(error_payload, dict):
            error_payload = {}

        code = str(error_payload.get("code") or "").lower()
        message = " ".join(
            [
                str(error_payload.get("message") or ""),
                str(getattr(response, "text", "") or ""),
            ]
        ).lower()
        if code in {"insufficient_quota", "quota_exceeded"}:
            return True
        return "insufficient_quota" in message or "quota" in message

    def _chat(self, prompt: str, system: str | None = None, temperature: float = 0.1, max_tokens: int = 1200) -> str:
        selected_model = CHAT_MODEL_OVERRIDE.get()
        if selected_model == "gemini-2.5-flash":
            return self._chat_completion(
                GEMINI_BASE_URL,
                GEMINI_API_KEY,
                "gemini-2.5-flash",
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        openai_model = selected_model or OPENAI_QA_MODEL
        try:
            return self._chat_completion(
                OPENAI_BASE_URL,
                OPENAI_API_KEY,
                openai_model,
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except requests.HTTPError as error:
            if not self._is_quota_exhausted(error):
                raise
            if not GEMINI_API_KEY:
                raise
            return self._chat_completion(
                GEMINI_BASE_URL,
                GEMINI_API_KEY,
                GEMINI_QA_MODEL,
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def _embedding_model(self):
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self.embedding_model

    def available_laws(self):
        rows = self._run_cypher("MATCH (l:Law) RETURN l.id AS id, l.name AS name, l.field AS field ORDER BY l.id")
        return {row["id"]: {"name": row.get("name"), "field": row.get("field")} for row in rows}

    def analyze_query(self, question: str) -> dict[str, Any]:
        laws = self.available_laws()
        prompt = f"""Bạn là bộ phân tích câu hỏi cho hệ thống Graph RAG pháp luật Việt Nam.

Nhiệm vụ: phân tích câu hỏi người dùng và trả JSON phục vụ truy xuất Neo4j.

Phạm vi dữ liệu nội bộ:
{json.dumps(laws or INTERNAL_LEGAL_DOMAINS, ensure_ascii=False, indent=2)}

Category:
- "GRAPH_LOOKUP": hỏi cấu trúc/đếm/liệt kê/tra cứu Điều-Khoản-Điểm cụ thể. Ví dụ: "Điều 123 quy định gì", "Luật hình sự có bao nhiêu chương".
- "CONSULTATION": hỏi tình huống pháp lý, cần tìm quy định liên quan rồi tư vấn.
- "NON_LEGAL": không liên quan pháp luật.

Yêu cầu JSON:
{{
  "is_relevant": true/false,
  "category": "GRAPH_LOOKUP|CONSULTATION|NON_LEGAL",
  "intent": "mô tả ngắn",
  "law_id": "một law_id nội bộ nếu xác định được, hoặc null",
  "law_ids": ["các law_id nội bộ có thể liên quan, gồm law_id chính nếu có"],
  "search_queries": ["4-8 cụm tìm kiếm pháp lý ngắn, có mở rộng từ viết tắt"],
  "extracted_facts": {{
    "legal_article": "Điều X nếu có",
    "clause": "Khoản Y nếu có",
    "law_name": "tên luật nếu có",
    "legal_events": ["sự kiện/hành vi chính"]
  }}
}}

Chỉ trả JSON hợp lệ, không markdown.

Câu hỏi: {question}"""
        fallback = {
            "is_relevant": True,
            "category": "CONSULTATION",
            "intent": question,
            "law_id": None,
            "search_queries": [question],
            "extracted_facts": {},
        }
        try:
            data = extract_json_object(self._chat(prompt, temperature=0, max_tokens=700))
        except Exception:
            data = fallback

        if data.get("category") not in {"GRAPH_LOOKUP", "CONSULTATION", "NON_LEGAL"}:
            data["category"] = "CONSULTATION"
        if not isinstance(data.get("search_queries"), list) or not data["search_queries"]:
            data["search_queries"] = [question]
        if data.get("law_id") not in laws:
            data["law_id"] = self.detect_law(question)
        if isinstance(data.get("law_ids"), list):
            data["law_ids"] = [law_id for law_id in data["law_ids"] if law_id in laws]
        else:
            data["law_ids"] = []
        data.setdefault("is_relevant", data["category"] != "NON_LEGAL")
        data.setdefault("extracted_facts", {})
        data.setdefault("intent", question)
        return data

    def detect_law(self, question: str):
        candidates = self.available_laws()
        if not candidates:
            return None
        prompt = f"""Chọn đúng một law_id phù hợp nhất cho câu hỏi.
Nếu không chắc, chọn law_id gần nhất theo lĩnh vực.

Các luật:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Câu hỏi: {question}

Chỉ trả JSON: {{"law_id":"..."}}"""
        try:
            data = extract_json_object(self._chat(prompt, temperature=0, max_tokens=200))
            if data.get("law_id") in candidates:
                return data["law_id"]
        except Exception:
            pass
        return next(iter(candidates))

    def ensure_chunk_index_for_law(self, law_id: str):
        rows = self._run_cypher(
            "MATCH (chunk:Chunk {law_id:$law_id}) RETURN count(chunk) AS count",
            {"law_id": law_id},
        )
        chunk_count = rows[0]["count"] if rows else 0
        if chunk_count == 0:
            raise FileNotFoundError(f"Missing Neo4j vector chunks for {law_id}. Run: python -m graph_rag.build_index")

    def has_chunk_search_text(self, law_id: str) -> bool:
        if law_id in self.search_text_laws:
            return True
        rows = self._run_cypher(
            """
            MATCH (chunk:Chunk {law_id:$law_id})
            WHERE 'search_text' IN keys(chunk)
            RETURN count(chunk) AS count
            LIMIT 1
            """,
            {"law_id": law_id},
        )
        if rows and rows[0]["count"] > 0:
            self.search_text_laws.add(law_id)
            return True
        return False

    def _keyword_terms(self, question: str, analysis: dict[str, Any]) -> list[str]:
        stopwords = {
            "minh",
            "ong",
            "anh",
            "chi",
            "phat",
            "hien",
            "nen",
            "thi",
            "ta",
            "se",
            "chiu",
            "nhung",
            "theo",
            "bo",
            "luat",
            "hinh",
            "su",
            "xac",
            "dinh",
            "truong",
            "hop",
            "quy",
        }
        query_parts = [question, analysis.get("intent") or ""]
        query_parts.extend(str(item) for item in analysis.get("search_queries", []))
        facts = analysis.get("extracted_facts") or {}
        query_parts.extend(str(item) for item in facts.get("legal_events", []) if item)
        source = normalize_search_text(" ".join(query_parts))
        source = re.sub(r"[^\w\s]", " ", source)
        raw_tokens = [token for token in re.split(r"\s+", source) if token]
        tokens = [token for token in raw_tokens if token not in stopwords]

        terms = set()
        for query in analysis.get("search_queries", []):
            normalized = normalize_search_text(query)
            if len(normalized) >= 5:
                terms.add(normalized)
        for size in (4, 3, 2):
            for index in range(0, max(0, len(tokens) - size + 1)):
                term = " ".join(tokens[index : index + size]).strip()
                if len(term) >= 5:
                    terms.add(term)
        generic_single_tokens = {"nguoi", "phat", "hinh", "su", "luat", "bo", "theo", "nhung", "chiu"}
        for token in tokens:
            if token in generic_single_tokens:
                continue
            if len(token) >= 4 or token in {"ma", "tuy"}:
                terms.add(token)
        return sorted(terms, key=lambda item: (-len(item.split()), -len(item), item))[:40]

    def _article_numbers_from_analysis(self, question: str, analysis: dict[str, Any]) -> list[str]:
        facts = analysis.get("extracted_facts") or {}
        parts = [
            question,
            analysis.get("intent") or "",
            facts.get("legal_article") or "",
            *[str(item) for item in analysis.get("search_queries", [])],
            *[str(item) for item in facts.get("legal_events", []) if item],
        ]
        normalized = normalize_search_text(" ".join(parts))
        numbers = set()
        for pattern in (r"\bdieu\s+(\d{1,4})\b", r"\?i\?u\s+(\d{1,4})\b", r"\blq\.(\d{1,4})\b"):
            numbers.update(re.findall(pattern, normalized))
        return sorted(numbers, key=lambda value: int(value))

    def _combined_query_text(self, question: str, analysis: dict[str, Any]) -> str:
        return normalize_search_text(
            " ".join(
                [
                    question,
                    analysis.get("intent") or "",
                    *[str(item) for item in analysis.get("search_queries", [])],
                    *[str(item) for item in (analysis.get("extracted_facts") or {}).get("legal_events", []) if item],
                ]
            )
        )

    def _article_numbers_for_law(self, question: str, analysis: dict[str, Any], law_id: str) -> list[str]:
        numbers = set(self._article_numbers_from_analysis(question, analysis))
        text = self._combined_query_text(question, analysis)

        has_divorce = "ly hon" in text
        has_asset = any(term in text for term in ["tai san", "chia tai san", "tai san chung", "tai san rieng", "vo chong"])
        has_child = any(term in text for term in ["nuoi con", "truc tiep nuoi con", "quyen nuoi con", "cap duong", "con sau"])

        if law_id == "law_hon_nhan_gia_dinh":
            if has_divorce and has_asset:
                numbers.update({"33", "43", "59", "60", "61", "62"})
            if has_divorce and has_child:
                numbers.update({"81", "82", "83", "84", "110"})
        elif law_id == "law_dan_su" and has_asset:
            numbers.update({"213", "219"})

        return sorted(numbers, key=lambda value: int(re.sub(r"\D", "", value) or 0))

    def related_law_ids(self, question: str, analysis: dict[str, Any], primary_law_id: str | None) -> list[str]:
        available_laws = self.available_laws()
        law_ids = []

        def add_law(law_id: str | None):
            if law_id in available_laws and law_id not in law_ids:
                law_ids.append(law_id)

        add_law(primary_law_id)
        for law_id in analysis.get("law_ids", []):
            add_law(law_id)

        text = self._combined_query_text(question, analysis)
        has_family_issue = any(term in text for term in ["ly hon", "nuoi con", "vo chong", "cap duong"])
        has_asset_issue = any(term in text for term in ["tai san", "chia tai san", "tai san chung", "so huu chung"])
        if has_family_issue:
            add_law("law_hon_nhan_gia_dinh")
        if has_asset_issue and has_family_issue:
            add_law("law_dan_su")

        return law_ids

    def article_number_search_chunks(self, question: str, analysis: dict[str, Any], law_id: str):
        article_numbers = self._article_numbers_for_law(question, analysis, law_id)
        if not article_numbers:
            return []
        explicit_article_numbers = self._article_numbers_from_analysis(question, analysis)
        return self._run_cypher(
            """
            MATCH (chunk:Chunk {law_id:$law_id})
            WHERE any(article IN $article_numbers
                      WHERE chunk.article_source = 'LQ'
                        AND chunk.article_number = article)
               OR any(article IN $explicit_article_numbers WHERE chunk.article_number = article)
            RETURN chunk.id AS chunk_id,
                   1 AS hit_count,
                   $article_numbers AS hits,
                   chunk.article_number AS article_number,
                   chunk.article_code AS article_code,
                   chunk.article_title AS article_title
            ORDER BY chunk.article_title, chunk.clause_number, chunk.point_letter, chunk.id
            LIMIT 80
            """,
            {
                "law_id": law_id,
                "article_numbers": article_numbers,
                "explicit_article_numbers": explicit_article_numbers,
            },
        )

    def keyword_search_chunks(self, question: str, analysis: dict[str, Any], law_id: str, top_k: int = 20):
        terms = self._keyword_terms(question, analysis)
        if not terms:
            return []
        candidate_limit = max(top_k * 200, 2000)
        if self.has_chunk_search_text(law_id):
            rows = self._run_cypher(
                """
                MATCH (chunk:Chunk {law_id:$law_id})
                WITH chunk,
                     [term IN $terms WHERE chunk.search_text CONTAINS term] AS hits
                WHERE size(hits) > 0
                RETURN chunk.id AS chunk_id, size(hits) AS hit_count, hits,
                       chunk.article_number AS article_number,
                       chunk.article_code AS article_code,
                       chunk.article_title AS article_title
                ORDER BY hit_count DESC, chunk.article_number, chunk.clause_number, chunk.point_letter
                LIMIT $candidate_limit
                """,
                {"law_id": law_id, "terms": terms, "candidate_limit": candidate_limit},
            )
            if rows:
                return rows

        # Backward-compatible fallback for Chunk nodes created before search_text
        # existed. Re-run graph_rag.build_index to make this faster in Neo4j.
        candidates = self._run_cypher(
            """
            MATCH (chunk:Chunk {law_id:$law_id})
            RETURN chunk.id AS chunk_id,
                   chunk.text AS text,
                   chunk.article_number AS article_number,
                   chunk.article_code AS article_code,
                   chunk.article_title AS article_title
            """,
            {"law_id": law_id},
        )
        scored = []
        for row in candidates:
            title = normalize_search_text(row.get("article_title") or "")
            searchable = normalize_search_text(" ".join([row.get("article_title") or "", row.get("text") or ""]))
            hits = [term for term in terms if term in searchable]
            if hits:
                title_bonus = sum(10 for term in hits if term in title)
                scored.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "hit_count": len(hits),
                        "keyword_score": len(hits) + title_bonus,
                        "hits": hits,
                        "article_number": row.get("article_number"),
                        "article_code": row.get("article_code"),
                        "article_title": row.get("article_title"),
                    }
                )
        scored.sort(key=lambda item: (-item["keyword_score"], str(item.get("article_number") or ""), str(item.get("chunk_id") or "")))
        return scored[:candidate_limit]

    def vector_search_chunks(self, queries: list[str], law_id: str, top_k: int = 20, threshold: float = 0.35):
        model = self._embedding_model()
        rows_by_query = []
        for query in queries:
            emb = model.encode([query], convert_to_numpy=True).astype("float32")
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            rows = self._run_cypher(
                """
                CALL db.index.vector.queryNodes($index_name, $search_k, $embedding)
                YIELD node, score
                WHERE node:Chunk AND node.law_id = $law_id AND score >= $threshold
                RETURN node.id AS chunk_id, score,
                       node.article_number AS article_number,
                       node.article_code AS article_code,
                       node.article_title AS article_title
                ORDER BY score DESC
                LIMIT $top_k
                """,
                {
                    "index_name": NEO4J_VECTOR_INDEX_NAME,
                    "search_k": max(top_k * 30, 200),
                    "embedding": emb[0].tolist(),
                    "law_id": law_id,
                    "threshold": threshold,
                    "top_k": top_k,
                },
            )
            rows_by_query.extend(rows)
        return rows_by_query

    def retrieve_chunks(self, question: str, analysis: dict[str, Any], law_id: str, top_k: int = 16):
        scores: dict[str, float] = {}
        metadata: dict[str, dict[str, Any]] = {}
        terms = self._keyword_terms(question, analysis)
        exact_title_terms = {
            normalize_search_text(item)
            for item in [
                *analysis.get("search_queries", []),
                *((analysis.get("extracted_facts") or {}).get("legal_events") or []),
            ]
            if len(normalize_search_text(item)) >= 5
        }

        for row in self.article_number_search_chunks(question, analysis, law_id):
            chunk_id = row["chunk_id"]
            title = normalize_search_text(row.get("article_title") or "")
            code = normalize_search_text(row.get("article_code") or "")
            article_bonus = sum(
                1000
                for number in self._article_numbers_for_law(question, analysis, law_id)
                if row.get("article_number") == number or f"lq.{number}" in code or f"lq.{number}" in title
            )
            scores[chunk_id] = max(scores.get(chunk_id, 0.0), float(row["hit_count"]) + article_bonus)
            metadata[chunk_id] = row

        for row in self.keyword_search_chunks(question, analysis, law_id, top_k=top_k):
            chunk_id = row["chunk_id"]
            title = normalize_search_text(row.get("article_title") or "")
            title_bonus = sum(100 for term in exact_title_terms if term in title)
            title_bonus += sum(10 for term in terms if term in title)
            scores[chunk_id] = max(scores.get(chunk_id, 0.0), float(row["hit_count"]) + 1.0 + title_bonus)
            metadata[chunk_id] = row

        vector_queries = [question, *[str(item) for item in analysis.get("search_queries", []) if str(item).strip()]]
        for row in self.vector_search_chunks(vector_queries, law_id, top_k=top_k):
            chunk_id = row["chunk_id"]
            scores[chunk_id] = max(scores.get(chunk_id, 0.0), float(row["score"]))
            metadata[chunk_id] = row

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        diverse = []
        duplicates = []
        seen_articles = set()
        for chunk_id, score in ranked:
            row = {"chunk_id": chunk_id, "score": score, **metadata.get(chunk_id, {})}
            article_key = (row.get("article_number"), row.get("article_title"))
            if article_key not in seen_articles:
                seen_articles.add(article_key)
                diverse.append(row)
            else:
                duplicates.append(row)
        return (diverse + duplicates)[:top_k]

    def expand_graph_context(self, chunk_ids: list[str], law_id: str, max_articles: int = 8):
        contexts = []
        seen_articles = set()
        for chunk_id in chunk_ids:
            rows = self._run_cypher(
                """
                MATCH (chunk:Chunk {id:$chunk_id, law_id:$law_id})
                WITH chunk
                OPTIONAL MATCH (sibling:Chunk {law_id:$law_id})
                WHERE sibling.article_number = chunk.article_number
                  AND sibling.article_title = chunk.article_title
                WITH chunk, sibling
                ORDER BY sibling.clause_number, sibling.point_letter, sibling.id
                WITH chunk, collect(DISTINCT sibling.text)[0..18] AS article_texts
                RETURN chunk.id AS chunk_id,
                       chunk.law_id AS law_id,
                       chunk.law_name AS law_name,
                       chunk.chapter_number AS chapter_number,
                       chunk.chapter_title AS chapter_title,
                       chunk.article_number AS article_number,
                       chunk.article_code AS article_code,
                       coalesce(chunk.article_title, chunk.article_heading) AS article_title,
                       chunk.clause_number AS clause_number,
                       chunk.point_letter AS point_letter,
                       reduce(text = '', item IN article_texts |
                              text + CASE WHEN text = '' THEN '' ELSE '\n\n' END + item) AS context_text
                """,
                {"chunk_id": chunk_id, "law_id": law_id},
            )
            for row in rows:
                article_key = (row.get("law_id"), row.get("article_number"), row.get("article_title"))
                if article_key in seen_articles:
                    continue
                seen_articles.add(article_key)
                contexts.append(row)
                if len(contexts) >= max_articles:
                    return contexts
        return contexts

    def run_article_graph_lookup(self, question: str, analysis: dict[str, Any], law_id: str):
        retrieval_rows = self.article_number_search_chunks(question, analysis, law_id)
        chunk_ids = [row["chunk_id"] for row in retrieval_rows]
        contexts = self.expand_graph_context(chunk_ids, law_id, max_articles=12)
        payload = {
            "question": question,
            "analysis": analysis,
            "law_id": law_id,
            "article_numbers": self._article_numbers_from_analysis(question, analysis),
            "chunk_ids": chunk_ids,
            "contexts": contexts,
        }
        answer = self._chat(
            f"""Chỉ dùng context từ Neo4j Graph RAG trong JSON sau để trả lời câu hỏi tra cứu điều luật.
Nếu context không có nội dung điều luật được hỏi, trả lời đúng câu: "Không đủ dữ liệu từ đồ thị tri thức để kết luận."

Yêu cầu:
- Trả lời trực tiếp nội dung điều luật được hỏi.
- Nêu rõ Luật/Chương/Điều/Khoản/Điểm khi context có dữ liệu.
- Không viện dẫn nguồn ngoài context.

{json.dumps(payload, ensure_ascii=False, indent=2)}""",
            system="Bạn là trợ lý pháp luật Việt Nam. Trả lời đúng theo dữ liệu graph, không bịa nguồn.",
            temperature=0.1,
            max_tokens=1600,
        )
        return answer, payload

    def generate_direct_cypher(self, question: str, analysis: dict[str, Any], law_id: str):
        schema = """
(l:Law {id, name, field, mode, source, article_count})
  -[:HAS_CHAPTER]->(ch:Chapter {uuid, law_id, number, title, source_id})
    -[:HAS_ARTICLE]->(a:Article {uuid, law_id, number, title, heading, preamble, source_id})
      -[:HAS_CLAUSE]->(cl:Clause {uuid, number, text})
        -[:HAS_POINT]->(p:Point {uuid, letter, text})
(source:Article|Clause|Point)-[:HAS_CHUNK]->(chunk:Chunk {id, law_id, text, search_text, article_number, article_code})
"""
        prompt = f"""Bạn là chuyên gia Neo4j Cypher cho graph pháp luật Việt Nam.

Câu hỏi: {question}
law_id bắt buộc: {law_id}
Phân tích:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

Schema:
{schema}

Yêu cầu:
- Chỉ sinh một câu Cypher đọc dữ liệu, không sửa dữ liệu.
- Bắt buộc lọc Law bằng (l:Law {{id:$law_id}}) hoặc lọc law_id bằng tham số $law_id.
- Không dùng property/label ngoài schema.
- Nếu tìm tiêu đề luật/điều gần đúng, dùng CONTAINS/toLower.
- RETURN alias rõ ràng, LIMIT hợp lý.
- Chỉ trả Cypher, không markdown.
"""
        cypher = clean_cypher(self._chat(prompt, temperature=0, max_tokens=700))
        if not re.match(r"(?is)^\s*(match|with|call)\b", cypher):
            raise ValueError("Generated Cypher is not read-only")
        forbidden = r"(?is)\b(create|merge|delete|detach|set|remove|drop|alter)\b"
        if re.search(forbidden, cypher):
            raise ValueError("Generated Cypher contains write operations")
        return cypher

    def run_direct_graph_lookup(self, question: str, analysis: dict[str, Any], law_id: str):
        try:
            cypher = self.generate_direct_cypher(question, analysis, law_id)
            rows = self._run_cypher(cypher, {"law_id": law_id})
        except Exception as exc:
            cypher = ""
            rows = [{"error": str(exc)}]

        law_rows = self._run_cypher("MATCH (l:Law {id:$law_id}) RETURN l AS law", {"law_id": law_id})
        payload = {
            "question": question,
            "analysis": analysis,
            "law_id": law_id,
            "cypher": cypher,
            "results": rows,
            "law_metadata": dict(law_rows[0]["law"]) if law_rows else {},
        }
        answer = self._chat(
            f"""Chỉ dùng JSON sau để trả lời câu hỏi. Nếu dữ liệu không đủ, nói rõ dữ liệu trong graph chưa đủ.

{json.dumps(payload, ensure_ascii=False, indent=2)}""",
            system="Bạn là trợ lý pháp luật Việt Nam. Trả lời ngắn gọn, đúng theo dữ liệu graph, không bịa nguồn.",
            temperature=0.1,
            max_tokens=1200,
        )
        return answer, payload

    def compose_answer(self, question: str, analysis: dict[str, Any], contexts: list[dict[str, Any]]):
        payload = {
            "question": question,
            "intent": analysis.get("intent"),
            "search_queries": analysis.get("search_queries", []),
            "contexts": contexts,
        }
        return self._chat(
            f"""Chỉ dùng context từ Neo4j Graph RAG trong JSON sau để trả lời.
Nếu context không có quy định liên quan trực tiếp, trả lời đúng câu: "Không đủ dữ liệu từ đồ thị tri thức để kết luận."

Yêu cầu:
- Trả lời trực tiếp tình huống người dùng hỏi.
- Nêu rõ Luật/Chương/Điều/Khoản/Điểm khi context có dữ liệu.
- Nếu có nhiều tội/hành vi, tách từng hành vi và nêu khung hình phạt tương ứng.
- Không viện dẫn nguồn ngoài context.

{json.dumps(payload, ensure_ascii=False, indent=2)}""",
            system="Bạn là trợ lý pháp luật Việt Nam, không viện dẫn nguồn ngoài context.",
            temperature=0.1,
            max_tokens=1800,
        )

    def generate_direct_answer(self, question: str, context: str | None = None, model: str | None = None):
        if model and model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported chat model: {model}")
        token = CHAT_MODEL_OVERRIDE.set(model) if model else None
        try:
            prompt = question if not context else f"Context người dùng cung cấp:\n{context}\n\nCâu hỏi:\n{question}"
            return self._chat(prompt, system="Trả lời trực tiếp bằng tiếng Việt.", temperature=0.2, max_tokens=1200)
        finally:
            if token is not None:
                CHAT_MODEL_OVERRIDE.reset(token)

    def consultation_result(self, question: str, analysis: dict[str, Any], law_id: str, question_type: str = "consultation"):
        law_ids = self.related_law_ids(question, analysis, law_id)
        retrieval_rows = []
        contexts = []
        chunk_ids = []
        for current_law_id in law_ids:
            self.ensure_chunk_index_for_law(current_law_id)
            is_primary_law = current_law_id == law_id
            article_hint_count = len(self._article_numbers_for_law(question, analysis, current_law_id))
            current_top_k = 16 if is_primary_law else max(4, article_hint_count)
            if is_primary_law:
                max_articles = max(8, min(article_hint_count, 12)) if article_hint_count else 8
            else:
                max_articles = max(2, min(article_hint_count, 4))
            current_rows = self.retrieve_chunks(question, analysis, current_law_id, top_k=current_top_k)
            retrieval_rows.extend([{**row, "law_id": current_law_id} for row in current_rows])
            current_chunk_ids = [row["chunk_id"] for row in current_rows]
            chunk_ids.extend(current_chunk_ids)
            contexts.extend(self.expand_graph_context(current_chunk_ids, current_law_id, max_articles=max_articles))
        answer = self.compose_answer(question, analysis, contexts)
        citations = [
            {
                "id": item.get("chunk_id"),
                "title": item.get("article_title"),
                "content": item.get("context_text", ""),
                "demuc_id": item.get("law_id"),
                "chuong_id": item.get("chapter_number"),
            }
            for item in contexts
        ]
        return {
            "status": "success",
            "question_type": question_type,
            "category": analysis.get("category", "CONSULTATION"),
            "law_id": law_id,
            "law_ids": law_ids,
            "analysis": analysis,
            "search_queries": analysis.get("search_queries", []),
            "chunk_ids": chunk_ids,
            "concept_ids": chunk_ids,  # Backward-compatible API alias; values are Chunk IDs.
            "retrieval": retrieval_rows,
            "contexts": contexts,
            "citation": citations,
            "response": answer,
        }

    def graph_lookup_payload_is_empty(self, payload: dict[str, Any]) -> bool:
        results = payload.get("results")
        if not results:
            return True
        return len(results) == 1 and isinstance(results[0], dict) and bool(results[0].get("error"))

    def should_use_direct_graph_lookup(self, question: str, analysis: dict[str, Any]) -> bool:
        text = self._combined_query_text(question, analysis)
        structural_terms = [
            "bao nhieu",
            "dem",
            "so luong",
            "liet ke",
            "danh sach",
            "co nhung chuong",
            "gom nhung chuong",
            "chuong nao",
            "dieu nao",
        ]
        content_terms = [
            "quy dinh gi",
            "noi dung",
            "cho toi biet",
            "cho biet them",
            "tu van",
            "che do",
            "uu dai",
            "ho tro",
            "muc huong",
            "dieu kien",
            "thu tuc",
        ]
        return any(term in text for term in structural_terms) and not any(term in text for term in content_terms)

    def ask(self, question: str, model: str | None = None):
        if model and model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported chat model: {model}")
        token = CHAT_MODEL_OVERRIDE.set(model) if model else None
        try:
            return self._ask(question)
        finally:
            if token is not None:
                CHAT_MODEL_OVERRIDE.reset(token)

    def _ask(self, question: str):
        analysis = self.analyze_query(question)
        category = analysis.get("category", "CONSULTATION")

        if category == "NON_LEGAL" or not analysis.get("is_relevant", True):
            answer = self.generate_direct_answer(question)
            return {
                "status": "success",
                "question_type": "non-legal",
                "category": category,
                "law_id": None,
                "law_ids": [],
                "analysis": analysis,
                "search_queries": analysis.get("search_queries", []),
                "chunk_ids": [],
                "concept_ids": [],
                "contexts": [],
                "citation": [],
                "response": answer,
            }

        law_id = analysis.get("law_id") or self.detect_law(question)
        if category == "GRAPH_LOOKUP":
            if self._article_numbers_from_analysis(question, analysis):
                answer, payload = self.run_article_graph_lookup(question, analysis, law_id)
                chunk_ids = payload.get("chunk_ids", [])
                contexts = payload.get("contexts", [])
                citations = [
                    {
                        "id": item.get("chunk_id"),
                        "title": item.get("article_title"),
                        "content": item.get("context_text", ""),
                        "demuc_id": item.get("law_id"),
                        "chuong_id": item.get("chapter_number"),
                    }
                    for item in contexts
                ]
                return {
                    "status": "success",
                    "question_type": "graph_lookup",
                    "category": category,
                    "law_id": law_id,
                    "law_ids": [law_id] if law_id else [],
                    "analysis": analysis,
                    "search_queries": analysis.get("search_queries", []),
                    "chunk_ids": chunk_ids,
                    "concept_ids": chunk_ids,
                    "contexts": contexts,
                    "citation": citations,
                    "response": answer,
                }
            if not self.should_use_direct_graph_lookup(question, analysis):
                return self.consultation_result(question, analysis, law_id, question_type="graph_lookup")
            answer, payload = self.run_direct_graph_lookup(question, analysis, law_id)
            if self.graph_lookup_payload_is_empty(payload):
                return self.consultation_result(question, analysis, law_id, question_type="graph_lookup")
            return {
                "status": "success",
                "question_type": "graph_lookup",
                "category": category,
                "law_id": law_id,
                "law_ids": [law_id] if law_id else [],
                "analysis": analysis,
                "search_queries": analysis.get("search_queries", []),
                "chunk_ids": [],
                "concept_ids": [],
                "contexts": [payload],
                "citation": [],
                "response": answer,
            }

        return self.consultation_result(question, analysis, law_id)
