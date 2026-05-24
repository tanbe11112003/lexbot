from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import NormalizeRequest, SearchRequest, SearchResponse
from app.services.answer_generator import generate_answer
from app.services.context_builder import citations_from_contexts
from app.services.graph_retriever import fetch_contexts
from app.services.fact_extractor import extract_facts
from app.services.fulltext_retriever import search_fulltext
from app.services.hybrid_retriever import retrieve_candidates
from app.services.legal_matcher import detect_missing_facts
from app.services.legal_reasoner import reason_over_contexts
from app.services.normalizer import normalize_endpoint_payload, normalize_text_with_graph
from app.services.query_rewriter import rewrite_queries
from app.services.decomposer import decompose_query
from app.services.reranker import rerank
from app.services.validator import validate_answer

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if req.search_type == "fulltext":
        candidates = search_fulltext(req.query, req.top_k)
        contexts = fetch_contexts([str(c.get("article_code")) for c in candidates if c.get("article_code")])
        facts = extract_facts(req.query)
        missing = detect_missing_facts(facts, req.query)
        reasoning = reason_over_contexts(contexts, facts, [], missing)
        answer = generate_answer(req.query, facts, contexts, reasoning, missing)
        answer, _, warnings = validate_answer(answer, contexts, missing, reasoning, 0.5)
        debug = {"mode": "fulltext", "warnings": warnings} if req.include_debug else None
        return SearchResponse(
            query=req.query,
            candidates=candidates,
            final_answer=answer,
            missing_facts=missing,
            citations=citations_from_contexts(contexts),
            debug=debug,
        )
    facts = extract_facts(req.query)
    normalized = normalize_text_with_graph(req.query)
    sub = decompose_query(req.query, facts)
    rewritten = rewrite_queries(req.query, facts, sub, normalized)
    candidates, debug = retrieve_candidates(rewritten, facts, normalized, req.top_k)
    candidates = rerank(req.query, candidates, req.top_k)
    contexts = fetch_contexts([str(c.get("article_code")) for c in candidates if c.get("article_code")])
    missing = detect_missing_facts(facts, req.query)
    reasoning = reason_over_contexts(contexts, facts, normalized, missing)
    answer = generate_answer(req.query, facts, contexts, reasoning, missing)
    answer, _, warnings = validate_answer(answer, contexts, missing, reasoning, 0.5)
    if req.include_debug:
        debug["warnings"] = warnings
        debug["facts"] = facts.model_dump()
    return SearchResponse(
        query=req.query,
        candidates=candidates,
        final_answer=answer,
        missing_facts=missing,
        citations=citations_from_contexts(contexts),
        debug=debug if req.include_debug else None,
    )


@router.post("/normalize")
def normalize(req: NormalizeRequest) -> dict:
    return normalize_endpoint_payload(req.text)
