from __future__ import annotations

from app.core.config import settings
from app.models.facts import ExtractedFacts
from app.services.fulltext_retriever import fallback_contains, search_conditions, search_fulltext
from app.services.graph_retriever import search_exact_articles, search_related_from_signals
from app.services.rrf import reciprocal_rank_fusion
from app.services.vector_retriever import vector_search


def retrieve_candidates(
    query_items: list[dict],
    facts: ExtractedFacts,
    normalized: list[dict],
    top_k: int,
) -> tuple[list[dict], dict]:
    rankings: list[list[dict]] = []
    debug: dict = {"query_items": query_items, "sources": {}}
    exact = search_exact_articles(facts.article_refs, top_k)
    if exact:
        rankings.append(exact)
        debug["sources"]["exact_article"] = len(exact)
    signal = search_related_from_signals(normalized, top_k)
    if signal:
        rankings.append(signal)
        debug["sources"]["normalized_signal_graph"] = len(signal)
    for item in query_items[:12]:
        q = item["text"]
        ft = search_fulltext(q, top_k)
        cond = search_conditions(q, top_k)
        vec = vector_search(q, top_k) if settings.use_vector_search else []
        if ft:
            rankings.append(ft)
        if cond:
            rankings.append(cond)
        if vec:
            rankings.append(vec)
    if not rankings:
        rankings.append(fallback_contains(query_items[0]["text"], top_k))
    fused = reciprocal_rank_fusion(rankings, k=settings.rrf_k)[:top_k]
    debug["ranking_count"] = len(rankings)
    debug["fused_count"] = len(fused)
    return fused, debug
