from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    search_type: Literal["hybrid", "fulltext", "graph", "vector"] = "hybrid"
    include_debug: bool = False


class SearchResponse(BaseModel):
    query: str
    candidates: list[dict]
    final_answer: str | None = None
    missing_facts: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    debug: dict | None = None


class AnalyzeScenarioRequest(BaseModel):
    scenario: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=30)
    include_debug: bool = False
    answer_style: Literal["auto", "balanced", "conversational", "brief", "educational", "structured"] = "auto"


class NormalizeRequest(BaseModel):
    text: str = Field(min_length=1)
