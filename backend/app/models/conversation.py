from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.facts import ExtractedFacts
from app.models.legal_output import CandidateArticle, LegalReasoningItem


class CaseStatus(str, Enum):
    collecting_facts = "collecting_facts"
    ready_to_answer = "ready_to_answer"
    answered = "answered"
    insufficient_information = "insufficient_information"


class MissingFactItem(BaseModel):
    key: str
    label: str
    description: str
    critical: bool = False
    domain: str | None = None
    question: str | None = None


class ConversationTurn(BaseModel):
    user_message: str
    extracted_facts: ExtractedFacts
    bot_response_summary: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CollectedFacts(BaseModel):
    facts: ExtractedFacts = Field(default_factory=ExtractedFacts)


class DialogueState(BaseModel):
    status: CaseStatus = CaseStatus.collecting_facts
    collected_facts: CollectedFacts = Field(default_factory=CollectedFacts)
    missing_facts: list[MissingFactItem] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CaseSession(BaseModel):
    case_id: str
    status: CaseStatus = CaseStatus.collecting_facts
    facts: ExtractedFacts = Field(default_factory=ExtractedFacts)
    scenario_text: str = ""
    turns: list[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class LegalChatResponse(BaseModel):
    case_id: str
    status: CaseStatus
    facts: ExtractedFacts
    missing_facts: list[MissingFactItem] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    candidate_articles: list[CandidateArticle] = Field(default_factory=list)
    legal_reasoning: list[LegalReasoningItem] = Field(default_factory=list)
    final_answer: str
    confidence: float = 0.0
    citations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None
