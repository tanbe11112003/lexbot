from __future__ import annotations

from pydantic import BaseModel, Field


class Actor(BaseModel):
    name: str
    role: str | None = None
    age: int | None = None
    notes: str | None = None


class Quantity(BaseModel):
    value: float | None = None
    unit: str | None = None
    raw_text: str
    object: str | None = None


class SubstanceFact(BaseModel):
    name: str
    alias: str | None = None
    quantity: Quantity | None = None
    confidence: float = 0.8


class ExhibitFact(BaseModel):
    status: str
    description: str
    quantity: Quantity | None = None
    source_text: str | None = None


class ExtractedFacts(BaseModel):
    actors: list[Actor] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    substances: list[SubstanceFact] = Field(default_factory=list)
    exhibits: list[ExhibitFact] = Field(default_factory=list)
    quantities: list[Quantity] = Field(default_factory=list)
    consequences: list[str] = Field(default_factory=list)
    age_info: list[str] = Field(default_factory=list)
    intent: list[str] = Field(default_factory=list)
    mental_state: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    location: list[str] = Field(default_factory=list)
    article_refs: list[str] = Field(default_factory=list)
    crime_hints: list[str] = Field(default_factory=list)
    mitigating_signals: list[str] = Field(default_factory=list)
    aggravating_signals: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
