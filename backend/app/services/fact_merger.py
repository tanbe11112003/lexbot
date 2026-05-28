from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from app.models.facts import Actor, ExhibitFact, ExtractedFacts, Quantity, SubstanceFact
from app.utils.text import normalize_text

T = TypeVar("T", bound=BaseModel)


def _key(value: object) -> str:
    return normalize_text(str(value or "")).strip()


def _is_specific(text: str | None) -> bool:
    if not text:
        return False
    norm = normalize_text(text)
    return not any(term in norm for term in ["chua ro", "khong ro", "khong biet", "khong xac dinh", "co"])


def _merge_strings(old: list[str], new: list[str]) -> list[str]:
    merged = list(old)
    seen = {_key(item) for item in merged}
    for item in new:
        key = _key(item)
        if key and key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


def _merge_quantities(old: list[Quantity], new: list[Quantity]) -> list[Quantity]:
    merged = list(old)
    seen = {(_key(q.raw_text), _key(q.unit), q.value) for q in merged}
    for quantity in new:
        key = (_key(quantity.raw_text), _key(quantity.unit), quantity.value)
        if key not in seen:
            merged.append(quantity)
            seen.add(key)
    return merged


def _merge_actors(old: list[Actor], new: list[Actor]) -> list[Actor]:
    merged = [actor.model_copy(deep=True) for actor in old]
    by_name = {_key(actor.name): actor for actor in merged}
    for actor in new:
        key = _key(actor.name)
        if not key:
            continue
        current = by_name.get(key)
        if not current:
            copied = actor.model_copy(deep=True)
            merged.append(copied)
            by_name[key] = copied
            continue
        if actor.age is not None:
            current.age = actor.age
        if _is_specific(actor.role) and not _is_specific(current.role):
            current.role = actor.role
        if _is_specific(actor.notes) and not _is_specific(current.notes):
            current.notes = actor.notes
    return merged


def _merge_substances(old: list[SubstanceFact], new: list[SubstanceFact]) -> list[SubstanceFact]:
    merged = [substance.model_copy(deep=True) for substance in old]
    by_name = {_key(substance.name): substance for substance in merged}
    for substance in new:
        key = _key(substance.name)
        if not key:
            continue
        current = by_name.get(key)
        if not current:
            copied = substance.model_copy(deep=True)
            merged.append(copied)
            by_name[key] = copied
            continue
        if not current.quantity and substance.quantity:
            current.quantity = substance.quantity
        if not current.alias and substance.alias:
            current.alias = substance.alias
        current.confidence = max(current.confidence, substance.confidence)
    return merged


def _merge_exhibits(old: list[ExhibitFact], new: list[ExhibitFact]) -> list[ExhibitFact]:
    merged = [exhibit.model_copy(deep=True) for exhibit in old]
    seen = {(_key(exhibit.status), _key(exhibit.description)) for exhibit in merged}
    for exhibit in new:
        key = (_key(exhibit.status), _key(exhibit.description))
        if key in seen:
            continue
        merged.append(exhibit.model_copy(deep=True))
        seen.add(key)
    return merged


def _infer_exhibit_from_forensic(facts: ExtractedFacts) -> None:
    has_forensic = any("giám định" in item or "dương tính" in item for item in facts.evidence)
    if has_forensic and facts.quantities and facts.substances and not facts.exhibits:
        facts.exhibits.append(ExhibitFact(
            status="seized",
            description="Có kết luận giám định và định lượng được nêu trong hồ sơ.",
            quantity=facts.quantities[0],
            source_text="kết luận giám định/định lượng",
        ))


def merge_facts(old: ExtractedFacts | None, new: ExtractedFacts) -> ExtractedFacts:
    base = (old or ExtractedFacts()).model_copy(deep=True)
    base.actors = _merge_actors(base.actors, new.actors)
    base.actions = _merge_strings(base.actions, new.actions)
    base.objects = _merge_strings(base.objects, new.objects)
    base.substances = _merge_substances(base.substances, new.substances)
    base.exhibits = _merge_exhibits(base.exhibits, new.exhibits)
    base.quantities = _merge_quantities(base.quantities, new.quantities)
    base.consequences = _merge_strings(base.consequences, new.consequences)
    base.age_info = _merge_strings(base.age_info, new.age_info)
    base.intent = _merge_strings(base.intent, new.intent)
    base.mental_state = _merge_strings(base.mental_state, new.mental_state)
    base.evidence = _merge_strings(base.evidence, new.evidence)
    base.location = _merge_strings(base.location, new.location)
    base.article_refs = _merge_strings(base.article_refs, new.article_refs)
    base.crime_hints = _merge_strings(base.crime_hints, new.crime_hints)
    base.mitigating_signals = _merge_strings(base.mitigating_signals, new.mitigating_signals)
    base.aggravating_signals = _merge_strings(base.aggravating_signals, new.aggravating_signals)
    base.unknowns = _merge_strings(base.unknowns, new.unknowns)
    _infer_exhibit_from_forensic(base)
    return base
