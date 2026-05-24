from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

from app.core.config import settings
from app.models.facts import Actor, ExtractedFacts, Quantity, SubstanceFact
from app.prompts.fact_extraction_prompt import FACT_EXTRACTION_SYSTEM, FACT_EXTRACTION_USER
from app.utils.text import dedupe_keep_order, normalize_text

logger = logging.getLogger(__name__)

ACTION_TERMS = [
    "tàng trữ", "vận chuyển", "mua bán", "mua", "cung cấp", "sản xuất", "chiếm đoạt", "sử dụng", "tổ chức sử dụng",
    "chứa chấp", "lôi kéo", "cưỡng bức", "che giấu", "không tố giác", "giúp sức", "xúi giục",
    "chủ mưu", "cầm đầu", "rủ", "chuẩn bị", "chưa đạt", "khai thác", "tự thú",
]
SUBSTANCE_ALIASES = {
    "ketamin": "ketamine", "ketamine": "ketamine", "kẹo": "MDMA", "thuốc lắc": "MDMA",
    "mdma": "MDMA", "đá": "methamphetamine", "meth": "methamphetamine", "cần sa": "cần sa",
    "heroin": "heroin", "ma túy": "ma túy",
}
CONSEQUENCE_TERMS = ["chết người", "tử vong", "thương tích", "thiệt hại", "dương tính"]
LOCATION_TERMS = ["karaoke", "quán bar", "nhà nghỉ", "phòng", "khách sạn"]
MITIGATING_TERMS = ["tự thú", "thành khẩn", "ăn năn", "đủ 70 tuổi", "người đủ 70 tuổi"]
AGGRAVATING_TERMS = ["có tổ chức", "tái phạm", "tái phạm nguy hiểm", "côn đồ", "lợi dụng chức vụ"]


def _parse_float(raw: str) -> float | None:
    try:
        return float(raw.replace(".", "").replace(",", "."))
    except ValueError:
        return None


def _extract_actors(text: str) -> list[Actor]:
    actors: list[Actor] = []
    seen: set[str] = set()
    for name in re.findall(r"\b([A-ZĐ][A-ZĐ0-9]{0,2})\b", text):
        if name not in seen and name not in {"BLHS", "MDMA"}:
            seen.add(name)
            actors.append(Actor(name=name))
    for m in re.finditer(r"([A-ZĐ][A-ZĐ0-9]{0,2})\s*(?:đủ\s*)?(\d{1,2})\s*tuổi", text):
        for actor in actors:
            if actor.name == m.group(1):
                actor.age = int(m.group(2))
    return actors


def _extract_quantities(text: str) -> list[Quantity]:
    patterns = [
        r"(\d+(?:[\.,]\d+)?)\s*(kg|g|gam|viên|gói|m3|m³|mét khối)",
        r"(\d+(?:[\.,]\d+)?)\s*(triệu|tỷ)(?:\s*đồng)?",
    ]
    quantities: list[Quantity] = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.I):
            value = _parse_float(m.group(1))
            unit = m.group(2).lower()
            if unit == "tỷ" and value is not None:
                value *= 1_000_000_000
                unit = "đồng"
            if unit == "triệu" and value is not None:
                value *= 1_000_000
                unit = "đồng"
            quantities.append(Quantity(value=value, unit=unit, raw_text=m.group(0)))
    return quantities


def _regex_extract(text: str) -> ExtractedFacts:
    norm = normalize_text(text)
    lowered = (text or "").lower()
    facts = ExtractedFacts()
    facts.actors = _extract_actors(text)
    facts.quantities = _extract_quantities(text)
    facts.actions = [term for term in ACTION_TERMS if normalize_text(term) in norm]
    facts.consequences = [term for term in CONSEQUENCE_TERMS if normalize_text(term) in norm]
    facts.location = [term for term in LOCATION_TERMS if normalize_text(term) in norm]
    facts.mitigating_signals = [term for term in MITIGATING_TERMS if normalize_text(term) in norm]
    facts.aggravating_signals = [term for term in AGGRAVATING_TERMS if normalize_text(term) in norm]
    facts.article_refs = dedupe_keep_order(re.findall(r"[Đđ]iều\s+(\d+[a-zA-Z]?)", text))
    facts.age_info = dedupe_keep_order([m.group(0) for m in re.finditer(r"(?:\d{1,2}\s*tuổi|dưới\s*\d{1,2}|từ\s*đủ\s*\d{1,2}|đủ\s*70\s*tuổi)", text, flags=re.I)])
    facts.intent = [term for term in ["mục đích", "hưởng lợi", "cho bạn", "để bán", "để sử dụng"] if normalize_text(term) in norm]
    facts.mental_state = [term for term in ["cố ý", "vô ý", "biết", "không biết"] if normalize_text(term) in norm]
    facts.evidence = [term for term in ["giám định", "kết luận giám định", "dương tính", "camera", "lời khai"] if normalize_text(term) in norm]
    for alias, name in SUBSTANCE_ALIASES.items():
        alias_norm = normalize_text(alias)
        if len(alias_norm) <= 2:
            matched = bool(re.search(rf"(?<!\w){re.escape(alias.lower())}(?!\w)", lowered))
        else:
            matched = alias_norm in norm
        if matched:
            quantity = facts.quantities[0] if facts.quantities else None
            facts.substances.append(SubstanceFact(name=name, alias=alias, quantity=quantity, confidence=0.9))
    facts.objects = [s.name for s in facts.substances]
    if any(x in norm for x in ["go", "lam san", "rung"]):
        facts.objects.append("gỗ/lâm sản")
        facts.crime_hints.append("tội vi phạm quy định về khai thác, bảo vệ rừng và lâm sản")
    if facts.substances:
        facts.crime_hints.append("nhóm tội phạm về ma túy")
    return facts


def _safe_json(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```[a-zA-Z]*", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        return json.loads(m.group(0)) if m else {}


def _llm_extract(text: str) -> ExtractedFacts | None:
    if not settings.use_llm_fact_extractor or not settings.openai_api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": FACT_EXTRACTION_SYSTEM},
                {"role": "user", "content": FACT_EXTRACTION_USER.format(scenario=text)},
            ],
        )
        return ExtractedFacts.model_validate(_safe_json(resp.choices[0].message.content or ""))
    except Exception as exc:
        logger.warning("LLM fact extraction skipped: %s", exc)
        return None


@lru_cache(maxsize=256)
def extract_facts(text: str) -> ExtractedFacts:
    base = _regex_extract(text)
    llm = _llm_extract(text)
    if not llm:
        return base
    for field in ExtractedFacts.model_fields:
        current = getattr(base, field)
        extra = getattr(llm, field)
        if isinstance(current, list):
            current.extend([x for x in extra if x not in current])
    return base
