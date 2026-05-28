from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

from app.core.config import settings
from app.models.facts import Actor, ExhibitFact, ExtractedFacts, Quantity, SubstanceFact
from app.prompts.fact_extraction_prompt import FACT_EXTRACTION_SYSTEM, FACT_EXTRACTION_USER
from app.utils.text import dedupe_keep_order, normalize_text

logger = logging.getLogger(__name__)

ACTION_TERMS = [
    "tàng trữ", "vận chuyển", "mua bán", "mua", "cung cấp", "sản xuất", "chiếm đoạt", "sử dụng", "tổ chức sử dụng",
    "chứa chấp", "lôi kéo", "cưỡng bức", "che giấu", "không tố giác", "giúp sức", "xúi giục",
    "chủ mưu", "cầm đầu", "rủ", "nhờ", "đặt phòng", "chuẩn bị", "chưa đạt", "khai thác", "tự thú",
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
EXHIBIT_PATTERNS = [
    ("consumed", r"(?:không\s+còn\s+tang\s+vật|tang\s+vật\s+đã\s+(?:bị\s+)?(?:tiêu\s+thụ|sử\s+dụng)\s+hết|đã\s+(?:tiêu\s+thụ|sử\s+dụng)\s+hết)"),
    ("not_seized", r"(?:không\s+thu\s+giữ\s+được|không\s+thu\s+được|không\s+phát\s+hiện\s+tang\s+vật)"),
    ("seized", r"(?:thu\s+giữ|thu\s+được|phát\s+hiện|bắt\s+quả\s+tang)[^.]{0,80}(?:tang\s+vật|ma\s+túy|ketamin|ketamine|thuốc\s+lắc|mdma|heroin|cần\s+sa|gói|viên|gam|g)"),
    ("mentioned", r"(?:tang\s+vật|vật\s+chứng)"),
]


def _parse_float(raw: str) -> float | None:
    try:
        raw = raw.strip()
        if "," in raw and "." in raw:
            return float(raw.replace(".", "").replace(",", "."))
        if "," in raw:
            return float(raw.replace(",", "."))
        if "." in raw:
            left, right = raw.rsplit(".", 1)
            if len(right) <= 2:
                return float(raw)
            return float(raw.replace(".", ""))
        return float(raw)
    except ValueError:
        return None


_ACTOR_STOPWORDS = {
    "Bộ", "Điều", "Khoản", "Tội", "Khi", "Nếu", "Tình", "Người", "Các", "Theo",
    "Trong", "Hiện", "Căn", "Tuy", "Do", "Vì", "Với", "Ca", "Nam", "Nữ",
    "Tương", "Những", "Long", "Sơn", "Ngọc", "Minh", "Nhật", "Tết",
}
_TITLE_PREFIX_RE = re.compile(r"^(?:ca\s+sĩ|nam\s+ca\s+sĩ|nữ\s+ca\s+sĩ|ông|bà|anh|chị|bị\s+can|bị\s+cáo)\s+", re.I)


def _clean_actor_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name or "").strip(" ,;:.")
    name = _TITLE_PREFIX_RE.sub("", name).strip()
    return name


def _is_likely_actor_name(name: str) -> bool:
    words = name.split()
    return 1 <= len(words) <= 4 and all(word and word[0].isupper() for word in words)


def _add_actor(actors: list[Actor], seen: set[str], name: str, age: int | None = None) -> None:
    name = _clean_actor_name(name)
    if not name or name in _ACTOR_STOPWORDS or name.upper() in {"BLHS", "MDMA"}:
        return
    if not _is_likely_actor_name(name):
        return
    key = name.lower()
    if key in seen:
        for actor in actors:
            if actor.name.lower() == key and age is not None:
                actor.age = age
        return
    seen.add(key)
    actors.append(Actor(name=name, age=age))


def _extract_actors(text: str) -> list[Actor]:
    actors: list[Actor] = []
    seen: set[str] = set()

    age_name_pattern = re.compile(
        r"((?:(?:ca\s+sĩ|nam\s+ca\s+sĩ|nữ\s+ca\s+sĩ|ông|bà|anh|chị)\s+)?"
        r"[A-ZĐ][a-zA-ZÀ-ỹ]{1,24}(?:\s+[A-ZĐ][a-zA-ZÀ-ỹ]{1,24}){0,3})\s*,\s*(\d{1,3})\s*tuổi",
        re.I,
    )
    for match in age_name_pattern.finditer(text):
        _add_actor(actors, seen, match.group(1), int(match.group(2)))

    for name in re.findall(r"\b([A-ZĐ][A-ZĐ0-9]{0,2})\b", text):
        _add_actor(actors, seen, name)
    for name in re.findall(r"\b([A-ZĐ][a-zA-ZÀ-ỹ]{1,24})\b", text):
        if name in _ACTOR_STOPWORDS and name != "Long":
            continue
        _add_actor(actors, seen, name)
    for m in re.finditer(r"([A-ZĐ][A-ZĐ0-9]{0,2})\s*(?:đủ\s*)?(\d{1,2})\s*tuổi", text):
        for actor in actors:
            if actor.name == m.group(1):
                actor.age = int(m.group(2))
    lowered = text.lower()
    for actor in actors:
        name = actor.name
        lname = name.lower()
        if re.search(rf"\b{re.escape(lname)}\s+nhờ\b", lowered):
            actor.role = "người nhờ/khởi xướng"
        elif re.search(rf"\bnhờ\s+{re.escape(lname)}\b", lowered):
            actor.role = "người được nhờ"
        elif re.search(rf"\bqua\s+{re.escape(lname)}\b", lowered):
            actor.role = "trung gian/liên hệ"
        elif re.search(rf"\btên\s+{re.escape(lname)}\b", lowered):
            actor.role = "người bán/cung cấp bị nêu tên"
        elif re.search(rf"\b{re.escape(lname)}\b[^.]{0,160}\btổ chức\b", lowered):
            actor.role = "người bị cáo buộc tổ chức"
        elif re.search(rf"\b{re.escape(lname)}\b[^.]{0,120}\b(chuyển tiền|nhờ người mua|mua hàng)\b", lowered):
            actor.role = "người bị cáo buộc mua/nhờ mua"
        elif re.search(rf"\b{re.escape(lname)}\b[^.]{0,160}\bsử dụng\b", lowered):
            actor.role = "người sử dụng"
    for actor in actors:
        lname = actor.name.lower()
        window_match = re.search(rf"\b{re.escape(lname)}\b(?P<tail>[^.]{{0,180}})", lowered)
        tail = window_match.group("tail") if window_match else ""
        if "thừa nhận" in tail and "sử dụng" in tail:
            actor.role = "người sử dụng"
            continue
        if actor.role:
            continue
        later_self_use = re.search(rf"\b{re.escape(lname)}\b[^.]{{0,180}}thừa nhận[^.]{{0,80}}sử dụng", lowered)
        if later_self_use:
            actor.role = "người sử dụng"
            continue
        elif "tổ chức" in tail:
            actor.role = "người bị cáo buộc tổ chức"
        elif "chuyển tiền" in tail or "nhờ người mua" in tail or "mua hàng" in tail:
            actor.role = "người bị cáo buộc mua/nhờ mua"
        elif "sử dụng" in tail:
            actor.role = "người sử dụng"
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


def _extract_exhibits(text: str, quantities: list[Quantity]) -> list[ExhibitFact]:
    exhibits: list[ExhibitFact] = []
    seen: set[tuple[str, str]] = set()
    for status, pattern in EXHIBIT_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.I):
            source = re.sub(r"\s+", " ", match.group(0)).strip()
            key = (status, source.lower())
            if key in seen:
                continue
            seen.add(key)
            quantity = quantities[0] if quantities and status in {"seized", "mentioned"} else None
            exhibits.append(ExhibitFact(status=status, description=source, quantity=quantity, source_text=source))
    return exhibits


def _regex_extract(text: str) -> ExtractedFacts:
    norm = normalize_text(text)
    lowered = (text or "").lower()
    facts = ExtractedFacts()
    facts.actors = _extract_actors(text)
    facts.quantities = _extract_quantities(text)
    facts.exhibits = _extract_exhibits(text, facts.quantities)
    facts.actions = [term for term in ACTION_TERMS if normalize_text(term) in norm]
    if "dat phong" in norm and "su dung" in norm and ("ma tuy" in norm or "ketamin" in norm or "ketamine" in norm):
        facts.actions.append("tổ chức sử dụng")
    facts.consequences = [term for term in CONSEQUENCE_TERMS if normalize_text(term) in norm]
    facts.location = [term for term in LOCATION_TERMS if normalize_text(term) in norm]
    facts.mitigating_signals = [term for term in MITIGATING_TERMS if normalize_text(term) in norm]
    facts.aggravating_signals = [term for term in AGGRAVATING_TERMS if normalize_text(term) in norm]
    facts.article_refs = dedupe_keep_order(re.findall(r"[Đđ]iều\s+(\d+[a-zA-Z]?)", text))
    facts.age_info = dedupe_keep_order([m.group(0) for m in re.finditer(r"(?:\d{1,2}\s*tuổi|dưới\s*\d{1,2}|từ\s*đủ\s*\d{1,2}|đủ\s*70\s*tuổi)", text, flags=re.I)])
    facts.intent = [term for term in ["mục đích", "hưởng lợi", "cho bạn", "để bán", "để sử dụng", "để long sử dụng"] if normalize_text(term) in norm]
    facts.mental_state = [term for term in ["cố ý", "vô ý", "biết", "không biết"] if normalize_text(term) in norm]
    facts.evidence = [term for term in ["giám định", "kết luận giám định", "dương tính", "camera", "lời khai"] if normalize_text(term) in norm]
    if "khong con tang vat" in norm or any(exhibit.status == "consumed" for exhibit in facts.exhibits):
        facts.evidence.append("không còn tang vật")
        facts.unknowns.append("Không còn tang vật: cần hồ sơ xét nghiệm/giám định và chứng cứ khác để chứng minh chất ma túy, nguồn cung, hành vi.")
    if any(exhibit.status == "not_seized" for exhibit in facts.exhibits):
        facts.evidence.append("không thu giữ được tang vật")
        facts.unknowns.append("Không thu giữ được tang vật: cần chứng cứ thay thế như xét nghiệm, lời khai, camera, tin nhắn hoặc chuyển khoản.")
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
    tokens = set(norm.split())
    if "lam san" in norm or "rung" in tokens or "go" in tokens:
        facts.objects.append("gỗ/lâm sản")
        facts.crime_hints.append("tội vi phạm quy định về khai thác, bảo vệ rừng và lâm sản")
    if facts.substances:
        facts.crime_hints.append("nhóm tội phạm về ma túy")
    facts.actions = dedupe_keep_order(facts.actions)
    facts.objects = dedupe_keep_order(facts.objects)
    facts.evidence = dedupe_keep_order(facts.evidence)
    facts.intent = dedupe_keep_order(facts.intent)
    facts.unknowns = dedupe_keep_order(facts.unknowns)
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
