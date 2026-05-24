from __future__ import annotations

from app.models.facts import ExtractedFacts
from app.models.legal_output import MatchedElement
from app.utils.scoring import clamp
from app.utils.text import normalize_text


def detect_missing_facts(facts: ExtractedFacts, scenario: str) -> list[str]:
    norm = normalize_text(scenario)
    missing: list[str] = []
    if facts.substances or "ma tuy" in norm:
        if not any("giám định" in e or "dương tính" in e for e in facts.evidence):
            missing.append("Ma túy: thiếu kết luận giám định về loại chất.")
        if not facts.quantities:
            missing.append("Ma túy: thiếu khối lượng/hàm lượng hoặc số lượng để xác định khoản.")
        missing.append("Ma túy: cần làm rõ ai cung cấp, ai tổ chức, ai sử dụng, mục đích và hưởng lợi.")
    if "go" in norm or "lam san" in norm or "rung" in norm:
        if not any(q.unit in {"m3", "m³", "mét khối"} for q in facts.quantities):
            missing.append("Lâm sản/gỗ: thiếu khối lượng m3.")
        missing.append("Lâm sản/gỗ: thiếu loại gỗ/nhóm IA-IIA, nguồn gốc và hành vi chính xác.")
    if facts.age_info and not facts.actors:
        missing.append("Tuổi: cần xác định tuổi gắn với từng người cụ thể.")
    if len(facts.actors) >= 2 and not any(a.role for a in facts.actors):
        missing.append("Đồng phạm: thiếu vai trò cụ thể của từng người.")
    if any(x in norm for x in ["thuong tich", "chet nguoi", "tu vong", "thiet hai"]) and not facts.consequences:
        missing.append("Tội có hậu quả: thiếu hậu quả, tỷ lệ thương tật hoặc thiệt hại tài sản.")
    if not facts.intent and not facts.mental_state:
        missing.append("Yếu tố lỗi/mục đích: cần làm rõ cố ý/vô ý, biết hay không biết, mục đích thực hiện.")
    return list(dict.fromkeys(missing))


def score_context(ctx: dict, facts: ExtractedFacts, normalized: list[dict], missing: list[str]) -> tuple[float, list[MatchedElement]]:
    text_parts: list[str] = []
    for key in ["article", "crime"]:
        node = ctx.get(key) or {}
        text_parts.extend(str(v) for v in node.values() if isinstance(v, str))
    for key in ["conditions", "act_requirements", "object_requirements", "consequence_requirements", "quantity_thresholds"]:
        for node in ctx.get(key) or []:
            text_parts.extend(str(v) for v in node.values() if isinstance(v, str))
    haystack = normalize_text(" ".join(text_parts))
    score = 0.0
    matched: list[MatchedElement] = []
    def add(kind: str, value: str, points: float, reason: str) -> None:
        nonlocal score
        if value and normalize_text(value) in haystack:
            score += points
            matched.append(MatchedElement(type=kind, text=value, score=points, reason=reason))
    for action in facts.actions:
        add("action", action, 0.25, "Hành vi trong tình huống khớp context.")
    for obj in facts.objects:
        add("object", obj, 0.20, "Đối tượng/vật chứng khớp context.")
    for sub in facts.substances:
        add("substance", sub.name, 0.20, "Chất/nhóm chất khớp context.")
    if facts.quantities and (ctx.get("quantity_thresholds") or ctx.get("conditions")):
        score += 0.20
        matched.append(MatchedElement(type="quantity", text=", ".join(q.raw_text for q in facts.quantities), score=0.20, reason="Có định lượng cần đối chiếu ngưỡng."))
    if facts.age_info and (ctx.get("subject_requirements") or (ctx.get("article") or {}).get("article_code") == "12"):
        score += 0.15
    for consequence in facts.consequences:
        add("consequence", consequence, 0.15, "Hậu quả khớp context.")
    code = str((ctx.get("article") or {}).get("article_code") or "")
    if code in facts.article_refs:
        score += 0.30
    title = str((ctx.get("article") or {}).get("title") or "")
    if title and any(normalize_text(title) in normalize_text(h) or normalize_text(h) in normalize_text(title) for h in facts.crime_hints):
        score += 0.20
    if normalized:
        score += 0.10
    if ctx.get("penalty_frames"):
        score += 0.05
    score -= 0.15 * len(missing)
    return clamp(score), matched
