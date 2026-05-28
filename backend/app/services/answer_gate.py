from __future__ import annotations

from app.models.conversation import CaseStatus, MissingFactItem
from app.models.facts import ExtractedFacts
from app.services.clarifying_questions import user_declines_or_lacks_more_info
from app.utils.text import normalize_text


DRUG_ACTIONS = {"tàng trữ", "mua", "mua bán", "vận chuyển", "tổ chức sử dụng", "chứa chấp", "sử dụng", "cung cấp"}


def _is_drug_case(facts: ExtractedFacts, scenario: str) -> bool:
    norm = normalize_text(scenario)
    return bool(facts.substances or "ma tuy" in norm or "ketamin" in norm or "thuoc lac" in norm or "mdma" in norm)


def _has_forensic(facts: ExtractedFacts) -> bool:
    return any("giám định" in item or "dương tính" in item for item in facts.evidence)


def _has_role_info(facts: ExtractedFacts) -> bool:
    if len(facts.actors) < 2:
        return True
    return any(actor.role for actor in facts.actors) or any(action in facts.actions for action in ["rủ", "nhờ", "đặt phòng", "giúp sức", "xúi giục", "chủ mưu", "cầm đầu"])


def _has_purpose(facts: ExtractedFacts) -> bool:
    action_norms = {normalize_text(action) for action in facts.actions}
    return bool(facts.intent or {"su dung", "mua ban", "van chuyen", "to chuc su dung"} & action_norms)


def _drug_core_ready(facts: ExtractedFacts) -> bool:
    action_norms = {normalize_text(action) for action in facts.actions}
    has_action = bool(action_norms & {normalize_text(action) for action in DRUG_ACTIONS})
    has_exhibit_or_substitute = bool(facts.exhibits) or (_has_forensic(facts) and bool(facts.quantities))
    return all([
        facts.substances,
        facts.quantities,
        has_exhibit_or_substitute,
        _has_forensic(facts),
        has_action,
        _has_role_info(facts),
        _has_purpose(facts),
    ])


def _critical_from_text(item: str, facts: ExtractedFacts, scenario: str) -> bool:
    norm = normalize_text(item)
    if "ma tuy" in norm:
        if any(term in norm for term in ["giam dinh", "khoi luong", "ham luong", "so luong", "tang vat", "muc dich"]):
            return True
        if any(term in norm for term in ["cung cap", "to chuc", "su dung", "huong loi"]):
            return _is_drug_case(facts, scenario)
    if "dong pham" in norm or "vai tro" in norm:
        return True
    if "tuoi" in norm and "tung nguoi" in norm:
        return True
    if "lam san" in norm or "go" in norm:
        return True
    if "yeu to loi" in norm or "muc dich" in norm:
        return True
    return False


def to_missing_items(missing: list[str], clarifying_questions: list[str], facts: ExtractedFacts, scenario: str) -> list[MissingFactItem]:
    items: list[MissingFactItem] = []
    for idx, text in enumerate(missing):
        norm = normalize_text(text)
        domain = "drug" if "ma tuy" in norm else "forestry" if ("lam san" in norm or "go" in norm) else "general"
        question = clarifying_questions[idx] if idx < len(clarifying_questions) else None
        items.append(MissingFactItem(
            key=f"missing_{idx + 1}",
            label=text.split(":", 1)[0],
            description=text,
            critical=_critical_from_text(text, facts, scenario),
            domain=domain,
            question=question,
        ))
    return items


def evaluate_answer_gate(
    facts: ExtractedFacts,
    scenario: str,
    missing: list[str],
    clarifying_questions: list[str],
) -> tuple[CaseStatus, list[MissingFactItem], list[str]]:
    missing_items = to_missing_items(missing, clarifying_questions, facts, scenario)
    warnings: list[str] = []
    if user_declines_or_lacks_more_info(scenario):
        warnings.append("Người dùng cho biết không có thêm thông tin; dừng hỏi lặp và chỉ phân tích giới hạn theo hồ sơ hiện có.")
        return CaseStatus.insufficient_information, missing_items, warnings

    if _is_drug_case(facts, scenario):
        if not _drug_core_ready(facts):
            warnings.append("Thiếu dữ kiện trọng yếu của nhóm tội ma túy; không chốt tội danh/khoản hoặc khung hình phạt.")
            return CaseStatus.collecting_facts, missing_items, warnings
        if missing_items:
            warnings.append("Dữ kiện cốt lõi của nhóm tội ma túy đã đủ để phân tích, nhưng vẫn còn điểm phụ cần nêu điều kiện.")
        return CaseStatus.ready_to_answer, missing_items, warnings

    if any(item.critical for item in missing_items):
        warnings.append("Còn dữ kiện trọng yếu; câu trả lời cuối cùng bị chặn để hỏi làm rõ.")
        return CaseStatus.collecting_facts, missing_items, warnings

    return CaseStatus.ready_to_answer, missing_items, warnings
