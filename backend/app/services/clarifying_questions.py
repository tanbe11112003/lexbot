from __future__ import annotations

from app.models.facts import ExtractedFacts
from app.utils.text import dedupe_keep_order, normalize_text


def _is_drug_related(facts: ExtractedFacts, scenario: str, missing: list[str]) -> bool:
    norm = normalize_text(scenario)
    return bool(
        facts.substances
        or "ma tuy" in norm
        or "thuoc lac" in norm
        or "ketamin" in norm
        or any(item.startswith("Ma túy:") for item in missing)
    )


def build_clarifying_questions(facts: ExtractedFacts, scenario: str, missing: list[str]) -> list[str]:
    if not missing:
        return []

    questions: list[str] = []
    norm = normalize_text(scenario)
    action_norms = {normalize_text(action) for action in facts.actions}

    if _is_drug_related(facts, scenario, missing):
        if not facts.quantities:
            questions.append("Chất ma túy có khối lượng, hàm lượng hoặc số lượng cụ thể là bao nhiêu?")
        if not any("giám định" in item or "dương tính" in item for item in facts.evidence):
            questions.append("Đã có kết luận giám định xác định loại chất ma túy chưa?")
        if not any(term in norm for term in ["nguoi mua", "mua cua ai", "mua tu ai"]):
            questions.append("Ai là người mua hoặc đặt mua chất ma túy?")
        if not any(term in norm for term in ["nguoi ban", "cung cap", "ban cho"]):
            questions.append("Ai là người bán, giao hoặc cung cấp chất ma túy?")
        if "to chuc su dung" not in action_norms:
            questions.append("Có ai rủ rê, chuẩn bị địa điểm, dụng cụ hoặc phân công người khác sử dụng ma túy không?")
        if not any(term in norm for term in ["su dung", "duong tinh"]):
            questions.append("Ai là người trực tiếp sử dụng hoặc bị xác định dương tính với ma túy?")
        if not facts.intent:
            questions.append("Mục đích giữ chất ma túy là để sử dụng, bán lại, vận chuyển hay mục đích khác?")
        questions.append("Có ai hưởng lợi, nhận tiền công hoặc được chia lợi ích từ việc mua bán/tổ chức sử dụng không?")

    if len(facts.actors) >= 2 and not any(actor.role for actor in facts.actors):
        questions.append("Vai trò cụ thể của từng người là gì: người khởi xướng, người mua, người bán, người giúp sức hay người sử dụng?")
    if facts.age_info and not facts.actors:
        questions.append("Tuổi cụ thể gắn với từng người trong tình huống là bao nhiêu?")
    if any("Yếu tố lỗi/mục đích" in item for item in missing):
        questions.append("Người thực hiện có biết rõ hành vi và hậu quả pháp lý của việc mình làm không?")

    return dedupe_keep_order(questions)
