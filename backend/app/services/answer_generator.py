from __future__ import annotations

import json
import logging

from app.core.config import settings
from app.models.facts import ExtractedFacts
from app.models.legal_output import LegalReasoningItem
from app.prompts.answer_prompt import ANSWER_SYSTEM, ANSWER_USER
from app.services.context_builder import build_context_text

logger = logging.getLogger(__name__)


STYLE_GUIDANCE: dict[str, str] = {
    "balanced": "Trả lời tự nhiên, có đoạn ngắn và gạch đầu dòng khi thật sự cần.",
    "conversational": "Trả lời như đang trao đổi với người dùng, mềm hơn nhưng vẫn thận trọng pháp lý.",
    "brief": "Trả lời ngắn gọn, ưu tiên kết luận có điều kiện và các điểm cần hỏi thêm.",
    "educational": "Giải thích theo hướng học thuật dễ hiểu, nêu vì sao dữ kiện đó quan trọng.",
    "structured": "Dùng các mục rõ ràng, nhưng không lặp lại khuôn 6 phần cố định.",
}


def _resolve_answer_style(answer_style: str, facts: ExtractedFacts, reasoning: list[LegalReasoningItem], missing: list[str]) -> str:
    if answer_style != "auto":
        return answer_style if answer_style in STYLE_GUIDANCE else "balanced"
    if missing:
        return "conversational"
    if len(reasoning) > 3:
        return "educational"
    if len(facts.actions) <= 1 and len(facts.actors) <= 1:
        return "brief"
    return "balanced"


def _top_articles(reasoning: list[LegalReasoningItem]) -> str:
    articles = [f"Điều {item.article_code} ({item.title})" for item in reasoning[:4]]
    return ", ".join(articles) if articles else "chưa xác định được điều luật ứng viên đủ tin cậy"


def _fallback_summary(facts: ExtractedFacts) -> str:
    actors = ", ".join(a.name + (f" ({a.age} tuổi)" if a.age else "") for a in facts.actors) or "chủ thể chưa rõ"
    actions = ", ".join(facts.actions) or "hành vi chưa rõ"
    objects = ", ".join(facts.objects + [s.name for s in facts.substances] + facts.consequences) or "đối tượng/hậu quả chưa rõ"
    return f"Hiện mình nhận diện được {actors}; hành vi/tín hiệu là {actions}; đối tượng hoặc hậu quả liên quan là {objects}."


def _question_block(clarifying_questions: list[str]) -> list[str]:
    if not clarifying_questions:
        return []
    return ["", "Để chắc hơn, mình cần hỏi thêm:", *[f"- {question}" for question in clarifying_questions[:6]]]


def _template_answer(
    facts: ExtractedFacts,
    reasoning: list[LegalReasoningItem],
    missing: list[str],
    answer_style: str = "auto",
    clarifying_questions: list[str] | None = None,
) -> str:
    clarifying_questions = clarifying_questions or []
    style = _resolve_answer_style(answer_style, facts, reasoning, missing)
    summary = _fallback_summary(facts)
    articles = _top_articles(reasoning)
    frames = [
        f"Điều {item.article_code}, khung [{pf.get('id')}]: {pf.get('text')}"
        for item in reasoning
        for pf in item.possible_penalty_frames[:2]
        if pf.get("text")
    ]

    if style == "brief":
        lines = [
            f"{summary} Với dữ kiện hiện có, có thể xem xét {articles}, nhưng chưa nên chốt tội danh/khoản nếu các dữ kiện trọng yếu chưa rõ.",
        ]
        if missing:
            lines.append("Điểm còn thiếu chính: " + "; ".join(missing[:3]))
        lines.extend(_question_block(clarifying_questions))
        return "\n".join(lines)

    if style == "conversational":
        lines = [
            f"Mình chưa muốn kết luận quá sớm ở tình huống này. {summary}",
            f"Hướng pháp lý có thể đặt ra là {articles}. Tuy vậy, kết luận cuối cùng còn phụ thuộc vào chứng cứ, kết quả giám định và vai trò cụ thể của từng người.",
        ]
        if frames:
            lines.append("Một số khung phạt có thể phải đối chiếu: " + "; ".join(frames[:3]))
        if missing:
            lines.append("Những điểm đang làm kết luận chưa chắc: " + "; ".join(missing[:4]))
        lines.extend(_question_block(clarifying_questions))
        return "\n".join(lines)

    if style == "educational":
        lines = [
            f"Nhìn theo từng bước, trước hết cần tách dữ kiện khỏi kết luận. {summary}",
            f"Sau đó mới đối chiếu với điều luật. Các điều nổi bật hiện tại là {articles}.",
        ]
        if missing:
            lines.append("Các dữ kiện còn thiếu quan trọng vì chúng quyết định đúng tội danh, đúng khoản và đúng vai trò: " + "; ".join(missing[:5]))
        if frames:
            lines.append("Khung hình phạt chỉ nên xem là khả năng tham khảo lúc này: " + "; ".join(frames[:4]))
        lines.append("Vì vậy, câu trả lời nên dừng ở mức có dấu hiệu/có thể xem xét, chưa đủ căn cứ để kết luận chắc chắn.")
        lines.extend(_question_block(clarifying_questions))
        return "\n".join(lines)

    if style == "structured":
        lines = [
            "Nhận định sơ bộ",
            summary,
            "",
            "Điều luật cần đối chiếu",
            f"- {articles}",
            "",
            "Lưu ý trước khi kết luận",
        ]
        lines.extend([f"- {m}" for m in missing[:6]] or ["- Chưa phát hiện thiếu dữ kiện trọng yếu, nhưng vẫn cần kiểm tra chứng cứ thực tế."])
        if frames:
            lines.extend(["", "Khung phạt có thể liên quan", *[f"- {frame}" for frame in frames[:5]]])
        lines.extend(_question_block(clarifying_questions))
        return "\n".join(lines)

    lines: list[str] = []
    lines.append(summary)
    lines.append(f"Các điều luật có thể liên quan gồm {articles}. Đây mới là hướng đối chiếu, không phải kết luận chắc chắn.")
    if frames:
        lines.append("Khung phạt có thể phải kiểm tra thêm: " + "; ".join(frames[:4]))
    if missing:
        lines.append("Những dữ kiện còn thiếu đang ảnh hưởng trực tiếp đến kết luận: " + "; ".join(missing[:5]))
    lines.append("Kết luận nên giữ ở mức thận trọng cho đến khi làm rõ chứng cứ, vai trò từng người và điều kiện trong điều luật.")
    lines.extend(_question_block(clarifying_questions))
    return "\n".join(lines)


def generate_answer(
    scenario: str,
    facts: ExtractedFacts,
    contexts: list[dict],
    reasoning: list[LegalReasoningItem],
    missing: list[str],
    answer_style: str = "auto",
    clarifying_questions: list[str] | None = None,
) -> str:
    clarifying_questions = clarifying_questions or []
    resolved_style = _resolve_answer_style(answer_style, facts, reasoning, missing)
    if not settings.openai_api_key:
        return _template_answer(facts, reasoning, missing, resolved_style, clarifying_questions)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.35 if resolved_style in {"conversational", "educational"} else 0.25,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": ANSWER_USER.format(
                    scenario=scenario,
                    facts=json.dumps(facts.model_dump(), ensure_ascii=False),
                    context=build_context_text(contexts),
                    missing_facts=json.dumps(missing, ensure_ascii=False),
                    clarifying_questions=json.dumps(clarifying_questions, ensure_ascii=False),
                    answer_style=f"{resolved_style}: {STYLE_GUIDANCE[resolved_style]}",
                )},
            ],
        )
        return resp.choices[0].message.content or _template_answer(facts, reasoning, missing, resolved_style, clarifying_questions)
    except Exception as exc:
        logger.warning("LLM answer skipped: %s", exc)
        return _template_answer(facts, reasoning, missing, resolved_style, clarifying_questions)
