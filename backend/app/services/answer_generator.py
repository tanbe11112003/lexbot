from __future__ import annotations

import json
import logging
import zlib

from app.core.config import settings
from app.models.facts import ExtractedFacts
from app.models.legal_output import LegalReasoningItem
from app.prompts.answer_prompt import ANSWER_SYSTEM, ANSWER_USER
from app.services.clarifying_questions import user_declines_or_lacks_more_info
from app.services.context_builder import build_context_text

logger = logging.getLogger(__name__)


STYLE_GUIDANCE: dict[str, str] = {
    "balanced": "Trả lời tự nhiên, có đoạn ngắn và gạch đầu dòng khi thật sự cần.",
    "conversational": "Trả lời như đang trao đổi với người dùng, mềm hơn nhưng vẫn thận trọng pháp lý.",
    "brief": "Trả lời ngắn gọn, ưu tiên kết luận có điều kiện và các điểm cần hỏi thêm.",
    "educational": "Giải thích theo hướng học thuật dễ hiểu, nêu vì sao dữ kiện đó quan trọng.",
    "structured": "Dùng các mục rõ ràng, nhưng không lặp lại khuôn 6 phần cố định.",
}


def _resolve_answer_style(answer_style: str, facts: ExtractedFacts, reasoning: list[LegalReasoningItem], missing: list[str], scenario: str = "") -> str:
    if answer_style != "auto":
        return answer_style if answer_style in STYLE_GUIDANCE else "balanced"
    if user_declines_or_lacks_more_info(scenario):
        return "brief"
    if missing:
        return "conversational" if zlib.crc32(scenario.encode("utf-8")) % 2 else "educational"
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
    exhibits = ", ".join(exhibit.description for exhibit in facts.exhibits) or "tang vật chưa rõ"
    return f"Hiện mình nhận diện được {actors}; hành vi/tín hiệu là {actions}; đối tượng hoặc hậu quả liên quan là {objects}; tình trạng tang vật: {exhibits}."


def _pick(seed: str, options: list[str]) -> str:
    return options[zlib.crc32(seed.encode("utf-8")) % len(options)]


def _question_block(clarifying_questions: list[str]) -> list[str]:
    if not clarifying_questions:
        return []
    return ["", "Để chắc hơn, mình cần hỏi thêm:", *[f"- {question}" for question in clarifying_questions[:6]]]


def _template_answer(
    scenario: str,
    facts: ExtractedFacts,
    reasoning: list[LegalReasoningItem],
    missing: list[str],
    answer_style: str = "auto",
    clarifying_questions: list[str] | None = None,
) -> str:
    clarifying_questions = clarifying_questions or []
    style = _resolve_answer_style(answer_style, facts, reasoning, missing, scenario)
    no_more_info = user_declines_or_lacks_more_info(scenario)
    summary = _fallback_summary(facts)
    articles = _top_articles(reasoning)
    frames = [
        f"Điều {item.article_code}, khung [{pf.get('id')}]: {pf.get('text')}"
        for item in reasoning
        for pf in item.possible_penalty_frames[:2]
        if pf.get("text")
    ]

    if style == "brief":
        opener = _pick(scenario, [
            "Mình chốt ở mức sơ bộ như sau:",
            "Với phần dữ kiện hiện có, hướng xử lý thận trọng là:",
            "Nếu chưa có thêm tài liệu, có thể kết luận tạm thời:",
        ])
        lines = [
            f"{opener} {summary} Có thể xem xét {articles}, nhưng chưa nên chốt tội danh/khoản nếu các dữ kiện trọng yếu chưa rõ.",
        ]
        if missing:
            prefix = "Do bạn chưa có thêm dữ liệu, các điểm này được ghi nhận như giới hạn của kết luận: " if no_more_info else "Điểm còn thiếu chính: "
            lines.append(prefix + "; ".join(missing[:3]))
        lines.extend(_question_block(clarifying_questions))
        return "\n".join(lines)

    if style == "conversational":
        opener = _pick(scenario, [
            "Mình sẽ đi chậm một nhịp để tránh kết luận quá tay.",
            "Ở tình huống này, điểm quan trọng là tách điều đã biết khỏi điều còn phải chứng minh.",
            "Có cơ sở để phân tích, nhưng chưa nên xem đây là kết luận cuối.",
        ])
        lines = [
            f"{opener} {summary}",
            f"Hướng pháp lý có thể đặt ra là {articles}. Tuy vậy, kết luận cuối cùng còn phụ thuộc vào chứng cứ, kết quả giám định và vai trò cụ thể của từng người.",
        ]
        if frames:
            lines.append("Một số khung phạt có thể phải đối chiếu: " + "; ".join(frames[:3]))
        if missing:
            label = "Vì chưa có thêm thông tin, mình coi đây là giới hạn của kết luận: " if no_more_info else "Những điểm đang làm kết luận chưa chắc: "
            lines.append(label + "; ".join(missing[:4]))
        lines.extend(_question_block(clarifying_questions))
        return "\n".join(lines)

    if style == "educational":
        opener = _pick(scenario, [
            "Có thể đọc tình huống này theo ba lớp: dữ kiện, điều luật, rồi mức độ chắc chắn.",
            "Cách chắc nhất là bắt đầu từ các yếu tố cấu thành trước khi nói đến khung phạt.",
            "Mình sẽ xem đây là nhận định có điều kiện, vì một vài dữ kiện còn quyết định trực tiếp đến khoản áp dụng.",
        ])
        lines = [
            f"{opener} {summary}",
            f"Sau đó mới đối chiếu với điều luật. Các điều nổi bật hiện tại là {articles}.",
        ]
        if missing:
            lines.append("Các dữ kiện còn thiếu quan trọng vì chúng quyết định đúng tội danh, đúng khoản và đúng vai trò: " + "; ".join(missing[:5]))
        if frames:
            lines.append("Khung hình phạt chỉ nên xem là khả năng tham khảo lúc này: " + "; ".join(frames[:4]))
        if no_more_info:
            lines.append("Vì bạn chưa có thêm thông tin, kết luận nên dừng ở mức có dấu hiệu/có thể xem xét theo dữ kiện hiện có.")
        else:
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
    lines.append(_pick(scenario, [
        summary,
        f"Tóm lại phần dữ kiện trước: {summary}",
        f"Mình đang nhìn thấy các điểm chính này: {summary}",
    ]))
    lines.append(f"Các điều luật có thể liên quan gồm {articles}. Đây mới là hướng đối chiếu, không phải kết luận chắc chắn.")
    if frames:
        lines.append("Khung phạt có thể phải kiểm tra thêm: " + "; ".join(frames[:4]))
    if missing:
        prefix = "Do chưa có thêm dữ liệu, kết luận bị giới hạn bởi: " if no_more_info else "Những dữ kiện còn thiếu đang ảnh hưởng trực tiếp đến kết luận: "
        lines.append(prefix + "; ".join(missing[:5]))
    lines.append("Kết luận nên giữ ở mức thận trọng: có dấu hiệu/có thể xem xét theo điều luật ứng viên, nhưng chưa đủ căn cứ để khẳng định chắc chắn tội danh hoặc khung cụ thể.")
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
    resolved_style = _resolve_answer_style(answer_style, facts, reasoning, missing, scenario)
    if not settings.openai_api_key:
        return _template_answer(scenario, facts, reasoning, missing, resolved_style, clarifying_questions)
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
        return resp.choices[0].message.content or _template_answer(scenario, facts, reasoning, missing, resolved_style, clarifying_questions)
    except Exception as exc:
        logger.warning("LLM answer skipped: %s", exc)
        return _template_answer(scenario, facts, reasoning, missing, resolved_style, clarifying_questions)
