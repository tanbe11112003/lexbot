from __future__ import annotations

import json
import logging

from app.core.config import settings
from app.models.facts import ExtractedFacts
from app.models.legal_output import LegalReasoningItem
from app.prompts.answer_prompt import ANSWER_SYSTEM, ANSWER_USER
from app.services.context_builder import build_context_text

logger = logging.getLogger(__name__)


def _template_answer(facts: ExtractedFacts, reasoning: list[LegalReasoningItem], missing: list[str]) -> str:
    lines: list[str] = []
    lines.append("1. Dữ kiện đã nhận diện")
    lines.append(f"- Chủ thể: {', '.join(a.name + (f' ({a.age} tuổi)' if a.age else '') for a in facts.actors) or 'chưa rõ'}")
    lines.append(f"- Hành vi/tín hiệu: {', '.join(facts.actions) or 'chưa rõ'}")
    lines.append(f"- Đối tượng/chất/hậu quả: {', '.join(facts.objects + facts.consequences) or 'chưa rõ'}")
    lines.append("")
    lines.append("2. Điều luật có thể liên quan")
    for item in reasoning[:8]:
        role = "tội danh có thể xem xét" if item.classification == "crime_candidate" else "quy định hỗ trợ"
        lines.append(f"- Điều {item.article_code} - {item.title}: {role}.")
    lines.append("")
    lines.append("3. Phân tích theo từng khả năng")
    for item in reasoning[:5]:
        lines.append(f"- Điều {item.article_code}: có dấu hiệu liên quan, nhưng cần đối chiếu đủ mặt khách quan, chủ thể, lỗi/mục đích và các điều kiện trong điều luật.")
    lines.append("")
    lines.append("4. Khung hình phạt có thể áp dụng")
    frames = []
    for item in reasoning:
        for pf in item.possible_penalty_frames[:3]:
            if pf.get("text"):
                frames.append(f"- Điều {item.article_code}, khung [{pf.get('id')}]: {pf.get('text')}")
    lines.extend(frames[:8] or ["- Chưa đủ dữ kiện để chọn khoản/khung cụ thể."])
    lines.append("")
    lines.append("5. Dữ kiện còn thiếu")
    lines.extend([f"- {m}" for m in missing] or ["- Chưa phát hiện thiếu dữ kiện trọng yếu, nhưng vẫn cần kiểm tra chứng cứ thực tế."])
    lines.append("")
    lines.append("6. Kết luận thận trọng")
    lines.append("Tình huống có thể được xem xét theo các điều nêu trên, nhưng chưa đủ căn cứ để kết luận chắc chắn một tội danh hoặc một khoản cụ thể nếu các dữ kiện còn thiếu chưa được làm rõ. Kết luận cuối cùng tùy kết quả giám định/điều tra và chứng cứ.")
    return "\n".join(lines)


def generate_answer(scenario: str, facts: ExtractedFacts, contexts: list[dict], reasoning: list[LegalReasoningItem], missing: list[str]) -> str:
    if not settings.openai_api_key:
        return _template_answer(facts, reasoning, missing)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": ANSWER_USER.format(
                    scenario=scenario,
                    facts=json.dumps(facts.model_dump(), ensure_ascii=False),
                    context=build_context_text(contexts),
                    missing_facts=json.dumps(missing, ensure_ascii=False),
                )},
            ],
        )
        return resp.choices[0].message.content or _template_answer(facts, reasoning, missing)
    except Exception as exc:
        logger.warning("LLM answer skipped: %s", exc)
        return _template_answer(facts, reasoning, missing)
