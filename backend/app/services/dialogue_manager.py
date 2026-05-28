from __future__ import annotations

from app.models.conversation import CaseSession, CaseStatus, ConversationTurn, LegalChatResponse
from app.services.answer_gate import evaluate_answer_gate
from app.services.clarifying_questions import build_clarifying_questions
from app.services.fact_extractor import extract_facts
from app.services.fact_merger import merge_facts
from app.services.legal_matcher import detect_missing_facts
from app.services.legal_pipeline import run_legal_analysis
from app.services.session_store import session_store


def _append_message(existing: str, message: str) -> str:
    if not existing:
        return message
    return f"{existing}\n{message}"


def _collecting_answer(questions: list[str], warnings: list[str]) -> str:
    lines = ["Chưa đủ dữ kiện để kết luận cuối cùng."]
    if warnings:
        lines.append("Vì còn thiếu dữ kiện trọng yếu, mình chưa chốt tội danh/khoản hoặc khung hình phạt.")
    if questions:
        lines.append("Cần làm rõ:")
        lines.extend(f"{idx}. {question}" for idx, question in enumerate(questions[:8], start=1))
    else:
        lines.append("Hiện chưa có thêm thông tin để hỏi tiếp; chỉ có thể ghi nhận vụ việc ở mức sơ bộ.")
    lines.append("Phân tích dưới đây, nếu có, chỉ là tham khảo và không thay thế kết luận của cơ quan có thẩm quyền.")
    return "\n".join(lines)


def _response_from_analysis(session: CaseSession, analysis, status: CaseStatus, debug: dict | None = None) -> LegalChatResponse:
    merged_debug = analysis.debug
    if debug:
        merged_debug = {**(merged_debug or {}), **debug}
    return LegalChatResponse(
        case_id=session.case_id,
        status=status,
        facts=analysis.facts,
        missing_facts=[],
        clarifying_questions=analysis.clarifying_questions,
        candidate_articles=analysis.candidate_articles,
        legal_reasoning=analysis.legal_reasoning,
        final_answer=analysis.final_answer,
        confidence=analysis.confidence,
        citations=analysis.citations,
        warnings=analysis.warnings + ["Phân tích tham khảo; không thay thế kết luận của cơ quan có thẩm quyền."],
        debug=merged_debug,
    )


def handle_legal_chat(
    message: str,
    case_id: str | None = None,
    top_k: int = 8,
    include_debug: bool = False,
    answer_style: str = "auto",
) -> LegalChatResponse:
    session = session_store.get_or_create(case_id)
    extracted = extract_facts(message)
    merged = merge_facts(session.facts, extracted)
    scenario_text = _append_message(session.scenario_text, message)

    missing = detect_missing_facts(merged, scenario_text)
    clarifying_questions = build_clarifying_questions(merged, message, missing)
    status, missing_items, gate_warnings = evaluate_answer_gate(merged, scenario_text, missing, clarifying_questions)

    turn = ConversationTurn(user_message=message, extracted_facts=extracted)
    session.facts = merged
    session.scenario_text = scenario_text
    session.status = status

    if status == CaseStatus.ready_to_answer:
        analysis = run_legal_analysis(
            scenario=scenario_text,
            facts=merged,
            top_k=top_k,
            include_debug=include_debug,
            answer_style=answer_style,
        )
        session.status = CaseStatus.answered
        turn.bot_response_summary = analysis.final_answer[:300]
        session.turns.append(turn)
        session_store.save(session)
        debug = {"dialogue_status_before_answer": status.value, "case_turns": len(session.turns)} if include_debug else None
        return _response_from_analysis(session, analysis, CaseStatus.answered, debug)

    final_answer = _collecting_answer(clarifying_questions, gate_warnings)
    if status == CaseStatus.insufficient_information:
        final_answer = (
            "Chưa đủ dữ kiện để kết luận cuối cùng. Người dùng cho biết không biết thêm thông tin, "
            "nên hệ thống dừng hỏi lặp. Chỉ có thể phân tích sơ bộ theo dữ kiện hiện có và không chốt "
            "tội danh/khoản hoặc khung hình phạt.\n"
            "Phân tích tham khảo; không thay thế kết luận của cơ quan có thẩm quyền."
        )

    turn.bot_response_summary = final_answer[:300]
    session.turns.append(turn)
    session_store.save(session)
    debug = None
    if include_debug:
        debug = {
            "extracted_facts": extracted.model_dump(),
            "case_turns": len(session.turns),
            "scenario_text": scenario_text,
            "gate_warnings": gate_warnings,
        }
    return LegalChatResponse(
        case_id=session.case_id,
        status=status,
        facts=merged,
        missing_facts=missing_items,
        clarifying_questions=clarifying_questions,
        final_answer=final_answer,
        confidence=0.25 if missing_items else 0.45,
        warnings=gate_warnings + ["Phân tích tham khảo; không thay thế kết luận của cơ quan có thẩm quyền."],
        debug=debug,
    )
