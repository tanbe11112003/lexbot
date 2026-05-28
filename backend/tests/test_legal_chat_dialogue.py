from app.models.conversation import CaseStatus
from app.services.dialogue_manager import handle_legal_chat


def test_legal_chat_drug_missing_quantity_collects_facts():
    response = handle_legal_chat("A rủ B đi bay phòng, có ketamin và thuốc lắc.")

    assert response.status == CaseStatus.collecting_facts
    assert any("giám định" in item.description for item in response.missing_facts)
    assert any("khối lượng" in item.description or "hàm lượng" in item.description for item in response.missing_facts)
    assert response.clarifying_questions
    assert "Chưa đủ dữ kiện" in response.final_answer
    assert "khung hình phạt" not in response.final_answer.lower() or "chưa chốt" in response.final_answer.lower()


def test_legal_chat_merges_added_drug_quantities(monkeypatch):
    captured = {}

    def fake_run_legal_analysis(scenario, facts=None, top_k=8, include_debug=False, answer_style="auto"):
        captured["scenario"] = scenario
        captured["facts"] = facts
        from app.models.legal_output import ScenarioAnalysisResponse

        return ScenarioAnalysisResponse(
            facts=facts,
            final_answer="Tóm tắt dữ kiện đã xác định và phân tích điều luật ở mức tham khảo.",
            confidence=0.7,
            warnings=[],
        )

    monkeypatch.setattr("app.services.dialogue_manager.run_legal_analysis", fake_run_legal_analysis)

    first = handle_legal_chat("A rủ B đi bay phòng, có ketamin và thuốc lắc.")
    second = handle_legal_chat(
        "Có kết luận giám định, ketamine 1g, MDMA 0.5g, A đặt phòng và nhờ người mua, B cùng sử dụng.",
        case_id=first.case_id,
    )

    assert second.status in {CaseStatus.ready_to_answer, CaseStatus.answered}
    assert captured["facts"].quantities
    assert len(captured["facts"].substances) >= 2
    assert "A rủ B" in captured["scenario"]
    assert "ketamine 1g" in captured["scenario"]
    assert "phân tích điều luật" in second.final_answer


def test_legal_chat_accomplice_missing_roles():
    response = handle_legal_chat("A và B cùng tham gia vụ trộm nhưng chưa rõ ai làm gì.")

    assert response.status == CaseStatus.collecting_facts
    assert any("vai trò" in item.description.lower() for item in response.missing_facts)
    assert any("Vai trò" in question or "vai trò" in question for question in response.clarifying_questions)
    assert "Chưa đủ dữ kiện" in response.final_answer


def test_legal_chat_user_does_not_know_more_stops_reasking():
    response = handle_legal_chat("Tôi không biết thêm thông tin.")

    assert response.status == CaseStatus.insufficient_information
    assert response.clarifying_questions == []
    assert "dừng hỏi lặp" in response.final_answer
    assert "sơ bộ" in response.final_answer
