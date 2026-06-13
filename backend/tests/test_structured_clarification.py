from __future__ import annotations

import pytest

from app.core.config import settings
from app.models.conversation import CaseStatus, ClarificationAnswer
from app.services.dialogue_manager import handle_legal_chat
from app.services.fact_extractor import extract_facts
from app.services.session_store import session_store


DRUG_SCENARIO = (
    "Tân và Thuận là bạn thân. Long nhờ Tân đặt phòng karaoke qua Thuận để Long sử dụng ma túy "
    "ở phòng với nhiều người khác. Long đồng thời nhờ Tân mua qua Thuận để Thuận mua ma túy từ "
    "một người tên Bí và đem đến cho Long. Khi Long cùng Văn, Tiến, Sang và bốn nhân viên bị công "
    "an bắt, thu giữ một gói nghi Ketamine và hai viên ma túy tổng hợp. Các đối tượng đều dương tính "
    "với ma túy."
)


@pytest.fixture(autouse=True)
def clean_state(monkeypatch: pytest.MonkeyPatch):
    session_store.clear()
    extract_facts.cache_clear()
    monkeypatch.setattr(settings, "use_llm_fact_extractor", False)
    monkeypatch.setattr(settings, "openai_api_key", "")
    yield
    session_store.clear()
    extract_facts.cache_clear()


def test_first_turn_returns_structured_questions_and_legacy_texts():
    response = handle_legal_chat(DRUG_SCENARIO)

    assert response.status == CaseStatus.collecting_facts
    assert response.case_version == 1
    assert response.clarification is not None
    assert len(response.clarification.questions) <= 5
    ids = {question.id for question in response.clarification.questions}
    assert {"q_powder_forensic_substance", "q_tablets_forensic_substance", "q_tan_knowledge"} <= ids
    tablets = next(question for question in response.clarification.questions if question.id == "q_tablets_forensic_substance")
    assert tablets.input_type == "single_choice"
    assert tablets.fact_path == "exhibits.tablets.forensic_substance"
    assert [option.id for option in tablets.options] == [
        "mdma",
        "methamphetamine",
        "ketamine",
        "other",
        "not_narcotic",
        "no_forensic_report",
        "unknown",
    ]
    assert response.clarifying_questions == [question.text for question in response.clarification.questions]


def test_drug_facts_do_not_infer_mdma_or_forensic_from_toxicology():
    response = handle_legal_chat(DRUG_SCENARIO)

    tablets = next(exhibit for exhibit in response.facts.exhibits if exhibit.id == "tablets")
    assert tablets.suspected_substance == "ma túy tổng hợp"
    assert tablets.confirmed_substance is None
    assert tablets.forensic_status != "forensic_confirmed"
    assert any("dương tính" in item.description for item in response.missing_facts)
    assert "tổ chức sử dụng" not in response.facts.actions
    assert response.status == CaseStatus.collecting_facts


def test_second_turn_answers_merge_and_activate_dependent_mass_questions():
    first = handle_legal_chat(DRUG_SCENARIO)
    second = handle_legal_chat(
        "",
        case_id=first.case_id,
        case_version=first.case_version,
        answers=[
            ClarificationAnswer(question_id="q_powder_forensic_substance", selected_option_ids=["ketamine"]),
            ClarificationAnswer(question_id="q_tablets_forensic_substance", selected_option_ids=["mdma"]),
            ClarificationAnswer(question_id="q_tan_knowledge", selected_option_ids=["knew_group_use"]),
            ClarificationAnswer(question_id="q_money_source", selected_option_ids=["long"]),
        ],
    )

    assert second.case_version == 2
    assert second.facts.structured_facts["exhibits.powder.confirmed_substance"] == "Ketamine"
    assert second.facts.structured_facts["exhibits.tablets.confirmed_substance"] == "MDMA"
    assert second.facts.structured_facts["transactions.drug_purchase.money_source"] == "Long"
    ids = [question.id for question in second.clarification.questions]
    assert "q_tablets_forensic_substance" not in ids
    assert "q_tan_knowledge" not in ids
    assert "q_powder_net_mass" in ids
    mass_question = next(question for question in second.clarification.questions if question.id == "q_powder_net_mass")
    assert mass_question.depends_on_question_id == "q_powder_forensic_substance"
    assert mass_question.input_type == "number"


def test_unknown_answer_is_stored_and_not_reasked_immediately():
    first = handle_legal_chat(DRUG_SCENARIO)
    second = handle_legal_chat(
        "",
        case_id=first.case_id,
        case_version=first.case_version,
        answers=[ClarificationAnswer(question_id="q_tablets_forensic_substance", selected_option_ids=["unknown"])],
    )

    session = session_store.get(first.case_id)
    assert session is not None
    assert "q_tablets_forensic_substance" in session.answered_unknown_question_ids
    assert all(question.id != "q_tablets_forensic_substance" for question in second.clarification.questions)


def test_backend_runs_without_openai_key():
    settings.openai_api_key = ""
    settings.use_llm_fact_extractor = False

    response = handle_legal_chat("A rủ B đi bay phòng, có hai viên ma túy tổng hợp.")

    assert response.status == CaseStatus.collecting_facts
    assert response.clarification is not None
    assert response.clarifying_questions
