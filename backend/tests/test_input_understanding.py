from __future__ import annotations

import pytest

from app.core.config import settings
from app.models.conversation import CaseStatus
from app.services.dialogue_manager import handle_legal_chat
from app.services.fact_extractor import extract_facts
from app.services.input_understanding import understand_input
from app.services.session_store import session_store


@pytest.fixture(autouse=True)
def clean_state(monkeypatch: pytest.MonkeyPatch):
    session_store.clear()
    extract_facts.cache_clear()
    monkeypatch.setattr(settings, "use_llm_fact_extractor", False)
    monkeypatch.setattr(settings, "openai_api_key", "")
    yield
    session_store.clear()
    extract_facts.cache_clear()


def test_greeting_returns_fast_blhs_prompt_without_pipeline():
    response = handle_legal_chat("Xin chào", include_debug=True)

    assert response.status == CaseStatus.answered
    assert response.clarification is None
    assert response.clarifying_questions == []
    assert "Bộ luật Hình sự Việt Nam" in response.final_answer
    assert response.debug["input_understanding"]["scope"] == "greeting"


def test_out_of_scope_weather_returns_redirect_to_blhs():
    response = handle_legal_chat("Hôm nay thời tiết ở Hà Nội thế nào?", include_debug=True)

    assert response.status == CaseStatus.answered
    assert response.clarifying_questions == []
    assert "chưa liên quan" in response.final_answer
    assert "Bộ luật Hình sự Việt Nam" in response.final_answer
    assert response.debug["input_understanding"]["scope"] == "out_of_scope"


def test_rule_understanding_detects_slang_location_and_no_fake_actor():
    understanding = understand_input("Vận chuyển 50 gram hàng trắng vào Việt Nam bị xử như thế nào?")

    assert understanding.scope == "criminal_law"
    assert "Việt Nam" in understanding.locations
    assert "Việt" not in understanding.actors
    assert any(term.canonical == "heroin" for term in understanding.slang_terms)
    assert "heroin" in understanding.normalized_message


def test_short_slang_does_not_match_common_words_after_accent_normalization():
    understanding = understand_input("Tôi đã có thông tin về hợp đồng.")

    assert {term.raw for term in understanding.slang_terms}.isdisjoint({"đá", "cỏ"})
    assert understanding.scope == "legal_other"


def test_legal_input_runs_pipeline_with_understanding_debug():
    response = handle_legal_chat(
        "Đăng thông tin bịa đặt gây ảnh hưởng tới danh dự người khác thì bị xử lý thế nào?",
        include_debug=True,
    )

    assert response.status in {CaseStatus.collecting_facts, CaseStatus.answered, CaseStatus.ready_to_answer}
    assert response.debug["input_understanding"]["scope"] == "criminal_law"
    assert "Đăng" not in {actor.name for actor in response.facts.actors}
    assert "đăng" in response.facts.actions
