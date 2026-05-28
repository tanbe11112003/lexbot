from app.services.fact_extractor import extract_facts
from app.services.clarifying_questions import build_clarifying_questions
from app.services.legal_matcher import detect_missing_facts


def test_fact_extractor_drug_age_quantity():
    facts = extract_facts("A 17 tuổi mua 2 viên thuốc lắc cho bạn dùng trong karaoke")
    assert facts.actors[0].age == 17
    assert facts.quantities[0].raw_text == "2 viên"
    assert facts.substances


def test_missing_drug_forensics():
    facts = extract_facts("A rủ B đi bay phòng, có ketamin trong phòng karaoke")
    missing = detect_missing_facts(facts, "")
    assert any("giám định" in m for m in missing)


def test_drug_clarifying_questions_cover_core_gaps():
    scenario = "A rủ B đi bay phòng, có ketamin trong phòng karaoke"
    facts = extract_facts(scenario)
    missing = detect_missing_facts(facts, scenario)
    questions = build_clarifying_questions(facts, scenario, missing)

    assert any("khối lượng" in question or "số lượng" in question for question in questions)
    assert any("tang vật" in question and "tiêu thụ" in question for question in questions)
    assert any("người bán" in question or "cung cấp" in question for question in questions)
    assert any("rủ rê" in question or "địa điểm" in question for question in questions)


def test_drug_clarifying_questions_for_consumed_exhibit():
    scenario = "A rủ B sử dụng ketamin, tang vật đã tiêu thụ hết nên không còn tang vật khi bị bắt"
    facts = extract_facts(scenario)
    missing = detect_missing_facts(facts, scenario)
    questions = build_clarifying_questions(facts, scenario, missing)

    assert any("xét nghiệm dương tính" in question for question in questions)
    assert not any("trường hợp nào" in question for question in questions)
