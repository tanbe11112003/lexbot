from app.services.fact_extractor import extract_facts
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
