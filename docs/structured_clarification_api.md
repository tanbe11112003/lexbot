# Structured Clarification API

Tài liệu này mô tả contract tích hợp `/chat/legal` cho backend nghiệp vụ và frontend. API vẫn tương thích response cũ qua `clarifying_questions: string[]`, nhưng client mới nên dùng `clarification.questions`.

## 1. Vòng đời case

1. Client gửi mô tả ban đầu bằng `message`.
2. AI backend trích xuất facts, phát hiện missing legal slots và trả `case_id`, `case_version`, `clarification`.
3. Frontend render form theo `clarification.questions`.
4. Client gửi lại `case_id`, `case_version` và `answers`; `message` có thể rỗng.
5. Server validate answers theo question set đã phát hành, tự ánh xạ option/value thành fact patch, merge vào facts, tăng `case_version`.
6. Server hỏi tiếp hoặc trả phân tích có điều kiện nếu đủ dữ kiện.

Session hiện dùng in-memory store. Backend nghiệp vụ cần lưu `case_id` và `case_version` ở phía mình nếu muốn tiếp tục case sau reload UI.

## 2. Request lượt đầu

```http
POST /chat/legal
Content-Type: application/json
```

```json
{
  "message": "Tân đặt phòng karaoke, công an thu giữ một gói nghi Ketamine và hai viên ma túy tổng hợp.",
  "top_k": 8,
  "include_debug": false
}
```

## 3. Response collecting facts

```json
{
  "case_id": "2e6b9c70-5d55-4d4d-986b-1dbb860f71b5",
  "case_version": 1,
  "status": "collecting_facts",
  "facts": {},
  "provisional_findings": [
    {
      "status": "insufficient_evidence",
      "text": "Chưa đủ dữ kiện trọng yếu để chuyển giả thuyết pháp lý thành kết luận.",
      "affected_articles": [],
      "confidence": 0.2
    }
  ],
  "missing_facts": [
    {
      "key": "exhibits.tablets.forensic_substance",
      "label": "Hoạt chất của viên nén",
      "description": "Tang vật và giám định: thiếu hoạt chất của viên nén theo kết luận giám định.",
      "critical": true,
      "domain": "drug"
    }
  ],
  "clarification": {
    "type": "form",
    "question_set_id": "qs-69e2b8d0-3adf-42f4-b733-5d45a4f117a1",
    "can_submit_partial": true,
    "questions": [
      {
        "id": "q_tablets_forensic_substance",
        "fact_path": "exhibits.tablets.forensic_substance",
        "group": "Tang vật và giám định",
        "text": "Kết luận giám định xác định hoạt chất trong hai viên nén là chất nào?",
        "input_type": "single_choice",
        "required": true,
        "critical": true,
        "allow_free_text": true,
        "reason": "Cụm 'ma túy tổng hợp' không tự động đồng nghĩa với MDMA; cần kết luận giám định.",
        "affected_articles": ["249", "250", "251", "255"],
        "options": [
          {"id": "mdma", "label": "MDMA"},
          {"id": "methamphetamine", "label": "Methamphetamine"},
          {"id": "ketamine", "label": "Ketamine"},
          {"id": "other", "label": "Chất khác", "requires_value": true, "value_type": "text"},
          {"id": "not_narcotic", "label": "Không phải chất ma túy"},
          {"id": "no_forensic_report", "label": "Chưa có kết luận giám định"},
          {"id": "unknown", "label": "Không biết"}
        ]
      }
    ]
  },
  "clarifying_questions": [
    "Kết luận giám định xác định hoạt chất trong hai viên nén là chất nào?"
  ],
  "final_answer": "Chưa đủ dữ kiện để kết luận cuối cùng."
}
```

## 4. Submit answers

Client chỉ gửi ID câu hỏi và ID option/value. Không gửi `fact_path`, object facts hoặc patch tự do.

```json
{
  "case_id": "2e6b9c70-5d55-4d4d-986b-1dbb860f71b5",
  "case_version": 1,
  "message": "",
  "answers": [
    {
      "question_id": "q_tablets_forensic_substance",
      "selected_option_ids": ["mdma"],
      "value": null,
      "free_text": null
    }
  ],
  "top_k": 8,
  "include_debug": false
}
```

Nếu chọn option `other`, client phải gửi `free_text`:

```json
{
  "question_id": "q_tablets_forensic_substance",
  "selected_option_ids": ["other"],
  "free_text": "Nimetazepam"
}
```

Với `number`, gửi `value`:

```json
{
  "question_id": "q_powder_net_mass",
  "selected_option_ids": [],
  "value": 1.23
}
```

## 5. Response sau merge

```json
{
  "case_id": "2e6b9c70-5d55-4d4d-986b-1dbb860f71b5",
  "case_version": 2,
  "status": "collecting_facts",
  "facts": {
    "structured_facts": {
      "exhibits.tablets.confirmed_substance": "MDMA",
      "exhibits.tablets.forensic_status": "forensic_confirmed"
    },
    "exhibits": [
      {
        "id": "tablets",
        "form": "tablets",
        "suspected_substance": "ma túy tổng hợp",
        "confirmed_substance": "MDMA",
        "forensic_status": "forensic_confirmed",
        "evidence_source": "forensic_report",
        "confidence": 0.97
      }
    ]
  },
  "clarification": {
    "type": "form",
    "question_set_id": "qs-next",
    "can_submit_partial": true,
    "questions": [
      {
        "id": "q_tablets_net_mass",
        "fact_path": "exhibits.tablets.quantity.value",
        "group": "Định lượng tang vật",
        "text": "Tổng khối lượng tịnh của hoạt chất trong các viên nén là bao nhiêu gam?",
        "input_type": "number",
        "unit": "g",
        "min_value": 0,
        "depends_on_question_id": "q_tablets_forensic_substance",
        "depends_on_option_ids": ["mdma", "methamphetamine", "ketamine", "other"],
        "required": true,
        "critical": true
      }
    ]
  }
}
```

## 6. Input types

- `single_choice`: radio/select, gửi một option ID.
- `multi_choice`: checkbox, gửi nhiều option IDs.
- `number`: ô số, gửi `value`; server kiểm tra `min_value` và `max_value`.
- `text`: ô nhập tự do, gửi `value` hoặc `free_text`.
- `date`: ngày hoặc khoảng ngày; gửi `value`.
- `boolean`: yes/no nếu template phát hành.
- `actor_matrix`: bảng actor x vai trò nếu template phát hành.

## 7. Versioning và lỗi

- `case_version` khớp: request được xử lý và response tăng version.
- `case_version` cũ hoặc mới hơn state hiện tại: HTTP 409 với `current_case_version`.
- `question_id` chưa từng phát hành cho case: HTTP 422.
- `question_id` thuộc case khác: HTTP 400.
- Option không nằm trong câu hỏi đã phát hành: HTTP 422.
- Option yêu cầu nhập thêm mà thiếu `free_text/value`: HTTP 422.
- Request không có cả `message` lẫn `answers`: HTTP 422.
- Extra fields như `fact_path` trong answer bị từ chối.

## 8. Quy tắc pháp lý quan trọng

- “Nghi Ketamine” chỉ là nghi vấn, không phải kết luận giám định.
- “Hai viên ma túy tổng hợp” không tự động được chuẩn hóa thành MDMA.
- “Dương tính với ma túy” là xét nghiệm trên cơ thể người, không thay thế giám định tang vật.
- Có nhiều người cùng sử dụng không tự động đủ căn cứ kết luận tội tổ chức sử dụng trái phép chất ma túy.
- LLM là optional; khi không có OpenAI key, backend vẫn dùng rule-based extraction, planner và answer gate.

## 9. Gợi ý tích hợp backend nghiệp vụ/frontend

Backend nghiệp vụ nên:

- Lưu `case_id`, `case_version`, `question_set_id` và raw response theo hồ sơ nghiệp vụ.
- Không tự tạo fact patch từ frontend. Chỉ forward answers gồm question/option/value.
- Khi nhận HTTP 409, tải lại state mới nhất hoặc yêu cầu người dùng refresh form.
- Với `can_submit_partial=true`, có thể gửi các câu đã trả lời trước; backend sẽ không hỏi lại ngay các câu đã trả lời hoặc chọn “Không biết”.

Frontend nên:

- Render theo `input_type`.
- Hiển thị `group`, `text`, `reason`, `affected_articles` nếu cần giải thích nghiệp vụ.
- Nếu option có `requires_value=true`, bắt nhập thêm đúng `value_type`.
- Không dựa vào thứ tự câu hỏi để map answers; luôn dùng `question.id` và `option.id`.
