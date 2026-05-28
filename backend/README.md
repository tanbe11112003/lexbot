# BLHS Graph Chatbot Backend

Backend FastAPI phân tích tình huống hình sự Việt Nam dựa trên Neo4j graph BLHS 2025 đã import sẵn. Neo4j là nguồn sự thật pháp lý; LLM chỉ dùng để trích xuất dữ kiện hoặc diễn giải context đã retrieve.

## Chạy

```powershell
cd backend
copy .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Chạy Backend Bằng Docker

Backend đọc cấu hình Neo4j Aura và OpenAI từ `backend/.env`.

```powershell
cd C:\Users\Admin\Downloads\2\blhs_graph_chatbot_full_upgrade_v1
docker compose up --build backend
```

Sau khi container chạy:

```text
API docs : http://127.0.0.1:8000/docs
UI demo  : http://127.0.0.1:8000/ui
Health   : http://127.0.0.1:8000/health
```

Nếu muốn chạy Neo4j local trong compose thay vì Neo4j Aura, bật profile riêng:

```powershell
docker compose --profile local-neo4j up --build
```

## Upload CSV Lên Neo4j Aura

Script `scripts/import_to_aura.py` đọc cấu hình Aura từ `backend/.env` và import dữ liệu trong `neo4j_import` bằng Neo4j driver. Mặc định script dùng `MERGE`, không xoá dữ liệu có sẵn.

```powershell
cd C:\Users\Admin\Downloads\2\blhs_graph_chatbot_full_upgrade_v1
python scripts\import_to_aura.py
```

Nếu muốn rebuild database Aura từ đầu, chỉ dùng khi chắc chắn target có thể xoá:

```powershell
python scripts\import_to_aura.py --reset
```

Nếu bạn mở `test_client.html` trực tiếp từ trình duyệt và gặp lỗi `OPTIONS ... 405`, backend này đã hỗ trợ CORS. Biến mặc định là:

```env
CORS_ALLOW_ORIGINS=*
```

Neo4j mặc định:

```text
bolt://localhost:7687
neo4j / password123456
database: neo4j
```

## Curl Test

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/articles/108
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d "{\"query\":\"ma túy\",\"top_k\":10,\"search_type\":\"hybrid\",\"include_debug\":true}"
curl -X POST http://127.0.0.1:8000/normalize -H "Content-Type: application/json" -d "{\"text\":\"bay phòng, kẹo, khay\"}"
curl -X POST http://127.0.0.1:8000/analyze-scenario -H "Content-Type: application/json" -d "{\"scenario\":\"A 17 tuổi mua 2 viên thuốc lắc cho bạn dùng trong karaoke\",\"top_k\":8,\"include_debug\":true}"
```

## Test Cases Gợi Ý

```bash
curl -X POST http://127.0.0.1:8000/analyze-scenario -H "Content-Type: application/json" -d "{\"scenario\":\"A 15 tuổi giúp sức cho B che giấu tang vật sau khi phạm tội\",\"top_k\":8,\"include_debug\":true}"
curl -X POST http://127.0.0.1:8000/analyze-scenario -H "Content-Type: application/json" -d "{\"scenario\":\"A rủ B đi bay phòng, có ketamin và 2 viên thuốc lắc trong phòng karaoke\",\"top_k\":8,\"include_debug\":true}"
curl -X POST http://127.0.0.1:8000/analyze-scenario -H "Content-Type: application/json" -d "{\"scenario\":\"Người đủ 70 tuổi phạm tội và đã tự thú\",\"top_k\":8,\"include_debug\":true}"
curl -X POST http://127.0.0.1:8000/analyze-scenario -H "Content-Type: application/json" -d "{\"scenario\":\"A khai thác 2m3 gỗ nhóm IA\",\"top_k\":8,\"include_debug\":true}"
```

## Thuật Toán

- Fast response: chào hỏi, cảm ơn, ngoài phạm vi BLHS.
- Hybrid fact extraction: regex chạy mặc định, underthesea/LLM là optional.
- Query decomposition theo actor và vai trò.
- Query rewriting gồm original, action, actor, crime hint, article ref, substance, slang-normalized, quantity và HyDE rule-based.
- Retrieval: exact article, Neo4j fulltext/fallback CONTAINS, condition search, graph signal search, vector optional.
- RRF fusion: `sum(1 / (k + rank))`, mặc định `k=60`.
- Cross-encoder reranker optional.
- Legal matcher scoring minh bạch và reasoner phân loại crime/supporting/general rule.
- Validator chống hallucination theo điều luật, ngôn ngữ kết luận chắc chắn khi thiếu dữ kiện.

## Giới Hạn

- Không import lại dữ liệu, không parse PDF, không dùng `deepseek_merged.json`.
- Vector search chỉ chạy nếu đã có embedding/index và `USE_VECTOR_SEARCH=true`.
- Reranker/underthesea/OpenAI đều optional; nếu thiếu model hoặc key, hệ thống fallback template.
- Kết quả là phân tích hỗ trợ, không thay thế kết luận điều tra/tòa án.
