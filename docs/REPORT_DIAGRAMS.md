# Bộ sơ đồ dùng cho báo cáo

Tài liệu này tổng hợp các sơ đồ Mermaid mô tả hệ thống **BLHS Graph Chatbot Full Upgrade v1**. Có thể chèn trực tiếp vào báo cáo Markdown hoặc render sang PNG/SVG bằng Mermaid.

## 1. Sơ đồ tổng quan hệ thống

```mermaid
flowchart LR
    User["Người dùng / Sinh viên luật"] --> UI["Web UI demo<br/>/ui"]
    User --> APIClient["Client API<br/>Postman / curl / Frontend"]

    UI --> Backend["FastAPI Backend<br/>BLHS Graph Chatbot"]
    APIClient --> Backend

    Backend --> Neo4j["Neo4j Graph Database<br/>Nguồn sự thật pháp lý"]
    Backend -. optional .-> OpenAI["OpenAI LLM<br/>Trích xuất dữ kiện / diễn giải"]
    Backend -. optional .-> Embedding["Embedding / Reranker Models<br/>Vector search / rerank"]

    PDF["PDF Bộ luật Hình sự"] --> Parser["Pipeline parse + enrich<br/>blhs_pdf_to_neo4j_full_pipeline.py"]
    Parser --> CSV["CSV import files<br/>neo4j_import/"]
    CSV --> Neo4j
```

## 2. Sơ đồ triển khai

```mermaid
flowchart TB
    subgraph Local["Môi trường local / demo"]
        Browser["Browser<br/>http://localhost:8000/ui"]
        BackendContainer["Backend container<br/>blhs-graph-chatbot-backend<br/>Port 8000"]
        Neo4jContainer["Neo4j container<br/>blhs-neo4j-full-upgrade<br/>Ports 7474, 7687"]
        ImportVolume["Volumes<br/>neo4j_import, cypher,<br/>neo4j_data, neo4j_logs"]
    end

    Browser --> BackendContainer
    BackendContainer -->|"Bolt 7687"| Neo4jContainer
    ImportVolume --> Neo4jContainer

    subgraph Cloud["Triển khai Railway / Aura"]
        Railway["Railway Backend<br/>start.sh + uvicorn"]
        Aura["Neo4j Aura<br/>Managed graph database"]
    end

    Railway -->|"NEO4J_URI"| Aura
```

## 3. Sơ đồ pipeline xây dựng dữ liệu

```mermaid
flowchart TD
    SourcePDF["source_pdf/<br/>PDF BLHS"] --> Extract["Trích xuất nội dung PDF"]
    Extract --> Normalize["Chuẩn hóa dữ liệu<br/>data/blhs_from_pdf_normalized.base.json"]
    Normalize --> Enrich["Rule-based extractor<br/>tội danh, điều kiện, khung phạt,<br/>ngưỡng định lượng, ngoại lệ"]
    Enrich --> CSVGen["Sinh CSV import<br/>neo4j_import/"]
    CSVGen --> Constraints["Tạo constraints / indexes<br/>cypher/01_constraints_indexes.cypher"]
    Constraints --> Import["Import graph<br/>cypher/02_import_csv.cypher<br/>hoặc scripts/import_to_aura.py"]
    Import --> Verify["Kiểm tra dữ liệu<br/>cypher/03_verify_import.cypher"]
    Verify --> Graph["Neo4j BLHS Graph"]
```

## 4. Sơ đồ tầng dữ liệu pháp lý

```mermaid
flowchart TB
    L1["Tầng 1: Legal Structure<br/>Law → Part → Chapter → Section → Article → Clause → Point"]
    L2["Tầng 2: Legal Meaning<br/>Article → Crime → Requirements"]
    L3["Tầng 3: Penalty<br/>Clause/Point → PenaltyFrame → Penalty"]
    L4["Tầng 4: General Rules<br/>Concepts, exceptions, mitigating/aggravating factors"]
    L5["Tầng 5: NLP Mapping<br/>Slang, action alias, substance alias, legal signal"]
    Runtime["Runtime<br/>ScenarioFact, MatchedCondition"]

    L1 --> L2
    L2 --> L3
    L4 --> L2
    L5 --> L2
    Runtime --> L2
```

## 5. Sơ đồ graph schema chính

```mermaid
erDiagram
    Law ||--o{ Part : HAS_PART
    Part ||--o{ Chapter : HAS_CHAPTER
    Chapter ||--o{ Section : HAS_SECTION
    Chapter ||--o{ Article : HAS_ARTICLE
    Section ||--o{ Article : HAS_ARTICLE
    Article ||--o{ Clause : HAS_CLAUSE
    Clause ||--o{ Point : HAS_POINT
    Article ||--o| Crime : DEFINES_CRIME
    Article ||--o{ Rule : HAS_RULE
    Clause ||--o{ Rule : REPRESENTS_RULE
    Point ||--o{ Rule : REPRESENTS_RULE
    Clause ||--o{ Condition : HAS_CONDITION
    Point ||--o{ Condition : HAS_CONDITION
    Clause ||--o{ PenaltyFrame : HAS_PENALTY_FRAME
    Point ||--o{ PenaltyFrame : HAS_PENALTY_FRAME
    PenaltyFrame ||--o{ Penalty : HAS_MAIN_OR_ADDITIONAL_PENALTY
    Article ||--o{ Article : REFERENCES
```

## 6. Sơ đồ yêu cầu cấu thành tội phạm

```mermaid
flowchart LR
    Article["Article<br/>Điều luật"] --> Crime["Crime<br/>Tội danh"]
    Crime --> Act["ActRequirement<br/>Hành vi"]
    Crime --> Subject["SubjectRequirement<br/>Chủ thể"]
    Crime --> Object["ObjectRequirement<br/>Khách thể"]
    Crime --> Consequence["ConsequenceRequirement<br/>Hậu quả"]
    Crime --> Quantity["QuantityThreshold<br/>Ngưỡng định lượng"]

    Article --> Exception["Exception<br/>Ngoại lệ"]
    Article --> Mitigating["MitigatingFactor<br/>Tình tiết giảm nhẹ"]
    Article --> Aggravating["AggravatingFactor<br/>Tình tiết tăng nặng"]
    Article --> Judicial["JudicialMeasure<br/>Biện pháp tư pháp"]
```

## 7. Sơ đồ NLP mapping

```mermaid
flowchart LR
    Slang["SlangTerm<br/>kẹo, bay phòng, khay"] -->|NORMALIZES_TO| Concept["LegalConcept / Substance"]
    Slang -->|MAY_INDICATE| Signal["LegalSignal"]
    ActionAlias["ActionAlias<br/>rủ đi bay, mua, giúp sức"] -->|MAY_INDICATE| Signal
    SubstanceAlias["SubstanceAlias<br/>MDMA, ketamin"] -->|NORMALIZES_TO| Substance["Substance"]
    Signal -->|RELATED_TO| Article["Article<br/>Điều luật liên quan"]
    Substance --> Article
    Concept --> Article
```

## 8. Sơ đồ kiến trúc backend

```mermaid
flowchart TB
    App["FastAPI app.main"] --> Routers["Routers"]
    Routers --> Health["/health"]
    Routers --> Articles["/articles/{article_code}"]
    Routers --> Search["/search<br/>/normalize"]
    Routers --> Chat["/analyze-scenario<br/>/chat/legal"]

    Search --> SearchServices["Search services"]
    Chat --> LegalPipeline["legal_pipeline"]
    Chat --> Dialogue["dialogue_manager"]
    Articles --> GraphRetriever["graph_retriever"]

    SearchServices --> FactExtractor["fact_extractor"]
    SearchServices --> HybridRetriever["hybrid_retriever"]
    SearchServices --> Reasoner["legal_reasoner"]
    SearchServices --> AnswerGenerator["answer_generator"]
    SearchServices --> Validator["validator"]

    LegalPipeline --> FactExtractor
    LegalPipeline --> HybridRetriever
    LegalPipeline --> GraphRetriever
    LegalPipeline --> Reasoner
    LegalPipeline --> AnswerGenerator
    LegalPipeline --> Validator

    Dialogue --> SessionStore["session_store<br/>in-memory"]
    Dialogue --> FactMerger["fact_merger"]
    Dialogue --> AnswerGate["answer_gate"]
    Dialogue --> LegalPipeline

    GraphRetriever --> Neo4jCore["core.neo4j<br/>Neo4j driver"]
    HybridRetriever --> Neo4jCore
```

## 9. Sơ đồ luồng phân tích một tình huống

```mermaid
sequenceDiagram
    actor User as Người dùng
    participant API as FastAPI /analyze-scenario
    participant Fast as fast_response
    participant Fact as fact_extractor
    participant Norm as normalizer
    participant Rewrite as decomposer + query_rewriter
    participant Retrieve as hybrid_retriever + reranker
    participant Graph as Neo4j Graph
    participant Reason as legal_reasoner
    participant Answer as answer_generator + validator

    User->>API: Gửi scenario, top_k
    API->>Fast: Kiểm tra chào hỏi / ngoài phạm vi
    alt Có phản hồi nhanh
        Fast-->>API: Trả lời template
        API-->>User: final_answer
    else Cần phân tích pháp lý
        API->>Fact: Trích xuất actors, actions, substances, ages, signals
        API->>Norm: Chuẩn hóa slang / alias bằng graph
        API->>Rewrite: Tách truy vấn và viết lại truy vấn
        API->>Retrieve: Tìm ứng viên điều luật
        Retrieve->>Graph: Exact article, fulltext, condition, signal, vector optional
        Graph-->>Retrieve: Candidate articles
        Retrieve-->>API: Danh sách ứng viên đã fusion/rerank
        API->>Graph: fetch_contexts(article_codes)
        Graph-->>API: Điều luật, tội danh, điều kiện, khung phạt
        API->>Reason: So khớp dữ kiện với context pháp lý
        API->>Answer: Sinh câu trả lời và kiểm định hallucination
        Answer-->>API: final_answer, confidence, warnings, citations
        API-->>User: Kết quả phân tích
    end
```

## 10. Sơ đồ luồng chat pháp lý nhiều lượt

```mermaid
sequenceDiagram
    actor User as Người dùng
    participant Chat as /chat/legal
    participant Store as session_store
    participant Extract as fact_extractor
    participant Merge as fact_merger
    participant Gate as answer_gate
    participant Pipeline as legal_pipeline

    User->>Chat: Lượt 1: message, không có case_id
    Chat->>Store: Tạo CaseSession mới
    Chat->>Extract: Trích xuất dữ kiện lượt 1
    Chat->>Merge: Merge vào facts của vụ việc
    Chat->>Gate: Kiểm tra thiếu dữ kiện trọng yếu
    alt Thiếu dữ kiện
        Gate-->>Chat: collecting_facts + missing_facts
        Chat->>Store: Lưu session và turn
        Chat-->>User: Câu hỏi làm rõ + case_id
    else Đủ dữ kiện
        Gate-->>Chat: ready_to_answer
        Chat->>Pipeline: Chạy phân tích đầy đủ
        Pipeline-->>Chat: final_answer
        Chat->>Store: Lưu trạng thái answered
        Chat-->>User: Kết luận tham khảo + citations
    end

    User->>Chat: Lượt bổ sung: case_id + thông tin mới
    Chat->>Store: Nạp CaseSession cũ
    Chat->>Extract: Trích xuất dữ kiện mới
    Chat->>Merge: Gộp facts cũ và mới
    Chat->>Gate: Đánh giá lại khả năng trả lời
```

## 11. Sơ đồ hybrid retrieval

```mermaid
flowchart TD
    Query["Câu hỏi / tình huống"] --> Facts["ExtractedFacts"]
    Query --> Normalized["Normalized signals"]
    Facts --> Exact["Exact article search"]
    Normalized --> Signal["Graph signal search"]
    Facts --> Title["Legal action title search"]
    Query --> Rewrite["Query decomposition + rewriting"]
    Rewrite --> Fulltext["Neo4j fulltext search"]
    Rewrite --> Condition["Condition search"]
    Rewrite -. optional .-> Vector["Vector search"]

    Exact --> Rankings["Rankings"]
    Signal --> Rankings
    Title --> Rankings
    Fulltext --> Rankings
    Condition --> Rankings
    Vector --> Rankings

    Rankings --> RRF["Reciprocal Rank Fusion<br/>k = 60"]
    RRF --> Rerank["Cross-encoder rerank<br/>optional"]
    Rerank --> Candidates["Candidate articles"]
    Candidates --> Context["Fetch legal context"]
    Context --> Reasoning["Legal reasoning"]
```

## 12. Sơ đồ kiểm soát chất lượng câu trả lời

```mermaid
flowchart TD
    Context["Legal contexts<br/>Điều luật, điều kiện, khung phạt"] --> Generate["answer_generator"]
    Facts["Extracted facts"] --> Generate
    Reasoning["Legal reasoning"] --> Generate
    Missing["Missing facts"] --> Generate
    Questions["Clarifying questions"] --> Generate

    Generate --> Draft["Draft final_answer"]
    Draft --> Validate["validator"]
    Context --> Validate
    Missing --> Validate
    Reasoning --> Validate

    Validate --> SafeAnswer["Câu trả lời cuối cùng"]
    Validate --> Confidence["Confidence"]
    Validate --> Warnings["Warnings<br/>thiếu dữ kiện, ngoài phạm vi,<br/>không thay thế kết luận cơ quan có thẩm quyền"]
    Context --> Citations["Citations"]
```

## 13. Sơ đồ use case

```mermaid
flowchart LR
    Actor["Người dùng"] --> UC1["Tra cứu điều luật"]
    Actor --> UC2["Tìm kiếm pháp lý"]
    Actor --> UC3["Chuẩn hóa thuật ngữ đời thường"]
    Actor --> UC4["Phân tích tình huống một lượt"]
    Actor --> UC5["Chat pháp lý nhiều lượt"]

    UC1 --> E1["GET /articles/{article_code}"]
    UC2 --> E2["POST /search"]
    UC3 --> E3["POST /normalize"]
    UC4 --> E4["POST /analyze-scenario"]
    UC5 --> E5["POST /chat/legal"]

    E1 --> Graph["Neo4j BLHS Graph"]
    E2 --> Graph
    E3 --> Graph
    E4 --> Graph
    E5 --> Graph
```

## 14. Sơ đồ trạng thái phiên chat

```mermaid
stateDiagram-v2
    [*] --> NewCase: Người dùng gửi message mới
    NewCase --> CollectingFacts: Thiếu dữ kiện trọng yếu
    CollectingFacts --> CollectingFacts: Người dùng bổ sung nhưng vẫn thiếu
    CollectingFacts --> ReadyToAnswer: Đủ dữ kiện
    NewCase --> ReadyToAnswer: Đủ dữ kiện ngay từ đầu
    ReadyToAnswer --> Answered: Chạy legal_pipeline
    CollectingFacts --> InsufficientInformation: Người dùng không biết thêm / dừng hỏi lặp
    Answered --> [*]
    InsufficientInformation --> [*]
```

## 15. Sơ đồ bảo vệ chống kết luận quá mức

```mermaid
flowchart TD
    Input["Tình huống đầu vào"] --> Extract["Trích xuất dữ kiện"]
    Extract --> MissingCheck["detect_missing_facts"]
    MissingCheck --> Gate["answer_gate"]

    Gate -->|Thiếu dữ kiện trọng yếu| Ask["Sinh câu hỏi làm rõ"]
    Ask --> Collecting["status = collecting_facts"]

    Gate -->|Đủ dữ kiện| Analyze["Chạy retrieval + reasoning"]
    Analyze --> Validate["validate_answer"]

    Validate -->|Có căn cứ từ graph| Conclusion["Kết luận tham khảo có citation"]
    Validate -->|Thiếu căn cứ / rủi ro hallucination| Warning["Giảm confidence + thêm warning"]
```

