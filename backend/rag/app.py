import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def default_huggingface_cache_dir():
    virtual_env = os.getenv("VIRTUAL_ENV")
    if virtual_env:
        return Path(virtual_env) / ".cache" / "huggingface"

    for parent in [BASE_DIR, *BASE_DIR.parents]:
        venv_path = parent / ".venv"
        if venv_path.exists():
            return venv_path / ".cache" / "huggingface"

    return BASE_DIR.parent.parent / ".cache" / "huggingface"


HF_CACHE_DIR = Path(os.getenv("SAULAI_HF_CACHE_DIR") or os.getenv("HF_HOME") or default_huggingface_cache_dir())
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR / "sentence-transformers"))

from flask import *
from flask_cors import CORS, cross_origin
from playhouse.shortcuts import model_to_dict
from models import *
from directory import *
from cache import *
import chromadb
import json
import jwt
import  re
from waitress import serve
import requests
from dotenv import load_dotenv
import sys
import torch
from sentence_transformers import SentenceTransformer


load_dotenv()


def env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_MODEL = "vistral-7b-chat"
QUERY_STRUCTURE_MODEL = os.getenv("QUERY_STRUCTURE_MODEL", "qwen2.5-7b-instruct")
USE_QUERY_STRUCTURE = env_bool("USE_QUERY_STRUCTURE", False)
EMBEDDING_MODEL = LLM_MODEL
EMBEDDING_PROVIDER = "sentence-transformer"
ST_MODEL_PATH = "keepitreal/vietnamese-sbert"
EMBEDDING_DEVICE = "auto"
LLM_TIMEOUT = env_int("LLM_TIMEOUT", 600)
LLM_TEMPERATURE = env_float("LLM_TEMPERATURE", 0.1)
LLM_TOP_P = env_float("LLM_TOP_P", 0.85)
RETRIEVAL_K = env_int("RETRIEVAL_K", 4)
MAX_CONTEXT_CHARS = env_int("MAX_CONTEXT_CHARS", 5000)
MAX_DOC_CHARS = env_int("MAX_DOC_CHARS", 700)
MAX_ANSWER_TOKENS = env_int("MAX_ANSWER_TOKENS", 768)
CHROMA_COLLECTION_NAME = "langchain"
QUERY_STRUCTURE_INTENTS = {
    "rights",
    "procedure",
    "penalty",
    "deadline",
    "condition",
    "definition",
    "authority",
    "unknown",
}
STRICT_FILTER_CONFIDENCE = 0.8
llm_session = requests.Session()
chroma_client = chromadb.PersistentClient(path=TOPIC_DB_PATH or "chroma_db_law")
vectordb = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
sentence_embedding_model = None

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def normalize_result_string(page_content):
    result_string = page_content or ""
    result_string = result_string.replace("\n", " ")
    result_string = re.sub(r"\s+", r" ", result_string)
    return result_string.strip()

def extract_json_object(text):
    if not text:
        raise ValueError("Empty model output")
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end + 1])

def empty_structured_question(question):
    return {
        "original_question": question,
        "search_query": normalize_result_string(question),
        "intent": "unknown",
        "legal_domain": None,
        "entities": {
            "actor": None,
            "opponent": None,
            "issue": None,
            "time": None,
            "amount": None,
        },
        "filters": {
            "document_type": None,
            "article_number": None,
            "demuc_id": None,
            "chuong_id": None,
        },
        "confidence": 0,
    }

def sanitize_structured_question(question, structured_question):
    fallback = empty_structured_question(question)
    if not isinstance(structured_question, dict):
        return fallback

    result = fallback
    result["original_question"] = str(structured_question.get("original_question") or question)
    search_query = normalize_result_string(structured_question.get("search_query") or question)
    result["search_query"] = search_query or normalize_result_string(question)

    intent = structured_question.get("intent")
    if intent not in QUERY_STRUCTURE_INTENTS:
        intent = "unknown"
    result["intent"] = intent

    legal_domain = structured_question.get("legal_domain")
    result["legal_domain"] = str(legal_domain).strip() if legal_domain else None

    entities = structured_question.get("entities") if isinstance(structured_question.get("entities"), dict) else {}
    for key in result["entities"]:
        value = entities.get(key)
        result["entities"][key] = str(value).strip() if value else None

    filters = structured_question.get("filters") if isinstance(structured_question.get("filters"), dict) else {}
    for key in result["filters"]:
        value = filters.get(key)
        result["filters"][key] = str(value).strip() if value else None

    try:
        confidence = float(structured_question.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0
    result["confidence"] = max(0, min(confidence, 1))
    return result

def structure_question(question):
    prompt = f"""Bạn là bộ phân tích truy vấn cho hệ thống RAG pháp luật Việt Nam.

Nhiệm vụ:
- Chuyển câu hỏi người dùng thành JSON hợp lệ.
- Không trả lời câu hỏi pháp luật.
- Không bịa điều luật, số điều, nghị định, thông tư nếu người dùng không nêu.
- Nếu không xác định được field nào, đặt null.
- search_query phải là câu truy vấn ngắn, rõ, dùng để tìm văn bản pháp luật.

intent chỉ được chọn một trong các giá trị:
- rights: hỏi quyền/lợi ích/nghĩa vụ
- procedure: hỏi phải làm gì, thủ tục, cách xử lý
- penalty: hỏi mức phạt/chế tài
- deadline: hỏi thời hạn
- condition: hỏi điều kiện
- definition: hỏi khái niệm
- authority: hỏi cơ quan/thẩm quyền
- unknown: không xác định

Không dùng null cho intent. Nếu không chắc, dùng "unknown".

Chỉ trả JSON. Không markdown.

Schema:
{{
  "original_question": string,
  "search_query": string,
  "intent": string,
  "legal_domain": string|null,
  "entities": {{
    "actor": string|null,
    "opponent": string|null,
    "issue": string|null,
    "time": string|null,
    "amount": string|null
  }},
  "filters": {{
    "document_type": string|null,
    "article_number": string|null,
    "demuc_id": string|null,
    "chuong_id": string|null
  }},
  "confidence": number
}}

Câu hỏi:
{question}"""
    payload = {
        "model": QUERY_STRUCTURE_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 512
    }
    try:
        output = llm_session.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        output.raise_for_status()
        content = output.json()["choices"][0]["message"]["content"].strip()
        return sanitize_structured_question(question, extract_json_object(content))
    except Exception:
        return empty_structured_question(question)

def normalize_document_type_filter(document_type):
    if not document_type:
        return None
    value = str(document_type).strip()
    mapping = {
        "luật": "Luật",
        "lq": "Luật",
        "nghị định": "Nghị định",
        "nđ": "Nghị định",
        "nd": "Nghị định",
        "thông tư": "Thông tư",
        "tt": "Thông tư",
        "quyết định": "Quyết định",
        "qđ": "Quyết định",
        "qd": "Quyết định",
    }
    return mapping.get(value.lower(), value)

def build_chroma_where(structured_question):
    if not structured_question or structured_question.get("confidence", 0) < STRICT_FILTER_CONFIDENCE:
        return None

    filters = structured_question.get("filters") or {}
    conditions = []

    document_type = normalize_document_type_filter(filters.get("document_type"))
    if document_type:
        conditions.append({"document_type": document_type})

    article_number = filters.get("article_number")
    if article_number:
        conditions.append({"article_number": str(article_number)})

    demuc_id = filters.get("demuc_id")
    if demuc_id:
        conditions.append({"demuc_id": str(demuc_id)})

    chuong_id = filters.get("chuong_id")
    if chuong_id:
        conditions.append({"chuong_id": str(chuong_id)})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}

def get_query_embedding(text):
    global sentence_embedding_model
    if EMBEDDING_PROVIDER == "sentence-transformer":
        if sentence_embedding_model is None:
            device = EMBEDDING_DEVICE
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            sentence_embedding_model = SentenceTransformer(
                ST_MODEL_PATH,
                device=device,
                cache_folder=str(HF_CACHE_DIR / "sentence-transformers"),
            )
        return sentence_embedding_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    output = llm_session.post(
        f"{LLM_BASE_URL}/embeddings",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT,
    )
    output.raise_for_status()
    data = output.json()["data"][0]["embedding"]
    return data

def query_vectordb(query_embedding, k, where=None):
    query = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": ["documents", "metadatas"]
    }
    if where:
        query["where"] = where
    return vectordb.query(**query)

def retrieve_docs(question, k=RETRIEVAL_K, structured_question=None):
    query_embedding = get_query_embedding(question)
    where = build_chroma_where(structured_question)
    if EMBEDDING_PROVIDER == "sentence-transformer":
        try:
            output = query_vectordb(query_embedding, k, where=where)
            if where and not output.get("documents", [[]])[0]:
                output = query_vectordb(query_embedding, k)
        except Exception:
            output = query_vectordb(query_embedding, k)
    else:
        try:
            output = query_vectordb(query_embedding, k, where=where)
            if where and not output.get("documents", [[]])[0]:
                output = query_vectordb(query_embedding, k)
        except Exception:
            output = query_vectordb(query_embedding, k)
    docs = []
    documents = output.get("documents", [[]])[0]
    metadatas = output.get("metadatas", [[]])[0]
    for index, page_content in enumerate(documents):
        metadata = metadatas[index] or {}
        result_string = normalize_result_string(page_content)[:MAX_DOC_CHARS]
        docs.append({
            "id": metadata.get("id", ""),
            "title": metadata.get("title", ""),
            "content": result_string,
            "demuc_id": metadata.get("demuc_id"),
            "chuong_id": metadata.get("chuong_id"),
            "article_id": metadata.get("article_id"),
            "article_number": metadata.get("article_number"),
            "article_code": metadata.get("article_code"),
            "article_heading": metadata.get("article_heading"),
            "document_type": metadata.get("document_type"),
            "document_type_code": metadata.get("document_type_code"),
            "chunk_index": metadata.get("chunk_index"),
            "chunk_total": metadata.get("chunk_total"),
        })
    return docs

def generate_response(question, context):
    prompt = f"""Bạn đang trả lời một tình huống pháp luật bằng tiếng Việt.
Dựa trên các căn cứ pháp luật được cung cấp bên dưới, hãy:
- Xác định phần căn cứ nào liên quan trực tiếp hoặc gián tiếp đến tình huống.
- Trả lời thực tế người hỏi nên làm gì tiếp theo.
- Nêu rõ nếu căn cứ chỉ hỗ trợ một phần và cần thêm giấy tờ/thông tin.
- Không bịa số điều hoặc nội dung không có trong căn cứ.
- Không đưa mã định danh nội bộ trong ngoặc vào câu trả lời, ví dụ id hoặc chuỗi số dài.

Các căn cứ pháp luật:
{context[:MAX_CONTEXT_CHARS]}

Câu hỏi/tình huống:
{question}"""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Bạn là trợ lý pháp luật Việt Nam. Trả lời có cấu trúc, thực tế, chỉ dựa trên căn cứ được cung cấp và nói rõ mức độ chắc chắn của căn cứ."},
            {"role": "user", "content": prompt}
        ],
        "temperature": LLM_TEMPERATURE,
        "top_p": LLM_TOP_P,
        "max_tokens": MAX_ANSWER_TOKENS,
    }
    output = llm_session.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=LLM_TIMEOUT,
    )
    output.raise_for_status()
    return output.json()["choices"][0]["message"]["content"].strip()

def build_context(citation):
    parts = []
    for index, item in enumerate(citation, start=1):
        title = item.get("title") or item.get("id") or f"Căn cứ {index}"
        heading = f"[{index}] {title}"
        parts.append(f"{heading}\n{item.get('content', '')}")
    return "\n\n".join(parts).strip()[:MAX_CONTEXT_CHARS]

def read_cached_answer(question):
    cached = redisClient.get(question)
    if not cached:
        return None
    return json.loads(cached.decode("utf-8"))

def write_cached_answer(question, result):
    redisClient.setex(question, redis_ttl_seconds, json.dumps(result, ensure_ascii=False))

def answer_question(question, should_generate=True, use_cache=True):
    if should_generate and use_cache:
        cached = read_cached_answer(question)
        if cached:
            cached["cache_hit"] = True
            return cached

    structured_question = structure_question(question) if USE_QUERY_STRUCTURE else empty_structured_question(question)
    retrieval_query = structured_question.get("search_query") or question
    citation = retrieve_docs(retrieval_query, structured_question=structured_question)
    context = build_context(citation)
    if not context:
        raise RuntimeError("No context retrieved from Chroma DB")
    response = generate_response(question, context) if should_generate else ""
    result = {
        "status": "success",
        "question": question,
        "structured_question": structured_question,
        "citation": citation,
        "response": response,
    }
    if should_generate and use_cache:
        write_cached_answer(question, result)
    return result

def print_terminal_result(result):
    print("\n=== Retrieved documents ===")
    for index, citation in enumerate(result["citation"], start=1):
        title = citation.get("title") or citation.get("id") or f"Document {index}"
        print(f"\n[{index}] {title}")
        print(citation.get("content", ""))

    if result.get("response"):
        print("\n=== Answer ===")
        if result.get("cache_hit"):
            print("(from Redis cache)")
        print(result["response"])
    print()

def run_terminal_test():
    print("QNA terminal test mode")
    print("Type a question and press Enter.")
    print("Commands: :q to quit, :retrieve to toggle retrieval-only mode.")
    retrieval_only = False

    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {":q", ":quit", "exit", "quit"}:
            break
        if question.lower() in {":retrieve", ":retrieval"}:
            retrieval_only = not retrieval_only
            mode = "retrieval only" if retrieval_only else "retrieval + generation"
            print(f"Mode: {mode}")
            continue

        try:
            result = answer_question(question, should_generate=not retrieval_only)
            print_terminal_result(result)
        except Exception as error:
            print(f"Error: {error}")

def save_references(question_id, citation):
    for item in citation:
        Reference.create(**{
            'question_id': question_id,
            'mapc': item.get('id', ''),
            'noidung': item.get('content', ''),
            'ten': item.get('title', ''),
        })

@app.route('/api/v1/question', methods=['GET'])
def get_question(email=None):
    token = request.headers.get('Authorization')

    if token.startswith('Bearer '):
        token = token[7:]
        
    decoded = jwt.decode(token, ACCESS_TOKEN_KEY, algorithms=['HS256'])
    email = decoded['email']
    if email:
        query = QuestionModel.select().where(email == QuestionModel.email).dicts()
        res = []
        for row in query:
            answer = []
            query1 = Reference.select().where(row['id'] == Reference.question_id).dicts()
            for r in query1: 
                answer.append({
                    "id": r['mapc'],
                    "title": r['ten'],
                    "content": r['noidung'],
                    "demuc_id": None,
                    "chuong_id": None,
                })
            res.append({
                "id": row['id'],
                "email": row['email'],
                "question": row['question'],
                "updatedAt": row['updatedAt'].strftime("%m/%d/%Y"),
                "response": row['response'],
                "answer": answer
            })
        return res, 201
    
@app.route('/api/v1/question', methods=['POST'])
def add_question():
    token = request.headers.get('Authorization')

    if token.startswith('Bearer '):
        token = token[7:]
    data = request.get_json()
    decoded = jwt.decode(token, ACCESS_TOKEN_KEY, algorithms=['HS256'])

    email = decoded['email']

    try:
        question = data["question"]
    except:
        return {
            "status": "error",
            "response": "No question in payload",
        }, 400
    
    if not question:
        return {
            "status": "error",
            "response": "Question can not be empty",
        }, 400

    try:
        result = answer_question(question)
    except Exception:
        return {
            "status": "error",
            "response": "Error while retrieving context from DB or generating answer",
        }, 500

    citation = result["citation"]
    response = result["response"]
    query = QuestionModel.create(**{"email": email, "question": question ,"response": response})
    save_references(query.id, citation)
    return result, 200

@app.route('/api/v1/question-with-context', methods=['POST'])
def add_question_with_context():
    try: 
        token = request.headers.get('Authorization')

        if token.startswith('Bearer '):
            token = token[7:]
        data = request.get_json()
        decoded = jwt.decode(token, ACCESS_TOKEN_KEY, algorithms=['HS256'])

        email = decoded['email']
    except: 
         return {
            "status": "error",
            "response": "Need authencation",
        }, 400
         
    try:
        question = data["question"]
        context = data["context"]
    except:
        return {
            "status": "error",
            "response": "Question or Context not found in the payload",
        }, 400
    
    if not question:
        return {
            "status": "error",
            "response": "Question can not be empty",
        }, 400
    if not context:
        return {
            "status": "error",
            "response": "Context can not be empty",
        }, 400
    cached = read_cached_answer(question)
    if cached:
        cached["cache_hit"] = True
        return cached, 200

    structured_question = structure_question(question) if USE_QUERY_STRUCTURE else empty_structured_question(question)
    retrieval_query = structured_question.get("search_query") or question
    
    try:
        citation = retrieve_docs(retrieval_query, structured_question=structured_question)
    except Exception:
        return {
            "status": "error",
            "response": "Error while retrieving context from DB",
        }, 500

    try:
        response = generate_response(question, context)
    except Exception:
        return {
            "status": "error",
            "response": "Error while generating answer",
        }, 500

    query = QuestionModel.create(**{"email": email, "question": question ,"response": response})
    save_references(query.id, citation)
    res = {
        "status": "success",
        "question": question,
        "structured_question": structured_question,
        "citation": citation,
        "response": response,
    }
    write_cached_answer(question, res)
    return res, 200


@app.route('/api/v1/question/<int:question_id>', methods=['PUT'])
def update_question(question_id):
    data = request.get_json()
    QuestionModel.update(**data).where(QuestionModel.id == question_id).execute()
    return '', 204

@app.route('/api/v1/question/<int:question_id>', methods=['DELETE'])
def delete_question(question_id):
    QuestionModel.delete().where(QuestionModel.id == question_id).execute()
    return '', 204

if __name__ == "__main__":
    if "--terminal" in sys.argv or "--test" in sys.argv:
        run_terminal_test()
    else:
        print('QNA server is running. ')
        serve(app, host='0.0.0.0', port=5001, threads=1)
