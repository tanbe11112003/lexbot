import atexit
import json
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1] if len(BASE_DIR.parents) > 1 else BASE_DIR

load_dotenv(REPO_ROOT / ".env")
load_dotenv(BASE_DIR / ".env", override=False)


def default_huggingface_cache_dir():
    return REPO_ROOT / ".venv" / ".cache" / "huggingface"


HF_CACHE_DIR = Path(os.getenv("SAULAI_HF_CACHE_DIR") or os.getenv("HF_HOME") or default_huggingface_cache_dir())
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR / "sentence-transformers"))

from flask import Flask, request, send_from_directory
from flask_cors import CORS
import jwt
from waitress import serve

from cache import redisClient, redis_ttl_seconds
from graph_rag.pipeline import GraphLawPipeline, SUPPORTED_CHAT_MODELS
from models import ACCESS_TOKEN_KEY, QuestionModel, Reference

app = Flask(__name__)
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

pipeline = None
RAG_CACHE_VERSION = "graph-rag-v4"
_redis_cache_cleared = False
WEB_DIR = Path(os.getenv("WEB_DIR") or REPO_ROOT / "web")
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "gpt-5.5")
QNA_THREADS = int(os.getenv("QNA_THREADS", "4"))


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = GraphLawPipeline()
    return pipeline


def normalize_chat_model(model):
    model = (model or DEFAULT_CHAT_MODEL or "gpt-5.5").strip()
    if model not in SUPPORTED_CHAT_MODELS:
        raise ValueError(f"Unsupported chat model: {model}")
    return model


def read_cached_answer(question, model):
    cached = redisClient.get(cache_key(question, model))
    if not cached:
        return None
    return json.loads(cached.decode("utf-8"))


def cache_key(question, model):
    return f"{RAG_CACHE_VERSION}:{model}:{question}"


def clear_redis_answer_cache():
    global _redis_cache_cleared
    if _redis_cache_cleared:
        return 0

    pattern = f"{RAG_CACHE_VERSION}:*"
    deleted = 0
    batch = []
    for key in redisClient.scan_iter(match=pattern, count=500):
        batch.append(key)
        if len(batch) >= 500:
            deleted += redisClient.delete(*batch)
            batch = []
    if batch:
        deleted += redisClient.delete(*batch)

    _redis_cache_cleared = True
    return deleted


def cleanup_on_shutdown():
    try:
        deleted = clear_redis_answer_cache()
        print(f"Cleared Redis answer cache: {deleted} keys")
    except Exception as error:
        print(f"Error while clearing Redis answer cache: {error}")


def handle_shutdown_signal(signum, frame):
    cleanup_on_shutdown()
    raise SystemExit(0)


def write_cached_answer(question, model, result):
    redisClient.setex(cache_key(question, model), redis_ttl_seconds, json.dumps(result, ensure_ascii=False))


def answer_question(question, model=None, should_generate=True, use_cache=True):
    model = normalize_chat_model(model)
    if should_generate and use_cache:
        cached = read_cached_answer(question, model)
        if cached:
            cached["cache_hit"] = True
            return cached

    result = get_pipeline().ask(question, model=model)
    result["model"] = model
    if not should_generate:
        result["response"] = ""

    if should_generate and use_cache:
        write_cached_answer(question, model, result)
    return result


def print_terminal_result(result):
    print("\n=== Graph RAG result ===")
    print(f"Question type: {result.get('question_type')}")
    print(f"Law ID: {result.get('law_id')}")
    if result.get("law_ids"):
        print("Law IDs:", ", ".join(result["law_ids"]))
    if result.get("search_queries"):
        print("Search queries:", ", ".join(result["search_queries"][:8]))
    if result.get("chunk_ids"):
        print("Chunk IDs:", ", ".join(result["chunk_ids"][:10]))
    if result.get("response"):
        print("\n=== Answer ===")
        if result.get("cache_hit"):
            print("(from Redis cache)")
        print(result["response"])
    print()


def run_terminal_test():
    print("Graph RAG terminal test mode")
    print("Type a question and press Enter. Commands: :q, :quit, :shutdown to quit and clear Redis cache.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {":q", ":quit", ":shutdown", "exit", "quit", "shutdown"}:
            cleanup_on_shutdown()
            break
        try:
            print_terminal_result(answer_question(question))
        except Exception as error:
            print(f"Error: {error}")


def decode_email_from_request():
    token = request.headers.get("Authorization", "")
    if token.startswith("Bearer "):
        token = token[7:]
    decoded = jwt.decode(token, ACCESS_TOKEN_KEY, algorithms=["HS256"])
    return decoded["email"]


def save_references(question_id, citation):
    for item in citation:
        Reference.create(
            **{
                "question_id": question_id,
                "mapc": item.get("id", ""),
                "noidung": item.get("content", ""),
                "ten": item.get("title", ""),
            }
        )


def build_chat_payload(question_rows, reference_rows):
    references_by_question = {}
    for reference in reference_rows:
        references_by_question.setdefault(reference["question_id"], []).append(
            {
                "id": reference["mapc"],
                "title": reference["ten"],
                "content": reference["noidung"],
                "demuc_id": None,
                "chuong_id": None,
            }
        )

    sessions = {}
    for row in sorted(question_rows, key=lambda item: (item.get("chat_id") or item["id"], item["id"])):
        chat_id = row.get("chat_id") or row["id"]
        answer = references_by_question.get(row["id"], [])
        turn = {
            "id": row["id"],
            "question_id": row["id"],
            "question": row["question"],
            "updatedAt": row["updatedAt"].strftime("%m/%d/%Y"),
            "response": row["response"],
            "model": row.get("model") or "",
            "answer": answer,
        }
        session = sessions.setdefault(
            chat_id,
            {
                "id": chat_id,
                "chat_id": chat_id,
                "email": row["email"],
                "question": row["question"],
                "updatedAt": row["updatedAt"].strftime("%m/%d/%Y"),
                "response": row["response"],
                "model": row.get("model") or "",
                "answer": answer,
                "messages": [],
                "_updatedAt": row["updatedAt"],
            },
        )
        session["messages"].append(turn)
        if row["updatedAt"] >= session["_updatedAt"]:
            session["updatedAt"] = row["updatedAt"].strftime("%m/%d/%Y")
            session["response"] = row["response"]
            session["model"] = row.get("model") or ""
            session["answer"] = answer
            session["_updatedAt"] = row["updatedAt"]

    res = sorted(sessions.values(), key=lambda item: item["_updatedAt"], reverse=True)
    for item in res:
        item.pop("_updatedAt", None)
    return res


def build_chat_summary_payload(question_rows):
    sessions = {}
    for row in sorted(question_rows, key=lambda item: (item.get("chat_id") or item["id"], item["id"])):
        chat_id = row.get("chat_id") or row["id"]
        session = sessions.setdefault(
            chat_id,
            {
                "id": chat_id,
                "chat_id": chat_id,
                "email": row["email"],
                "question": row["question"],
                "updatedAt": row["updatedAt"].strftime("%m/%d/%Y"),
                "response": "",
                "model": row.get("model") or "",
                "answer": [],
                "messages": [],
                "_updatedAt": row["updatedAt"],
            },
        )
        if row["updatedAt"] >= session["_updatedAt"]:
            session["updatedAt"] = row["updatedAt"].strftime("%m/%d/%Y")
            session["model"] = row.get("model") or ""
            session["_updatedAt"] = row["updatedAt"]

    res = sorted(sessions.values(), key=lambda item: item["_updatedAt"], reverse=True)
    for item in res:
        item.pop("_updatedAt", None)
    return res


def latest_chat_id_for_email(email):
    last_row = (
        QuestionModel.select(QuestionModel.chat_id, QuestionModel.id)
        .where(QuestionModel.email == email)
        .order_by(QuestionModel.updatedAt.desc(), QuestionModel.id.desc())
        .first()
    )
    if not last_row:
        return None
    return last_row.chat_id or last_row.id


@app.route("/", methods=["GET"])
def serve_web_index():
    response = send_from_directory(WEB_DIR, "index.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/web/<path:path>", methods=["GET"])
def serve_web_asset(path):
    response = send_from_directory(WEB_DIR, path)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/v1/models", methods=["GET"])
def get_models():
    return {"models": sorted(SUPPORTED_CHAT_MODELS), "default": normalize_chat_model(None)}, 200


@app.route("/api/v1/question", methods=["GET"])
def get_question(email=None):
    email = decode_email_from_request()
    question_rows = list(
        QuestionModel.select(
            QuestionModel.id,
            QuestionModel.email,
            QuestionModel.question,
            QuestionModel.updatedAt,
            QuestionModel.model,
            QuestionModel.chat_id,
        )
        .where(QuestionModel.email == email)
        .dicts()
    )
    if not question_rows:
        return [], 200

    return build_chat_summary_payload(question_rows), 200


@app.route("/api/v1/question/<int:chat_id>", methods=["GET"])
def get_chat(chat_id):
    try:
        email = decode_email_from_request()
    except Exception:
        return {"status": "error", "response": "Need authentication"}, 400
    question_rows = list(
        QuestionModel.select().where((QuestionModel.email == email) & (QuestionModel.chat_id == chat_id)).dicts()
    )
    if not question_rows:
        return {"status": "error", "response": "Chat not found"}, 404
    question_ids = [row["id"] for row in question_rows]
    reference_rows = list(Reference.select().where(Reference.question_id.in_(question_ids)).dicts())
    payload = build_chat_payload(question_rows, reference_rows)
    return (payload[0] if payload else {"id": chat_id, "chat_id": chat_id, "messages": []}), 200


@app.route("/api/v1/question", methods=["POST"])
def add_question():
    try:
        email = decode_email_from_request()
    except Exception:
        return {"status": "error", "response": "Need authentication"}, 400

    data = request.get_json() or {}
    question = data.get("question")
    chat_id = data.get("chat_id")
    new_chat = bool(data.get("new_chat"))
    try:
        model = normalize_chat_model(data.get("model"))
    except ValueError as error:
        return {"status": "error", "response": str(error)}, 400
    if not question:
        return {"status": "error", "response": "Question can not be empty"}, 400
    if chat_id:
        try:
            chat_id = int(chat_id)
        except (TypeError, ValueError):
            return {"status": "error", "response": "Invalid chat_id"}, 400
        chat = QuestionModel.get_or_none((QuestionModel.chat_id == chat_id) & (QuestionModel.email == email))
        if not chat:
            return {"status": "error", "response": "Chat not found"}, 404
    elif not new_chat:
        chat_id = latest_chat_id_for_email(email)

    try:
        result = answer_question(question, model=model)
    except Exception as error:
        return {"status": "error", "response": f"Error while running Graph RAG pipeline: {error}"}, 500

    query = QuestionModel.create(
        **{
            "email": email,
            "question": question,
            "response": result.get("response", ""),
            "model": model,
            "chat_id": chat_id,
        }
    )
    if not chat_id:
        chat_id = query.id
        query.chat_id = chat_id
        query.save()
    save_references(query.id, result.get("citation", []))
    result["question_id"] = query.id
    result["chat_id"] = chat_id
    return result, 200


@app.route("/api/v1/question-with-context", methods=["POST"])
def add_question_with_context():
    try:
        email = decode_email_from_request()
    except Exception:
        return {"status": "error", "response": "Need authentication"}, 400

    data = request.get_json() or {}
    question = data.get("question")
    context = data.get("context")
    try:
        model = normalize_chat_model(data.get("model"))
    except ValueError as error:
        return {"status": "error", "response": str(error)}, 400
    if not question:
        return {"status": "error", "response": "Question can not be empty"}, 400
    if not context:
        return {"status": "error", "response": "Context can not be empty"}, 400

    try:
        response = get_pipeline().generate_direct_answer(question, context=context, model=model)
    except Exception as error:
        return {"status": "error", "response": f"Error while generating answer: {error}"}, 500

    query = QuestionModel.create(**{"email": email, "question": question, "response": response, "model": model})
    query.chat_id = query.id
    query.save()
    result = {
        "status": "success",
        "question": question,
        "question_type": "direct-context",
        "law_id": None,
        "citation": [],
        "response": response,
        "model": model,
    }
    save_references(query.id, [])
    result["question_id"] = query.id
    result["chat_id"] = query.chat_id
    write_cached_answer(question, model, result)
    return result, 200


@app.route("/api/v1/question/<int:question_id>", methods=["PUT"])
def update_question(question_id):
    try:
        email = decode_email_from_request()
    except Exception:
        return {"status": "error", "response": "Need authentication"}, 400
    data = request.get_json()
    QuestionModel.update(**data).where((QuestionModel.id == question_id) & (QuestionModel.email == email)).execute()
    return "", 204


@app.route("/api/v1/question/<int:question_id>", methods=["DELETE"])
def delete_question(question_id):
    try:
        email = decode_email_from_request()
    except Exception:
        return {"status": "error", "response": "Need authentication"}, 400
    question = QuestionModel.get_or_none((QuestionModel.id == question_id) & (QuestionModel.email == email))
    if question:
        Reference.delete().where(Reference.question_id == question.id).execute()
        question.delete_instance()
    return "", 204


@app.route("/api/v1/question/chat/<int:chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    try:
        email = decode_email_from_request()
    except Exception:
        return {"status": "error", "response": "Need authentication"}, 400

    question_rows = list(
        QuestionModel.select().where((QuestionModel.email == email) & (QuestionModel.chat_id == chat_id)).dicts()
    )
    if not question_rows:
        return {"status": "error", "response": "Chat not found"}, 404

    question_ids = [row["id"] for row in question_rows]
    Reference.delete().where(Reference.question_id.in_(question_ids)).execute()
    QuestionModel.delete().where(QuestionModel.id.in_(question_ids)).execute()
    return "", 204


@app.route("/api/v1/account-data", methods=["DELETE"])
def delete_account_data():
    try:
        email = decode_email_from_request()
    except Exception:
        return {"status": "error", "response": "Need authentication"}, 400

    question_ids = [
        row.id
        for row in QuestionModel.select(QuestionModel.id).where(QuestionModel.email == email)
    ]
    if question_ids:
        Reference.delete().where(Reference.question_id.in_(question_ids)).execute()
        QuestionModel.delete().where(QuestionModel.id.in_(question_ids)).execute()
    return "", 204


if __name__ == "__main__":
    atexit.register(cleanup_on_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    if "--terminal" in sys.argv or "--test" in sys.argv:
        run_terminal_test()
    elif "--server" in sys.argv or len(sys.argv) == 1:
        print("Graph RAG QNA server is running at http://localhost:5001")
        serve(app, host="0.0.0.0", port=5001, threads=QNA_THREADS)
    else:
        print("Unknown mode. Use --server or --terminal.")
        raise SystemExit(2)
