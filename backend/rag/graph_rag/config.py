import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parents[1] if len(BASE_DIR.parents) > 1 else BASE_DIR

load_dotenv(REPO_ROOT / ".env")
load_dotenv(BASE_DIR / ".env", override=False)

CORPUS_PATH = BASE_DIR / "corpus" / "pddieu.csv"
CRAWLER_DATA_DIR = REPO_ROOT / "law-crawler" / "phap-dien"
GRAPH_RAG_DIR = BASE_DIR / "graph_rag_data"
STRUCTURE_DIR = GRAPH_RAG_DIR / "data"
FILTERED_CORPUS_PATH = GRAPH_RAG_DIR / "corpus_5_topics.csv"
HF_CACHE_DIR = Path(os.getenv("SAULAI_HF_CACHE_DIR") or os.getenv("HF_HOME") or REPO_ROOT / ".venv" / ".cache" / "huggingface")

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE_DIR / "sentence-transformers"))

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
NEO4J_VECTOR_INDEX_NAME = os.getenv("NEO4J_VECTOR_INDEX_NAME", "chunk_embedding_index")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_QA_MODEL = os.getenv("OPENAI_QA_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.5"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
GEMINI_QA_MODEL = os.getenv("GEMINI_QA_MODEL", os.getenv("GEMINI_ENRICH_MODEL", "gemini-2.5-flash"))
GEMINI_ENRICH_MODEL = os.getenv("GEMINI_ENRICH_MODEL", "gemini-2.5-flash")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "600"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "123456789")

LAW_SPECS = [
    {
        "id": "law_hinh_su",
        "name": "Hình sự",
        "field": "Hình sự",
        "mode": "criminal",
        "topic_text": "Hình sự",
        "demuc_texts": ["Hình sự"],
        "keywords": ["hình sự", "tội phạm", "trách nhiệm hình sự", "hình phạt"],
    },
    {
        "id": "law_dan_su",
        "name": "Dân sự",
        "field": "Dân sự",
        "mode": "civil",
        "topic_text": "Dân sự",
        "demuc_texts": [
            "Dân sự",
            "Đăng ký biện pháp bảo đảm",
            "Quy định thi hành Bộ luật Dân sự về bảo đảm thực hiện nghĩa vụ",
        ],
        "keywords": ["dân sự", "hợp đồng", "bồi thường", "thừa kế", "tài sản", "nghĩa vụ"],
    },
    {
        "id": "law_hon_nhan_gia_dinh",
        "name": "Hôn nhân và gia đình",
        "field": "Hôn nhân và gia đình",
        "mode": "family",
        "topic_text": "Dân số, gia đình, trẻ em, bình đẳng giới",
        "demuc_texts": ["Hôn nhân và gia đình"],
        "keywords": ["hôn nhân", "gia đình", "ly hôn", "nuôi con", "cấp dưỡng", "vợ chồng"],
    },
    {
        "id": "law_an_ninh_mang",
        "name": "An ninh mạng",
        "field": "An ninh mạng",
        "mode": "cybersecurity",
        "topic_text": "An ninh quốc gia",
        "demuc_texts": ["An ninh mạng"],
        "keywords": ["an ninh mạng", "không gian mạng", "dữ liệu", "thông tin mạng"],
    },
    {
        "id": "law_giao_duc",
        "name": "Giáo dục",
        "field": "Giáo dục",
        "mode": "education",
        "topic_text": "Giáo dục, đào tạo",
        "demuc_texts": ["Giáo dục", "Giáo dục đại học"],
        "keywords": ["giáo dục", "đào tạo", "học sinh", "sinh viên", "nhà giáo", "trường học"],
    },
]


def ensure_dirs():
    GRAPH_RAG_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
