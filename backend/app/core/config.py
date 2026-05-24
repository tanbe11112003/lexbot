from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123456"
    neo4j_database: str = "neo4j"

    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"

    use_llm_fact_extractor: bool = False
    use_vector_search: bool = False
    use_reranker: bool = False
    use_hyde: bool = False

    embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    rrf_k: int = 60
    cors_allow_origins: str = "*"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password123456"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        use_llm_fact_extractor=os.getenv("USE_LLM_FACT_EXTRACTOR", "false").lower() == "true",
        use_vector_search=os.getenv("USE_VECTOR_SEARCH", "false").lower() == "true",
        use_reranker=os.getenv("USE_RERANKER", "false").lower() == "true",
        use_hyde=os.getenv("USE_HYDE", "false").lower() == "true",
        embedding_model=os.getenv("EMBEDDING_MODEL", "bkai-foundation-models/vietnamese-bi-encoder"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        cors_allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*"),
    )


settings = get_settings()
