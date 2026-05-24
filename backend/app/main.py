from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging
from app.core.neo4j import neo4j_db
from app.routers import articles, chat, health, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        neo4j_db.ensure_indexes()
    except Exception:
        # Startup must not mutate beyond indexes and must explain failures at endpoints.
        pass
    yield
    neo4j_db.close()


app = FastAPI(
    title="BLHS Graph Chatbot Backend",
    version="2.0.0",
    description="FastAPI backend for Vietnamese criminal law scenario analysis over Neo4j.",
    lifespan=lifespan,
)

allow_origins = [origin.strip() for origin in settings.cors_allow_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(articles.router)
app.include_router(search.router)
app.include_router(chat.router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "BLHS Graph Chatbot Backend",
        "health": "/health",
        "docs": "/docs",
        "neo4j_database": settings.neo4j_database,
    }
