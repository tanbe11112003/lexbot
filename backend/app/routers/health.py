from __future__ import annotations

from fastapi import APIRouter

from app.core.neo4j import neo4j_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    labels = ["Article", "Crime", "Clause", "Point", "Condition", "Rule", "PenaltyFrame", "Penalty"]
    try:
        counts = neo4j_db.count_labels(labels)
        return {"status": "ok", "neo4j_connected": True, "database_counts": counts}
    except Exception as exc:
        return {"status": "error", "neo4j_connected": False, "database_counts": {}, "error": str(exc)}
