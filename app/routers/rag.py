"""Endpoint /rag/query (JSON) va /rag/query/stream (SSE)."""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse
from app.pipeline.orchestrator import run_pipeline, run_pipeline_fast, run_pipeline_stream
from app.pipeline.pdf_textbook import run_pdf_lookup_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query", response_model=ChatResponse)
async def rag_query(request: ChatRequest) -> ChatResponse:
    """Goi pipeline RAG va tra ve ChatResponse day du.

    Tuong thich voi backend /chat/query da forward sang day.
    """
    try:
        if request.chat_mode == "tra_cuu_pdf":
            return await asyncio.to_thread(
                run_pdf_lookup_pipeline,
                request.question,
                request.include_debug,
            )
        if request.chat_mode == "phan_tich":
            return await asyncio.to_thread(
                run_pipeline,
                request.question,
                request.top_k,
                request.include_debug,
            )
        runner = run_pipeline_fast if request.query_mode == "fast" else run_pipeline
        return await asyncio.to_thread(
            runner,
            question=request.question,
            top_k=request.top_k,
            include_debug=request.include_debug,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Loi pipeline RAG")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _format_sse(event: str, data: dict) -> bytes:
    """Encode 1 sse event."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


@router.post("/query/stream")
async def rag_query_stream(request: ChatRequest):
    """Stream tien trinh xu ly qua SSE.

    Browser hoac client co the dung EventSource de doc tung su kien stage.
    """

    async def _generator():
        try:
            async for event in run_pipeline_stream(
                question=request.question,
                top_k=request.top_k,
                include_debug=request.include_debug,
                chat_mode=request.chat_mode,
            ):
                yield _format_sse(event.stage, event.payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Stream RAG loi")
            yield _format_sse("error", {"detail": str(exc)})

    return StreamingResponse(_generator(), media_type="text/event-stream")
