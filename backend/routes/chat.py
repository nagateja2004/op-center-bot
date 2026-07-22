"""Chat request and server-sent event routes."""

from __future__ import annotations

import base64
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.dependencies import get_graph
from src.groq_limits import get_groq_limiter, groq_request_scope
from src.inference import InferenceBusyError
from src.observability import observe
from src.config import settings


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(max_length=8_000)
    session_id: UUID | None = None
    conversation_id: UUID | None = None
    thread_id: UUID | None = None
    diagram_enabled: bool = True
    diagram_type: Literal["auto", "hierarchy", "relationship", "process", "decision", "architecture"] = "auto"

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value

    @model_validator(mode="after")
    def identifiers_must_be_paired(self):
        if (self.conversation_id is None) != (self.thread_id is None):
            raise ValueError("conversation_id and thread_id must be provided together")
        return self


class ChatAccepted(BaseModel):
    request_id: str
    session_id: str
    conversation_id: str
    thread_id: str


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _updated_nodes(event: Any) -> list[str]:
    if not isinstance(event, dict):
        return []
    data = event.get("data") if event.get("type") == "updates" else event
    return list(data) if isinstance(data, dict) else []


def _source_payload(state: dict[str, Any]) -> list[dict[str, Any]]:
    documents = state.get("reranked_docs", [])
    payload: list[dict[str, Any]] = []
    for source in state.get("sources", []):
        item = source.model_dump() if hasattr(source, "model_dump") else dict(source)
        source_id = str(item.get("source_id", ""))
        if source_id.startswith("S") and source_id[1:].isdigit():
            index = int(source_id[1:]) - 1
            if 0 <= index < len(documents):
                document = documents[index]
                item["table_rows"] = document.get("metadata", {}).get("table_rows", [])
        payload.append(item)
    return payload


@lru_cache(maxsize=2)
def _load_figure_catalog(path: str, modified_ns: int) -> list[dict[str, Any]]:
    del modified_ns
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _manual_figure_payload(state: dict[str, Any]) -> list[dict[str, Any]]:
    if not (
        state.get("diagram_requested")
        or state.get("diagram_useful", state.get("needs_diagram", False))
    ):
        return []
    catalog_path = settings.manual_figures_path
    try:
        catalog = _load_figure_catalog(str(catalog_path), catalog_path.stat().st_mtime_ns)
    except OSError:
        return []
    source_pages: list[tuple[str, int]] = []
    for source in state.get("sources", []):
        item = source.model_dump() if hasattr(source, "model_dump") else dict(source)
        if item.get("source_file") and item.get("pdf_page"):
            source_pages.append((str(item["source_file"]), int(item["pdf_page"])))
    for document in state.get("reranked_docs", []):
        metadata = document.get("metadata", {})
        if metadata.get("source_file") and metadata.get("pdf_page"):
            source_pages.append((str(metadata["source_file"]), int(metadata["pdf_page"])))
    ranked_pages = list(dict.fromkeys(source_pages))
    figures_root = settings.manual_figures_dir.resolve()
    payload: list[dict[str, Any]] = []
    for source_file, pdf_page in ranked_pages:
        for figure in catalog:
            if (
                figure.get("source_file") != source_file
                or int(figure.get("pdf_page", 0)) != pdf_page
            ):
                continue
            image_path = (settings.indexes_dir / str(figure.get("path", ""))).resolve()
            if not image_path.is_relative_to(figures_root):
                continue
            try:
                image = image_path.read_bytes()
            except OSError:
                continue
            if not image or len(image) > 2_000_000:
                continue
            payload.append(
                {
                    "figure_id": str(figure.get("figure_id", "")),
                    "manual": str(figure.get("manual", "")),
                    "pdf_page": pdf_page,
                    "caption": str(figure.get("caption", "Diagram from the manual")),
                    "image_base64": base64.b64encode(image).decode("ascii"),
                }
            )
            if len(payload) == 3:
                return payload
    return payload


def _align_answer_with_manual_figures(
    answer: str, manual_figures: list[dict[str, Any]]
) -> str:
    if not manual_figures:
        return answer
    return re.sub(
        r"(?im)^.*(?:no (?:original )?diagram|no diagram is included|diagram is not included).*$",
        "**Original manual diagram** – Shown below.",
        answer,
    )


@router.post("/chat", response_model=ChatAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_chat(payload: ChatRequest, request: Request) -> ChatAccepted:
    request_id = getattr(request.state, "request_id", str(uuid4()))
    session_id = payload.session_id or uuid4()
    conversation_id = payload.conversation_id or uuid4()
    thread_id = payload.thread_id or uuid4()
    store = getattr(request.app.state, "request_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Chat service is not ready.")
    if not await store.claim_thread(str(thread_id), str(session_id)):
        raise HTTPException(status_code=403, detail="Conversation is not available in this session.")
    accepted_payload = payload.model_copy(
        update={"session_id": session_id, "conversation_id": conversation_id, "thread_id": thread_id}
    )
    await store.save_request(request_id, accepted_payload.model_dump_json())
    await get_groq_limiter().set_status(request_id, "accepted")
    logger.info(
        "chat_accepted request_id=%s session_id=%s conversation_id=%s",
        request_id,
        session_id,
        conversation_id,
    )
    return ChatAccepted(
        request_id=request_id,
        session_id=str(session_id),
        conversation_id=str(conversation_id),
        thread_id=str(thread_id),
    )


@router.get("/chat/{request_id}/stream")
async def stream_chat(
    request_id: str,
    request: Request,
    graph=Depends(get_graph),
) -> StreamingResponse:
    store = getattr(request.app.state, "request_store", None)
    raw_payload = await store.pop_request(request_id) if store is not None else None
    if raw_payload is None:
        raise HTTPException(status_code=404, detail="Unknown or already streamed request")
    payload = ChatRequest.model_validate_json(raw_payload)

    async def events():
        graph_started = time.perf_counter()
        config = {
            "configurable": {
                "thread_id": f"{payload.conversation_id}:{payload.thread_id}",
                "conversation_id": str(payload.conversation_id),
                "client_thread_id": str(payload.thread_id),
            }
        }
        try:
            await get_groq_limiter().set_status(request_id, "running")
            with groq_request_scope(request_id):
                async for event in graph.astream(
                    {
                        "messages": [HumanMessage(content=payload.message)],
                        "retry_count": 0,
                        "diagram_enabled": payload.diagram_enabled,
                        "diagram_type_override": payload.diagram_type,
                    },
                    config=config,
                    stream_mode="updates",
                    version="v2",
                ):
                    for node in _updated_nodes(event):
                        yield _sse("progress", {"node": node})

                state = dict((await graph.aget_state(config)).values)
            answer = state.get("answer", "No answer was generated.")
            basic_chat = bool(state.get("basic_chat"))
            sources = [] if basic_chat else _source_payload(state)
            manual_figures = [] if basic_chat else _manual_figure_payload(state)
            answer = _align_answer_with_manual_figures(answer, manual_figures)
            for chunk in re.findall(r"\S+\s*", answer):
                yield _sse("answer", {"text": chunk})
            yield _sse(
                "complete",
                {
                    "answer": answer,
                    "sources": sources,
                    "evidence": {
                        "status": "" if basic_chat else state.get("evidence_status", ""),
                        "manuals": list(dict.fromkeys(
                            str(source.get("manual") or source.get("source") or "")
                            for source in sources
                            if source.get("manual") or source.get("source")
                        )),
                        "sections": list(dict.fromkeys(
                            str(source.get("section"))
                            for source in sources
                            if source.get("section")
                        )),
                    },
                    "diagram": {
                        "generated": False if basic_chat or manual_figures else bool(state.get("diagram_generated")),
                        "dot": "" if basic_chat or manual_figures else _safe_dot(state.get("diagram_dot", "")),
                    },
                    "manual_figures": manual_figures,
                },
            )
            await get_groq_limiter().set_status(request_id, "complete")
        except InferenceBusyError:
            await get_groq_limiter().set_status(request_id, "rejected", reason="inference_busy")
            yield _sse("error", {"message": "The service is busy. Please try again shortly."})
        except Exception:
            logger.error(
                "rag_request_failed request_id=%s session_id=%s conversation_id=%s",
                request_id,
                payload.session_id,
                payload.conversation_id,
            )
            await get_groq_limiter().set_status(request_id, "failed")
            yield _sse(
                "error",
                {"message": "The request could not be completed. Please try again."},
            )
        finally:
            duration = time.perf_counter() - graph_started
            observe(
                "opcenter_langgraph_total_duration_seconds",
                duration,
            )
            logger.info(
                "langgraph_total request_id=%s session_id=%s conversation_id=%s duration_ms=%.1f",
                request_id,
                payload.session_id,
                payload.conversation_id,
                duration * 1000,
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _safe_dot(dot: str) -> str:
    if len(dot) > 20_000 or not re.fullmatch(r"\s*digraph\s+[\w-]+\s*\{[\s\S]*\}\s*", dot):
        return ""
    return dot
