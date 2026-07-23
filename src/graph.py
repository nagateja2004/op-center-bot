"""Compiled LangGraph workflow with pluggable conversation checkpoints."""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import settings
from src.nodes import (
    abroaden_query,
    aretrieve_documents,
    arerank_documents,
    agenerate_answer,
    agenerate_diagram,
    agrade_evidence,
    aunderstand_question,
    averify_answer,
    broaden_query,
    expand_context,
    generate_fallback,
    generate_answer,
    generate_diagram,
    grade_evidence,
    rerank_documents,
    retrieve_documents,
    understand_question,
    verify_answer,
)
from src.schemas import RAGState
from src.observability import timed_async_node, timed_sync_node


Route = Literal["generate_answer", "broaden_query", "generate_fallback"]
InitialRoute = Literal["retrieve_documents", "end"]
DiagramRoute = Literal["generate_diagram", "end"]
logger = logging.getLogger(__name__)


def route_after_understand(state: RAGState) -> InitialRoute:
    return "end" if state.get("basic_chat", False) else "retrieve_documents"


def route_after_grade(state: RAGState) -> Route:
    """Route once toward answer, one retry, or fallback."""
    if state.get("evidence_status") in {"sufficient", "partial"}:
        return "generate_answer"
    if state.get("evidence_status") == "retry" and state.get("retry_count", 0) == 0:
        return "broaden_query"
    return "generate_fallback"


def route_after_verify(state: RAGState) -> DiagramRoute:
    """Route grounded answers to an explicitly requested or useful diagram."""
    enabled = state.get("diagram_enabled", state.get("allow_diagrams", True))
    useful = state.get("diagram_useful", state.get("needs_diagram", False))
    route = (
        "generate_diagram"
        if enabled
        and state.get("grounded", False)
        and (state.get("diagram_requested", False) or useful)
        else "end"
    )
    logger.info(
        "diagram_route requested=%s enabled=%s type=%s grounded=%s useful=%s decision=%s",
        state.get("diagram_requested", False),
        enabled,
        state.get("requested_diagram_type", "auto"),
        state.get("grounded", False),
        useful,
        route,
    )
    return route


def _create_local_checkpointer() -> BaseCheckpointSaver:
    if settings.checkpoint_backend != "sqlite":
        raise RuntimeError("A checkpointer must be provided outside local SQLite mode.")

    import sqlite3

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        settings.sqlite_path,
        check_same_thread=False,
        timeout=30,
    )
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    saver = SqliteSaver(
        connection,
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=[("src.schemas", "SourceInfo")]
        ),
    )
    saver.setup()
    return saver


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    workflow = StateGraph(RAGState)
    def node(name, sync_operation, async_operation=None):
        return RunnableLambda(
            timed_sync_node(name, sync_operation),
            afunc=timed_async_node(name, async_operation) if async_operation else None,
        )

    workflow.add_node("understand_question", node("understand_question", understand_question, aunderstand_question))
    workflow.add_node("retrieve_documents", node("retrieve_documents", retrieve_documents, aretrieve_documents))
    workflow.add_node("expand_context", node("expand_context", expand_context))
    workflow.add_node("rerank_documents", node("rerank_documents", rerank_documents, arerank_documents))
    workflow.add_node("grade_evidence", node("grade_evidence", grade_evidence, agrade_evidence))
    workflow.add_node("broaden_query", node("broaden_query", broaden_query, abroaden_query))
    workflow.add_node("generate_answer", node("generate_answer", generate_answer, agenerate_answer))
    workflow.add_node("verify_answer", node("verify_answer", verify_answer, averify_answer))
    workflow.add_node("generate_diagram", node("generate_diagram", generate_diagram, agenerate_diagram))
    workflow.add_node("generate_fallback", node("generate_fallback", generate_fallback))

    workflow.add_edge(START, "understand_question")
    workflow.add_conditional_edges(
        "understand_question",
        route_after_understand,
        {"retrieve_documents": "retrieve_documents", "end": END},
    )
    workflow.add_edge("retrieve_documents", "expand_context")
    workflow.add_edge("expand_context", "rerank_documents")
    workflow.add_edge("rerank_documents", "grade_evidence")
    workflow.add_conditional_edges(
        "grade_evidence",
        route_after_grade,
        {
            "generate_answer": "generate_answer",
            "broaden_query": "broaden_query",
            "generate_fallback": "generate_fallback",
        },
    )
    workflow.add_edge("broaden_query", "retrieve_documents")
    workflow.add_edge("generate_answer", "verify_answer")
    workflow.add_conditional_edges(
        "verify_answer",
        route_after_verify,
        {"generate_diagram": "generate_diagram", "end": END},
    )
    workflow.add_edge("generate_diagram", END)
    workflow.add_edge("generate_fallback", END)
    return workflow.compile(
        checkpointer=checkpointer if checkpointer is not None else _create_local_checkpointer()
    )


graph = build_graph() if settings.checkpoint_backend == "sqlite" else None
