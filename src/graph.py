"""Compiled LangGraph workflow with SQLite conversation memory."""

from __future__ import annotations

import sqlite3
from typing import Literal

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import settings
from src.nodes import (
    broaden_query,
    expand_context,
    generate_answer,
    generate_diagram,
    generate_fallback,
    grade_evidence,
    rerank_documents,
    retrieve_documents,
    understand_question,
    verify_answer,
)
from src.schemas import RAGState


Route = Literal["generate_answer", "broaden_query", "generate_fallback"]


def route_after_grade(state: RAGState) -> Route:
    """Route once toward answer, one retry, or fallback."""
    if state.get("evidence_status") in {"sufficient", "partial"}:
        return "generate_answer"
    if state.get("evidence_status") == "retry" and state.get("retry_count", 0) == 0:
        return "broaden_query"
    return "generate_fallback"


def _create_checkpointer() -> SqliteSaver:
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


def build_graph(checkpointer: SqliteSaver | None = None) -> CompiledStateGraph:
    workflow = StateGraph(RAGState)
    workflow.add_node("understand_question", understand_question)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("expand_context", expand_context)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("grade_evidence", grade_evidence)
    workflow.add_node("broaden_query", broaden_query)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("verify_answer", verify_answer)
    workflow.add_node("generate_diagram", generate_diagram)
    workflow.add_node("generate_fallback", generate_fallback)

    workflow.add_edge(START, "understand_question")
    workflow.add_edge("understand_question", "retrieve_documents")
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
    workflow.add_edge("verify_answer", "generate_diagram")
    workflow.add_edge("generate_diagram", END)
    workflow.add_edge("generate_fallback", END)
    return workflow.compile(checkpointer=checkpointer or _create_checkpointer())


graph = build_graph()
