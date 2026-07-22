from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph import graph, route_after_grade, route_after_understand, route_after_verify
from src.nodes import basic_chat_response
from src.schemas import SourceInfo


def test_retry_route_is_bounded() -> None:
    assert route_after_grade({"evidence_status": "retry", "retry_count": 0}) == "broaden_query"
    assert route_after_grade({"evidence_status": "retry", "retry_count": 1}) == "generate_fallback"
    assert route_after_grade({"evidence_status": "sufficient"}) == "generate_answer"


def test_basic_chat_skips_retrieval_and_returns_no_artifacts() -> None:
    assert basic_chat_response("Hellow!")
    assert basic_chat_response("What is a Resource?") is None
    assert route_after_understand({"basic_chat": True}) == "end"

    result = graph.invoke(
        {"messages": [HumanMessage(content="hello")]},
        {"configurable": {"thread_id": f"pytest-{uuid4()}"}},
    )

    assert result["answer"].startswith("Hello!")
    assert result["sources"] == []
    assert result["retrieved_docs"] == []


def test_graph_has_expected_terminal_paths() -> None:
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert ("generate_diagram", "__end__") in edges
    assert ("generate_fallback", "__end__") in edges
    assert ("broaden_query", "retrieve_documents") in edges


def test_explicit_diagram_route_overrides_usefulness() -> None:
    assert route_after_verify({
        "diagram_enabled": True,
        "diagram_requested": True,
        "diagram_useful": False,
        "grounded": True,
    }) == "generate_diagram"


def test_diagram_route_honors_toggle_and_grounding() -> None:
    assert route_after_verify({
        "diagram_enabled": False,
        "diagram_requested": True,
        "grounded": True,
    }) == "end"
    assert route_after_verify({
        "diagram_enabled": True,
        "diagram_requested": True,
        "grounded": False,
    }) == "end"


def test_graph_uses_sqlite_thread_memory() -> None:
    assert isinstance(graph.checkpointer, SqliteSaver)
    snapshot = graph.get_state({"configurable": {"thread_id": "pytest-thread"}})

    assert snapshot.values == {}


def test_checkpoint_serializer_explicitly_allows_source_info() -> None:
    source = SourceInfo(source_id="S1", source="manual.pdf")

    restored = graph.checkpointer.serde.loads_typed(
        graph.checkpointer.serde.dumps_typed(source)
    )

    assert restored == source


def test_sqlite_checkpointer_allows_streamlit_threads() -> None:
    thread_ids = [f"pytest-{uuid4()}" for _ in range(8)]

    def read_state(thread_id: str) -> dict:
        return dict(
            graph.get_state({"configurable": {"thread_id": thread_id}}).values
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(read_state, thread_ids))

    assert states == [{} for _ in thread_ids]
