from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph import graph, route_after_grade


def test_retry_route_is_bounded() -> None:
    assert route_after_grade({"evidence_status": "retry", "retry_count": 0}) == "broaden_query"
    assert route_after_grade({"evidence_status": "retry", "retry_count": 1}) == "generate_fallback"
    assert route_after_grade({"evidence_status": "sufficient"}) == "generate_answer"


def test_graph_has_expected_terminal_paths() -> None:
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert ("generate_diagram", "__end__") in edges
    assert ("generate_fallback", "__end__") in edges
    assert ("broaden_query", "retrieve_documents") in edges


def test_graph_uses_sqlite_thread_memory() -> None:
    assert isinstance(graph.checkpointer, SqliteSaver)
    snapshot = graph.get_state({"configurable": {"thread_id": "pytest-thread"}})

    assert snapshot.values == {}


def test_sqlite_checkpointer_allows_streamlit_threads() -> None:
    thread_ids = [f"pytest-{uuid4()}" for _ in range(8)]

    def read_state(thread_id: str) -> dict:
        return dict(
            graph.get_state({"configurable": {"thread_id": thread_id}}).values
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(read_state, thread_ids))

    assert states == [{} for _ in thread_ids]
