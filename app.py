"""Streamlit chat UI backed by the compiled LangGraph and SQLite memory."""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
import streamlit as st

from src.graph import graph
from src.config import settings
from src.ingest import IndexSchemaMismatchError, REINGEST_COMMAND, validate_indexes
from src.llm import GroqRequestError


logger = logging.getLogger(__name__)
st.set_page_config(page_title="Opcenter Chatbot", page_icon="📘", layout="wide")
st.title("Opcenter Chatbot")
st.caption("Answers are grounded in the indexed Opcenter manuals.")


@st.cache_resource
def validate_search_indexes() -> int:
    """Validate deployment prerequisites before accepting a question."""
    settings.validate()
    if not settings.manuals_dir.is_dir() or not any(settings.manuals_dir.glob("*.pdf")):
        raise FileNotFoundError("manuals")
    required = (
        settings.evidence_units_path,
        settings.retrieval_segments_path,
        settings.bm25_path,
        settings.indexes_dir / "manifest.json",
        settings.chroma_dir / "chroma.sqlite3",
    )
    if any(not path.exists() for path in required):
        raise FileNotFoundError("indexes")
    return validate_indexes()


try:
    validate_search_indexes()
except EnvironmentError:
    st.error("GROQ_API_KEY is missing. Add it to .env before starting the chatbot.")
    st.stop()
except FileNotFoundError:
    st.error("Required manuals or retrieval indexes are unavailable.")
    st.info(
        "Place the PDF manuals in `manuals/`, then run the explicit re-ingestion command:"
    )
    st.code(REINGEST_COMMAND, language="bash")
    st.stop()
except IndexSchemaMismatchError:
    st.error("The document index schema changed. Existing indexes were left untouched.")
    st.info("Rebuild them explicitly before starting the chatbot:")
    st.code(REINGEST_COMMAND, language="bash")
    st.stop()
except Exception:
    logger.exception("Startup index validation failed")
    st.error("The retrieval indexes are invalid or not aligned.")
    st.info("Run the explicit re-ingestion command before starting the chatbot:")
    st.code(REINGEST_COMMAND, language="bash")
    st.stop()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())


def thread_config() -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def value(item: Any, field: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def render_history(messages: list[Any]) -> None:
    for message in messages:
        if isinstance(message, HumanMessage) or value(message, "type") == "human":
            role = "user"
        elif isinstance(message, AIMessage) or value(message, "type") == "ai":
            role = "assistant"
        else:
            continue
        with st.chat_message(role):
            st.markdown(str(value(message, "content", "")))


def cited_documents(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    documents = state.get("reranked_docs", [])
    cited: list[tuple[str, dict[str, Any]]] = []
    for source in state.get("sources", []):
        source_id = str(value(source, "source_id", ""))
        if not source_id.startswith("S") or not source_id[1:].isdigit():
            continue
        index = int(source_id[1:]) - 1
        if 0 <= index < len(documents):
            cited.append((source_id, documents[index]))
    return cited


def render_table(rows: list[list[Any]]) -> None:
    if not rows:
        return
    width = max(len(row) for row in rows)
    raw_headers = list(rows[0]) + [""] * (width - len(rows[0]))
    headers = [str(header).strip() or f"Column {index + 1}" for index, header in enumerate(raw_headers)]
    unique_headers: list[str] = []
    for index, header in enumerate(headers):
        unique_headers.append(header if header not in unique_headers else f"{header} {index + 1}")
    records = [
        dict(zip(unique_headers, list(row) + [""] * (width - len(row)), strict=True))
        for row in rows[1:]
    ]
    st.dataframe(records, hide_index=True, use_container_width=True)


def render_sources(state: dict[str, Any]) -> None:
    for source_id, document in cited_documents(state):
        metadata = document.get("metadata", {})
        source = next(
            (
                item
                for item in state.get("sources", [])
                if str(value(item, "source_id", "")) == source_id
            ),
            {},
        )
        manual = value(source, "manual") or metadata.get("manual") or metadata.get("source_file") or "Manual"
        printed_page = value(source, "printed_page") or metadata.get("printed_page") or "—"
        with st.expander(f"{source_id} · {manual} · page {printed_page}"):
            st.markdown(
                "\n".join(
                    [
                        f"- **Manual:** {manual}",
                        f"- **Release:** {value(source, 'release') or metadata.get('release') or '—'}",
                        f"- **Chapter:** {value(source, 'chapter') or metadata.get('chapter') or '—'}",
                        f"- **Section:** {value(source, 'section') or metadata.get('section') or '—'}",
                        f"- **Printed page:** {printed_page}",
                        f"- **PDF page:** {value(source, 'pdf_page') or metadata.get('pdf_page') or '—'}",
                        f"- **Content type:** {value(source, 'content_type') or document.get('content_type') or '—'}",
                    ]
                )
            )
            rows = metadata.get("table_rows")
            if document.get("content_type") in {"table", "field_definition"} and rows:
                st.markdown("**Structured table**")
                render_table(rows)


def render_artifacts(state: dict[str, Any], show_sources: bool) -> None:
    if state.get("diagram_dot"):
        st.graphviz_chart(state["diagram_dot"], use_container_width=True)
    if show_sources and state.get("sources"):
        st.subheader("Cited sources")
        render_sources(state)


PROGRESS_LABELS = {
    "understand_question": "Understanding question",
    "retrieve_documents": "Searching manuals",
    "expand_context": "Expanding context",
    "rerank_documents": "Reranking evidence",
    "grade_evidence": "Checking coverage",
    "generate_answer": "Preparing answer",
    "verify_answer": "Verifying answer",
    "generate_diagram": "Preparing diagram",
    "generate_fallback": "Preparing answer",
}


def updated_nodes(event: Any) -> list[str]:
    if not isinstance(event, dict):
        return []
    data = event.get("data") if event.get("type") == "updates" else event
    return [name for name in data if name in PROGRESS_LABELS] if isinstance(data, dict) else []


def answer_chunks(answer: str):
    """Stream only the completed, verified answer into the live chat message."""
    yield from (part for part in re.split(r"(\s+)", answer) if part)


with st.sidebar:
    st.header("Conversation")
    if st.button("New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid4())
        st.rerun()
    if st.button("Delete current conversation", use_container_width=True):
        graph.checkpointer.delete_thread(st.session_state.thread_id)
        st.session_state.thread_id = str(uuid4())
        st.rerun()
    show_sources = st.checkbox("Show sources", value=True)
    allow_diagrams = st.checkbox("Generate diagrams when useful", value=True)
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}`")

config = thread_config()
snapshot = graph.get_state(config)
stored_state = dict(snapshot.values)
render_history(list(stored_state.get("messages", [])))

prompt = st.chat_input("Ask a question about the Opcenter manuals")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.status("Searching and checking the manuals...", expanded=True) as status:
            seen_nodes: set[str] = set()
            for event in graph.stream(
                {
                    "messages": [HumanMessage(content=prompt)],
                    "retry_count": 0,
                    "allow_diagrams": allow_diagrams,
                },
                config=config,
                stream_mode="updates",
                version="v2",
            ):
                for node in updated_nodes(event):
                    if node not in seen_nodes:
                        status.write(PROGRESS_LABELS[node])
                        seen_nodes.add(node)
            final_state = dict(graph.get_state(config).values)
            status.update(label="Answer ready", state="complete", expanded=False)
    except GroqRequestError:
        logger.exception("Groq could not complete the RAG request")
        st.error("The language model could not complete this request. Please try again.")
    except Exception:
        logger.exception("The RAG request failed")
        st.error("The request could not be completed. Check the server logs and try again.")
    else:
        with st.chat_message("assistant"):
            st.write_stream(
                answer_chunks(final_state.get("answer", "No answer was generated."))
            )
            render_artifacts(final_state, show_sources)
elif stored_state.get("answer"):
    render_artifacts(stored_state, show_sources)
