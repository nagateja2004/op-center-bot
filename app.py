"""Streamlit UI for the FastAPI-backed Opcenter chatbot."""

from __future__ import annotations

import json
import os
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
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


def _request(path: str, payload: dict[str, Any] | None = None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"{BACKEND_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    return urlopen(request, timeout=120)


def create_chat(
    message: str,
    session_id: str,
    conversation_id: str | None,
    thread_id: str | None,
    diagram_enabled: bool,
    diagram_type: str,
) -> dict[str, str]:
    with _request(
        "/v1/chat",
        {
            "message": message,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "diagram_enabled": diagram_enabled,
            "diagram_type": diagram_type,
        },
    ) as response:
        return json.load(response)


def stream_events(request_id: str) -> Iterator[tuple[str, dict[str, Any]]]:
    with _request(f"/v1/chat/{request_id}/stream") as response:
        event = "message"
        data: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
            elif not line and data:
                yield event, json.loads("\n".join(data))
                event, data = "message", []


def render_table(rows: list[list[Any]]) -> None:
    if not rows:
        return
    width = max(len(row) for row in rows)
    headers = [str(value).strip() or f"Column {index + 1}" for index, value in enumerate(rows[0])]
    headers += [f"Column {index + 1}" for index in range(len(headers), width)]
    normalized = [list(row) + [""] * (width - len(row)) for row in rows[1:]]
    escape = lambda value: str(value).replace("|", r"\|").replace("\n", "<br>")
    st.markdown(
        "\n".join(
            [
                f"| {' | '.join(map(escape, headers))} |",
                f"| {' | '.join(['---'] * width)} |",
                *(f"| {' | '.join(map(escape, row))} |" for row in normalized),
            ]
        )
    )


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    st.subheader("Cited sources")
    for source in sources:
        manual = source.get("manual") or source.get("source_file") or source.get("source") or "Manual"
        page = source.get("printed_page") or source.get("page") or "—"
        with st.expander(f"{source.get('source_id', '')} · {manual} · page {page}"):
            st.markdown(
                "\n".join(
                    [
                        f"- **Manual:** {manual}",
                        f"- **Release:** {source.get('release') or '—'}",
                        f"- **Chapter:** {source.get('chapter') or '—'}",
                        f"- **Section:** {source.get('section') or '—'}",
                        f"- **Printed page:** {page}",
                        f"- **PDF page:** {source.get('pdf_page') or '—'}",
                        f"- **Content type:** {source.get('content_type') or '—'}",
                    ]
                )
            )
            if source.get("table_rows"):
                st.markdown("**Structured table**")
                render_table(source["table_rows"])


def render_artifacts(message: dict[str, Any], show_sources: bool) -> None:
    evidence = message.get("evidence", {})
    if evidence.get("status"):
        labels = {
            "sufficient": "Fully supported",
            "partial": "Partially supported",
            "in_scope_insufficient": "Insufficient manual evidence",
            "out_of_scope": "Outside indexed manuals",
        }
        details = [f"Evidence: {labels.get(evidence['status'], evidence['status'])}"]
        if evidence.get("manuals"):
            details.append(f"Manuals: {', '.join(evidence['manuals'])}")
        if evidence.get("sections"):
            details.append(f"Sections: {', '.join(evidence['sections'][:3])}")
        st.caption(" · ".join(details))
    diagram = message.get("diagram", {})
    if diagram.get("generated") and diagram.get("dot"):
        st.graphviz_chart(diagram["dot"], width="stretch")
        st.download_button(
            "Download diagram (.dot)",
            diagram["dot"],
            file_name="opcenter-diagram.dot",
            mime="text/vnd.graphviz",
            key=f"diagram-{abs(hash(diagram['dot']))}",
        )
    if show_sources:
        render_sources(message.get("sources", []))


st.set_page_config(page_title="Opcenter Chatbot", page_icon="📘", layout="wide")
st.title("Opcenter Chatbot")
st.caption("Answers are grounded in the indexed Opcenter manuals.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "conversation_id" not in st.session_state or "thread_id" not in st.session_state:
    st.session_state.conversation_id = None
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Conversation")
    if st.button("New conversation"):
        st.session_state.conversation_id = None
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.rerun()
    show_sources = st.checkbox("Show sources", value=True)
    diagram_enabled = st.checkbox("Generate diagrams when useful", value=True)
    diagram_type = st.selectbox(
        "Diagram type",
        ("Automatic", "Hierarchy", "Relationship", "Process", "Decision", "Architecture"),
        disabled=not diagram_enabled,
    ).casefold().replace("automatic", "auto")
    thread_label = st.session_state.thread_id[:8] if st.session_state.thread_id else "new"
    st.caption(f"Thread: `{thread_label}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_artifacts(message, show_sources)

prompt = st.chat_input("Ask a question about the Opcenter manuals")
if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        accepted = create_chat(
            prompt,
            st.session_state.session_id,
            st.session_state.conversation_id,
            st.session_state.thread_id,
            diagram_enabled,
            diagram_type,
        )
        st.session_state.conversation_id = accepted["conversation_id"]
        st.session_state.thread_id = accepted["thread_id"]
        completed: dict[str, Any] = {}
        with st.chat_message("assistant"):
            status = st.status("Searching and checking the manuals...", expanded=True)

            def answer_chunks():
                for event, data in stream_events(accepted["request_id"]):
                    if event == "progress" and data.get("node") in PROGRESS_LABELS:
                        status.write(PROGRESS_LABELS[data["node"]])
                    elif event == "answer":
                        yield data.get("text", "")
                    elif event == "complete":
                        completed.update(data)
                    elif event == "error":
                        raise RuntimeError(data.get("message", "The request failed."))

            answer = st.write_stream(answer_chunks())
            status.update(label="Answer ready", state="complete", expanded=False)
            assistant_message = {
                "role": "assistant",
                "content": completed.get("answer") or answer or "No answer was generated.",
                "sources": completed.get("sources", []),
                "evidence": completed.get("evidence", {}),
                "diagram": completed.get("diagram", {}),
            }
            st.session_state.messages.append(assistant_message)
            render_artifacts(assistant_message, show_sources)
    except (HTTPError, URLError, RuntimeError, TimeoutError):
        st.error("The request could not be completed. Check the backend and try again.")
