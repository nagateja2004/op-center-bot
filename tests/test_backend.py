import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient

from backend.main import app
from backend import dependencies
import backend.main as backend_main


class FakeGraph:
    config = None

    async def astream(self, inputs, **kwargs):
        assert inputs["messages"][0].content == "What is a Factory?"
        assert kwargs["stream_mode"] == "updates"
        self.config = kwargs["config"]
        yield {"type": "updates", "data": {"verify_answer": {}}}

    async def aget_state(self, config):
        return SimpleNamespace(
            values={
                "answer": "A Factory is a division [S1].",
                "sources": [
                    {
                        "source_id": "S1",
                        "source": "Modeling Guide",
                        "printed_page": "2-3",
                    }
                ],
                "reranked_docs": [{"metadata": {}}],
                "diagram_generated": True,
                "diagram_dot": "digraph G { factory; }",
                "evidence_status": "sufficient",
            }
        )


class FakeRequestStore:
    def __init__(self):
        self.owners = {}
        self.requests = {}

    async def claim_thread(self, thread_id, session_id):
        owner = self.owners.setdefault(thread_id, session_id)
        return owner == session_id

    async def save_request(self, request_id, payload):
        self.requests[request_id] = payload

    async def pop_request(self, request_id):
        return self.requests.pop(request_id, None)


def test_health_ready_and_async_chat_stream(monkeypatch) -> None:
    graph = FakeGraph()
    app.state.graph = graph
    app.state.ready_error = None
    app.state.request_store = FakeRequestStore()
    async def ready(_app):
        return {"graph": True}
    monkeypatch.setattr(backend_main, "readiness_checks", ready)
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready", "checks": {"graph": True}}
    assert client.get("/metrics").status_code == 200

    accepted = client.post("/v1/chat", json={"message": "What is a Factory?"})
    assert accepted.status_code == 202
    identifiers = accepted.json()
    request_id = identifiers["request_id"]
    UUID(identifiers["conversation_id"])
    UUID(identifiers["thread_id"])
    UUID(identifiers["session_id"])

    streamed = client.get(f"/v1/chat/{request_id}/stream")
    assert streamed.status_code == 200
    assert "event: progress" in streamed.text
    assert "event: answer" in streamed.text
    assert "event: complete" in streamed.text
    assert "A Factory is a division [S1]." in streamed.text
    assert "digraph G" in streamed.text
    assert "Fully supported" not in streamed.text
    assert '"status": "sufficient"' in streamed.text
    assert graph.config == {
        "configurable": {
            "thread_id": (
                f"{identifiers['conversation_id']}:{identifiers['thread_id']}"
            ),
            "conversation_id": identifiers["conversation_id"],
            "client_thread_id": identifiers["thread_id"],
        }
    }
    assert client.get(f"/v1/chat/{request_id}/stream").status_code == 404

    follow_up = client.post(
        "/v1/chat",
        json={
            "message": "What is a Factory?",
            "session_id": identifiers["session_id"],
            "conversation_id": identifiers["conversation_id"],
            "thread_id": identifiers["thread_id"],
        },
    )
    assert follow_up.status_code == 202
    assert follow_up.json()["thread_id"] == identifiers["thread_id"]

    forbidden = client.post(
        "/v1/chat",
        json={
            "message": "What is a Factory?",
            "session_id": str(UUID(int=1)),
            "conversation_id": identifiers["conversation_id"],
            "thread_id": identifiers["thread_id"],
        },
    )
    assert forbidden.status_code == 403


def test_chat_identifiers_are_both_new_or_both_reused() -> None:
    app.state.graph = FakeGraph()
    app.state.request_store = FakeRequestStore()
    client = TestClient(app)

    first = client.post("/v1/chat", json={"message": "one"}).json()
    second = client.post("/v1/chat", json={"message": "two"}).json()
    assert first["thread_id"] != second["thread_id"]
    assert first["conversation_id"] != second["conversation_id"]

    invalid = client.post(
        "/v1/chat",
        json={"message": "bad", "thread_id": first["thread_id"]},
    )
    assert invalid.status_code == 422


def test_postgres_pool_opens_once_and_closes_cleanly(monkeypatch) -> None:
    calls: list[object] = []

    class FakePool:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        async def open(self, wait=False):
            calls.append(("open", wait))

        async def close(self):
            calls.append("close")

    class FakeSaver:
        def __init__(self, pool, serde=None):
            calls.append(("saver", pool, serde))

    monkeypatch.setattr(dependencies, "AsyncConnectionPool", FakePool)
    monkeypatch.setattr(
        "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
        FakeSaver,
    )
    monkeypatch.setattr(
        dependencies,
        "settings",
        SimpleNamespace(
            checkpoint_backend="postgres",
            database_url="postgresql://example/test",
        ),
    )

    async def exercise():
        stack = AsyncExitStack()
        await dependencies._open_checkpointer(stack)
        await stack.aclose()

    asyncio.run(exercise())

    assert calls[0]["conninfo"] == "postgresql://example/test"
    assert calls[0]["open"] is False
    assert calls[1] == ("open", True)
    assert calls[2][0] == "saver"
    assert isinstance(calls[2][1], FakePool)
    assert calls[3] == "close"
