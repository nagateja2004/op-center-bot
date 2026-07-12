from typing import cast

from groq import APITimeoutError
import httpx
from langchain_core.messages import AIMessage
import pytest

import src.embeddings as embeddings
import src.retrieval as retrieval
from src.config import Settings
import src.llm as llm
from src.llm import (
    GroqRequestError,
    GroqRole,
    RoleConfig,
    _call_groq,
    _status_error_kind,
    call_llm,
    call_structured,
    create_llm,
    role_config,
)
from src.schemas import QueryPlan


def test_llm_is_cached_and_uses_role_request_limits() -> None:
    config = Settings(groq_api_key="test-key")
    captured: dict[str, object] = {}
    client = object()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(llm, "_chat_groq_class", lambda: lambda **kwargs: captured.update(kwargs) or client)
    create_llm.cache_clear()

    first = create_llm("planner", "planner-model", config)
    second = create_llm("planner", "planner-model", config)

    assert first is second
    assert captured["temperature"] == 0
    assert captured["max_tokens"] == 1024
    assert captured["timeout"] == 30
    assert captured["max_retries"] == 0
    monkeypatch.undo()


def test_groq_rejects_non_generation_tasks() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _call_groq(cast(GroqRole, "retrieval"), lambda: "unused")


def test_structured_call_uses_one_distinct_fallback_model(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []

    class Invocation:
        def __init__(self, model: str):
            self.model = model

        def invoke(self, prompt):
            calls.append(self.model)
            if self.model == "primary":
                raise GroqRequestError("rate_limit", "planner", 429)
            return {"standalone_question": "What is a CDO?", "intent": "definition"}

    class Client:
        def __init__(self, model: str):
            self.model = model

        def with_structured_output(self, schema, **kwargs):
            assert schema is QueryPlan
            assert kwargs == {"method": "json_schema", "strict": False}
            return Invocation(self.model)

    monkeypatch.setattr(
        llm,
        "role_config",
        lambda role: RoleConfig("primary", "fallback", 0, 100, 10, True),
    )
    monkeypatch.setattr(llm, "create_llm", lambda role, model: Client(model))

    result = call_structured("private manual prompt", QueryPlan, task="planner")

    assert result.standalone_question == "What is a CDO?"
    assert calls == ["primary", "fallback"]
    assert "recovered=True" in caplog.text
    assert all(record.levelname != "ERROR" for record in caplog.records)


def test_rate_limited_model_is_not_retried_when_fallback_matches_primary(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[str] = []

    class Client:
        def invoke(self, prompt):
            calls.append("same")
            raise GroqRequestError("rate_limit", "answer", 429)

    monkeypatch.setattr(
        llm,
        "role_config",
        lambda role: RoleConfig("same", "same", 0, 100, 10, False),
    )
    monkeypatch.setattr(llm, "create_llm", lambda role, model: Client())

    with pytest.raises(GroqRequestError, match="kind=rate_limit"):
        call_llm("private manual prompt", task="answer", evidence_count=4)

    assert calls == ["same"]
    assert "evidence_count=4" in caplog.text
    assert "private manual prompt" not in caplog.text
    assert any(record.levelname == "ERROR" for record in caplog.records)


@pytest.mark.parametrize(
    ("status", "detail", "kind"),
    [
        (429, "quota exceeded for organization org_secret", "rate_limit"),
        (400, "tool_use_failed: failed to call a function", "function_call"),
        (400, "requested model does not exist", "unavailable_model"),
        (404, "not found", "unavailable_model"),
        (400, "invalid request", "bad_request"),
    ],
)
def test_status_errors_are_classified_without_exposing_provider_details(
    status: int, detail: str, kind: str
) -> None:
    assert _status_error_kind(status, detail) == kind
    error = GroqRequestError(kind, "planner", status)  # type: ignore[arg-type]
    assert "org_secret" not in str(error)
    assert "API key" not in str(error)


def test_invalid_structured_output_is_sanitized_after_one_fallback(monkeypatch) -> None:
    calls: list[str] = []

    class Invalid:
        def __init__(self, model: str):
            self.model = model

        def invoke(self, prompt):
            calls.append(self.model)
            return {"unexpected": "payload containing private prompt"}

    class Client:
        def __init__(self, model: str):
            self.model = model

        def with_structured_output(self, *args, **kwargs):
            return Invalid(self.model)

    monkeypatch.setattr(
        llm,
        "role_config",
        lambda role: RoleConfig("primary", "fallback", 0, 100, 10, True),
    )
    monkeypatch.setattr(llm, "create_llm", lambda role, model: Client(model))

    with pytest.raises(GroqRequestError) as captured:
        call_structured("private prompt", QueryPlan, task="planner")

    assert calls == ["primary", "fallback"]
    assert captured.value.kind == "invalid_structured"
    assert "private prompt" not in str(captured.value)


def test_timeout_is_sanitized() -> None:
    timeout = APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))

    with pytest.raises(GroqRequestError) as captured:
        _call_groq("verifier", lambda: (_ for _ in ()).throw(timeout))

    assert captured.value.kind == "timeout"
    assert "api.groq.com" not in str(captured.value)


def test_role_configuration_and_structured_permissions(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_ANSWER_PRIMARY_MODEL", "answer-primary")
    monkeypatch.setenv("GROQ_ANSWER_FALLBACK_MODEL", "")
    monkeypatch.setenv("GROQ_ANSWER_TEMPERATURE", "0.25")
    monkeypatch.setenv("GROQ_ANSWER_MAX_OUTPUT_TOKENS", "3000")
    monkeypatch.setenv("GROQ_ANSWER_TIMEOUT", "45")

    policy = role_config("answer")

    assert policy == RoleConfig("answer-primary", "", 0.25, 3000, 45, False)
    with pytest.raises(ValueError, match="not allowed"):
        call_structured("prompt", QueryPlan, task="answer")


def test_compose_model_aliases_are_supported(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_VERIFIER_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("GROQ_VERIFIER_FALLBACK_MODEL", raising=False)
    monkeypatch.setenv("GROQ_VERIFY_MODEL", "verify-primary")
    monkeypatch.setenv("GROQ_VERIFY_FALLBACK_MODEL", "verify-fallback")

    policy = role_config("verifier")

    assert policy.primary_model == "verify-primary"
    assert policy.fallback_model == "verify-fallback"


def test_index_artifact_paths_follow_the_configured_index_directory(
    monkeypatch, tmp_path
) -> None:
    for name in ("BM25_INDEX_PATH", "EVIDENCE_UNITS_PATH", "RETRIEVAL_SEGMENTS_PATH"):
        monkeypatch.delenv(name, raising=False)
    config = Settings(groq_api_key="test", indexes_dir=tmp_path)

    assert config.bm25_path == tmp_path / "bm25.pkl"
    assert config.evidence_units_path == tmp_path / "evidence_units.json"
    assert config.retrieval_segments_path == tmp_path / "retrieval_segments.json"


def test_embeddings_are_local_and_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_embeddings(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(embeddings, "_embedding_class", lambda: fake_embeddings)
    embeddings.create_embedding_model.cache_clear()
    embeddings.create_embedding_model(Settings(groq_api_key=""))

    assert captured["model_kwargs"] == {"device": "cpu"}
    assert captured["encode_kwargs"] == {"normalize_embeddings": True}


def test_cross_encoder_scores_are_scalar_and_finite() -> None:
    class Scalar:
        def item(self) -> float:
            return 0.25

    assert embeddings.finite_scores([Scalar(), -0.5], 2) == [0.25, -0.5]
    with pytest.raises(ValueError, match="invalid scores"):
        embeddings.finite_scores([float("nan")], 1)


def test_expensive_clients_have_single_entry_caches() -> None:
    assert create_llm.cache_info().maxsize == 16
    assert embeddings.create_embedding_model.cache_info().maxsize == 1
    assert embeddings.create_reranker.cache_info().maxsize == 1
    assert retrieval.load_resources.cache_info().maxsize == 1
