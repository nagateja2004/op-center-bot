import pytest

from src.config import Settings
from src.schemas import DocumentChunk, EvidenceGrade, QueryPlan, RAGState, RetrievalSegment


def test_document_chunk_defaults() -> None:
    chunk = DocumentChunk(id="manual-1", text="Example", source="manual.pdf", page=1)

    assert chunk.chapter is None
    assert chunk.metadata == {}


def test_requested_schema_contracts() -> None:
    plan = QueryPlan(standalone_question="What is AQL?", intent="definition")
    grade = EvidenceGrade(status="sufficient", reason="Manual evidence found")
    state: RAGState = {"messages": [], "retry_count": 0}

    assert plan.needs_diagram is False
    assert plan.exact_phrases == []
    assert plan.canonical_terms == []
    assert plan.aliases == []
    assert plan.manual_hints == []
    assert grade.missing_concepts == []
    assert state["retry_count"] == 0
    assert "effective_embedding_limit" in RetrievalSegment.__required_keys__


def test_structured_schemas_are_flat() -> None:
    for schema in (QueryPlan, EvidenceGrade):
        definitions = schema.model_json_schema().get("$defs", {})
        assert not definitions
    assert "aspect_queries" not in QueryPlan.model_fields
    assert "manual_filters" not in QueryPlan.model_fields
    assert {"exact_phrases", "canonical_terms", "aliases", "manual_hints"}.issubset(
        QueryPlan.model_fields
    )


def test_settings_reports_missing_api_key() -> None:
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        Settings(groq_api_key="").validate()


def test_settings_repr_does_not_expose_credentials() -> None:
    config = Settings(
        groq_api_key="gsk_fixture_secret",
        evaluation_api_token="fixture_evaluation_secret",
        database_url="postgresql://user:fixture_db_secret@localhost/test",
        redis_url="redis://:fixture_redis_secret@localhost/0",
    )
    rendered = repr(config)
    for secret in ("gsk_fixture_secret", "fixture_evaluation_secret", "fixture_db_secret", "fixture_redis_secret"):
        assert secret not in rendered


def test_settings_rejects_a_non_groq_api_key() -> None:
    with pytest.raises(EnvironmentError, match="beginning with 'gsk_'"):
        Settings(groq_api_key="another-provider-key").validate()


def test_postgres_requires_database_url_but_explicit_sqlite_does_not() -> None:
    with pytest.raises(EnvironmentError, match="DATABASE_URL"):
        Settings(
            groq_api_key="gsk_test",
            checkpoint_backend="postgres",
            database_url="",
        ).validate()

    Settings(
        groq_api_key="gsk_test",
        checkpoint_backend="sqlite",
        database_url="",
    ).validate()
