import pytest

from src.config import Settings
from src.schemas import DocumentChunk, EvidenceGrade, QueryPlan, RAGState


def test_document_chunk_defaults() -> None:
    chunk = DocumentChunk(id="manual-1", text="Example", source="manual.pdf", page=1)

    assert chunk.chapter is None
    assert chunk.metadata == {}


def test_requested_schema_contracts() -> None:
    plan = QueryPlan(standalone_question="What is AQL?", intent="definition")
    grade = EvidenceGrade(status="sufficient", reason="Manual evidence found")
    state: RAGState = {"messages": [], "retry_count": 0}

    assert plan.needs_diagram is False
    assert grade.missing_concepts == []
    assert state["retry_count"] == 0


def test_structured_schemas_are_flat() -> None:
    for schema in (QueryPlan, EvidenceGrade):
        definitions = schema.model_json_schema().get("$defs", {})
        assert not definitions
    assert "aspect_queries" not in QueryPlan.model_fields
    assert "manual_filters" not in QueryPlan.model_fields


def test_settings_reports_missing_api_key() -> None:
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        Settings(groq_api_key="").validate()
