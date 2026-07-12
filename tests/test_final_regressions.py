from inspect import getsource
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import chromadb
import pytest

import src.llm as llm
import src.nodes as nodes
import src.retrieval as retrieval
from src.config import Settings, settings
from src.ingest import (
    CHROMA_COLLECTION,
    EMBEDDING_TOKEN_LIMIT,
    Element,
    IndexSchemaMismatchError,
    SectionGroup,
    _build_evidence,
    _markdown_table,
    _require_index_schema,
)
from src.llm import GroqRequestError, RoleConfig
from src.schemas import EvidenceGrade


def build(group: SectionGroup):
    return _build_evidence(
        [group],
        source_file="manual.pdf",
        file_hash="hash",
        manual="Manual",
        release="2510",
        config=Settings(groq_api_key="test"),
    )


def test_700_token_section_is_one_evidence_unit_with_safe_segments() -> None:
    units, segments = build(
        SectionGroup(
            chapter="Concepts",
            section="Large concept",
            elements=[Element("configuration " * 700, "concept", 1, "1-1")],
        )
    )

    assert len(units) == 1
    assert len(segments) > 1
    assert {segment["evidence_id"] for segment in segments} == {units[0]["evidence_id"]}
    assert all(segment["embedding_token_count"] < EMBEDDING_TOKEN_LIMIT for segment in segments)


def test_procedures_split_between_steps_and_resolve_complete_unit(monkeypatch) -> None:
    steps = [f"{number}. Perform complete operation number {number}." for number in range(1, 121)]
    units, segments = build(
        SectionGroup(
            chapter="Procedures",
            section="How to configure",
            elements=[Element("\n".join(steps), "procedure", 2, "2-1")],
        )
    )

    assert len(units) == 1 and len(segments) > 1
    assert all(sum(step in segment["searchable_text"] for segment in segments) == 1 for step in steps)
    monkeypatch.setattr(
        retrieval,
        "load_resources",
        lambda config: SimpleNamespace(evidence_units_by_id={units[0]["evidence_id"]: units[0]}),
    )
    selected = retrieval._result(segments[len(segments) // 2], {"reranker_score": 1.0})

    resolved = retrieval.resolve_evidence_units([selected])

    assert len(resolved) == 1
    assert resolved[0]["text"] == units[0]["text"]
    assert resolved[0]["metadata"]["procedure_steps"] == steps


def test_table_rows_keep_headers_and_field_definitions() -> None:
    rows = [
        ["Field", "Definition"],
        *[[f"Field {number}", f"Definition {number} " + "detail " * 30] for number in range(15)],
    ]
    units, segments = build(
        SectionGroup(
            chapter="Fields",
            section="Factory fields",
            elements=[
                Element(
                    _markdown_table("Factory fields", rows),
                    "field_definition",
                    3,
                    "3-1",
                    rows,
                )
            ],
        )
    )

    assert units[0]["structured_table"]["headers"] == rows[0]
    assert all(segment["metadata"]["table_rows"][0] == rows[0] for segment in segments)
    recovered = [row for segment in segments for row in segment["metadata"]["table_rows"][1:]]
    assert recovered == rows[1:]


def test_live_chroma_and_bm25_contain_only_aligned_retrieval_segments() -> None:
    segments = json.loads((settings.indexes_dir / "retrieval_segments.json").read_text())
    segment_ids = {segment["segment_id"] for segment in segments}
    with (settings.indexes_dir / "bm25.pkl").open("rb") as handle:
        bm25_ids = set(pickle.load(handle)["segment_ids"])
    collection = chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(
        CHROMA_COLLECTION
    )
    chroma = collection.get(include=["metadatas"])

    assert segment_ids == bm25_ids == set(chroma["ids"])
    assert all(segment_id.startswith("s_") for segment_id in segment_ids)
    assert all(
        metadata.get("segment_id") in segment_ids and metadata.get("evidence_id")
        for metadata in chroma["metadatas"]
    )


def test_planner_grader_and_diagram_policies_exclude_answer_model() -> None:
    answer_models = {
        llm.ROLE_DEFAULTS["answer"].primary_model,
        llm.ROLE_DEFAULTS["answer"].fallback_model,
    }
    for role in ("planner", "grader", "diagram"):
        policy = llm.ROLE_DEFAULTS[role]
        assert policy.primary_model not in answer_models
        assert policy.fallback_model not in answer_models
    assert 'task="planner"' in getsource(nodes.understand_question)
    assert 'task="grader"' in getsource(nodes.grade_evidence)
    assert 'task="diagram"' in getsource(nodes.generate_diagram)


def test_grader_429_uses_only_grader_fallback(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class Invocation:
        def __init__(self, model: str):
            self.model = model

        def invoke(self, prompt):
            calls.append(("grader", self.model))
            if self.model == "grader-primary":
                raise GroqRequestError("rate_limit", "grader", 429)
            return EvidenceGrade(status="sufficient", reason="fallback")

    class Client:
        def __init__(self, model: str):
            self.model = model

        def with_structured_output(self, *args, **kwargs):
            return Invocation(self.model)

    def policy(role):
        assert role == "grader"
        return RoleConfig("grader-primary", "grader-fallback", 0, 100, 10, True)

    monkeypatch.setattr(llm, "role_config", policy)
    monkeypatch.setattr(llm, "create_llm", lambda role, model: Client(model))

    grade = llm.call_structured("prompt", EvidenceGrade, task="grader")

    assert grade.status == "sufficient"
    assert calls == [("grader", "grader-primary"), ("grader", "grader-fallback")]


def test_plain_generation_roles_never_enable_function_calling() -> None:
    for role in ("answer", "verifier", "diagram"):
        assert llm.ROLE_DEFAULTS[role].allow_structured_output is False
        assert "call_structured" not in getsource(
            {
                "answer": nodes.generate_answer,
                "verifier": nodes.verify_answer,
                "diagram": nodes.generate_diagram,
            }[role]
        )


@pytest.mark.parametrize(
    ("function", "budget_name"),
    [
        (nodes.understand_question, "planner_input_token_budget"),
        (nodes.broaden_query, "query_broadening_input_token_budget"),
        (nodes.grade_evidence, "grader_input_token_budget"),
        (nodes.generate_answer, "answer_input_token_budget"),
        (nodes.verify_answer, "verifier_input_token_budget"),
        (nodes.generate_diagram, "diagram_input_token_budget"),
    ],
)
def test_every_llm_node_enforces_its_configured_prompt_budget(function, budget_name) -> None:
    source = getsource(function)
    assert "_trim_to_token_budget" in source
    assert f"settings.{budget_name}" in source


def test_schema_mismatch_requires_explicit_reingestion_without_deleting(tmp_path: Path) -> None:
    indexes = tmp_path / "indexes"
    indexes.mkdir()
    marker = indexes / "existing-index.bin"
    marker.write_bytes(b"keep")
    (indexes / "manifest.json").write_text('{"version": 3}', encoding="utf-8")
    config = Settings(
        groq_api_key="test",
        indexes_dir=indexes,
        chroma_dir=indexes / "chroma",
    )

    with pytest.raises(IndexSchemaMismatchError, match="python -m src.ingest"):
        _require_index_schema(config)

    assert marker.read_bytes() == b"keep"
