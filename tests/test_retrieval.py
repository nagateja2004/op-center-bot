import pytest
from types import SimpleNamespace

import src.retrieval as retrieval
from src.retrieval import deduplicate_results, reciprocal_rank_fusion
from src.schemas import RetrievedDocument


def result(chunk_id: str, scores: dict[str, float]) -> RetrievedDocument:
    return {
        "chunk_id": chunk_id,
        "text": chunk_id,
        "content_type": "text",
        "metadata": {},
        "retrieval_scores": scores,
    }


def test_rrf_uses_ranks_not_raw_scores() -> None:
    vector = [result("shared", {"vector_distance": 999.0}), result("vector", {})]
    bm25 = [result("shared", {"bm25_score": 0.01}), result("bm25", {})]

    fused = reciprocal_rank_fusion(vector, bm25)

    assert fused[0]["chunk_id"] == "shared"
    assert fused[0]["retrieval_scores"]["rrf_score"] == 2 / 61


def test_deduplication_merges_scores_by_chunk_id() -> None:
    results = deduplicate_results(
        [result("same", {"vector_rank": 1}), result("same", {"bm25_rank": 2})]
    )

    assert len(results) == 1
    assert results[0]["retrieval_scores"] == {"vector_rank": 1, "bm25_rank": 2}


def test_original_query_gets_slightly_more_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retrieval,
        "vector_search",
        lambda query, *args, **kwargs: [result(query, {"vector_rank": 1})],
    )
    monkeypatch.setattr(
        retrieval,
        "bm25_search",
        lambda query, *args, **kwargs: [result(query, {"bm25_rank": 1})],
    )

    results = retrieval.retrieve_multiple_queries("original", ["variation"])

    assert results[0]["chunk_id"] == "original"


def test_near_duplicate_field_question_prefers_structured_table() -> None:
    prose = result("prose", {"final_score": 1.0})
    prose["text"] = "Field Description Name The object name used by the factory."
    table = result("table", {"final_score": 0.9})
    table["text"] = "Field Description Name The object name used by the factory. Extra"
    table["content_type"] = "field_definition"
    table["metadata"]["table_rows"] = [["Field", "Description"]]

    deduplicated = deduplicate_results([prose, table], intent="field table")

    assert [item["chunk_id"] for item in deduplicated] == ["table"]


def test_near_matches_for_different_aspects_are_preserved() -> None:
    left = result("left", {"final_score": 1.0})
    right = result("right", {"final_score": 0.9})
    left["text"] = "A shared manual sentence with enough matching words for overlap detection."
    right["text"] = "A shared manual sentence with enough matching words for overlap detection. Extra."
    left["metadata"]["aspects"] = ["configuration"]
    right["metadata"]["aspects"] = ["runtime"]

    assert len(deduplicate_results([left, right])) == 2


def test_repeated_table_headers_do_not_remove_distinct_tables() -> None:
    first = result("table-one", {"final_score": 1.0})
    second = result("table-two", {"final_score": 0.9})
    first["content_type"] = second["content_type"] = "table"
    first["text"] = "Field Description Name First factory field definition value"
    second["text"] = "Field Description Name First factory field definition value Extra row"

    assert len(deduplicate_results([first, second], intent="table")) == 2


def test_complementary_evidence_and_segment_are_preserved() -> None:
    parent = result("evidence", {"final_score": 1.0})
    child = result("segment", {"final_score": 0.9})
    parent["metadata"]["chunk_level"] = "evidence"
    child["metadata"]["chunk_level"] = "segment"
    parent["text"] = "Resource configuration manual context with several shared relevant words."
    child["text"] = "Resource configuration manual context with several shared relevant words. Detail."

    assert len(deduplicate_results([parent, child])) == 2


def test_reranker_failure_keeps_order_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(retrieval, "_disabled_rerankers", set())
    duplicate = result("second", {"final_score": 0.5})
    duplicate["text"] = "same normalized text"
    first = result("first", {"final_score": 1.0})
    first["text"] = "Same   normalized text"
    monkeypatch.setattr(
        retrieval,
        "create_reranker",
        lambda config: (_ for _ in ()).throw(OSError("unavailable")),
    )

    reranked = retrieval.rerank_documents("question", [first, duplicate])

    assert [item["chunk_id"] for item in reranked] == ["first"]
    assert reranked[0]["retrieval_scores"]["reranker_fallback"] == 1.0


def test_invalid_reranker_scores_use_one_time_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(retrieval, "_disabled_rerankers", set())
    calls = 0

    class InvalidReranker:
        def predict(self, pairs: object, **kwargs: object) -> list[float]:
            nonlocal calls
            calls += 1
            return [float("nan")]

    monkeypatch.setattr(retrieval, "create_reranker", lambda config: InvalidReranker())

    reranked = retrieval.rerank_documents("question", [result("first", {})])
    repeated = retrieval.rerank_documents("question", [result("first", {})])

    assert reranked[0]["retrieval_scores"]["reranker_fallback"] == 1.0
    assert repeated[0]["retrieval_scores"]["reranker_fallback"] == 1.0
    assert "reranker_score" not in reranked[0]["retrieval_scores"]
    assert calls == 1
    assert caplog.text.count("Cross-encoder unavailable") == 1


def test_segment_expansion_hydrates_complete_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = result("segment", {"final_score": 1.0})
    seed["text"] = "Short searchable segment."
    seed["metadata"]["evidence_id"] = "evidence"
    evidence = {
        "evidence_id": "evidence",
        "text": "Complete evidence definition with its related warning.",
        "content_type": "definition",
        "metadata": {
            "manual": "Designer Guide",
            "section": "CDO definition",
            "printed_page": "2-4",
            "pdf_page": 10,
        },
        "structured_table": None,
        "procedure_steps": [],
        "annotations": [{"type": "warning", "text": "Related warning"}],
    }
    previous = {
        "segment_id": "previous",
        "evidence_id": "evidence",
        "searchable_text": "Previous searchable segment.",
        "content_type": "definition",
        "metadata": evidence["metadata"],
        "segment_index": 0,
        "previous_segment_id": None,
        "next_segment_id": "segment",
        "word_count": 3,
        "embedding_token_count": 5,
    }
    segment = {
        **previous,
        "segment_id": "segment",
        "searchable_text": "Short searchable segment.",
        "segment_index": 1,
        "previous_segment_id": "previous",
        "next_segment_id": None,
    }
    resources = SimpleNamespace(
        evidence_units_by_id={"evidence": evidence},
        segments_by_id={"previous": previous, "segment": segment},
    )
    monkeypatch.setattr(retrieval, "load_resources", lambda config: resources)

    expanded = retrieval.expand_context([seed])

    assert [item["chunk_id"] for item in expanded] == ["segment", "previous"]
    assert expanded[1]["metadata"]["context_relation"] == "previous"

    resolved = retrieval.resolve_evidence_units(expanded)

    assert [item["chunk_id"] for item in resolved] == ["evidence"]
    assert resolved[0]["text"].startswith("Complete evidence definition")
    assert resolved[0]["metadata"]["context_relation"] == "evidence_unit"


def test_evidence_resolution_deduplicates_by_evidence_id_and_merges_aspects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = result("segment-1", {"reranker_score": 0.9})
    second = result("segment-2", {"reranker_score": 0.8})
    first["metadata"] = {"evidence_id": "e1", "aspects": ["configuration"]}
    second["metadata"] = {"evidence_id": "e1", "aspects": ["runtime"]}
    unit = {
        "evidence_id": "e1",
        "text": "Complete procedure\n1. Configure.\n2. Run.",
        "content_type": "procedure",
        "metadata": {"manual": "Manual", "source_file": "manual.pdf"},
        "structured_table": None,
        "procedure_steps": ["1. Configure.", "2. Run."],
        "annotations": [],
    }
    monkeypatch.setattr(
        retrieval,
        "load_resources",
        lambda config: SimpleNamespace(evidence_units_by_id={"e1": unit}),
    )

    resolved = retrieval.resolve_evidence_units([first, second])

    assert len(resolved) == 1
    assert resolved[0]["metadata"]["aspects"] == ["configuration", "runtime"]
    assert resolved[0]["metadata"]["procedure_steps"] == [
        "1. Configure.",
        "2. Run.",
    ]
