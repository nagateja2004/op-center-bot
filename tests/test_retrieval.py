import pytest
from types import SimpleNamespace

import src.retrieval as retrieval
from src.config import Settings
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


def test_production_chroma_client_uses_configured_http_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    client = object()

    def fake_http_client(**kwargs):
        captured.update(kwargs)
        return client

    monkeypatch.setattr(retrieval.chromadb, "HttpClient", fake_http_client)
    config = Settings(chroma_mode="server", chroma_host="chroma", chroma_port=8443, chroma_ssl=True)

    assert retrieval.create_chroma_client(config) is client
    assert captured == {"host": "chroma", "port": 8443, "ssl": True}


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
    exact_calls = 0
    exact_dedup = retrieval.deduplicate_exact_results

    def counted_exact(results):
        nonlocal exact_calls
        exact_calls += 1
        return exact_dedup(results)

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
    monkeypatch.setattr(retrieval, "heading_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval, "concept_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval, "representation_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(retrieval, "deduplicate_exact_results", counted_exact)

    results = retrieval.retrieve_multiple_queries("original", ["variation"])

    assert results[0]["chunk_id"] == "original"
    assert exact_calls == 1


def test_exact_body_heading_retrieval_excludes_toc() -> None:
    matches = retrieval.heading_search("Explain Defining Numbering Rules.")

    assert matches
    assert matches[0]["metadata"]["section"] == "Defining Numbering Rules"
    assert matches[0]["metadata"]["is_toc"] is False
    assert "Modeling" in matches[0]["metadata"]["manual"]
    assert matches[0]["retrieval_scores"]["heading_exact"] == 1.0


@pytest.mark.parametrize(
    ("question", "entities", "aliases", "manuals", "expected_sections"),
    [
        (
            "How are unique numbers automatically assigned to containers?",
            ["Numbering Rule", "Container"],
            ["unique numbers assigned to containers"],
            ["Modeling"],
            {"Defining Numbering Rules", "When Defining Numbering Rules for Containers"},
        ),
        (
            "What is the hierarchy of physical modelling?",
            ["Physical Modeling Sequence", "Factory Hierarchy"],
            ["physical modeling hierarchy"],
            ["Modeling"],
            {"Physical Modeling Sequence", "Factory Hierarchy", "Configuring a Factory Hierarchy"},
        ),
        (
            "What happens before a new value is assigned to a CDO field?",
            ["Validate Event", "Field Event", "CDO"],
            ["before a new value is assigned to a CDO field"],
            ["Designer"],
            {"Events", "Introduction to CLFs"},
        ),
        (
            "What UI components are placed inside Portal Studio web parts?",
            ["Portal Studio Control", "Web Part"],
            ["components inside Portal Studio web parts"],
            ["Portal Studio"],
            {"Controls", "Adding a Control to a Web Part", "Creating a Web Part"},
        ),
    ],
)
def test_concept_and_alias_paths_retrieve_expected_evidence(
    question, entities, aliases, manuals, expected_sections
) -> None:
    matches = retrieval.retrieve_multiple_queries(
        question,
        entities=entities,
        aliases=aliases,
        preferred_manuals=manuals,
        limit=10,
    )

    matched_sections = [str(match["metadata"].get("section", "")) for match in matches]
    assert any(
        section == expected or section.endswith(f": {expected}")
        for section in matched_sections
        for expected in expected_sections
    )
    assert any(
        match["retrieval_scores"].get("concept_exact")
        or match["retrieval_scores"].get("alias_match")
        for match in matches
    )


def test_near_duplicate_field_question_preserves_content_type_diversity() -> None:
    prose = result("prose", {"final_score": 1.0})
    prose["text"] = "Field Description Name The object name used by the factory."
    table = result("table", {"final_score": 0.9})
    table["text"] = "Field Description Name The object name used by the factory. Extra"
    table["content_type"] = "field_definition"
    table["metadata"]["table_rows"] = [["Field", "Description"]]

    deduplicated = deduplicate_results([prose, table], intent="field table")

    assert [item["chunk_id"] for item in deduplicated] == ["prose", "table"]


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


def test_reranker_failure_keeps_prepared_order_without_another_dedup(
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

    assert [item["chunk_id"] for item in reranked] == ["first", "second"]
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


def test_pre_rerank_cap_prefers_representation_type_diversity() -> None:
    heading = result("heading", {"final_score": 1.0})
    body = result("body", {"final_score": 0.9})
    definition = result("definition", {"final_score": 0.8})
    for candidate, representation_type in (
        (heading, "heading"),
        (body, "body"),
        (definition, "definition"),
    ):
        candidate["metadata"].update(
            {"evidence_id": "e1", "representation_type": representation_type}
        )

    prepared = retrieval.prepare_rerank_candidates(
        [heading, body, definition], limit=20
    )

    assert [candidate["chunk_id"] for candidate in prepared] == ["heading", "body"]


def test_post_rerank_preserves_near_text_and_uses_score_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lower = result("lower", {"final_score": 1.0})
    higher = result("higher", {"final_score": 0.9})
    lower["text"] = "A shared manual sentence with enough matching words for overlap detection."
    higher["text"] = lower["text"] + " Extra."
    lower["metadata"].update({"evidence_id": "e1", "manual": "Manual"})
    higher["metadata"].update({"evidence_id": "e2", "manual": "Manual"})
    class Reranker:
        def predict(self, pairs, **kwargs):
            return [0.2, 0.9]

    monkeypatch.setattr(retrieval, "_disabled_rerankers", set())
    monkeypatch.setattr(retrieval, "create_reranker", lambda config: Reranker())
    monkeypatch.setattr(
        retrieval,
        "deduplicate_results",
        lambda *args, **kwargs: pytest.fail("near-text dedup must not run"),
    )

    reranked = retrieval.rerank_documents("question", [lower, higher])

    assert [candidate["chunk_id"] for candidate in reranked] == ["higher", "lower"]


def test_representation_vector_result_retains_parent_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    representation = {
        "representation_id": "r1",
        "evidence_id": "e1",
        "representation_type": "procedure_title",
        "text": "Procedure: Configuring a Resource",
        "metadata": {
            "manual": "Modeling Guide",
            "source_file": "manual.pdf",
            "chapter": "Resources",
            "section": "Configuring a Resource",
            "pdf_page": 5,
        },
        "embedding_token_count": 8,
    }

    class Embedding:
        def embed_query(self, query):
            return [1.0]

    class Collection:
        def query(self, **kwargs):
            return {"ids": [["r1"]], "distances": [[0.1]]}

    resources = SimpleNamespace(
        embedding_model=Embedding(),
        representation_collection=Collection(),
        representations_by_id={"r1": representation},
    )
    monkeypatch.setattr(retrieval, "load_resources", lambda config: resources)

    matches = retrieval.representation_search("configure resource")

    assert matches[0]["metadata"]["evidence_id"] == "e1"
    assert matches[0]["metadata"]["representation_type"] == "procedure_title"
    assert matches[0]["metadata"]["chunk_level"] == "representation"


@pytest.mark.parametrize(
    "representation_type",
    ["heading", "definition", "procedure_title", "table_title_headers"],
)
def test_live_representation_vectors_retrieve_their_parent(
    representation_type: str,
) -> None:
    resources = retrieval.load_resources()
    target = next(
        item
        for item in resources.representations_by_id.values()
        if item["representation_type"] == representation_type
    )

    matches = retrieval.representation_search(target["text"], top_k=5)

    assert target["evidence_id"] in {
        match["metadata"]["evidence_id"] for match in matches
    }


def test_indirect_definition_wording_retrieves_direct_heading_parent() -> None:
    direct = retrieval.representation_search("Defining Numbering Rules", top_k=5)
    indirect = retrieval.representation_search(
        "Numbering Rule assigns unique tracking numbers to quality records and containers",
        top_k=5,
    )

    assert direct[0]["metadata"]["evidence_id"] == indirect[0]["metadata"]["evidence_id"]
    assert direct[0]["metadata"]["section"] == "Defining Numbering Rules"


def test_exact_only_dedup_precedes_reranking_and_keeps_near_text() -> None:
    exact = result("exact", {"final_score": 1.0})
    exact_duplicate = result("exact-duplicate", {"final_score": 0.9})
    near = result("near", {"final_score": 0.8})
    exact["text"] = "Same normalized representation text"
    exact_duplicate["text"] = "Same   normalized representation text"
    near["text"] = "Same normalized representation text with an extra detail"
    for candidate, representation_type in (
        (exact, "heading"),
        (exact_duplicate, "definition"),
        (near, "body"),
    ):
        candidate["metadata"].update(
            {"evidence_id": "e1", "representation_type": representation_type}
        )

    fused = retrieval.deduplicate_exact_results([exact, exact_duplicate, near])
    prepared = retrieval.prepare_rerank_candidates(fused, limit=20)

    assert [candidate["chunk_id"] for candidate in prepared] == ["exact", "near"]


def test_resolution_merges_selected_segments_and_representations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = result("s1", {"reranker_score": 0.9})
    representation = result("r1", {"reranker_score": 0.8})
    segment["metadata"] = {"evidence_id": "e1", "aspects": ["configuration"]}
    representation["metadata"] = {
        "evidence_id": "e1",
        "aspects": ["runtime"],
        "chunk_level": "representation",
        "representation_type": "heading",
    }
    unit = {
        "evidence_id": "e1",
        "text": "Complete evidence.",
        "content_type": "definition",
        "metadata": {"manual": "Manual", "source_file": "manual.pdf"},
        "structured_table": None,
        "procedure_steps": [],
        "annotations": [],
    }
    monkeypatch.setattr(
        retrieval,
        "load_resources",
        lambda config: SimpleNamespace(evidence_units_by_id={"e1": unit}),
    )

    resolved = retrieval.resolve_evidence_units([segment, representation])

    assert resolved[0]["metadata"]["aspects"] == ["configuration", "runtime"]
    assert resolved[0]["metadata"]["matched_representation_types"] == ["heading"]
    assert resolved[0]["metadata"]["selected_candidate_ids"] == ["s1", "r1"]
    assert resolved[0]["metadata"]["selected_segment_id"] == "s1"
    assert resolved[0]["metadata"]["selected_representation_id"] == "r1"
