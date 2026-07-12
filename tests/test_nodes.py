from langchain_core.messages import AIMessage, HumanMessage

import src.nodes as nodes
from src.schemas import (
    EvidenceGrade,
    QueryPlan,
)


def document(number: int) -> dict:
    return {
        "chunk_id": f"chunk-{number}",
        "text": f"Evidence {number}",
        "content_type": "text",
        "metadata": {
            "manual": "Modeling User Guide",
            "chapter": "Physical Model Definitions",
            "section": f"Section {number}",
            "printed_page": f"2-{number}",
            "pdf_page": number,
            "release": "2510+",
        },
        "retrieval_scores": {"final_score": 1 / number},
    }


def test_understand_question_uses_latest_six_messages_and_sets_diagram(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_call(prompt, schema, *, task):
        captured["prompt"] = prompt
        captured["task"] = task
        return QueryPlan(
            standalone_question="Show the factory hierarchy",
            intent="relationship",
            entities=["Factory"],
            search_queries=["factory hierarchy"],
        )

    monkeypatch.setattr(nodes, "call_structured", fake_call)
    messages = [HumanMessage(content=f"message-{number}") for number in range(8)]

    update = nodes.understand_question({"messages": messages})

    assert "message-1" not in captured["prompt"]
    assert "message-2" in captured["prompt"]
    assert captured["task"] == "planner"
    assert 2 <= len(update["search_queries"]) <= 4
    assert update["needs_diagram"] is True


def test_grade_does_not_treat_weak_opcenter_retrieval_as_out_of_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_structured",
        lambda *args, **kwargs: EvidenceGrade(
            status="out_of_scope", reason="Weak evidence", missing_concepts=[]
        ),
    )

    update = nodes.grade_evidence(
        {
            "standalone_question": "How do I define a Factory?",
            "entities": ["Factory"],
            "reranked_docs": [document(1)],
        }
    )

    assert update["evidence_status"] == "in_scope_insufficient"


def test_empty_evidence_skips_grading_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_structured",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call Groq")),
    )

    update = nodes.grade_evidence(
        {"standalone_question": "What is a Factory?", "reranked_docs": [], "retry_count": 0}
    )

    assert update["evidence_status"] == "retry"


def test_grade_returns_partial_when_only_one_aspect_is_supported(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_structured",
        lambda prompt, *args, **kwargs: EvidenceGrade(
            status="sufficient" if "Assigned aspect: supported" in prompt else "in_scope_insufficient",
            reason="graded",
        ),
    )

    update = nodes.grade_evidence(
        {
            "standalone_question": "Opcenter multi-part question",
            "required_aspects": ["supported", "missing"],
            "aspect_documents": {
                "supported": [document(1)],
                "missing": [document(2)],
            },
        }
    )

    assert update["evidence_status"] == "partial"
    assert update["missing_aspects"] == ["missing"]


def test_retry_retrieves_only_missing_aspects(monkeypatch) -> None:
    calls: list[str] = []

    def fake_retrieve(aspect, *args, **kwargs):
        calls.append(aspect)
        return [document(2)]

    monkeypatch.setattr(nodes, "retrieve_multiple_queries", fake_retrieve)
    existing = document(1)
    update = nodes.retrieve_documents(
        {
            "standalone_question": "Question",
            "required_aspects": ["supported", "missing"],
            "aspect_queries": {"supported": ["A"], "missing": ["B"]},
            "aspect_documents": {"supported": [existing]},
            "missing_aspects": ["missing"],
            "retry_count": 1,
        }
    )

    assert calls == ["missing"]
    assert update["aspect_documents"]["supported"][0]["chunk_id"] == "chunk-1"


def test_reranking_is_per_aspect_and_final_evidence_is_capped(monkeypatch) -> None:
    calls: list[str] = []

    def fake_rerank(aspect, documents, **kwargs):
        calls.append(aspect)
        return documents[: kwargs["limit"]]

    monkeypatch.setattr(nodes, "cross_encoder_rerank", fake_rerank)
    monkeypatch.setattr(
        nodes,
        "resolve_evidence_units",
        lambda documents, **kwargs: documents[: kwargs["limit"]],
    )
    aspect_documents = {
        aspect: [document(index * 10 + offset) for offset in range(1, 5)]
        for index, aspect in enumerate(("A", "B", "C", "D"), start=1)
    }

    update = nodes.rerank_documents(
        {"standalone_question": "Question", "aspect_documents": aspect_documents}
    )

    assert calls == ["A", "B", "C", "D"]
    assert all(len(items) == 3 for items in update["aspect_documents"].values())
    assert len(update["reranked_docs"]) == 10


def test_broaden_query_allows_only_one_retry(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_structured",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call LLM")),
    )

    update = nodes.broaden_query(
        {
            "standalone_question": "How do I define a Factory?",
            "retry_count": 1,
            "manual_filters": {"manuals": ["Modeling"]},
        }
    )

    assert update["evidence_status"] == "in_scope_insufficient"
    assert update["manual_filters"] == {}


def test_generate_answer_returns_only_valid_cited_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_llm",
        lambda *args, **kwargs: AIMessage(content="Supported [S2]. Invalid [S9]."),
    )

    update = nodes.generate_answer(
        {
            "standalone_question": "Question",
            "evidence_status": "sufficient",
            "reranked_docs": [document(1), document(2)],
        }
    )

    assert "[S9]" not in update["answer"]
    assert [source.source_id for source in update["sources"]] == ["S2"]
    assert update["sources"][0].pdf_page == 2
    assert update["sources"][0].content_type == "text"


def test_generate_answer_normalizes_grouped_citations(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_llm",
        lambda *args, **kwargs: AIMessage(content="Supported by both [S1, S2]."),
    )

    update = nodes.generate_answer(
        {
            "standalone_question": "Question",
            "evidence_status": "sufficient",
            "reranked_docs": [document(1), document(2)],
        }
    )

    assert update["answer"] == "Supported by both [S1] [S2]."
    assert [source.source_id for source in update["sources"]] == ["S1", "S2"]


def test_verify_uses_cleaned_answer_and_valid_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_llm",
        lambda *args, **kwargs: AIMessage(content="Supported [S1]. Invalid [S8]."),
    )

    update = nodes.verify_answer(
        {
            "standalone_question": "Question",
            "answer": "Supported [S1]. Unsupported [S8].",
            "reranked_docs": [document(1)],
        }
    )

    assert update["answer"] == "Supported [S1]. Invalid ."
    assert [source.source_id for source in update["sources"]] == ["S1"]
    assert update["grounded"] is False


def test_process_diagram_is_evidence_gated_and_left_to_right(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_llm",
        lambda *args, **kwargs: AIMessage(
            content="digraph G {\nrankdir=TB;\na [label=\"A [S1]\"];\nb [label=\"B [S1]\"];\nc [label=\"C [S1]\"];\na -> b;\nb -> c;\n}"
        ),
    )

    update = nodes.generate_diagram(
        {
            "standalone_question": "Show the process relationship",
            "needs_diagram": True,
            "grounded": True,
            "evidence_status": "sufficient",
            "answer": "A to B to C [S1].",
            "reranked_docs": [document(1)],
        }
    )

    assert "rankdir=LR" in update["diagram_dot"]


def test_malformed_or_external_graphviz_is_rejected() -> None:
    assert nodes._validated_dot("digraph G { a [label=\"A\"];", "TB") is None
    assert (
        nodes._validated_dot(
            'digraph G { a [image="manual.pdf"]; b [label="B"]; c [label="C"]; a -> b; b -> c; }',
            "TB",
        )
        is None
    )


def test_fallback_messages_are_distinct_and_never_include_diagram() -> None:
    unsupported = nodes.generate_fallback({"evidence_status": "in_scope_insufficient"})
    irrelevant = nodes.generate_fallback({"evidence_status": "out_of_scope"})

    assert unsupported["answer"] != irrelevant["answer"]
    assert unsupported["diagram_dot"] is None
    assert irrelevant["diagram_dot"] is None


def test_compatibility_evidence_is_limited_to_eight_sources() -> None:
    documents = [document(number) for number in range(1, 11)]
    for item in documents:
        item["text"] = "\n".join(f"Complete evidence line {line}." for line in range(400))

    formatted = nodes._format_evidence(documents)

    assert "[S8]" in formatted and "[S9]" not in formatted
    assert len(formatted) <= 12_000


def test_diagram_false_skips_groq(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call Groq")),
    )

    update = nodes.generate_diagram(
        {
            "standalone_question": "Explain a Factory",
            "needs_diagram": False,
            "grounded": True,
            "evidence_status": "sufficient",
            "reranked_docs": [document(1)],
        }
    )

    assert update["diagram_dot"] is None


def test_planner_failure_uses_deterministic_plan(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_structured",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            nodes.GroqRequestError("invalid_structured", "planner")
        ),
    )

    update = nodes.understand_question(
        {"messages": [HumanMessage(content="How do I define a Factory?")]}
    )

    assert update["standalone_question"] == "How do I define a Factory?"
    assert update["intent"] == "procedure"
    assert "procedure" in update["required_output"]


def test_grader_failure_uses_heuristic_coverage(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_structured",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            nodes.GroqRequestError("rate_limit", "grader", 429)
        ),
    )
    evidence = document(1)
    evidence["text"] = "A Factory is configured in Opcenter Modeling."

    update = nodes.grade_evidence(
        {
            "standalone_question": "How is a Factory configured in Opcenter?",
            "required_aspects": ["Factory configuration"],
            "entities": ["Factory"],
            "aspect_documents": {"Factory configuration": [evidence]},
        }
    )

    assert update["evidence_status"] == "sufficient"
    assert "matches the assigned aspect" in update["evidence_reason"]


def test_answer_failure_returns_temporary_message_and_skips_verifier(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            nodes.GroqRequestError("rate_limit", "answer", 429)
        ),
    )
    answer_update = nodes.generate_answer(
        {
            "standalone_question": "What is a Factory?",
            "required_aspects": ["Factory definition"],
            "required_output": ["explanation"],
            "evidence_status": "sufficient",
            "reranked_docs": [document(1)],
        }
    )
    verify_update = nodes.verify_answer(answer_update)

    assert answer_update["answer"] == nodes.TEMPORARY_LLM_MESSAGE
    assert verify_update["answer"] == nodes.TEMPORARY_LLM_MESSAGE
    assert verify_update["grounded"] is False
    assert verify_update["sources"] == []


def test_verifier_failure_runs_deterministic_citation_checks(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            nodes.GroqRequestError("timeout", "verifier")
        ),
    )

    update = nodes.verify_answer(
        {
            "standalone_question": "Question",
            "required_aspects": ["definition"],
            "answer": "Supported [S1]. Invalid [S9].",
            "reranked_docs": [document(1)],
        }
    )

    assert "[S1]" in update["answer"] and "[S9]" not in update["answer"]
    assert update["grounded"] is False
    assert [source.source_id for source in update["sources"]] == ["S1"]


def test_diagram_failure_does_not_fail_verified_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            nodes.GroqRequestError("unavailable_model", "diagram", 404)
        ),
    )

    update = nodes.generate_diagram(
        {
            "standalone_question": "Show the Factory hierarchy",
            "needs_diagram": True,
            "grounded": True,
            "answer": "Factory contains Resources [S1].",
            "reranked_docs": [document(1)],
        }
    )

    assert update == {"diagram_dot": None}


def test_required_manual_coverage_retries_then_downgrades_to_partial(monkeypatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_structured",
        lambda *args, **kwargs: EvidenceGrade(status="sufficient", reason="supported"),
    )
    modeling = document(1)
    modeling["metadata"]["manual"] = "Opcenter Execution Core Modeling User Guide"
    state = {
        "standalone_question": "Use Modeling and Shop Floor evidence for this Opcenter question",
        "required_aspects": ["configuration"],
        "required_manuals": ["Modeling", "Shop Floor"],
        "aspect_documents": {"configuration": [modeling]},
    }

    first = nodes.grade_evidence({**state, "retry_count": 0})
    final = nodes.grade_evidence({**state, "retry_count": 1})

    assert first["evidence_status"] == "retry"
    assert final["evidence_status"] == "partial"
    assert first["manual_coverage"] == {"Modeling": True, "Shop Floor": False}
    assert "Shop Floor manual evidence" in first["missing_aspects"]


def test_single_manual_citations_cannot_claim_cross_manual_synthesis() -> None:
    source = nodes.SourceInfo(
        source_id="S1",
        source="Modeling Guide",
        manual="Modeling Guide",
        pdf_page=1,
    )

    answer = nodes._sanitize_cross_manual_label(
        "**Cross-manual synthesis:** This is a view across cited manuals [S1].",
        [source],
    )

    assert "cross-manual" not in answer.casefold()
    assert "across cited manuals" not in answer.casefold()
