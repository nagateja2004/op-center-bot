from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
import pytest

import src.nodes as nodes
from src.graph import graph
from src.schemas import (
    EvidenceGrade,
    QueryPlan,
)


def evidence(
    chunk_id: str,
    text: str,
    *,
    manual: str = "Opcenter Execution Core Modeling User Guide",
    content_type: str = "text",
    table_rows: list[list[str]] | None = None,
) -> dict:
    metadata = {
        "manual": manual,
        "release": "Release 2510+ Rev. 1",
        "chapter": "Chapter 2: Physical Model Definitions",
        "section": "Defining a Factory",
        "printed_page": "2-5",
        "pdf_page": 75,
        "chunk_level": "child",
    }
    if table_rows:
        metadata["table_rows"] = table_rows
    return {
        "chunk_id": chunk_id,
        "text": text,
        "content_type": content_type,
        "metadata": metadata,
        "retrieval_scores": {"final_score": 1.0},
    }


def fake_documents(question: str) -> list[dict]:
    lowered = question.casefold()
    if "field" in lowered or "table" in lowered:
        return [
            evidence(
                "table",
                "| Field | Description |\n| Name | Factory name |",
                content_type="field_definition",
                table_rows=[["Field", "Description"], ["Name", "Factory name"]],
            )
        ]
    if "compare" in lowered or "difference" in lowered:
        return [
            evidence("factory", "A Factory is an accounting and reporting division."),
            evidence("resource", "A Resource represents manufacturing equipment."),
        ]
    if "installation and modeling" in lowered or "cross-manual" in lowered:
        return [
            evidence("modeling", "The Modeling guide defines Factory objects."),
            evidence(
                "installation",
                "The Installation Guide describes system components.",
                manual="Opcenter Execution Core Installation Guide",
            ),
        ]
    if "how" in lowered or "procedure" in lowered or "steps" in lowered:
        return [
            evidence(
                "procedure",
                "How to Define a Factory\n1. Open the Factory page.\n2. Click New.\n3. Enter the Factory name.",
                content_type="procedure",
            )
        ]
    return [evidence("definition", "A Factory is an accounting and reporting division.")]


@pytest.fixture
def mocked_pipeline(monkeypatch: pytest.MonkeyPatch):
    def fake_structured(prompt, schema, *, task, **kwargs):
        question_match = re.search(r"(?:Current question|Question): (.+)", str(prompt))
        question = question_match.group(1).strip() if question_match else ""
        lowered = question.casefold()
        if schema is QueryPlan:
            standalone = question
            if re.search(r"\b(?:it|that|this)\b", lowered) and "factory" in str(prompt).casefold():
                standalone = "How do I define a Factory?"
            intent = (
                "procedure"
                if "how" in lowered or "steps" in lowered
                else "field_table"
                if "field" in lowered or "table" in lowered
                else "comparison"
                if "compare" in lowered or "difference" in lowered
                else "definition"
            )
            return QueryPlan(
                standalone_question=standalone,
                intent=intent,
                entities=(
                    []
                    if any(term in lowered for term in ("cake", "weather", "poem"))
                    else ["Factory"]
                ),
                search_queries=[standalone, f"Opcenter {standalone}"],
                preferred_manuals=["Modeling"],
                needs_diagram=bool(re.search(r"hierarch|process|architecture|relationship", lowered)),
            )
        if schema is EvidenceGrade:
            if any(term in lowered for term in ("cake", "weather", "poem")):
                return EvidenceGrade(status="out_of_scope", reason="Unrelated request")
            if any(term in lowered for term in ("kubernetes", "predictive ai", "blockchain")):
                return EvidenceGrade(
                    status="in_scope_insufficient",
                    reason="The supplied manuals do not support this Opcenter claim.",
                )
            if "force retry" in lowered:
                return EvidenceGrade(status="retry", reason="Broader search needed")
            return EvidenceGrade(status="sufficient", reason="Supported by [S1]")
        raise AssertionError(f"Unexpected schema: {schema}")

    def fake_plain(prompt, *, task, **kwargs):
        question_match = re.search(r"Question: (.+)", str(prompt))
        question = question_match.group(1).strip() if question_match else ""
        lowered = question.casefold()
        if task == "answer":
            if "field" in lowered or "table" in lowered:
                answer = "The `Name` field stores the Factory name [S1]."
            elif "compare" in lowered or "difference" in lowered:
                answer = "A Factory is a division [S1]; a Resource is equipment [S2]."
            elif "installation and modeling" in lowered or "cross-manual" in lowered:
                answer = "Modeling defines Factory [S1], while installation describes components [S2]."
            elif "how" in lowered or "steps" in lowered:
                answer = "1. Open the Factory page.\n2. Click New.\n3. Enter the name [S1]."
            else:
                answer = "A Factory is an accounting and reporting division [S1]."
            return AIMessage(content=answer)
        if task == "verifier":
            answer_match = re.search(
                r"Draft answer:\n(.*?)\nCited EvidenceUnits:", str(prompt), re.S
            )
            answer = answer_match.group(1).strip() if answer_match else ""
            return AIMessage(content=answer.replace("UNSUPPORTED", "").strip())
        if task == "diagram":
            return AIMessage(
                content='digraph G {\na [label="Enterprise [S1]"];\nb [label="Factory [S1]"];\nc [label="Resource [S1]"];\na -> b;\nb -> c;\n}'
            )
        if task == "query_broadening":
            return AIMessage(content="Opcenter Factory related sections")
        raise AssertionError(task)

    monkeypatch.setattr(nodes, "call_structured", fake_structured)
    monkeypatch.setattr(nodes, "call_llm", fake_plain)
    monkeypatch.setattr(
        nodes,
        "retrieve_multiple_queries",
        lambda standalone_query, *args, **kwargs: fake_documents(standalone_query),
    )
    monkeypatch.setattr(
        nodes,
        "expand_retrieval_context",
        lambda documents, **kwargs: documents,
    )
    monkeypatch.setattr(
        nodes,
        "cross_encoder_rerank",
        lambda query, documents, **kwargs: documents[:8],
    )
    monkeypatch.setattr(
        nodes,
        "resolve_evidence_units",
        lambda documents, **kwargs: documents[: kwargs["limit"]],
    )
    return fake_structured


def invoke(question: str, thread_id: str | None = None) -> dict:
    return graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "retry_count": 0,
            "allow_diagrams": True,
        },
        config={"configurable": {"thread_id": thread_id or str(uuid4())}},
    )


def test_direct_definition(mocked_pipeline) -> None:
    result = invoke("What is a Factory?")
    assert "accounting and reporting division" in result["answer"]
    assert result["evidence_status"] == "sufficient"


def test_indirect_paraphrased_question(mocked_pipeline) -> None:
    result = invoke("Which modeling object represents an accounting division?")
    assert result["standalone_question"]
    assert "Factory" in result["answer"]


def test_follow_up_uses_same_thread_id(mocked_pipeline) -> None:
    thread_id = str(uuid4())
    invoke("What is a Factory?", thread_id)
    result = invoke("How do I create it?", thread_id)
    assert result["standalone_question"] == "How do I define a Factory?"
    assert "1. Open the Factory page" in result["answer"]


def test_procedure_answer_preserves_steps(mocked_pipeline) -> None:
    result = invoke("How do I define a Factory?")
    assert "1." in result["answer"] and "2." in result["answer"] and "3." in result["answer"]


def test_table_and_field_definition_question(mocked_pipeline) -> None:
    result = invoke("Show the Factory field definition table")
    assert "`Name`" in result["answer"]
    assert result["reranked_docs"][0]["metadata"]["table_rows"]


def test_comparison_question(mocked_pipeline) -> None:
    result = invoke("Compare a Factory and a Resource")
    assert "[S1]" in result["answer"] and "[S2]" in result["answer"]


def test_cross_manual_question(mocked_pipeline) -> None:
    result = invoke("Give a cross-manual view of installation and modeling")
    assert len({source.source for source in result["sources"]}) == 2


def test_one_retry_limit(mocked_pipeline) -> None:
    result = invoke("Force retry for Opcenter Factory")
    assert result["retry_count"] == 1
    assert result["evidence_status"] == "in_scope_insufficient"
    assert "do not provide enough evidence" in result["answer"]


def test_relevant_but_unsupported_fallback(mocked_pipeline) -> None:
    result = invoke("Does Opcenter support Kubernetes autoscaling?")
    assert result["evidence_status"] == "in_scope_insufficient"
    assert result["diagram_dot"] is None
    assert "Opcenter-related" in result["answer"]


def test_insufficient_evidence_never_calls_answer_generation(
    mocked_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    def guarded_call(prompt, *, task, **kwargs):
        if task == "answer":
            raise AssertionError("answer generation must not run")
        return AIMessage(content="Opcenter Factory related sections")

    monkeypatch.setattr(nodes, "call_llm", guarded_call)

    result = invoke("Does Opcenter support Kubernetes autoscaling?")

    assert result["evidence_status"] == "in_scope_insufficient"


def test_irrelevant_fallback(mocked_pipeline) -> None:
    result = invoke("How do I bake a cake?")
    assert result["evidence_status"] == "out_of_scope"
    assert "unrelated" in result["answer"]


def test_valid_citation_ids_only(mocked_pipeline) -> None:
    result = invoke("Compare a Factory and a Resource")
    cited = set(re.findall(r"\[S(\d+)\]", result["answer"]))
    source_ids = {source.source_id[1:] for source in result["sources"]}
    assert cited == source_ids == {"1", "2"}


def test_unsupported_claims_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        nodes,
        "call_llm",
        lambda *args, **kwargs: AIMessage(content="Supported [S1]."),
    )
    update = nodes.verify_answer(
        {
            "standalone_question": "Question",
            "answer": "Supported [S1]. UNSUPPORTED",
            "reranked_docs": fake_documents("factory"),
        }
    )
    assert "UNSUPPORTED" not in update["answer"]


def test_diagram_only_when_evidence_supports_it(mocked_pipeline) -> None:
    supported = invoke("Show the Factory hierarchy relationship")
    unsupported = invoke("Show an Opcenter Kubernetes architecture")
    assert supported["diagram_dot"]
    assert unsupported["diagram_dot"] is None


def test_no_image_processing_code() -> None:
    forbidden = ("get_images", "pixmap", "pytesseract", "st.image", "opencv", "cv2", "from pil")
    code = "\n".join(
        path.read_text(encoding="utf-8").casefold()
        for path in (Path("src/ingest.py"), Path("app.py"))
    )
    assert not any(term in code for term in forbidden)
