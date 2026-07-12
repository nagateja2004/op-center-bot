from langchain_core.messages import AIMessage

import src.nodes as nodes
from src.config import settings
from src.prompts import ANSWER_GENERATION_PROMPT
from src.schemas import EvidenceGrade


def document(
    number: int,
    *,
    aspect: str = "configuration",
    content_type: str = "text",
    text: str | None = None,
) -> dict:
    return {
        "chunk_id": f"e{number}",
        "text": text or (f"Relevant evidence {number}. " * 300),
        "content_type": content_type,
        "metadata": {
            "evidence_id": f"e{number}",
            "manual": "Modeling Guide",
            "release": "2510+",
            "section": f"Section {number}",
            "pdf_page": number,
            "aspects": [aspect],
        },
        "retrieval_scores": {
            "vector_rank": float(number),
            "bm25_score": 1 / number,
            "rrf_score": 1 / (60 + number),
            "reranker_score": 1 - number / 100,
        },
    }


def test_grader_gets_two_compact_summaries_without_duplicate_table_payload(
    monkeypatch,
) -> None:
    captured: dict[str, str] = {}
    rows = [["Field", "Description"], ["Name", "Factory name"], ["Type", "Factory type"]]
    documents = [document(index) for index in range(1, 4)]
    documents[0]["metadata"]["table_rows"] = rows
    documents[0]["text"] = "DUPLICATE_MARKDOWN " + ("full parent section " * 300)

    def fake_grade(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return EvidenceGrade(status="sufficient", reason="supported")

    monkeypatch.setattr(nodes, "call_structured", fake_grade)
    nodes.grade_evidence(
        {
            "standalone_question": "Show Factory fields",
            "required_aspects": ["Factory fields"],
            "aspect_documents": {"Factory fields": documents},
        }
    )

    prompt = captured["prompt"]
    assert "Source ID: S1" in prompt and "Source ID: S2" in prompt
    assert "Source ID: S3" not in prompt
    assert "Evidence ID:" in prompt and "Assigned aspect: Factory fields" in prompt
    assert all(name in prompt for name in ("vector_rank", "bm25_score", "rrf_score", "reranker_score"))
    assert "DUPLICATE_MARKDOWN" not in prompt
    assert "table_rows" not in prompt
    assert nodes._estimated_tokens(prompt) <= settings.grader_input_token_budget


def test_answer_budget_keeps_one_unique_source_per_supported_aspect() -> None:
    aspects = [f"aspect-{index}" for index in range(1, 7)]
    documents = [document(index, aspect=aspects[(index - 1) % 6]) for index in range(1, 11)]
    selected = nodes._select_answer_documents(documents, aspects)
    evidence = nodes._format_answer_evidence(
        selected,
        "Explain configuration and runtime",
        ["explanation"],
        max_chars=settings.answer_input_token_budget * 4 - 1_400,
    )
    prompt = nodes._trim_to_token_budget(
        ANSWER_GENERATION_PROMPT.format(
            standalone_question="Explain configuration and runtime",
            required_output="explanation",
            supported_aspects=" | ".join(aspects),
            missing_aspects="none",
            evidence=evidence,
            answer_structure="- Direct explanation",
        ),
        settings.answer_input_token_budget,
    )

    assert len(selected) == 10
    assert len({item["metadata"]["evidence_id"] for item in selected}) == 10
    assert all(any(aspect in item["metadata"]["aspects"] for item in selected) for aspect in aspects)
    assert all(f"[S{index}]" in prompt for index in range(1, 11))
    assert nodes._estimated_tokens(prompt) <= settings.answer_input_token_budget


def test_answer_uses_concise_table_rows_and_gates_complete_procedure() -> None:
    table = document(1, content_type="table", text="DUPLICATE_TABLE_MARKDOWN")
    table["metadata"]["table_rows"] = [
        ["Field", "Description"],
        ["Name", "Factory name"],
        ["Runtime", "Execution behavior"],
    ]
    steps = "\n".join(f"{number}. Perform operation {number}." for number in range(1, 80))
    procedure = document(2, content_type="procedure", text=steps)

    procedural = nodes._format_answer_evidence(
        [table, procedure], "Factory procedure", ["procedure"], max_chars=12_000
    )
    explanatory = nodes._format_answer_evidence(
        [table, procedure], "Factory concept", ["explanation"], max_chars=12_000
    )

    assert "DUPLICATE_TABLE_MARKDOWN" not in procedural
    assert procedural.count("| Field | Description |") == 1
    assert "79. Perform operation 79." in procedural
    assert "79. Perform operation 79." not in explanatory


def test_verifier_receives_only_cited_evidence_units(monkeypatch) -> None:
    captured: dict[str, str] = {}
    documents = [document(index) for index in range(1, 4)]

    def fake_verify(prompt, **kwargs):
        captured["prompt"] = prompt
        return AIMessage(content="Supported [S2].")

    monkeypatch.setattr(nodes, "call_llm", fake_verify)
    nodes.verify_answer(
        {
            "standalone_question": "Question",
            "required_aspects": ["runtime"],
            "answer": "Supported [S2].",
            "reranked_docs": documents,
        }
    )

    prompt = captured["prompt"]
    evidence = prompt.split("Cited EvidenceUnits:\n", 1)[1]
    assert "evidence_id=e2" in evidence
    assert "evidence_id=e1" not in evidence and "evidence_id=e3" not in evidence
    assert nodes._estimated_tokens(prompt) <= settings.verifier_input_token_budget


def test_diagram_receives_only_verified_entities_relationships_and_sources(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_diagram(prompt, **kwargs):
        captured["prompt"] = prompt
        return AIMessage(
            content='digraph G { a [label="Factory [S1]"]; b [label="Resource [S1]"]; c [label="Runtime [S1]"]; a -> b; b -> c; }'
        )

    monkeypatch.setattr(nodes, "call_llm", fake_diagram)
    nodes.generate_diagram(
        {
            "standalone_question": "Show the runtime process",
            "entities": ["Factory", "Resource"],
            "needs_diagram": True,
            "grounded": True,
            "answer": "Uncited private draft. Factory uses Resource at Runtime [S1].",
            "reranked_docs": [document(1)],
        }
    )

    prompt = captured["prompt"]
    assert "Diagram type: process" in prompt
    assert "Factory uses Resource at Runtime [S1]" in prompt
    assert "Uncited private draft" not in prompt
    assert "Supporting source IDs: S1" in prompt
    assert "Relevant evidence" not in prompt
    assert nodes._estimated_tokens(prompt) <= settings.diagram_input_token_budget
