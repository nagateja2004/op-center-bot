from copy import deepcopy

from src.compression import compress_evidence


def document(
    text: str,
    *,
    content_type: str = "definition",
    metadata: dict | None = None,
) -> dict:
    return {
        "chunk_id": "e1",
        "text": text,
        "content_type": content_type,
        "metadata": {
            "evidence_id": "e1",
            "manual": "Manual",
            "source_file": "manual.pdf",
            "chapter": "Chapter",
            "section": "Section",
            "pdf_page": 1,
            **(metadata or {}),
        },
        "retrieval_scores": {"reranker_score": 0.9},
    }


def test_concept_compression_is_deterministic_and_does_not_mutate_parent() -> None:
    parent = document(
        "A container holds shop-floor material. It has a unique identifier. "
        "If a container is closed, movement is blocked. Unrelated history is archived."
    )
    original = deepcopy(parent)

    first = compress_evidence(parent, "container movement", "When is movement blocked?")
    second = compress_evidence(parent, "container movement", "When is movement blocked?")

    assert first == second
    assert "movement is blocked" in first["compressed_text"]
    assert first["selected_sentence_indexes"]
    assert parent == original


def test_procedure_compression_preserves_selected_step_order_and_complete_request() -> None:
    steps = [f"{number}. Perform operation {number}." for number in range(1, 8)]
    parent = document(
        "\n".join(steps),
        content_type="procedure",
        metadata={"procedure_steps": steps},
    )

    concise = compress_evidence(parent, "operation 5", "How do I perform operation 5?")
    complete = compress_evidence(
        parent,
        "procedure",
        "Show the complete procedure",
        include_complete_procedure=True,
    )

    assert concise["selected_step_indexes"] == sorted(concise["selected_step_indexes"])
    assert len(concise["selected_step_indexes"]) <= 4
    assert complete["selected_step_indexes"] == list(range(7))
    assert complete["compressed_text"].endswith(steps[-1])


def test_table_compression_keeps_headers_and_complete_relevant_rows() -> None:
    rows = [
        ["Field", "Description"],
        ["Name", "Factory name"],
        ["Runtime", "Execution behavior"],
        ["Status", "Current runtime status"],
    ]
    parent = document(
        "Rendered table",
        content_type="field_definition",
        metadata={"table_rows": rows},
    )

    view = compress_evidence(parent, "runtime fields", "Explain runtime fields")

    assert "| Field | Description |" in view["compressed_text"]
    assert "| Runtime | Execution behavior |" in view["compressed_text"]
    assert view["selected_table_row_indexes"]
    assert all(len(row) == 2 for row in rows)
