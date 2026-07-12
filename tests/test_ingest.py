from pathlib import Path

import pytest

import src.ingest as ingest
from src.config import Settings
from src.ingest import (
    EMBEDDING_TOKEN_LIMIT,
    Element,
    SectionGroup,
    _bm25_tokens,
    _build_evidence,
    _embedding_token_count,
    _evidence_specs,
    _image_only_caption,
    _indexable_segments,
    _markdown_table,
    _noncontent_context,
)


def build(group: SectionGroup):
    return _build_evidence(
        [group],
        source_file="designer.pdf",
        file_hash="hash",
        manual="Designer Guide",
        release="2510",
        config=Settings(groq_api_key="test"),
    )


def test_evidence_unit_preserves_metadata_and_ordered_procedure_steps() -> None:
    steps = [f"{number}. Perform operation {number}." for number in range(1, 101)]
    units, segments = build(
        SectionGroup(
            chapter="Designer",
            section="How to configure a CDO",
            elements=[Element("\n".join(steps), "procedure", 10, "2-4")],
        )
    )

    ordered = [step for unit in units for step in unit["procedure_steps"]]
    assert ordered == steps
    assert all(unit["metadata"]["manual"] == "Designer Guide" for unit in units)
    assert all(unit["metadata"]["printed_page"] == "2-4" for unit in units)
    assert all(unit["metadata"]["pdf_page"] == 10 for unit in units)
    assert len(units) == 1
    assert all(segment["evidence_id"] in {unit["evidence_id"] for unit in units} for segment in segments)
    assert all(segment["embedding_token_count"] < EMBEDDING_TOKEN_LIMIT for segment in segments)
    combined = "\n".join(segment["searchable_text"] for segment in segments)
    assert all(combined.count(step) == 1 for step in steps)


def test_table_segments_repeat_title_and_headers_without_splitting_rows() -> None:
    rows = [["Field", "Description"], *[[f"Field {i}", "word " * 35] for i in range(12)]]
    units, segments = build(
        SectionGroup(
            chapter="Designer",
            section="CDO Field Definitions",
            elements=[
                Element(
                    _markdown_table("CDO Field Definitions", rows),
                    "field_definition",
                    20,
                    "3-8",
                    rows,
                )
            ],
        )
    )

    assert all(unit["structured_table"] for unit in units)
    assert len(segments) > 1
    for segment in segments:
        assert "### CDO Field Definitions" in segment["searchable_text"]
        assert "| Field | Description |" in segment["searchable_text"]
        assert segment["embedding_token_count"] < EMBEDDING_TOKEN_LIMIT
    recovered_rows = [
        row
        for segment in segments
        for row in segment["metadata"]["table_rows"][1:]
    ]
    assert recovered_rows == rows[1:]


def test_notes_and_warnings_stay_attached_to_related_evidence() -> None:
    specs = _evidence_specs(
        SectionGroup(
            chapter="Designer",
            section="CDO Events",
            elements=[
                Element("Events trigger configured behavior.", "concept", 5, "1-2"),
                Element("Warning: save the CDO first.", "warning", 5, "1-2"),
            ],
        )
    )

    assert len(specs) == 1
    assert "Events trigger" in specs[0].text
    assert "Warning:" in specs[0].text
    assert specs[0].annotations[0]["type"] == "warning"


def test_prose_segments_respect_word_and_token_limits() -> None:
    text = " ".join(f"Concept sentence {i} explains configurable behavior." for i in range(100))
    _, segments = build(
        SectionGroup(
            chapter="Designer",
            section="CDO concepts",
            elements=[Element(text, "concept", 2, "1-1")],
        )
    )

    assert len(segments) > 1
    assert all(segment["word_count"] <= 220 for segment in segments)
    assert all(segment["embedding_token_count"] < EMBEDDING_TOKEN_LIMIT for segment in segments)
    assert all(
        segment["embedding_token_count"]
        == _embedding_token_count(segment["searchable_text"])
        for segment in segments
    )


def test_noncontent_pages_and_image_only_captions_are_detected() -> None:
    assert _noncontent_context("Contents", "Contents")
    assert _noncontent_context("Appendix", "Index")
    assert _image_only_caption("Figure 4-2. Configuration screen")
    assert not _image_only_caption(
        "Figure behavior is configured by events and methods in this explanatory paragraph."
    )


def test_index_contract_rejects_duplicate_segment_ids() -> None:
    segments = [
        {"segment_id": "duplicate", "evidence_id": "e1"},
        {"segment_id": "duplicate", "evidence_id": "e1"},
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        _indexable_segments(segments)  # type: ignore[arg-type]


def test_index_contract_rejects_segments_over_embedding_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment = {
        "segment_id": "s1",
        "evidence_id": "e1",
        "searchable_text": "oversized",
        "metadata": {
            "manual": "Manual",
            "source_file": "manual.pdf",
            "chapter": "Chapter",
            "section": "Section",
            "pdf_page": 1,
        },
    }
    monkeypatch.setattr(ingest, "_embedding_token_count", lambda text, config: 512)

    with pytest.raises(ValueError, match="embedding tokens"):
        _indexable_segments([segment], Settings(groq_api_key="test"))  # type: ignore[arg-type]


def test_chroma_metadata_contains_evidence_and_source_fields() -> None:
    _, segments = build(
        SectionGroup(
            chapter="Designer",
            section="CDO concepts",
            elements=[Element("A configurable object concept.", "concept", 2, "1-1")],
        )
    )

    metadata = ingest._chroma_metadata(segments[0])

    assert metadata["segment_id"] == segments[0]["segment_id"]
    assert metadata["evidence_id"] == segments[0]["evidence_id"]
    assert metadata["manual"] == "Designer Guide"
    assert metadata["source_file"] == "designer.pdf"


def test_unchanged_hash_reuses_both_levels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manuals = tmp_path / "manuals"
    manuals.mkdir()
    (manuals / "manual.pdf").touch()
    config = Settings(
        groq_api_key="test",
        manuals_dir=manuals,
        indexes_dir=tmp_path / "indexes",
        chroma_dir=tmp_path / "indexes" / "chroma",
        sqlite_path=tmp_path / "data.sqlite",
    )
    unit = {
        "evidence_id": "e1",
        "text": "Definition",
        "content_type": "definition",
        "metadata": {"source_file": "manual.pdf", "manual": "Manual"},
        "token_count": 1,
        "structured_table": None,
        "procedure_steps": [],
        "annotations": [],
    }
    segment = {
        "segment_id": "s1",
        "evidence_id": "e1",
        "searchable_text": "Definition",
        "content_type": "definition",
        "metadata": {"source_file": "manual.pdf", "manual": "Manual"},
        "segment_index": 0,
        "previous_segment_id": None,
        "next_segment_id": None,
        "word_count": 1,
        "embedding_token_count": 3,
    }
    calls = 0

    def fake_ingest(path, passed_config):
        nonlocal calls
        calls += 1
        return [unit], [segment], {
            "sha256": "same",
            "manual": "Manual",
            "release": "",
            "pdf_pages": 1,
            "evidence_unit_count": 1,
            "retrieval_segment_count": 1,
        }

    monkeypatch.setattr(ingest, "_file_hash", lambda path: "same")
    monkeypatch.setattr(ingest, "_ingest_pdf", fake_ingest)
    monkeypatch.setattr(ingest, "build_indexes", lambda segments, config: 1)
    monkeypatch.setattr(ingest, "validate_indexes", lambda config: 1)

    first = ingest.ingest_manuals(config)
    second = ingest.ingest_manuals(config)

    assert calls == 1
    assert first["processed"] == ["manual.pdf"]
    assert second["skipped_unchanged"] == ["manual.pdf"]


def test_bm25_tokenizer_normalizes_searchable_table_text() -> None:
    assert _bm25_tokens("| Field Name | Lot-ID |") == ["field", "name", "lot", "id"]
