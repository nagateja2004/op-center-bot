from pathlib import Path

import fitz
import pytest

import src.ingest as ingest
from src.config import Settings
from src.ingest import (
    EMBEDDING_TOKEN_LIMIT,
    INGESTION_PIPELINE_VERSION,
    Element,
    SectionGroup,
    _bm25_tokens,
    _build_evidence,
    _embedding_token_count,
    _evidence_specs,
    _effective_embedding_limit,
    _finish_ingestion_audit,
    _build_concept_index,
    _build_heading_index,
    _build_search_representations,
    _image_only_caption,
    _indexable_segments,
    _indexable_representations,
    _markdown_table,
    _new_ingestion_audit,
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


def test_oversized_procedure_step_is_split_without_losing_evidence() -> None:
    long_step = "1. " + " ".join(f"configuration{i}" for i in range(700))
    units, segments = build(
        SectionGroup(
            chapter="Designer",
            section="Defining a large configuration",
            elements=[Element(long_step, "procedure", 10, "2-4")],
        )
    )

    assert units[0]["procedure_steps"] == [long_step]
    assert len(segments) > 1
    assert all("Section: Defining a large configuration" in item["searchable_text"] for item in segments)
    assert all("\n1. " in item["searchable_text"] for item in segments)
    assert "Step 1 — Part 1" in segments[0]["searchable_text"]
    assert "Step 1 — Part 2" in segments[1]["searchable_text"]
    assert all(item["embedding_token_count"] <= _effective_embedding_limit() for item in segments)
    assert all(item["effective_embedding_limit"] == _effective_embedding_limit() for item in segments)
    audit = _finish_ingestion_audit(
        _new_ingestion_audit(Path("manual.pdf"), "Manual", []), units, segments
    )
    assert audit["oversized_steps_split"] == 1


def test_subsection_heading_path_and_compact_prefix_are_preserved() -> None:
    units, segments = build(
        SectionGroup(
            chapter="Chapter 2: Physical Model",
            section="Resources",
            subsection="Resource conditions",
            heading_path=(
                "Designer Guide",
                "Chapter 2: Physical Model",
                "Resources",
                "Resource conditions",
            ),
            elements=[
                Element("A resource condition controls availability.", "definition", 8, "2-6")
            ],
        )
    )

    metadata = units[0]["metadata"]
    assert metadata["subsection"] == "Resource conditions"
    assert metadata["heading_path"][-2:] == ["Resources", "Resource conditions"]
    assert metadata["content_type"] == "definition"
    assert "Subsection: Resource conditions" in segments[0]["searchable_text"]
    assert "Type: definition" in segments[0]["searchable_text"]
    assert "release" not in segments[0]["searchable_text"].casefold()


def test_definition_keeps_conditions_and_exceptions() -> None:
    specs = _evidence_specs(
        SectionGroup(
            chapter="Designer",
            section="Validate Event",
            elements=[
                Element("The Validate event checks a proposed field value.", "definition", 3, "1-3"),
                Element("It applies only when a Validate CLF is configured.", "definition", 3, "1-3"),
                Element("Exception: read-only fields are not changed.", "definition", 3, "1-3"),
            ],
        )
    )

    assert len(specs) == 1
    assert "only when" in specs[0].text
    assert "Exception:" in specs[0].text


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
        assert "Manual: Designer Guide" in segment["searchable_text"]
        assert "Chapter: Designer" in segment["searchable_text"]
        assert "Section: CDO Field Definitions" in segment["searchable_text"]
        assert "### CDO Field Definitions" in segment["searchable_text"]
        assert "| Field | Description |" in segment["searchable_text"]
        assert segment["embedding_token_count"] < EMBEDDING_TOKEN_LIMIT
    recovered_rows = [
        row
        for segment in segments
        for row in segment["metadata"]["table_rows"][1:]
    ]
    assert recovered_rows == rows[1:]


def test_complete_logical_table_remains_one_evidence_unit() -> None:
    rows = [
        ["Field", "Description"],
        *[
            [f"Field {number}", " ".join(f"detail{word}" for word in range(80))]
            for number in range(30)
        ],
    ]
    units, segments = build(
        SectionGroup(
            chapter="Designer",
            section="Complete Field Definitions",
            elements=[
                Element(
                    _markdown_table("Complete Field Definitions", rows),
                    "field_definition",
                    20,
                    "3-8",
                    rows,
                )
            ],
        )
    )

    assert len(units) == 1
    assert units[0]["structured_table"]["rows"] == rows[1:]
    assert {segment["evidence_id"] for segment in segments} == {units[0]["evidence_id"]}


def test_adjacent_matching_tables_merge_into_one_logical_evidence_unit() -> None:
    header = ["Field", "Description"]
    units, _ = build(
        SectionGroup(
            chapter="Designer",
            section="Resource Fields",
            elements=[
                Element("", "field_definition", 20, "3-8", [header, ["Name", "Name field"]]),
                Element("", "field_definition", 21, "3-9", [header, ["Type", "Type field"]]),
            ],
        )
    )

    assert len(units) == 1
    assert units[0]["structured_table"]["rows"] == [
        ["Name", "Name field"],
        ["Type", "Type field"],
    ]


def test_oversized_table_row_keeps_complete_structured_metadata() -> None:
    row = ["LongField", " ".join(f"meaning{i}" for i in range(600))]
    rows = [["Field", "Description"], row]
    units, segments = build(
        SectionGroup(
            chapter="Designer",
            section="Long Field Definitions",
            elements=[
                Element(
                    _markdown_table("Long Field Definitions", rows),
                    "field_definition",
                    20,
                    "3-8",
                    rows,
                )
            ],
        )
    )

    assert units[0]["structured_table"]["rows"] == [row]
    assert len(segments) > 1
    assert all(item["metadata"]["table_rows"] == rows for item in segments)
    assert all("Column:" in item["searchable_text"] for item in segments)
    assert all(item["embedding_token_count"] <= _effective_embedding_limit() for item in segments)


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


def test_contextless_warning_does_not_create_a_search_segment() -> None:
    units, segments = build(
        SectionGroup(
            chapter="Designer",
            section="Warnings",
            elements=[Element("Warning: qualifying content is missing.", "warning", 5, "1-2")],
        )
    )

    assert units == []
    assert segments == []


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
    assert all("Type: concept" in segment["searchable_text"] for segment in segments)
    assert all(
        segment["effective_embedding_limit"] == _effective_embedding_limit()
        for segment in segments
    )


def test_prose_segments_do_not_repeat_content_as_fixed_overlap() -> None:
    text = " ".join(
        f"UniqueMarker{i} explains one configurable condition."
        for i in range(120)
    )
    _, segments = build(
        SectionGroup(
            chapter="Designer",
            section="Configuration conditions",
            elements=[Element(text, "concept", 2, "1-1")],
        )
    )

    combined = "\n".join(segment["searchable_text"] for segment in segments)
    assert all(combined.count(f"UniqueMarker{i} ") == 1 for i in range(120))


def test_heading_and_concept_indexes_are_manual_derived() -> None:
    units, _ = build(
        SectionGroup(
            chapter="Modeling",
            section="Defining Numbering Rules",
            elements=[Element("A Numbering Rule assigns values to a Container.", "definition", 7, "1-5")],
        )
    )

    headings = _build_heading_index(units)
    concepts = _build_concept_index(units)

    assert headings[0]["normalized_heading"] == "defining numbering rules"
    assert "numbering rules" in headings[0]["alternate_keys"]
    assert headings[0]["is_toc"] is False
    assert headings[0]["evidence_ids"] == [units[0]["evidence_id"]]
    assert any(item["canonical_name"] == "Defining Numbering Rules" for item in concepts)


def test_concept_index_anchors_curated_aliases_to_manual_evidence() -> None:
    units, _ = build(
        SectionGroup(
            chapter="Designer",
            section="Field Events",
            elements=[
                Element(
                    "The field Initialize, Update, and Validate events call configured CLFs.",
                    "definition",
                    9,
                    "2-7",
                )
            ],
        )
    )

    concept = next(
        item for item in _build_concept_index(units)
        if item["canonical_name"] == "Validate Event"
    )

    assert concept["evidence_ids"] == [units[0]["evidence_id"]]
    assert "validate field value" in concept["aliases"]


def test_noncontent_pages_and_image_only_captions_are_detected() -> None:
    assert _noncontent_context("Contents", "Contents")
    assert _noncontent_context("Appendix", "Index")
    assert _image_only_caption("Figure 4-2. Configuration screen")
    assert not _image_only_caption(
        "Figure behavior is configured by events and methods in this explanatory paragraph."
    )


def test_toc_entries_are_excluded_while_matching_body_is_retained(tmp_path: Path) -> None:
    path = tmp_path / "manual.pdf"
    doc = fitz.open()
    toc_page = doc.new_page()
    toc_page.insert_text((72, 72), "Table of Contents", fontsize=18)
    toc_page.insert_text(
        (72, 110),
        "Defining Numbering Rules ........ 2\nPhysical Modeling Sequence ........ 3",
        fontsize=11,
    )
    body_page = doc.new_page()
    body_page.insert_text((72, 72), "Defining Numbering Rules", fontsize=18)
    body_page.insert_text(
        (72, 110),
        "Numbering rules assign unique values to containers using prefixes and sequences.",
        fontsize=11,
    )
    doc.set_metadata({"title": "Modeling User Guide"})
    doc.set_toc([[1, "Modeling", 2], [2, "Defining Numbering Rules", 2]])
    doc.save(path)
    doc.close()

    units, segments = ingest.ingest_pdf(path, Settings(groq_api_key="test"))

    all_evidence = "\n".join(unit["text"] for unit in units)
    assert "assign unique values" in all_evidence
    assert "Physical Modeling Sequence ........ 3" not in all_evidence
    assert any("Section: Defining Numbering Rules" in item["searchable_text"] for item in segments)


def test_unique_top_of_page_heading_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "manual.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 28), "Unique Configuration Heading", fontsize=18)
    page.insert_text((72, 90), "This body explains a unique Opcenter configuration concept.")
    doc.set_metadata({"title": "Designer Guide"})
    doc.save(path)
    doc.close()

    units, _ = ingest.ingest_pdf(path, Settings(groq_api_key="test"))

    text = "\n".join(unit["text"] for unit in units)
    assert "Unique Configuration Heading" in text
    assert "unique Opcenter configuration concept" in text


def test_pdf_bookmarks_preserve_chapter_section_and_subsection(tmp_path: Path) -> None:
    path = tmp_path / "manual.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Availability Conditions", fontsize=16)
    page.insert_text((72, 110), "A resource is available only when its status permits use.")
    doc.set_metadata({"title": "Modeling Guide"})
    doc.set_toc(
        [
            [1, "Chapter 2: Physical Model", 1],
            [2, "Resources", 1],
            [3, "Availability Conditions", 1],
        ]
    )
    doc.save(path)
    doc.close()

    units, _ = ingest.ingest_pdf(path, Settings(groq_api_key="test"))

    metadata = units[0]["metadata"]
    assert metadata["chapter"] == "Chapter 2: Physical Model"
    assert metadata["section"] == "Resources"
    assert metadata["subsection"] == "Availability Conditions"
    assert metadata["heading_path"] == [
        "Modeling Guide",
        "Chapter 2: Physical Model",
        "Resources",
        "Availability Conditions",
    ]


def test_repeated_headers_and_discarded_blocks_are_audited(tmp_path: Path) -> None:
    path = tmp_path / "manual.pdf"
    doc = fitz.open()
    toc = doc.new_page()
    toc.insert_text((72, 72), "Table of Contents", fontsize=18)
    toc.insert_text((72, 110), "Configuration ........ 2")
    for number in range(3):
        page = doc.new_page()
        page.insert_text((72, 24), "Repeated Manual Header")
        page.insert_text((72, 820), "Repeated Manual Footer")
        page.insert_text((72, 90), f"Configuration Topic {number}", fontsize=16)
        page.insert_text((72, 130), f"Useful Opcenter body text for topic {number}.")
        if number == 0:
            page.insert_text((72, 180), "Figure 1. Configuration screen")
    doc.set_metadata({"title": "Designer Guide"})
    doc.save(path)
    doc.close()

    units, _, entry = ingest._ingest_pdf(path, Settings(groq_api_key="test"))
    audit = entry["_audit"]

    assert "Repeated Manual Header" not in "\n".join(unit["text"] for unit in units)
    assert "Repeated Manual Footer" not in "\n".join(unit["text"] for unit in units)
    assert audit["repeated_headers_removed"] == 3
    assert audit["repeated_footers_removed"] == 3
    assert audit["toc_index_pages_excluded"] == 1
    assert audit["captions_removed"] == 1
    assert audit["discarded_block_categories"]["image_only_caption"] == 1
    assert (
        audit["text_blocks_indexed"]
        + sum(audit["discarded_block_categories"].values())
        == audit["text_blocks_extracted"]
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
    for name in (
        "BM25_INDEX_PATH",
        "EVIDENCE_UNITS_PATH",
        "RETRIEVAL_SEGMENTS_PATH",
        "SEARCH_REPRESENTATIONS_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
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
        "metadata": {
            "source_file": "manual.pdf",
            "manual": "Manual",
            "chapter": "Chapter",
            "section": "Definition",
            "pdf_page": 1,
        },
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
        "metadata": {
            "source_file": "manual.pdf",
            "manual": "Manual",
            "chapter": "Chapter",
            "section": "Definition",
            "pdf_page": 1,
        },
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

    manifest_path = config.indexes_dir / "manifest.json"
    manifest = ingest._load_json(manifest_path, {})
    manifest["ingestion_pipeline_version"] = "old-pipeline"
    ingest._write_json(manifest_path, manifest)
    third = ingest.ingest_manuals(config)

    assert calls == 2
    assert third["processed"] == ["manual.pdf"]
    assert third["version"] == 8
    assert third["ingestion_pipeline_version"] == INGESTION_PIPELINE_VERSION
    audit = ingest._load_json(config.ingestion_audit_path, {})
    assert audit["schema_version"] == ingest.INDEX_SCHEMA_VERSION
    assert "manual.pdf" in audit["manuals"]


def test_unsupported_document_parser_is_rejected(tmp_path: Path) -> None:
    config = Settings(
        groq_api_key="test",
        manuals_dir=tmp_path,
        indexes_dir=tmp_path / "indexes",
        document_parser="docling",
    )

    with pytest.raises(ValueError, match="only 'pymupdf' is supported"):
        ingest.ingest_manuals(config)


def test_bm25_tokenizer_normalizes_searchable_table_text() -> None:
    assert _bm25_tokens("| Field Name | Lot-ID |") == ["field", "name", "lot", "id"]


def test_search_representations_cover_all_deterministic_types() -> None:
    source = {
        "source_file": "manual.pdf",
        "manual": "Modeling Guide",
        "chapter": "Definitions",
        "pdf_page": 4,
    }
    units = [
        {
            "evidence_id": "definition",
            "text": "Work In Process (WIP) means material currently being processed.",
            "content_type": "definition",
            "metadata": {**source, "section": "Work In Process"},
            "token_count": 10,
            "structured_table": None,
            "procedure_steps": [],
            "annotations": [],
        },
        {
            "evidence_id": "procedure",
            "text": "1. Open the page.\n2. Save the record.",
            "content_type": "procedure",
            "metadata": {**source, "section": "Configuring a Resource"},
            "token_count": 10,
            "structured_table": None,
            "procedure_steps": ["1. Open the page.", "2. Save the record."],
            "annotations": [],
        },
        {
            "evidence_id": "fields",
            "text": "Resource fields",
            "content_type": "field_definition",
            "metadata": {**source, "section": "Resource Fields"},
            "token_count": 5,
            "structured_table": {
                "title": "Resource Fields",
                "headers": ["Field", "Description"],
                "rows": [["Name", "Resource name"]],
            },
            "procedure_steps": [],
            "annotations": [],
        },
    ]

    representations = _build_search_representations(units, Settings(groq_api_key="test"))
    validated = _indexable_representations(
        representations, {unit["evidence_id"] for unit in units}
    )

    assert {item["representation_type"] for item in validated} == {
        "heading",
        "definition",
        "procedure_title",
        "table_title_headers",
        "field_name_description",
        "acronym_alias",
    }
    assert all(item["evidence_id"] in {"definition", "procedure", "fields"} for item in validated)
    assert all(item["embedding_token_count"] < EMBEDDING_TOKEN_LIMIT for item in validated)


def test_search_representation_rejects_missing_parent() -> None:
    with pytest.raises(ValueError, match="missing EvidenceUnits"):
        _indexable_representations(
            [
                {
                    "representation_id": "r1",
                    "evidence_id": "missing",
                    "representation_type": "heading",
                    "text": "Heading: Missing",
                    "metadata": {},
                    "embedding_token_count": 2,
                }
            ],
            set(),
        )


def test_search_representations_remove_global_exact_vector_duplicates() -> None:
    metadata = {
        "source_file": "manual.pdf",
        "manual": "Manual",
        "chapter": "Chapter",
        "section": "Repeated Heading",
        "pdf_page": 1,
    }
    units = [
        {
            "evidence_id": evidence_id,
            "text": text,
            "content_type": "text",
            "metadata": metadata,
            "token_count": 2,
            "structured_table": None,
            "procedure_steps": [],
            "annotations": [],
        }
        for evidence_id, text in (("e1", "First body."), ("e2", "Second body."))
    ]

    representations = _build_search_representations(
        units, Settings(groq_api_key="test")
    )

    assert [item["representation_type"] for item in representations] == ["heading"]
    assert representations[0]["evidence_id"] == "e1"
