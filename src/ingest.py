"""Deterministic, text-only ingestion for Opcenter PDF manuals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
import json
import logging
import math
from pathlib import Path
import pickle
import re
import shutil
from statistics import median
from typing import Any, Literal, Sequence

import fitz

from src.config import Settings, settings
from src.schemas import (
    EvidenceUnit,
    RetrievalSegment,
    SearchRepresentation,
    SearchRepresentationType,
)


ContentType = Literal[
    "text",
    "concept",
    "definition",
    "procedure",
    "prerequisite",
    "table",
    "field_definition",
    "note",
    "warning",
]

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PRINTED_PAGE_RE = re.compile(r"^(?:[A-Z]-)?\d+-\d+$|^[ivxlcdm]{1,12}$|^\d+$", re.I)
CHAPTER_RE = re.compile(r"^(?:chapter\s+\d+|appendix\s+[A-Z])\b", re.I)
PROCEDURE_RE = re.compile(r"^(?:how to\b|procedure\b|to\s+\w+)", re.I)
NOTE_RE = re.compile(r"^(?:note|important)\s*[:.]?\s*", re.I)
WARNING_RE = re.compile(r"^(?:warning|caution|danger)\s*[:.]?\s*", re.I)
DEFINITION_RE = re.compile(r"^(?:what is\b|definitions?\b|.+\bdefinitions?\b)", re.I)
PREREQUISITE_RE = re.compile(r"\b(?:prerequisites?|before you (?:begin|start)|requirements?)\b", re.I)
NONCONTENT_RE = re.compile(r"^(?:table of )?contents$|^index$|^list of (?:figures|tables)$", re.I)
TOC_LINE_RE = re.compile(r"^.{2,120}?(?:\.{2,}|\s{2,})\s*(?:[A-Z]-)?\d+(?:-\d+)?$", re.I)
COPYRIGHT_RE = re.compile(
    r"(?:\bcopyright\b|©|all rights reserved|siemens industry software)", re.I
)
IMAGE_CAPTION_RE = re.compile(
    r"^(?:figure|fig\.|image|illustration|screen(?:shot)?)\s+"
    r"(?:[A-Z]?\d[\w.-]*|[ivxlcdm]+)\b\s*[:.-]?",
    re.I,
)
STEP_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+", re.M)
STEP_MARKER_RE = re.compile(r"^\s*(\d{1,3})[.)]\s*$")
FIELD_RE = re.compile(r"\b(?:field|property|parameter|column|button)s?\b", re.I)
DESCRIPTION_RE = re.compile(r"\b(?:description|definition|meaning|value)s?\b", re.I)
TITLE_TERM_RE = re.compile(
    r"\b(?:[A-Z]{2,}|[A-Z][a-z]+)(?:\s+(?:[A-Z]{2,}|[A-Z][a-z]+)){0,4}\b"
)

TARGET_SEGMENT_WORDS = 185
MIN_SEGMENT_WORDS = 150
MAX_SEGMENT_WORDS = 220
MAX_EVIDENCE_TOKENS = 1_200
EMBEDDING_TOKEN_LIMIT = 512
EMBEDDING_SAFETY_MARGIN = 8
CHROMA_COLLECTION = "opcenter_manuals"
REPRESENTATION_COLLECTION = "opcenter_manual_representations"
INDEX_SCHEMA_VERSION = 8
INGESTION_PIPELINE_VERSION = "text-only-pymupdf-hierarchical-v8.0"
REINGEST_COMMAND = "python -m src.ingest"
logger = logging.getLogger(__name__)


class IndexSchemaMismatchError(RuntimeError):
    """Existing index artifacts require an explicit schema rebuild."""


def _validate_parser(config: Settings) -> None:
    if config.document_parser.casefold() != "pymupdf":
        raise ValueError(
            f"Unsupported DOCUMENT_PARSER {config.document_parser!r}; "
            "only 'pymupdf' is supported."
        )


def _require_index_schema(config: Settings) -> None:
    manifest = _load_json(config.indexes_dir / "manifest.json", {})
    version = manifest.get("version") if isinstance(manifest, dict) else None
    pipeline_version = manifest.get("ingestion_pipeline_version") if isinstance(manifest, dict) else None
    if version != INDEX_SCHEMA_VERSION or pipeline_version != INGESTION_PIPELINE_VERSION:
        raise IndexSchemaMismatchError(
            "Index schema or ingestion pipeline changed "
            f"(found schema={version or 'none'}, pipeline={pipeline_version or 'none'}; "
            f"required schema={INDEX_SCHEMA_VERSION}, pipeline={INGESTION_PIPELINE_VERSION}). "
            f"Existing indexes were not modified. Run `{REINGEST_COMMAND}` explicitly."
        )


@dataclass(slots=True)
class TextBlock:
    text: str
    size: float
    bbox: tuple[float, float, float, float]
    bold: bool = False


@dataclass(slots=True)
class PageData:
    pdf_page: int
    printed_page: str | None
    width: float
    height: float
    blocks: list[TextBlock]
    is_toc: bool = False


@dataclass(slots=True)
class Element:
    text: str
    content_type: ContentType
    pdf_page: int
    printed_page: str | None
    table_rows: list[list[str]] | None = None


@dataclass(slots=True)
class SectionGroup:
    chapter: str
    section: str
    subsection: str | None = None
    heading_path: tuple[str, ...] = ()
    elements: list[Element] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceSpec:
    text: str
    content_type: ContentType
    pdf_page: int
    printed_page: str | None
    table_rows: list[list[str]] | None = None
    end_pdf_page: int | None = None
    procedure_steps: list[str] = field(default_factory=list)
    annotations: list[dict[str, str]] = field(default_factory=list)


def _clean(value: Any) -> str:
    text = str(value or "").replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_margin_text(text: str) -> str:
    text = _clean(text).lower()
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"\s+", " ", text)


def _extract_page_data(doc: fitz.Document) -> tuple[list[PageData], set[str], float]:
    pages: list[PageData] = []
    margin_counts: Counter[str] = Counter()
    body_sizes: list[float] = []

    for pdf_page, page in enumerate(doc, start=1):
        blocks: list[TextBlock] = []
        page_dict = page.get_text("dict", sort=True, flags=fitz.TEXTFLAGS_TEXT)
        for raw_block in page_dict.get("blocks", []):
            if raw_block.get("type") != 0:
                continue
            lines: list[str] = []
            sizes: list[float] = []
            bold = False
            for line in raw_block.get("lines", []):
                spans = line.get("spans", [])
                line_text = _clean("".join(str(span.get("text", "")) for span in spans))
                if not line_text:
                    continue
                lines.append(line_text)
                sizes.extend(float(span.get("size", 0)) for span in spans if span.get("size"))
                bold = bold or any(
                    "bold" in str(span.get("font", "")).lower() for span in spans
                )
            text = "\n".join(lines)
            if not text:
                continue
            bbox = tuple(float(value) for value in raw_block.get("bbox", (0, 0, 0, 0)))
            size = max(sizes, default=10.0)
            blocks.append(TextBlock(text=text, size=size, bbox=bbox, bold=bold))
            if bbox[1] > page.rect.height * 0.08 and bbox[3] < page.rect.height * 0.9:
                body_sizes.extend(value for value in sizes if 7 <= value <= 14)
            if bbox[3] <= page.rect.height * 0.1 or bbox[1] >= page.rect.height * 0.9:
                margin_counts[_normalized_margin_text(text)] += 1

        printed_page = _printed_page(page, blocks)
        pages.append(
            PageData(
                pdf_page=pdf_page,
                printed_page=printed_page,
                width=float(page.rect.width),
                height=float(page.rect.height),
                blocks=blocks,
                is_toc=_is_toc_page(blocks),
            )
        )

    threshold = max(3, math.ceil(len(pages) * 0.15))
    repeated = {text for text, count in margin_counts.items() if text and count >= threshold}
    return pages, repeated, median(body_sizes) if body_sizes else 10.0


def _is_toc_page(blocks: list[TextBlock]) -> bool:
    """Identify navigation pages without suppressing matching body sections."""
    lines = [_clean(line) for block in blocks for line in block.text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return False
    first = " ".join(lines[:8])
    if re.search(
        r"\b(?:table of contents|contents|index|list of figures|list of tables)\b",
        first,
        re.I,
    ):
        return True
    if NONCONTENT_RE.fullmatch(lines[0]):
        return True
    patterned = sum(bool(TOC_LINE_RE.match(line)) for line in lines)
    short_numbered = sum(
        bool(re.match(r"^.{2,100}\s+(?:[A-Z]-)?\d+(?:-\d+)?$", line))
        for line in lines
    )
    return len(lines) >= 8 and max(patterned, short_numbered) / len(lines) >= 0.6


def _printed_page(page: fitz.Page, blocks: list[TextBlock]) -> str | None:
    label = _clean(page.get_label())
    if label:
        return label
    candidates: list[str] = []
    for block in blocks:
        if block.bbox[1] < page.rect.height * 0.86:
            continue
        for line in reversed(block.text.splitlines()):
            line = _clean(line)
            if PRINTED_PAGE_RE.fullmatch(line):
                candidates.append(line)
    return candidates[-1] if candidates else None


def _heading_path(
    manual: str, chapter: str, section: str, subsection: str | None = None
) -> tuple[str, ...]:
    path: list[str] = []
    for value in (manual, chapter, section, subsection or ""):
        cleaned = _clean(value)
        if cleaned and (not path or cleaned != path[-1]):
            path.append(cleaned)
    return tuple(path)


def _outline_contexts(
    doc: fitz.Document, manual_title: str
) -> dict[int, tuple[str, str, str | None, tuple[str, ...]]]:
    toc = sorted(doc.get_toc(), key=lambda item: (int(item[2]), int(item[0])))
    contexts: dict[int, tuple[str, str, str | None, tuple[str, ...]]] = {}
    path: dict[int, str] = {}
    cursor = 0
    for pdf_page in range(1, doc.page_count + 1):
        while cursor < len(toc) and int(toc[cursor][2]) <= pdf_page:
            level, title, _ = toc[cursor]
            path[int(level)] = _clean(title)
            path = {key: value for key, value in path.items() if key <= int(level)}
            cursor += 1
        chapter = path.get(1) or manual_title
        section = path.get(2) or chapter
        deeper = [path[level] for level in sorted(path) if level > 2]
        subsection = deeper[-1] if deeper else None
        contexts[pdf_page] = (
            chapter,
            section,
            subsection,
            _heading_path(manual_title, chapter, section, subsection),
        )
    return contexts


def _margin_exclusion_reason(
    block: TextBlock, page: PageData, repeated: set[str]
) -> str | None:
    """Remove only proven boilerplate; page position alone is not evidence."""
    normalized = _normalized_margin_text(block.text)
    near_top = block.bbox[3] <= page.height * 0.12
    near_bottom = block.bbox[1] >= page.height * 0.88
    if near_bottom and PRINTED_PAGE_RE.fullmatch(_clean(block.text)):
        return "page_number"
    if normalized in repeated and near_top:
        return "repeated_header"
    if normalized in repeated and near_bottom:
        return "repeated_footer"
    return None


def _copyright_boilerplate(text: str) -> bool:
    return len(text) <= 400 and bool(COPYRIGHT_RE.search(text))


def _exclude_block(audit: dict[str, Any], reason: str, count: int = 1) -> None:
    categories = audit["discarded_block_categories"]
    categories[reason] = categories.get(reason, 0) + count


def _is_heading(block: TextBlock, body_size: float) -> bool:
    text = _clean(block.text.replace("\n", " "))
    if not 3 <= len(text) <= 160 or len(block.text.splitlines()) > 3:
        return False
    if PROCEDURE_RE.match(text):
        return block.size >= max(10.5, body_size * 1.08) or block.bold
    if CHAPTER_RE.match(text):
        return True
    if text.startswith(("•", "-")) or re.match(r"^\d+[.)]\s", text):
        return False
    if text.endswith((".", ";", ",")):
        return False
    return block.size >= max(11.5, body_size * 1.18) or (
        block.bold and block.size >= max(11.0, body_size * 1.1)
    )


def _section_type(heading: str) -> ContentType:
    if PROCEDURE_RE.match(heading):
        return "procedure"
    if PREREQUISITE_RE.search(heading):
        return "prerequisite"
    if FIELD_RE.search(heading) and DESCRIPTION_RE.search(heading):
        return "field_definition"
    if DEFINITION_RE.match(heading):
        return "definition"
    return "concept"


def _block_type(text: str, active_type: ContentType) -> ContentType:
    if WARNING_RE.match(text):
        return "warning"
    if NOTE_RE.match(text):
        return "note"
    step_numbers = [int(number) for number in STEP_RE.findall(text)]
    if step_numbers[:2] == [1, 2]:
        return "procedure"
    return active_type


def _overlaps(rect: tuple[float, float, float, float], other: tuple[float, ...]) -> bool:
    x0, y0, x1, y1 = rect
    ox0, oy0, ox1, oy1 = (float(value) for value in other)
    intersection = max(0.0, min(x1, ox1) - max(x0, ox0)) * max(
        0.0, min(y1, oy1) - max(y0, oy0)
    )
    area = max(1.0, (x1 - x0) * (y1 - y0))
    return intersection / area >= 0.45


def _table_type(rows: list[list[str]], title: str) -> ContentType:
    header = " ".join(rows[0] if rows else [])
    return (
        "field_definition"
        if FIELD_RE.search(f"{title} {header}") and DESCRIPTION_RE.search(header)
        else "table"
    )


def _markdown_table(title: str, rows: list[list[str]]) -> str:
    width = max((len(row) for row in rows), default=0)
    if not width:
        return title

    def cells(row: list[str]) -> list[str]:
        padded = row + [""] * (width - len(row))
        return [cell.replace("|", "\\|").replace("\n", "<br>") for cell in padded]

    header = cells(rows[0])
    lines = [f"### {title}", "| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend("| " + " | ".join(cells(row)) + " |" for row in rows[1:])
    return "\n".join(lines)


def _table_elements(
    table: Any,
    title: str,
    pdf_page: int,
    printed_page: str | None,
    page_height: float,
) -> list[Element]:
    bbox = tuple(float(value) for value in table.bbox)
    if bbox[1] >= page_height * 0.9 or bbox[3] <= page_height * 0.08:
        return []
    rows = [[_clean(cell) for cell in row] for row in table.extract()]
    rows = [row for row in rows if any(row)]
    width = max((len(row) for row in rows), default=0)
    if len(rows) < 2 or width < 2 or sum(bool(cell) for row in rows for cell in row) < 4:
        return []
    rows = [row + [""] * (width - len(row)) for row in rows]
    content_type = _table_type(rows, title)
    return [
        Element(
            _markdown_table(title, rows),
            content_type,
            pdf_page,
            printed_page,
            rows,
        )
    ]


def _append_element(
    groups: list[SectionGroup],
    chapter: str,
    section: str,
    subsection: str | None,
    heading_path: tuple[str, ...],
    element: Element,
) -> None:
    chapter = _clean(chapter) or "Untitled chapter"
    section = _clean(section) or chapter
    subsection = _clean(subsection) or None
    if (
        not groups
        or groups[-1].chapter != chapter
        or groups[-1].section != section
        or groups[-1].subsection != subsection
    ):
        groups.append(
            SectionGroup(
                chapter=chapter,
                section=section,
                subsection=subsection,
                heading_path=heading_path,
            )
        )
    groups[-1].elements.append(element)


def _extract_groups(
    doc: fitz.Document,
    pages: list[PageData],
    repeated: set[str],
    body_size: float,
    manual_title: str,
    audit: dict[str, Any] | None = None,
) -> list[SectionGroup]:
    contexts = _outline_contexts(doc, manual_title)
    groups: list[SectionGroup] = []

    for page_data in pages:
        page = doc[page_data.pdf_page - 1]
        chapter, section, subsection, heading_path = contexts[page_data.pdf_page]
        if page_data.is_toc or _noncontent_context(chapter, subsection or section):
            if audit is not None:
                audit["toc_index_pages_excluded"] += 1
                _exclude_block(audit, "toc_or_index_page", len(page_data.blocks))
            continue
        active_type: ContentType = _section_type(subsection or section)
        try:
            tables = page.find_tables().tables
        except Exception:
            tables = []
        accepted_tables: list[tuple[Any, list[Element]]] = []
        if audit is not None:
            audit["tables_detected"] += len(tables)
        for table in tables:
            title = (
                subsection or section
                if section != chapter
                else f"Table on page {page_data.printed_page or page_data.pdf_page}"
            )
            elements = _table_elements(
                table,
                title,
                page_data.pdf_page,
                page_data.printed_page,
                page_data.height,
            )
            if elements:
                accepted_tables.append((table, elements))
                if audit is not None:
                    audit["tables_accepted"] += 1
            elif audit is not None:
                audit["tables_rejected"] += 1
        table_boxes = [
            tuple(float(value) for value in table.bbox)
            for table, _ in accepted_tables
        ]
        events: list[tuple[float, int, Any]] = []
        for block in page_data.blocks:
            margin_reason = _margin_exclusion_reason(block, page_data, repeated)
            if margin_reason:
                if audit is not None:
                    audit[f"{margin_reason}s_removed" if margin_reason != "page_number" else "page_numbers_removed"] += 1
                    _exclude_block(audit, margin_reason)
                continue
            if _copyright_boilerplate(block.text):
                if audit is not None:
                    audit["copyright_boilerplate_removed"] += 1
                    _exclude_block(audit, "copyright_boilerplate")
                continue
            if any(_overlaps(block.bbox, bbox) for bbox in table_boxes):
                if audit is not None:
                    audit["text_blocks_indexed"] += 1
                continue
            events.append((block.bbox[1], 1, block))
        events.extend(
            (float(table.bbox[1]), 0, elements)
            for table, elements in accepted_tables
        )

        for _, event_type, value in sorted(events, key=lambda event: (event[0], event[1])):
            if event_type == 0:
                for element in value:
                    if element.table_rows is not None:
                        element.text = _markdown_table(
                            subsection or section, element.table_rows
                        )
                    _append_element(
                        groups, chapter, section, subsection, heading_path, element
                    )
                continue

            block: TextBlock = value
            text = _clean(block.text)
            if _image_only_caption(text):
                if audit is not None:
                    audit["captions_removed"] += 1
                    _exclude_block(audit, "image_only_caption")
                continue
            if _is_heading(block, body_size):
                heading = _clean(text.replace("\n", " "))
                if CHAPTER_RE.match(heading):
                    chapter = heading
                    section = heading
                    subsection = None
                elif heading == section or heading == subsection:
                    pass
                elif section == chapter:
                    section = heading
                    subsection = None
                else:
                    subsection = heading
                heading_path = _heading_path(
                    manual_title, chapter, section, subsection
                )
                active_type = _section_type(heading)
                content_type = active_type
                if audit is not None:
                    audit["headings_detected"] += 1
            else:
                content_type = _block_type(text, active_type)
            _append_element(
                groups,
                chapter,
                section,
                subsection,
                heading_path,
                Element(text, content_type, page_data.pdf_page, page_data.printed_page),
            )
            if audit is not None:
                audit["text_blocks_indexed"] += 1
    return [group for group in groups if group.elements]


def _noncontent_context(chapter: str, section: str) -> bool:
    return bool(
        NONCONTENT_RE.fullmatch(_clean(chapter))
        or NONCONTENT_RE.fullmatch(_clean(section))
    )


def _image_only_caption(text: str) -> bool:
    return bool(IMAGE_CAPTION_RE.match(text) and len(text.split()) <= 25)


@lru_cache(maxsize=2)
def _embedding_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _embedding_token_count(text: str, config: Settings = settings) -> int:
    tokenizer = _embedding_tokenizer(config.embedding_model)
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


@lru_cache(maxsize=4)
def _model_sequence_limit(model_name: str) -> int:
    """Read the SentenceTransformers limit without loading the embedding model."""
    tokenizer = _embedding_tokenizer(model_name)
    limits = [int(tokenizer.model_max_length)]
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_name, "sentence_bert_config.json")
        model_config = json.loads(Path(path).read_text(encoding="utf-8"))
        limits.append(int(model_config["max_seq_length"]))
    except (KeyError, OSError, TypeError, ValueError):
        logger.warning(
            "Could not read SentenceTransformers max_seq_length for %s; "
            "using tokenizer/configured limits",
            model_name,
        )
    usable = [limit for limit in limits if 0 < limit < 1_000_000]
    return min(usable) if usable else EMBEDDING_TOKEN_LIMIT


def _effective_embedding_limit(config: Settings = settings) -> int:
    measured = _model_sequence_limit(config.embedding_model)
    return max(16, min(measured, config.embedding_safety_limit) - EMBEDDING_SAFETY_MARGIN)


def _procedure_steps(text: str) -> list[str]:
    steps: list[str] = []
    current: list[str] = []
    for line in (line.strip() for line in text.splitlines() if line.strip()):
        if STEP_RE.match(line) or STEP_MARKER_RE.match(line):
            if current:
                steps.append(" ".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        steps.append(" ".join(current))
    return steps


def _semantic_atoms(text: str, content_type: ContentType) -> list[str]:
    if content_type == "procedure":
        steps = _procedure_steps(text)
        if steps:
            introduction = text[: text.find(steps[0].split(" ", 1)[0])].strip()
            return ([introduction] if introduction else []) + steps
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n{2,}|(?=\n\s*[•*-]\s+)", text)
        if part.strip()
    ]


def _pack_evidence_atoms(
    atoms: list[str], content_type: ContentType
) -> list[str]:
    if content_type == "procedure":
        return ["\n\n".join(atoms)] if atoms else []
    parts: list[str] = []
    current: list[str] = []
    for atom in atoms:
        candidate = "\n\n".join([*current, atom])
        if current and _token_count(candidate) > MAX_EVIDENCE_TOKENS:
            parts.append("\n\n".join(current))
            current = []
        if _token_count(atom) > MAX_EVIDENCE_TOKENS:
            words = atom.split()
            parts.extend(
                " ".join(words[start : start + MAX_EVIDENCE_TOKENS])
                for start in range(0, len(words), MAX_EVIDENCE_TOKENS)
            )
        else:
            current.append(atom)
    if current:
        parts.append("\n\n".join(current))
    return parts


def _evidence_specs(group: SectionGroup) -> list[EvidenceSpec]:
    specs: list[EvidenceSpec] = []
    run: list[Element] = []

    def flush() -> None:
        if not run:
            return
        main = next(
            (item for item in run if item.content_type not in {"note", "warning"}),
            run[0],
        )
        text = "\n\n".join(item.text for item in run)
        annotations = [
            {"type": item.content_type, "text": item.text}
            for item in run
            if item.content_type in {"note", "warning"}
        ]
        for part in _pack_evidence_atoms(_semantic_atoms(text, main.content_type), main.content_type):
            specs.append(
                EvidenceSpec(
                    text=part,
                    content_type=main.content_type,
                    pdf_page=main.pdf_page,
                    printed_page=main.printed_page,
                    procedure_steps=(
                        _procedure_steps(part)
                        if main.content_type == "procedure"
                        else []
                    ),
                    annotations=[item for item in annotations if item["text"] in part],
                )
            )
        run.clear()

    pending_annotations: list[Element] = []
    for element in group.elements:
        if element.content_type in {"note", "warning"}:
            if run:
                run.append(element)
            elif specs:
                specs[-1].text = f"{specs[-1].text}\n\n{element.text}"
                specs[-1].annotations.append(
                    {"type": element.content_type, "text": element.text}
                )
            else:
                pending_annotations.append(element)
            continue
        if element.table_rows is not None:
            flush()
            if (
                not pending_annotations
                and specs
                and specs[-1].table_rows is not None
                and specs[-1].content_type == element.content_type
                and specs[-1].table_rows[0] == element.table_rows[0]
                and element.pdf_page <= (specs[-1].end_pdf_page or specs[-1].pdf_page) + 1
            ):
                previous = specs[-1]
                previous.table_rows.extend(element.table_rows[1:])
                previous.end_pdf_page = element.pdf_page
                previous.text = "\n\n".join(
                    [
                        *(item["text"] for item in previous.annotations),
                        _markdown_table(
                            group.subsection or group.section,
                            previous.table_rows,
                        ),
                    ]
                )
                continue
            annotations = [
                {"type": item.content_type, "text": item.text}
                for item in pending_annotations
            ]
            specs.append(
                EvidenceSpec(
                    text="\n\n".join(
                        [
                            *(item.text for item in pending_annotations),
                            _markdown_table(
                                group.subsection or group.section,
                                element.table_rows,
                            ),
                        ]
                    ),
                    content_type=element.content_type,
                    pdf_page=element.pdf_page,
                    printed_page=element.printed_page,
                    table_rows=element.table_rows,
                    end_pdf_page=element.pdf_page,
                    annotations=annotations,
                )
            )
            pending_annotations.clear()
            continue
        if pending_annotations:
            run.extend(pending_annotations)
            pending_annotations.clear()
        current_type = next(
            (item.content_type for item in run if item.content_type not in {"note", "warning"}),
            None,
        )
        if run and current_type != element.content_type:
            flush()
        run.append(element)
    flush()
    if pending_annotations and specs:
        for item in pending_annotations:
            specs[-1].text = f"{specs[-1].text}\n\n{item.text}"
            specs[-1].annotations.append({"type": item.content_type, "text": item.text})
    return specs


def _record_id(prefix: str, seed: str) -> str:
    return f"{prefix}_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


def _fits_segment(text: str, config: Settings) -> bool:
    return (
        len(text.split()) <= MAX_SEGMENT_WORDS
        and _fits_embedding(text, config)
    )


def _fits_embedding(text: str, config: Settings) -> bool:
    return _embedding_token_count(text, config) <= _effective_embedding_limit(config)


def _split_searchable_atom(
    atom: str, prefix: str, *, splittable: bool, config: Settings
) -> list[str]:
    if _fits_segment(f"{prefix}\n{atom}", config):
        return [atom]
    step = STEP_RE.match(atom) if not splittable else None
    step_number = step.group(1) if step else ""
    repeated_label = f"{step_number}." if step else ""
    body = atom[step.end() :] if step else atom
    if step:
        continuation_label = f"Step {step_number} — Part 999\n{repeated_label}"
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", body)
            if part.strip()
        ]
        sentence_parts = [
            part
            for sentence in sentences
            for part in _split_searchable_atom(
                sentence,
                f"{prefix}\n{continuation_label}",
                splittable=True,
                config=config,
            )
        ]
        packed: list[str] = []
        current: list[str] = []
        for sentence in sentence_parts:
            candidate = " ".join([*current, sentence])
            if current and not _fits_segment(
                f"{prefix}\n{continuation_label} {candidate}", config
            ):
                packed.append(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            packed.append(" ".join(current))
        return [
            f"Step {step_number} — Part {index}\n{repeated_label} {part}"
            for index, part in enumerate(packed, start=1)
        ]
    words = body.split()
    parts: list[str] = []
    while words:
        size = min(MAX_SEGMENT_WORDS, len(words))
        candidate_body = " ".join(words[:size])
        candidate = f"{repeated_label} {candidate_body}".strip()
        while size and not _fits_segment(f"{prefix}\n{candidate}", config):
            size -= 1
            candidate_body = " ".join(words[:size])
            candidate = f"{repeated_label} {candidate_body}".strip()
        if not size:
            raise ValueError("Searchable text cannot fit the embedding tokenizer limit")
        parts.append(candidate)
        words = words[size:]
    return parts


def _pack_searchable_atoms(
    atoms: list[str], prefix: str, *, splittable: bool, config: Settings
) -> list[str]:
    expanded = [
        part
        for atom in atoms
        for part in _split_searchable_atom(
            atom, prefix, splittable=splittable, config=config
        )
    ]
    packed: list[str] = []
    current: list[str] = []
    for atom in expanded:
        candidate = "\n\n".join([*current, atom])
        current_words = len(f"{prefix}\n{' '.join(current)}".split())
        candidate_words = len(f"{prefix}\n{candidate}".split())
        if current and (
            not _fits_segment(f"{prefix}\n{candidate}", config)
            or current_words >= TARGET_SEGMENT_WORDS
            or (candidate_words > TARGET_SEGMENT_WORDS and current_words >= MIN_SEGMENT_WORDS)
        ):
            packed.append("\n\n".join(current))
            current = []
        current.append(atom)
    if current:
        packed.append("\n\n".join(current))
    if len(packed) > 1 and len(packed[-1].split()) < MIN_SEGMENT_WORDS:
        combined = f"{packed[-2]}\n\n{packed[-1]}"
        if _fits_segment(f"{prefix}\n{combined}", config):
            packed[-2:] = [combined]
    return packed


def _table_segment_parts(
    title: str, rows: list[list[str]], config: Settings, *, prefix: str = ""
) -> list[tuple[str, list[list[str]]]]:
    header, data_rows = rows[0], rows[1:]
    parts: list[tuple[str, list[list[str]]]] = []
    current: list[list[str]] = []
    for row in data_rows:
        single = _markdown_table(title, [header, row])
        if not _fits_embedding(f"{prefix}\n{single}", config):
            if current:
                part_rows = [header, *current]
                parts.append((_markdown_table(title, part_rows), part_rows))
                current = []
            parts.extend(_oversized_table_row_parts(title, header, row, prefix, config))
            continue
        candidate_rows = [header, *current, row]
        candidate = _markdown_table(title, candidate_rows)
        if current and not _fits_segment(f"{prefix}\n{candidate}", config):
            part_rows = [header, *current]
            parts.append((_markdown_table(title, part_rows), part_rows))
            current = []
        current.append(row)
    if current:
        part_rows = [header, *current]
        parts.append((_markdown_table(title, part_rows), part_rows))
    return parts


def _oversized_table_row_parts(
    title: str,
    header: list[str],
    row: list[str],
    prefix: str,
    config: Settings,
) -> list[tuple[str, list[list[str]]]]:
    """Keep the full row in metadata while embedding header-labelled continuations."""
    row_label = next((cell for cell in row if cell), "table row")[:80]
    parts: list[tuple[str, list[list[str]]]] = []
    for index, (column, value) in enumerate(zip(header, row)):
        if not value:
            continue
        column_name = (column or f"Column {index + 1}")[:80]
        continuation_prefix = (
            f"### {title}\nColumn: {column_name}\nRow: {row_label}\nContinuation:"
        )
        fragments = _split_searchable_atom(
            value,
            f"{prefix}\n{continuation_prefix}",
            splittable=True,
            config=config,
        )
        parts.extend(
            (f"{continuation_prefix}\n{fragment}", [header, row])
            for fragment in fragments
        )
    return parts


def _segments_for_unit(
    unit: EvidenceUnit, config: Settings
) -> list[RetrievalSegment]:
    metadata = dict(unit["metadata"])
    prefix_lines = [
        f"Manual: {metadata['manual']}",
        f"Chapter: {metadata['chapter']}",
        f"Section: {metadata['section']}",
    ]
    if metadata.get("subsection"):
        prefix_lines.append(f"Subsection: {metadata['subsection']}")
    prefix_lines.append(f"Type: {unit['content_type']}")
    prefix = "\n".join(prefix_lines)
    effective_limit = _effective_embedding_limit(config)
    segment_parts: list[tuple[str, dict[str, Any]]] = []
    table = unit["structured_table"]
    if table:
        rows = [table["headers"], *table["rows"]]
        segment_parts = [
            (f"{prefix}\n{text}", {"table_rows": part_rows})
            for text, part_rows in _table_segment_parts(
                table["title"], rows, config, prefix=prefix
            )
        ]
    else:
        atoms = unit["procedure_steps"] or _semantic_atoms(
            unit["text"], unit["content_type"]  # type: ignore[arg-type]
        )
        parts = _pack_searchable_atoms(
            atoms,
            prefix,
            splittable=not bool(unit["procedure_steps"]),
            config=config,
        )
        segment_parts = [(f"{prefix}\n{part}", {}) for part in parts]

    segments: list[RetrievalSegment] = []
    for index, (text, extra_metadata) in enumerate(segment_parts):
        segment_id = _record_id("s", f"{unit['evidence_id']}:{index}:{text}")
        segments.append(
            {
                "segment_id": segment_id,
                "evidence_id": unit["evidence_id"],
                "searchable_text": text,
                "content_type": unit["content_type"],
                "metadata": {**metadata, **extra_metadata},
                "segment_index": index,
                "previous_segment_id": None,
                "next_segment_id": None,
                "word_count": len(text.split()),
                "embedding_token_count": _embedding_token_count(text, config),
                "effective_embedding_limit": effective_limit,
            }
        )
    for index, segment in enumerate(segments):
        segment["previous_segment_id"] = segments[index - 1]["segment_id"] if index else None
        segment["next_segment_id"] = segments[index + 1]["segment_id"] if index + 1 < len(segments) else None
    return segments


def _build_evidence(
    groups: list[SectionGroup],
    *,
    source_file: str,
    file_hash: str,
    manual: str,
    release: str,
    config: Settings,
) -> tuple[list[EvidenceUnit], list[RetrievalSegment]]:
    units: list[EvidenceUnit] = []
    segments: list[RetrievalSegment] = []
    for group_index, group in enumerate(groups):
        for unit_index, spec in enumerate(_evidence_specs(group)):
            evidence_id = _record_id(
                "e", f"{file_hash}:{group_index}:{unit_index}:{spec.text}"
            )
            metadata = {
                "manual": manual,
                "source_file": source_file,
                "release": release,
                "chapter": group.chapter,
                "section": group.section,
                "subsection": group.subsection,
                "heading_path": list(
                    group.heading_path
                    or _heading_path(manual, group.chapter, group.section, group.subsection)
                ),
                "content_type": spec.content_type,
                "printed_page": spec.printed_page,
                "pdf_page": spec.pdf_page,
                "is_toc": False,
            }
            table = (
                {
                    "title": group.subsection or group.section,
                    "headers": spec.table_rows[0],
                    "rows": spec.table_rows[1:],
                }
                if spec.table_rows
                else None
            )
            unit: EvidenceUnit = {
                "evidence_id": evidence_id,
                "text": spec.text,
                "content_type": spec.content_type,
                "metadata": metadata,
                "token_count": _token_count(spec.text),
                "structured_table": table,
                "procedure_steps": spec.procedure_steps,
                "annotations": spec.annotations,
            }
            units.append(unit)
            segments.extend(_segments_for_unit(unit, config))
    return units, segments


def _new_ingestion_audit(
    pdf_path: Path, manual: str, pages: list[PageData]
) -> dict[str, Any]:
    text_sizes = [sum(len(block.text.strip()) for block in page.blocks) for page in pages]
    return {
        "manual": manual,
        "source_file": pdf_path.name,
        "total_pages": len(pages),
        "pages_with_text": sum(size > 0 for size in text_sizes),
        "image_only_or_low_text_pages": sum(size < 40 for size in text_sizes),
        "text_blocks_extracted": sum(len(page.blocks) for page in pages),
        "text_blocks_indexed": 0,
        "headings_detected": 0,
        "procedures_detected": 0,
        "procedure_steps_detected": 0,
        "oversized_steps_split": 0,
        "tables_detected": 0,
        "tables_accepted": 0,
        "tables_rejected": 0,
        "toc_index_pages_excluded": 0,
        "repeated_headers_removed": 0,
        "repeated_footers_removed": 0,
        "page_numbers_removed": 0,
        "copyright_boilerplate_removed": 0,
        "captions_removed": 0,
        "unclassified_or_discarded_blocks": 0,
        "discarded_block_categories": {},
        "evidence_unit_count": 0,
        "retrieval_segment_count": 0,
        "minimum_segment_size_words": 0,
        "maximum_segment_size_words": 0,
        "average_segment_size_words": 0.0,
        "maximum_embedding_token_count": 0,
        "warnings": [],
    }


def _finish_ingestion_audit(
    audit: dict[str, Any],
    units: list[EvidenceUnit],
    segments: list[RetrievalSegment],
) -> dict[str, Any]:
    sizes = [int(segment["word_count"]) for segment in segments]
    audit["procedures_detected"] = sum(
        unit["content_type"] == "procedure" for unit in units
    )
    audit["procedure_steps_detected"] = sum(
        len(unit["procedure_steps"]) for unit in units
    )
    audit["oversized_steps_split"] = len(
        {
            (segment["evidence_id"], match.group(1))
            for segment in segments
            if (
                match := re.search(
                    r"Step (\d+) — Part (?:[2-9]|\d{2,})\b",
                    segment["searchable_text"],
                )
            )
        }
    )
    audit["evidence_unit_count"] = len(units)
    audit["retrieval_segment_count"] = len(segments)
    audit["minimum_segment_size_words"] = min(sizes, default=0)
    audit["maximum_segment_size_words"] = max(sizes, default=0)
    audit["average_segment_size_words"] = round(
        sum(sizes) / len(sizes), 2
    ) if sizes else 0.0
    audit["maximum_embedding_token_count"] = max(
        (int(segment["embedding_token_count"]) for segment in segments),
        default=0,
    )
    discarded = sum(audit["discarded_block_categories"].values())
    audit["unclassified_or_discarded_blocks"] = discarded
    accounted = audit["text_blocks_indexed"] + discarded
    extracted = audit["text_blocks_extracted"]
    if accounted != extracted:
        missing = extracted - accounted
        audit["discarded_block_categories"]["unclassified"] = max(0, missing)
        audit["unclassified_or_discarded_blocks"] += max(0, missing)
        audit["warnings"].append(
            f"Text block accounting mismatch: extracted={extracted}, accounted={accounted}."
        )
    if audit["image_only_or_low_text_pages"]:
        audit["warnings"].append(
            f"{audit['image_only_or_low_text_pages']} image-only or low-text pages."
        )
    if audit["tables_rejected"]:
        audit["warnings"].append(
            f"{audit['tables_rejected']} detected tables were rejected; their text blocks remained indexable."
        )
    return audit


def _audit_from_records(
    source_file: str,
    entry: dict[str, Any],
    units: list[EvidenceUnit],
    segments: list[RetrievalSegment],
) -> dict[str, Any]:
    audit = _new_ingestion_audit(
        Path(source_file),
        str(entry.get("manual", source_file)),
        [],
    )
    audit["total_pages"] = int(entry.get("pdf_pages", 0))
    audit["warnings"].append("Block-level audit data was unavailable; manual was rebuilt.")
    return _finish_ingestion_audit(audit, units, segments)


def _ingest_pdf(
    pdf_path: Path, config: Settings = settings
) -> tuple[list[EvidenceUnit], list[RetrievalSegment], dict[str, Any]]:
    _validate_parser(config)
    file_hash = _file_hash(pdf_path)
    with fitz.open(pdf_path) as doc:
        metadata = doc.metadata or {}
        manual = _clean(metadata.get("title")) or pdf_path.stem
        release = _clean(metadata.get("subject"))
        pages, repeated, body_size = _extract_page_data(doc)
        audit = _new_ingestion_audit(pdf_path, manual, pages)
        groups = _extract_groups(doc, pages, repeated, body_size, manual, audit)
        evidence_units, retrieval_segments = _build_evidence(
            groups,
            source_file=pdf_path.name,
            file_hash=file_hash,
            manual=manual,
            release=release,
            config=config,
        )
        _finish_ingestion_audit(audit, evidence_units, retrieval_segments)
        entry = {
            "sha256": file_hash,
            "manual": manual,
            "release": release,
            "pdf_pages": doc.page_count,
            "evidence_unit_count": len(evidence_units),
            "retrieval_segment_count": len(retrieval_segments),
            "_audit": audit,
        }
    return evidence_units, retrieval_segments, entry


def ingest_pdf(
    pdf_path: Path, config: Settings = settings
) -> tuple[list[EvidenceUnit], list[RetrievalSegment]]:
    """Extract semantic EvidenceUnits and searchable RetrievalSegments."""
    evidence_units, retrieval_segments, _ = _ingest_pdf(pdf_path, config)
    return evidence_units, retrieval_segments


def _normalize_heading(value: str) -> str:
    text = _clean(value).casefold().replace("modelling", "modeling")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _alternate_keys(value: str) -> list[str]:
    normalized = _normalize_heading(value)
    aliases = {normalized}
    if normalized.startswith("defining "):
        aliases.add(normalized.removeprefix("defining "))
    safe_plural = {
        "rules", "fields", "models", "controls", "permissions", "roles",
        "resources", "containers", "patterns", "events", "steps",
    }
    for alias in list(aliases):
        words = alias.split()
        if words and words[-1] in safe_plural:
            aliases.add(" ".join([*words[:-1], words[-1][:-1]]))
    return sorted(alias for alias in aliases if alias)


def _build_heading_index(units: list[EvidenceUnit]) -> list[dict[str, Any]]:
    records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for unit in units:
        metadata = unit["metadata"]
        section = _clean(metadata.get("section"))
        subsection = _clean(metadata.get("subsection"))
        heading = subsection or section
        if not heading:
            continue
        key = (
            str(metadata.get("source_file", "")),
            str(metadata.get("chapter", "")),
            section,
            subsection,
        )
        record = records.setdefault(
            key,
            {
                "original_heading": heading,
                "normalized_heading": _normalize_heading(heading),
                "alternate_keys": _alternate_keys(heading),
                "manual": metadata.get("manual", ""),
                "chapter": metadata.get("chapter", ""),
                "section": section,
                "subsection": subsection or None,
                "heading_path": metadata.get("heading_path", []),
                "source_file": metadata.get("source_file", ""),
                "evidence_ids": [],
                "pdf_page": metadata.get("pdf_page"),
                "printed_page": metadata.get("printed_page"),
                "is_toc": False,
            },
        )
        record["evidence_ids"].append(unit["evidence_id"])
    return sorted(
        records.values(),
        key=lambda item: (item["manual"], item["pdf_page"] or 0, item["section"]),
    )


def _build_concept_index(
    units: list[EvidenceUnit], config: Settings = settings
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    repeated_terms: Counter[str] = Counter()
    term_units: dict[str, set[str]] = {}
    term_names: dict[str, str] = {}

    try:
        raw_catalog = json.loads(config.alias_config_path.read_text(encoding="utf-8"))
        alias_catalog = raw_catalog if isinstance(raw_catalog, dict) else {}
    except (OSError, json.JSONDecodeError):
        alias_catalog = {}

    def add(
        name: str, unit: EvidenceUnit, aliases: Sequence[str] = ()
    ) -> None:
        normalized = _normalize_heading(name)
        if len(normalized) < 2 or normalized in {"note", "warning", "description"}:
            return
        metadata = unit["metadata"]
        record = candidates.setdefault(
            normalized,
            {
                "canonical_name": _clean(name),
                "aliases": set(_alternate_keys(name)),
                "evidence_ids": set(),
                "manuals": set(),
                "sections": set(),
            },
        )
        record["aliases"].update(
            alias
            for value in aliases
            for alias in _alternate_keys(str(value))
        )
        record["evidence_ids"].add(unit["evidence_id"])
        record["manuals"].add(str(metadata.get("manual", "")))
        record["sections"].add(
            str(metadata.get("subsection") or metadata.get("section", ""))
        )

    for unit in units:
        metadata = unit["metadata"]
        nearest_heading = str(
            metadata.get("subsection") or metadata.get("section", "")
        )
        add(nearest_heading, unit)
        table = unit.get("structured_table")
        if table:
            add(str(table.get("title", "")), unit)
            for field_name in table.get("headers", []):
                if 1 < len(str(field_name).split()) <= 6:
                    add(str(field_name), unit)
            if unit["content_type"] == "field_definition":
                for row in table.get("rows", []):
                    field_name = str(row[0]) if row else ""
                    if 1 <= len(field_name.split()) <= 8:
                        add(field_name, unit)
        if unit["content_type"] in {"procedure", "definition"}:
            add(nearest_heading, unit)
        for term in TITLE_TERM_RE.findall(unit["text"]):
            normalized = _normalize_heading(term)
            if normalized:
                repeated_terms[normalized] += 1
                term_units.setdefault(normalized, set()).add(unit["evidence_id"])
                term_names.setdefault(normalized, term)
        searchable = f" {_normalize_heading(nearest_heading + ' ' + unit['text'])} "
        for canonical, payload in alias_catalog.items():
            aliases = payload.get("aliases", []) if isinstance(payload, dict) else []
            terms = [str(canonical), *(str(alias) for alias in aliases)]
            variants = {
                variant
                for term in terms
                for normalized in [_normalize_heading(term)]
                for variant in (
                    normalized,
                    f"{normalized}s" if normalized.split()[-1:] and normalized.split()[-1] in {
                        "control", "event", "field", "model", "pattern", "permission",
                        "resource", "role", "rule", "test", "transaction",
                    } else "",
                )
                if variant
            }
            if any(f" {variant} " in searchable for variant in variants):
                add(str(canonical), unit, terms)

    units_by_id = {unit["evidence_id"]: unit for unit in units}
    for normalized, count in repeated_terms.items():
        if count < 3:
            continue
        for evidence_id in term_units[normalized]:
            add(term_names[normalized], units_by_id[evidence_id])

    output: list[dict[str, Any]] = []
    for record in candidates.values():
        output.append(
            {
                **record,
                "aliases": sorted(set(record["aliases"])),
                "evidence_ids": sorted(record["evidence_ids"]),
                "manuals": sorted(value for value in record["manuals"] if value),
                "sections": sorted(value for value in record["sections"] if value),
            }
        )
    return sorted(output, key=lambda item: _normalize_heading(item["canonical_name"]))


def _acronym_aliases(text: str) -> list[str]:
    pairs: set[tuple[str, str]] = set()
    for meaning, acronym in re.findall(
        r"\b([A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){1,7})\s+"
        r"\(([A-Z][A-Z0-9]{1,9})\)",
        text,
    ):
        pairs.add((acronym, _clean(meaning)))
    for acronym, meaning in re.findall(
        r"\b([A-Z][A-Z0-9]{1,9})\s+\(([A-Za-z][^()\n]{2,80})\)",
        text,
    ):
        pairs.add((acronym, _clean(meaning)))
    return [f"Acronym: {acronym}\nMeaning: {meaning}" for acronym, meaning in sorted(pairs)]


def _representation_texts(
    unit: EvidenceUnit, config: Settings
) -> list[tuple[SearchRepresentationType, str]]:
    metadata = unit["metadata"]
    section = _clean(metadata.get("subsection") or metadata.get("section"))
    table = unit.get("structured_table")
    raw: list[tuple[SearchRepresentationType, str, str]] = []
    if section:
        raw.append(("heading", "Heading:", section))
    if unit["content_type"] in {"concept", "definition"}:
        atoms = _semantic_atoms(unit["text"], "concept")
        if atoms:
            first = atoms[0]
            if section and first.casefold().startswith(section.casefold()):
                first = first[len(section) :].lstrip(" :.-") or first
            definition = " ".join([first, *atoms[1:2]]).strip()
            if definition:
                raw.append(("definition", "Definition:", definition))
    if unit["content_type"] == "procedure" and section:
        raw.append(("procedure_title", "Procedure:", section))
    if table:
        headers = [str(value) for value in table.get("headers", [])]
        raw.append(
            (
                "table_title_headers",
                f"Table: {_clean(table.get('title')) or section}\nColumns:",
                " | ".join(value for value in headers if value.strip()),
            )
        )
        if unit["content_type"] == "field_definition":
            for row in table.get("rows", []):
                cells = [str(cell) for cell in row]
                if not cells or not cells[0].strip():
                    continue
                field_name = _clean(cells[0].splitlines()[0])
                if len(field_name) > 120:
                    field_name = field_name[:120].rsplit(" ", 1)[0].strip()
                if not field_name:
                    continue
                details = " | ".join(
                    f"{headers[index] if index < len(headers) else f'Column {index + 1}'}: {cell}"
                    for index, cell in enumerate(cells)
                    if cell.strip()
                )
                raw.append(
                    (
                        "field_name_description",
                        f"Field: {field_name}",
                        details,
                    )
                )
    raw.extend(("acronym_alias", "", value) for value in _acronym_aliases(unit["text"]))

    output: list[tuple[SearchRepresentationType, str]] = []
    for representation_type, prefix, body in raw:
        for part in _split_searchable_atom(
            body,
            prefix,
            splittable=True,
            config=config,
        ):
            output.append(
                (representation_type, "\n".join(value for value in (prefix, part) if value))
            )
    return output


def _build_search_representations(
    units: list[EvidenceUnit], config: Settings = settings
) -> list[SearchRepresentation]:
    representations: list[SearchRepresentation] = []
    globally_seen: set[tuple[str, str]] = set()
    for unit in units:
        seen: set[tuple[str, str]] = set()
        for index, (representation_type, text) in enumerate(
            _representation_texts(unit, config)
        ):
            normalized = _normalized_margin_text(text)
            key = (representation_type, normalized)
            if not normalized or key in seen or key in globally_seen:
                continue
            seen.add(key)
            globally_seen.add(key)
            representation_id = _record_id(
                "r",
                f"{unit['evidence_id']}:{representation_type}:{index}:{text}",
            )
            representations.append(
                {
                    "representation_id": representation_id,
                    "evidence_id": unit["evidence_id"],
                    "representation_type": representation_type,
                    "text": text,
                    "metadata": dict(unit["metadata"]),
                    "embedding_token_count": _embedding_token_count(text, config),
                }
            )
    return representations


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2 if pretty else None),
        encoding="utf-8",
    )
    temporary.replace(path)


def _indexable_segments(
    segments: list[RetrievalSegment],
    config: Settings | None = None,
) -> list[RetrievalSegment]:
    counts = Counter(str(segment.get("segment_id", "")) for segment in segments)
    duplicates = sorted(
        segment_id
        for segment_id, count in counts.items()
        if not segment_id or count > 1
    )
    if duplicates:
        raise ValueError(
            f"Duplicate or empty retrieval segment IDs: {', '.join(duplicates[:10])}"
        )
    if any(not segment.get("evidence_id") for segment in segments):
        raise ValueError("Every retrieval segment must reference an EvidenceUnit")
    required_source_fields = {"manual", "source_file", "chapter", "section", "pdf_page"}
    for segment in segments:
        missing = required_source_fields - set(segment.get("metadata", {}))
        if missing:
            raise ValueError(
                f"RetrievalSegment {segment['segment_id']} is missing source metadata: "
                f"{', '.join(sorted(missing))}"
            )
        if config is not None:
            token_count = _embedding_token_count(segment["searchable_text"], config)
            limit = _effective_embedding_limit(config)
            if token_count > limit:
                raise ValueError(
                    f"RetrievalSegment {segment['segment_id']} has {token_count} embedding "
                    f"tokens; effective limit is {limit}"
                )
            if segment.get("embedding_token_count") != token_count:
                raise ValueError(
                    f"RetrievalSegment {segment['segment_id']} has stale embedding_token_count"
                )
            if segment.get("effective_embedding_limit") != limit:
                raise ValueError(
                    f"RetrievalSegment {segment['segment_id']} has stale "
                    "effective_embedding_limit"
                )
    return segments


def _indexable_representations(
    representations: list[SearchRepresentation],
    evidence_ids: set[str],
    config: Settings | None = None,
) -> list[SearchRepresentation]:
    counts = Counter(
        str(representation.get("representation_id", ""))
        for representation in representations
    )
    duplicates = sorted(
        representation_id
        for representation_id, count in counts.items()
        if not representation_id or count > 1
    )
    if duplicates:
        raise ValueError(
            f"Duplicate or empty search representation IDs: {', '.join(duplicates[:10])}"
        )
    missing_evidence = sorted(
        str(representation.get("evidence_id", ""))
        for representation in representations
        if representation.get("evidence_id") not in evidence_ids
    )
    if missing_evidence:
        raise ValueError(
            f"SearchRepresentations reference missing EvidenceUnits: {missing_evidence[:10]}"
        )
    required_source_fields = {"manual", "source_file", "chapter", "section", "pdf_page"}
    for representation in representations:
        missing = required_source_fields - set(representation.get("metadata", {}))
        if missing:
            raise ValueError(
                f"SearchRepresentation {representation['representation_id']} is missing "
                f"source metadata: {', '.join(sorted(missing))}"
            )
        if config is not None:
            token_count = _embedding_token_count(representation["text"], config)
            if token_count > _effective_embedding_limit(config):
                raise ValueError(
                    f"SearchRepresentation {representation['representation_id']} has "
                    f"{token_count} embedding tokens"
                )
            if representation.get("embedding_token_count") != token_count:
                raise ValueError(
                    f"SearchRepresentation {representation['representation_id']} has stale "
                    "embedding_token_count"
                )
    return representations


def _bm25_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.casefold())


def _chroma_metadata(segment: RetrievalSegment) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    values = {**segment["metadata"], **segment}
    for key, value in values.items():
        if key in {"searchable_text", "metadata"}:
            continue
        if value is None:
            metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = json.dumps(value, ensure_ascii=False)
    return metadata


def _representation_metadata(
    representation: SearchRepresentation,
) -> dict[str, str | int | float | bool]:
    return {
        **{
            key: value
            for key, value in representation["metadata"].items()
            if isinstance(value, (str, int, float, bool))
        },
        "representation_id": representation["representation_id"],
        "evidence_id": representation["evidence_id"],
        "representation_type": representation["representation_type"],
        "embedding_token_count": representation["embedding_token_count"],
    }


def build_indexes(
    segments: list[RetrievalSegment],
    config: Settings = settings,
    *,
    representations: list[SearchRepresentation] | None = None,
) -> int:
    """Rebuild body Chroma, representation Chroma, and BM25 indexes."""
    from langchain_chroma import Chroma
    from rank_bm25 import BM25Okapi

    from src.embeddings import create_embedding_model

    segments = _indexable_segments(segments, config)
    if not segments:
        raise ValueError("No RetrievalSegments are available for indexing")
    evidence_ids = {
        str(unit["evidence_id"])
        for unit in _load_json(config.evidence_units_path, [])
    }
    representation_records = representations if representations is not None else _load_json(
        config.search_representations_path, []
    )
    representation_records = _indexable_representations(
        representation_records if isinstance(representation_records, list) else [],
        evidence_ids,
        config,
    )
    if not representation_records:
        raise ValueError("No SearchRepresentations are available for indexing")
    ids = [str(segment["segment_id"]) for segment in segments]
    texts = [str(segment["searchable_text"]) for segment in segments]
    metadatas = [_chroma_metadata(segment) for segment in segments]
    representation_ids = [item["representation_id"] for item in representation_records]
    representation_texts = [item["text"] for item in representation_records]
    representation_metadatas = [
        _representation_metadata(item) for item in representation_records
    ]

    shutil.rmtree(config.chroma_dir, ignore_errors=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    (config.chroma_dir / ".gitkeep").write_text("\n", encoding="utf-8")
    embedding_model = create_embedding_model(config)
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding_model,
        persist_directory=str(config.chroma_dir),
    )
    for start in range(0, len(segments), 256):
        end = start + 256
        vector_store.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
    representation_store = Chroma(
        collection_name=REPRESENTATION_COLLECTION,
        embedding_function=embedding_model,
        persist_directory=str(config.chroma_dir),
    )
    for start in range(0, len(representation_records), 256):
        end = start + 256
        representation_store.add_texts(
            texts=representation_texts[start:end],
            metadatas=representation_metadatas[start:end],
            ids=representation_ids[start:end],
        )

    tokenized = [_bm25_tokens(text) for text in texts]
    bm25_payload = {"bm25": BM25Okapi(tokenized), "segment_ids": ids}
    bm25_path = config.bm25_path
    temporary = bm25_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(bm25_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(bm25_path)

    for manual, count in sorted(
        Counter(segment["metadata"]["manual"] for segment in segments).items()
    ):
        logger.info("Indexed %s: %d retrieval segments", manual, count)
    logger.info("Indexed %d deterministic search representations", len(representation_records))
    return validate_indexes(config, require_schema=False)


def validate_indexes(
    config: Settings = settings, *, require_schema: bool = True
) -> int:
    """Require Chroma, BM25, and retrieval_segments.json to share IDs."""
    import chromadb

    if require_schema:
        _require_index_schema(config)

    evidence_units = _load_json(config.evidence_units_path, [])
    evidence_ids = {
        str(unit["evidence_id"])
        for unit in evidence_units if isinstance(unit, dict) and unit.get("evidence_id")
    }
    raw_segments = _load_json(config.retrieval_segments_path, [])
    segments = _indexable_segments(raw_segments if isinstance(raw_segments, list) else [])
    segment_ids = [str(segment["segment_id"]) for segment in segments]
    raw_representations = _load_json(config.search_representations_path, [])
    representations = _indexable_representations(
        raw_representations if isinstance(raw_representations, list) else [],
        evidence_ids,
    )
    representation_ids = [item["representation_id"] for item in representations]

    bm25_path = config.bm25_path
    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
    with bm25_path.open("rb") as handle:
        payload = pickle.load(handle)
    bm25_ids = list(payload.get("segment_ids", []))

    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    collection = client.get_collection(CHROMA_COLLECTION)
    chroma_ids = list(collection.get(include=[])["ids"])
    representation_collection = client.get_collection(REPRESENTATION_COLLECTION)
    chroma_representation_ids = list(
        representation_collection.get(include=[])["ids"]
    )

    if len(bm25_ids) != len(set(bm25_ids)):
        raise ValueError("BM25 contains duplicate retrieval segment IDs")
    if len(chroma_ids) != len(set(chroma_ids)):
        raise ValueError("Chroma contains duplicate retrieval segment IDs")
    if len(chroma_representation_ids) != len(set(chroma_representation_ids)):
        raise ValueError("Chroma contains duplicate search representation IDs")
    if set(segment_ids) != set(bm25_ids) or set(segment_ids) != set(chroma_ids):
        raise ValueError(
            "Index ID mismatch: "
            f"retrieval_segments.json={len(segment_ids)}, "
            f"BM25={len(bm25_ids)}, Chroma={len(chroma_ids)}"
        )
    if set(representation_ids) != set(chroma_representation_ids):
        raise ValueError(
            "Representation index ID mismatch: "
            f"search_representations.json={len(representation_ids)}, "
            f"Chroma={len(chroma_representation_ids)}"
        )
    if set(segment_ids).intersection(representation_ids):
        raise ValueError("Body segment and search representation IDs overlap")
    logger.info(
        "Validated matching RetrievalSegment IDs across JSON, BM25, and Chroma: %d",
        len(segment_ids),
    )
    logger.info(
        "Validated SearchRepresentation references and Chroma IDs: %d",
        len(representation_ids),
    )
    return len(segment_ids)


def ingest_manuals(config: Settings = settings) -> dict[str, Any]:
    """Ingest changed manuals and reuse both record levels for unchanged hashes."""
    _validate_parser(config)
    config.indexes_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = config.evidence_units_path
    segments_path = config.retrieval_segments_path
    manifest_path = config.indexes_dir / "manifest.json"
    old_audit_root = _load_json(config.ingestion_audit_path, {"manuals": {}})
    old_audits = (
        old_audit_root.get("manuals", {})
        if isinstance(old_audit_root, dict)
        and old_audit_root.get("schema_version") == INDEX_SCHEMA_VERSION
        and old_audit_root.get("ingestion_pipeline_version") == INGESTION_PIPELINE_VERSION
        else {}
    )
    old_units = _load_json(evidence_path, [])
    old_segments = _load_json(segments_path, [])
    old_manifest = _load_json(manifest_path, {"manuals": {}})
    old_entries = (
        old_manifest.get("manuals", {})
        if old_manifest.get("version") == INDEX_SCHEMA_VERSION
        and old_manifest.get("ingestion_pipeline_version") == INGESTION_PIPELINE_VERSION
        else {}
    )
    old_units_by_file: dict[str, list[EvidenceUnit]] = {}
    old_segments_by_file: dict[str, list[RetrievalSegment]] = {}
    for unit in old_units if isinstance(old_units, list) else []:
        source = str(unit.get("metadata", {}).get("source_file", ""))
        old_units_by_file.setdefault(source, []).append(unit)
    for segment in old_segments if isinstance(old_segments, list) else []:
        source = str(segment.get("metadata", {}).get("source_file", ""))
        old_segments_by_file.setdefault(source, []).append(segment)

    all_units: list[EvidenceUnit] = []
    all_segments: list[RetrievalSegment] = []
    entries: dict[str, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    processed: list[str] = []
    skipped: list[str] = []
    pdf_paths = sorted(config.manuals_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF manuals found in {config.manuals_dir}")
    current_files = {path.name for path in pdf_paths}
    removed = sorted(set(old_entries) - current_files)

    for pdf_path in pdf_paths:
        file_hash = _file_hash(pdf_path)
        old_entry = old_entries.get(pdf_path.name, {})
        reusable = (
            old_entry.get("sha256") == file_hash
            and pdf_path.name in old_units_by_file
            and pdf_path.name in old_segments_by_file
            and pdf_path.name in old_audits
        )
        if reusable:
            all_units.extend(old_units_by_file[pdf_path.name])
            all_segments.extend(old_segments_by_file[pdf_path.name])
            entries[pdf_path.name] = old_entry
            audits[pdf_path.name] = old_audits[pdf_path.name]
            skipped.append(pdf_path.name)
            continue
        units, segments, entry = _ingest_pdf(pdf_path, config)
        all_units.extend(units)
        all_segments.extend(segments)
        audit = entry.pop("_audit", None)
        audits[pdf_path.name] = audit or _audit_from_records(
            pdf_path.name, entry, units, segments
        )
        entries[pdf_path.name] = entry
        processed.append(pdf_path.name)

    evidence_ids = {unit["evidence_id"] for unit in all_units}
    missing_evidence = sorted(
        segment["evidence_id"]
        for segment in all_segments
        if segment["evidence_id"] not in evidence_ids
    )
    if missing_evidence:
        raise ValueError(
            f"RetrievalSegments reference missing EvidenceUnits: {missing_evidence[:10]}"
        )
    _write_json(evidence_path, all_units)
    _write_json(segments_path, all_segments)
    heading_index = _build_heading_index(all_units)
    concept_index = _build_concept_index(all_units, config)
    search_representations = _indexable_representations(
        _build_search_representations(all_units, config),
        evidence_ids,
        config,
    )
    _write_json(config.search_representations_path, search_representations)
    _write_json(config.heading_index_path, heading_index, pretty=True)
    _write_json(config.concept_index_path, concept_index, pretty=True)
    representation_counts = Counter(
        str(item["metadata"].get("source_file", ""))
        for item in search_representations
    )
    for source_file, audit in audits.items():
        audit["search_representation_count"] = representation_counts[source_file]
    _write_json(
        config.ingestion_audit_path,
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "ingestion_pipeline_version": INGESTION_PIPELINE_VERSION,
            "generated_at": datetime.now(UTC).isoformat(),
            "manuals": audits,
        },
        pretty=True,
    )
    (config.indexes_dir / "chunks.json").unlink(missing_ok=True)
    index_artifacts_exist = (
        config.bm25_path.exists()
        and (config.chroma_dir / "chroma.sqlite3").exists()
        and config.heading_index_path.exists()
        and config.concept_index_path.exists()
        and config.ingestion_audit_path.exists()
        and config.search_representations_path.exists()
    )
    indexes_rebuilt = bool(
        processed
        or removed
        or old_manifest.get("version") != INDEX_SCHEMA_VERSION
        or old_manifest.get("ingestion_pipeline_version") != INGESTION_PIPELINE_VERSION
        or not index_artifacts_exist
    )
    indexed_segments = (
        build_indexes(all_segments, config)
        if indexes_rebuilt
        else validate_indexes(config)
    )

    manifest = {
        "version": INDEX_SCHEMA_VERSION,
        "ingestion_pipeline_version": INGESTION_PIPELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "manuals": entries,
        "total_manuals": len(entries),
        "total_evidence_units": len(all_units),
        "total_retrieval_segments": len(all_segments),
        "total_search_representations": len(search_representations),
        "processed": processed,
        "skipped_unchanged": skipped,
        "removed": removed,
        "indexes_rebuilt": indexes_rebuilt,
        "indexed_retrieval_segments": indexed_segments,
        "heading_record_count": len(heading_index),
        "concept_record_count": len(concept_index),
        "ingestion_audit_manual_count": len(audits),
        "effective_embedding_token_limit": _effective_embedding_limit(config),
        "chroma_collection": CHROMA_COLLECTION,
        "representation_chroma_collection": REPRESENTATION_COLLECTION,
    }
    _write_json(manifest_path, manifest, pretty=True)
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = ingest_manuals()
    print(
        f"Ingestion complete: {manifest['total_manuals']} manuals, "
        f"{manifest['total_evidence_units']} evidence units, "
        f"{manifest['indexed_retrieval_segments']} retrieval segments indexed, "
        f"{manifest['total_search_representations']} search representations indexed, "
        f"{len(manifest['processed'])} processed, "
        f"{len(manifest['skipped_unchanged'])} unchanged, "
        f"indexes rebuilt={manifest['indexes_rebuilt']}."
    )


if __name__ == "__main__":
    main()
