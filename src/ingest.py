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
from typing import Any, Literal

import fitz

from src.config import Settings, settings
from src.schemas import EvidenceUnit, RetrievalSegment


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
IMAGE_CAPTION_RE = re.compile(
    r"^(?:figure|fig\.|image|illustration|screen(?:shot)?)\s+"
    r"(?:[A-Z]?\d[\w.-]*|[ivxlcdm]+)\b\s*[:.-]?",
    re.I,
)
STEP_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+", re.M)
STEP_MARKER_RE = re.compile(r"^\s*(\d{1,3})[.)]\s*$")
FIELD_RE = re.compile(r"\b(?:field|property|parameter|column|button)s?\b", re.I)
DESCRIPTION_RE = re.compile(r"\b(?:description|definition|meaning|value)s?\b", re.I)

TARGET_SEGMENT_WORDS = 185
MIN_SEGMENT_WORDS = 150
MAX_SEGMENT_WORDS = 220
MAX_EVIDENCE_TOKENS = 1_200
EMBEDDING_TOKEN_LIMIT = 512
CHROMA_COLLECTION = "opcenter_manuals"
INDEX_SCHEMA_VERSION = 4
REINGEST_COMMAND = "python -m src.ingest"
logger = logging.getLogger(__name__)


class IndexSchemaMismatchError(RuntimeError):
    """Existing index artifacts require an explicit schema rebuild."""


def _require_index_schema(config: Settings) -> None:
    manifest = _load_json(config.indexes_dir / "manifest.json", {})
    version = manifest.get("version") if isinstance(manifest, dict) else None
    if version != INDEX_SCHEMA_VERSION:
        raise IndexSchemaMismatchError(
            f"Index schema changed (found {version or 'none'}, required {INDEX_SCHEMA_VERSION}). "
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
    elements: list[Element] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceSpec:
    text: str
    content_type: ContentType
    pdf_page: int
    printed_page: str | None
    table_rows: list[list[str]] | None = None
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
            )
        )

    threshold = max(3, math.ceil(len(pages) * 0.15))
    repeated = {text for text, count in margin_counts.items() if text and count >= threshold}
    return pages, repeated, median(body_sizes) if body_sizes else 10.0


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


def _outline_contexts(doc: fitz.Document, manual_title: str) -> dict[int, tuple[str, str]]:
    toc = sorted(doc.get_toc(), key=lambda item: (int(item[2]), int(item[0])))
    contexts: dict[int, tuple[str, str]] = {}
    path: dict[int, str] = {}
    cursor = 0
    for pdf_page in range(1, doc.page_count + 1):
        while cursor < len(toc) and int(toc[cursor][2]) <= pdf_page:
            level, title, _ = toc[cursor]
            path[int(level)] = _clean(title)
            path = {key: value for key, value in path.items() if key <= int(level)}
            cursor += 1
        chapter = path.get(1) or manual_title
        deeper = [path[level] for level in sorted(path) if level > 1]
        contexts[pdf_page] = (chapter, deeper[-1] if deeper else chapter)
    return contexts


def _in_margin(block: TextBlock, page: PageData, repeated: set[str]) -> bool:
    normalized = _normalized_margin_text(block.text)
    return (
        block.bbox[3] <= page.height * 0.065
        or block.bbox[1] >= page.height * 0.925
        or (
            normalized in repeated
            and (block.bbox[3] <= page.height * 0.12 or block.bbox[1] >= page.height * 0.88)
        )
    )


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
    element: Element,
) -> None:
    chapter = _clean(chapter) or "Untitled chapter"
    section = _clean(section) or chapter
    if not groups or groups[-1].chapter != chapter or groups[-1].section != section:
        groups.append(SectionGroup(chapter=chapter, section=section))
    groups[-1].elements.append(element)


def _extract_groups(
    doc: fitz.Document,
    pages: list[PageData],
    repeated: set[str],
    body_size: float,
    manual_title: str,
) -> list[SectionGroup]:
    contexts = _outline_contexts(doc, manual_title)
    groups: list[SectionGroup] = []

    for page_data in pages:
        page = doc[page_data.pdf_page - 1]
        chapter, section = contexts[page_data.pdf_page]
        if _noncontent_context(chapter, section):
            continue
        active_type: ContentType = _section_type(section)
        try:
            tables = page.find_tables().tables
        except Exception:
            tables = []
        table_boxes = [tuple(float(value) for value in table.bbox) for table in tables]
        events: list[tuple[float, int, Any]] = []
        for block in page_data.blocks:
            if _in_margin(block, page_data, repeated):
                continue
            if any(_overlaps(block.bbox, bbox) for bbox in table_boxes):
                continue
            events.append((block.bbox[1], 1, block))
        events.extend((float(table.bbox[1]), 0, table) for table in tables)

        for _, event_type, value in sorted(events, key=lambda event: (event[0], event[1])):
            if event_type == 0:
                title = section if section != chapter else f"Table on page {page_data.printed_page or page_data.pdf_page}"
                for element in _table_elements(
                    value,
                    title,
                    page_data.pdf_page,
                    page_data.printed_page,
                    page_data.height,
                ):
                    _append_element(groups, chapter, section, element)
                continue

            block: TextBlock = value
            text = _clean(block.text)
            if _image_only_caption(text):
                continue
            if _is_heading(block, body_size):
                heading = _clean(text.replace("\n", " "))
                if CHAPTER_RE.match(heading):
                    chapter = heading
                section = heading
                active_type = _section_type(heading)
                content_type = active_type
            else:
                content_type = _block_type(text, active_type)
            _append_element(
                groups,
                chapter,
                section,
                Element(text, content_type, page_data.pdf_page, page_data.printed_page),
            )
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
        for part in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
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
            annotations = [
                {"type": item.content_type, "text": item.text}
                for item in pending_annotations
            ]
            header, data_rows = element.table_rows[0], element.table_rows[1:]
            current_rows: list[list[str]] = []
            for row in data_rows:
                candidate_rows = [header, *current_rows, row]
                candidate = _markdown_table(group.section, candidate_rows)
                if current_rows and _token_count(candidate) > MAX_EVIDENCE_TOKENS:
                    part_rows = [header, *current_rows]
                    specs.append(
                        EvidenceSpec(
                            text="\n\n".join(
                                [
                                    *(item.text for item in pending_annotations),
                                    _markdown_table(group.section, part_rows),
                                ]
                            ),
                            content_type=element.content_type,
                            pdf_page=element.pdf_page,
                            printed_page=element.printed_page,
                            table_rows=part_rows,
                            annotations=annotations,
                        )
                    )
                    current_rows = []
                current_rows.append(row)
            if current_rows:
                part_rows = [header, *current_rows]
                specs.append(
                    EvidenceSpec(
                        text="\n\n".join(
                            [
                                *(item.text for item in pending_annotations),
                                _markdown_table(group.section, part_rows),
                            ]
                        ),
                        content_type=element.content_type,
                        pdf_page=element.pdf_page,
                        printed_page=element.printed_page,
                        table_rows=part_rows,
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
    elif pending_annotations:
        for item in pending_annotations:
            specs.append(
                EvidenceSpec(
                    text=item.text,
                    content_type=item.content_type,
                    pdf_page=item.pdf_page,
                    printed_page=item.printed_page,
                    annotations=[{"type": item.content_type, "text": item.text}],
                )
            )
    return specs


def _record_id(prefix: str, seed: str) -> str:
    return f"{prefix}_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"


def _fits_segment(text: str, config: Settings) -> bool:
    return (
        len(text.split()) <= MAX_SEGMENT_WORDS
        and _fits_embedding(text, config)
    )


def _fits_embedding(text: str, config: Settings) -> bool:
    return _embedding_token_count(text, config) < EMBEDDING_TOKEN_LIMIT


def _split_searchable_atom(
    atom: str, prefix: str, *, splittable: bool, config: Settings
) -> list[str]:
    if _fits_segment(f"{prefix}\n{atom}", config):
        return [atom]
    if not splittable:
        if _fits_embedding(f"{prefix}\n{atom}", config):
            return [atom]
        raise ValueError("An indivisible procedure step exceeds the embedding limit")
    words = atom.split()
    parts: list[str] = []
    while words:
        size = min(MAX_SEGMENT_WORDS, len(words))
        while size and not _fits_segment(f"{prefix}\n{' '.join(words[:size])}", config):
            size -= 1
        if not size:
            raise ValueError("Searchable text cannot fit the embedding tokenizer limit")
        parts.append(" ".join(words[:size]))
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
        if current and not _fits_segment(f"{prefix}\n{candidate}", config):
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
    title: str, rows: list[list[str]], config: Settings
) -> list[tuple[str, list[list[str]]]]:
    header, data_rows = rows[0], rows[1:]
    parts: list[tuple[str, list[list[str]]]] = []
    current: list[list[str]] = []
    for row in data_rows:
        single = _markdown_table(title, [header, row])
        if not _fits_embedding(single, config):
            raise ValueError("A complete table row exceeds the embedding tokenizer limit")
        candidate_rows = [header, *current, row]
        candidate = _markdown_table(title, candidate_rows)
        if current and not _fits_segment(candidate, config):
            part_rows = [header, *current]
            parts.append((_markdown_table(title, part_rows), part_rows))
            current = []
        current.append(row)
    if current:
        part_rows = [header, *current]
        parts.append((_markdown_table(title, part_rows), part_rows))
    return parts


def _segments_for_unit(
    unit: EvidenceUnit, config: Settings
) -> list[RetrievalSegment]:
    metadata = dict(unit["metadata"])
    prefix = f"{metadata['section']}"
    segment_parts: list[tuple[str, dict[str, Any]]] = []
    table = unit["structured_table"]
    if table:
        rows = [table["headers"], *table["rows"]]
        segment_parts = [
            (text, {"table_rows": part_rows})
            for text, part_rows in _table_segment_parts(table["title"], rows, config)
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
                "printed_page": spec.printed_page,
                "pdf_page": spec.pdf_page,
            }
            table = (
                {
                    "title": group.section,
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


def _ingest_pdf(
    pdf_path: Path, config: Settings = settings
) -> tuple[list[EvidenceUnit], list[RetrievalSegment], dict[str, Any]]:
    file_hash = _file_hash(pdf_path)
    with fitz.open(pdf_path) as doc:
        metadata = doc.metadata or {}
        manual = _clean(metadata.get("title")) or pdf_path.stem
        release = _clean(metadata.get("subject"))
        pages, repeated, body_size = _extract_page_data(doc)
        groups = _extract_groups(doc, pages, repeated, body_size, manual)
        evidence_units, retrieval_segments = _build_evidence(
            groups,
            source_file=pdf_path.name,
            file_hash=file_hash,
            manual=manual,
            release=release,
            config=config,
        )
        entry = {
            "sha256": file_hash,
            "manual": manual,
            "release": release,
            "pdf_pages": doc.page_count,
            "evidence_unit_count": len(evidence_units),
            "retrieval_segment_count": len(retrieval_segments),
        }
    return evidence_units, retrieval_segments, entry


def ingest_pdf(
    pdf_path: Path, config: Settings = settings
) -> tuple[list[EvidenceUnit], list[RetrievalSegment]]:
    """Extract semantic EvidenceUnits and searchable RetrievalSegments."""
    evidence_units, retrieval_segments, _ = _ingest_pdf(pdf_path, config)
    return evidence_units, retrieval_segments


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
            if token_count >= EMBEDDING_TOKEN_LIMIT:
                raise ValueError(
                    f"RetrievalSegment {segment['segment_id']} has {token_count} embedding "
                    f"tokens; limit is {EMBEDDING_TOKEN_LIMIT - 1}"
                )
    return segments


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


def build_indexes(
    segments: list[RetrievalSegment], config: Settings = settings
) -> int:
    """Rebuild Chroma and BM25 from the same RetrievalSegments."""
    from langchain_chroma import Chroma
    from rank_bm25 import BM25Okapi

    from src.embeddings import create_embedding_model

    segments = _indexable_segments(segments, config)
    if not segments:
        raise ValueError("No RetrievalSegments are available for indexing")
    ids = [str(segment["segment_id"]) for segment in segments]
    texts = [str(segment["searchable_text"]) for segment in segments]
    metadatas = [_chroma_metadata(segment) for segment in segments]

    shutil.rmtree(config.chroma_dir, ignore_errors=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=create_embedding_model(config),
        persist_directory=str(config.chroma_dir),
    )
    for start in range(0, len(segments), 256):
        end = start + 256
        vector_store.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
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
    return validate_indexes(config, require_schema=False)


def validate_indexes(
    config: Settings = settings, *, require_schema: bool = True
) -> int:
    """Require Chroma, BM25, and retrieval_segments.json to share IDs."""
    import chromadb

    if require_schema:
        _require_index_schema(config)

    raw_segments = _load_json(config.retrieval_segments_path, [])
    segments = _indexable_segments(raw_segments if isinstance(raw_segments, list) else [])
    segment_ids = [str(segment["segment_id"]) for segment in segments]

    bm25_path = config.bm25_path
    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found: {bm25_path}")
    with bm25_path.open("rb") as handle:
        payload = pickle.load(handle)
    bm25_ids = list(payload.get("segment_ids", []))

    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    collection = client.get_collection(CHROMA_COLLECTION)
    chroma_ids = list(collection.get(include=[])["ids"])

    if len(bm25_ids) != len(set(bm25_ids)):
        raise ValueError("BM25 contains duplicate retrieval segment IDs")
    if len(chroma_ids) != len(set(chroma_ids)):
        raise ValueError("Chroma contains duplicate retrieval segment IDs")
    if set(segment_ids) != set(bm25_ids) or set(segment_ids) != set(chroma_ids):
        raise ValueError(
            "Index ID mismatch: "
            f"retrieval_segments.json={len(segment_ids)}, "
            f"BM25={len(bm25_ids)}, Chroma={len(chroma_ids)}"
        )
    logger.info(
        "Validated matching RetrievalSegment IDs across JSON, BM25, and Chroma: %d",
        len(segment_ids),
    )
    return len(segment_ids)


def ingest_manuals(config: Settings = settings) -> dict[str, Any]:
    """Ingest changed manuals and reuse both record levels for unchanged hashes."""
    config.indexes_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = config.evidence_units_path
    segments_path = config.retrieval_segments_path
    manifest_path = config.indexes_dir / "manifest.json"
    old_units = _load_json(evidence_path, [])
    old_segments = _load_json(segments_path, [])
    old_manifest = _load_json(manifest_path, {"manuals": {}})
    old_entries = (
        old_manifest.get("manuals", {})
        if old_manifest.get("version") == INDEX_SCHEMA_VERSION
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
        )
        if reusable:
            all_units.extend(old_units_by_file[pdf_path.name])
            all_segments.extend(old_segments_by_file[pdf_path.name])
            entries[pdf_path.name] = old_entry
            skipped.append(pdf_path.name)
            continue
        units, segments, entry = _ingest_pdf(pdf_path, config)
        all_units.extend(units)
        all_segments.extend(segments)
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
    (config.indexes_dir / "chunks.json").unlink(missing_ok=True)
    index_artifacts_exist = (
        config.bm25_path.exists()
        and (config.chroma_dir / "chroma.sqlite3").exists()
    )
    indexes_rebuilt = bool(
        processed
        or removed
        or old_manifest.get("version") != INDEX_SCHEMA_VERSION
        or not index_artifacts_exist
    )
    indexed_segments = (
        build_indexes(all_segments, config) if indexes_rebuilt else validate_indexes(config)
    )

    manifest = {
        "version": INDEX_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "manuals": entries,
        "total_manuals": len(entries),
        "total_evidence_units": len(all_units),
        "total_retrieval_segments": len(all_segments),
        "processed": processed,
        "skipped_unchanged": skipped,
        "removed": removed,
        "indexes_rebuilt": indexes_rebuilt,
        "indexed_retrieval_segments": indexed_segments,
        "chroma_collection": CHROMA_COLLECTION,
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
        f"{len(manifest['processed'])} processed, "
        f"{len(manifest['skipped_unchanged'])} unchanged, "
        f"indexes rebuilt={manifest['indexes_rebuilt']}."
    )


if __name__ == "__main__":
    main()
