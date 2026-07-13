"""Deterministic post-resolution views over immutable EvidenceUnits."""

from __future__ import annotations

import re
from typing import Any, Sequence

from src.schemas import CompressedEvidenceView, RetrievedDocument


CONDITION_RE = re.compile(
    r"\b(?:after|before|except|if|only|required|unless|until|warning|when)\b",
    re.I,
)
ALL_ROWS_RE = re.compile(r"\b(?:all rows|complete table|entire table|full table)\b", re.I)


def _terms(*values: str) -> set[str]:
    return {
        term
        for value in values
        for term in re.findall(r"[a-z0-9_]+", value.casefold())
        if len(term) > 2
        and term not in {"and", "are", "for", "from", "the", "this", "what", "with"}
    }


def _ranked_indexes(values: Sequence[str], query_terms: set[str]) -> list[int]:
    return sorted(
        range(len(values)),
        key=lambda index: (
            len(query_terms & _terms(values[index])),
            -index,
        ),
        reverse=True,
    )


def _sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        if part.strip()
    ]


def _markdown_table(title: str, rows: list[list[str]]) -> str:
    width = max((len(row) for row in rows), default=0)
    if not width:
        return title
    padded = [row + [""] * (width - len(row)) for row in rows]
    lines = [title, "| " + " | ".join(padded[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend(
        "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
        for row in padded[1:]
    )
    return "\n".join(lines)


def _connected_annotations(
    annotations: list[dict[str, str]],
    query_terms: set[str],
    selected_text: str,
    *,
    include_all: bool,
) -> list[str]:
    selected_terms = _terms(selected_text)
    return [
        str(annotation.get("text", ""))
        for annotation in annotations
        if annotation.get("text")
        and (
            include_all
            or query_terms & _terms(str(annotation.get("text", "")))
            or selected_terms & _terms(str(annotation.get("text", "")))
        )
    ]


def compress_evidence(
    document: RetrievedDocument,
    aspect: str,
    question: str,
    *,
    canonical_terms: Sequence[str] = (),
    include_complete_procedure: bool = False,
    include_all_table_rows: bool = False,
    max_characters: int = 700,
) -> CompressedEvidenceView:
    """Create a lexical view without mutating the resolved parent document."""
    metadata = document["metadata"]
    query_terms = _terms(question, aspect, *canonical_terms)
    selected_sentences: list[int] = []
    selected_steps: list[int] = []
    selected_rows: list[int] = []
    section = str(metadata.get("section", "")).strip()
    annotations = list(metadata.get("annotations", []))
    rows = metadata.get("table_rows")
    steps = [str(step) for step in metadata.get("procedure_steps", [])]
    if not steps and document["content_type"] == "procedure":
        numbered = [
            line.strip()
            for line in document["text"].splitlines()
            if re.match(r"^\s*(?:\d+[.)]|[-*])\s+", line)
        ]
        if numbered:
            steps = numbered

    if rows:
        normalized_rows = [[str(cell) for cell in row] for row in rows]
        data_rows = normalized_rows[1:]
        if include_all_table_rows or ALL_ROWS_RE.search(question):
            selected_rows = list(range(len(data_rows)))
        else:
            ranked = _ranked_indexes(
                [" ".join(row) for row in data_rows], query_terms
            )
            selected_rows = sorted(
                index
                for index in ranked[:6]
                if query_terms & _terms(" ".join(data_rows[index]))
            ) or ranked[:1]
        compressed = _markdown_table(
            section,
            [normalized_rows[0], *(data_rows[index] for index in selected_rows)],
        )
        method = "deterministic_lexical_table_v1"
    elif steps:
        if include_complete_procedure:
            selected_steps = list(range(len(steps)))
        else:
            ranked = _ranked_indexes(steps, query_terms)
            selected_steps = sorted(
                index for index in ranked[:4] if query_terms & _terms(steps[index])
            ) or ranked[:1]
        compressed = "\n".join(
            [section, *(steps[index] for index in selected_steps)]
        ).strip()
        method = "deterministic_lexical_procedure_v1"
    else:
        sentences = _sentences(document["text"])
        ranked = _ranked_indexes(sentences, query_terms)
        direct = [
            index for index in ranked if query_terms & _terms(sentences[index])
        ][:3]
        if not direct and sentences:
            direct = [0]
        selected = set(direct)
        for index in direct[:1]:
            for adjacent in (index - 1, index + 1):
                if 0 <= adjacent < len(sentences) and (
                    CONDITION_RE.search(sentences[adjacent]) or len(direct) == 1
                ):
                    selected.add(adjacent)
        selected_sentences = sorted(selected)
        chosen: list[str] = []
        for index in selected_sentences:
            candidate = " ".join([*chosen, sentences[index]])
            if chosen and len(candidate) > max_characters:
                continue
            chosen.append(sentences[index])
        compressed = " ".join(chosen)
        method = "deterministic_lexical_sentences_v1"

    connected = _connected_annotations(
        annotations,
        query_terms,
        compressed,
        include_all=include_complete_procedure,
    )
    if connected:
        compressed = "\n\n".join([compressed, *connected])
    source_metadata = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "annotations",
            "aspects",
            "compressed_views",
            "procedure_steps",
            "table_rows",
        }
    }
    return {
        "evidence_id": str(metadata.get("evidence_id") or document["chunk_id"]),
        "aspect": aspect,
        "compressed_text": compressed,
        "selected_sentence_indexes": selected_sentences,
        "selected_step_indexes": selected_steps,
        "selected_table_row_indexes": selected_rows,
        "compression_method": method,
        "original_character_count": len(document["text"]),
        "compressed_character_count": len(compressed),
        "source_metadata": source_metadata,
    }
