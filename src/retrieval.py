"""Cached hybrid retrieval with weighted reciprocal-rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import pickle
import re
from typing import Any

import chromadb
from langchain_core.embeddings import Embeddings

from src.config import Settings, settings
from src.embeddings import create_embedding_model, create_reranker, finite_scores
from src.ingest import CHROMA_COLLECTION, _bm25_tokens, _require_index_schema
from src.schemas import RetrievedDocument


RRF_K = 60
ORIGINAL_QUERY_WEIGHT = 1.15
logger = logging.getLogger(__name__)
_disabled_rerankers: set[str] = set()
STOPWORDS = {
    "a", "an", "and", "are", "for", "how", "in", "is", "of", "the", "to", "what", "with"
}


@dataclass(slots=True)
class RetrievalResources:
    evidence_units_by_id: dict[str, dict[str, Any]]
    segments_by_id: dict[str, dict[str, Any]]
    bm25: Any
    bm25_ids: tuple[str, ...]
    chroma_collection: Any
    embedding_model: Embeddings


@lru_cache(maxsize=1)
def load_resources(config: Settings = settings) -> RetrievalResources:
    """Load EvidenceUnits, RetrievalSegments, BM25, Chroma, and embeddings."""
    _require_index_schema(config)
    evidence_path = config.evidence_units_path
    segments_path = config.retrieval_segments_path
    bm25_path = config.bm25_path
    if not evidence_path.exists() or not segments_path.exists() or not bm25_path.exists():
        raise FileNotFoundError("Run `python -m src.ingest` before retrieval")

    evidence_units = json.loads(evidence_path.read_text(encoding="utf-8"))
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    evidence_by_id = {str(unit["evidence_id"]): unit for unit in evidence_units}
    segments_by_id = {str(segment["segment_id"]): segment for segment in segments}
    if any(segment.get("evidence_id") not in evidence_by_id for segment in segments):
        raise ValueError("RetrievalSegments reference missing EvidenceUnits")
    with bm25_path.open("rb") as handle:
        payload = pickle.load(handle)
    bm25_ids = tuple(str(segment_id) for segment_id in payload["segment_ids"])
    collection = chromadb.PersistentClient(path=str(config.chroma_dir)).get_collection(
        CHROMA_COLLECTION
    )

    expected = set(segments_by_id)
    if expected != set(bm25_ids) or expected != set(collection.get(include=[])["ids"]):
        raise ValueError("Retrieval indexes do not match retrieval_segments.json")
    return RetrievalResources(
        evidence_units_by_id=evidence_by_id,
        segments_by_id=segments_by_id,
        bm25=payload["bm25"],
        bm25_ids=bm25_ids,
        chroma_collection=collection,
        embedding_model=create_embedding_model(config),
    )


def _result(
    record: dict[str, Any], retrieval_scores: dict[str, float]
) -> RetrievedDocument:
    if "segment_id" in record:
        metadata = {
            **record["metadata"],
            "segment_id": record["segment_id"],
            "evidence_id": record["evidence_id"],
            "segment_index": record["segment_index"],
            "previous_segment_id": record.get("previous_segment_id"),
            "next_segment_id": record.get("next_segment_id"),
            "word_count": record["word_count"],
            "embedding_token_count": record["embedding_token_count"],
            "chunk_level": "segment",
        }
        return {
            "chunk_id": str(record["segment_id"]),
            "text": str(record["searchable_text"]),
            "content_type": str(record["content_type"]),
            "metadata": metadata,
            "retrieval_scores": retrieval_scores,
        }

    table = record.get("structured_table")
    metadata = {
        **record["metadata"],
        "evidence_id": record["evidence_id"],
        "chunk_level": "evidence",
        "procedure_steps": record.get("procedure_steps", []),
        "annotations": record.get("annotations", []),
    }
    if table:
        metadata["table_rows"] = [table["headers"], *table["rows"]]
    return {
        "chunk_id": str(record["evidence_id"]),
        "text": str(record["text"]),
        "content_type": str(record["content_type"]),
        "metadata": metadata,
        "retrieval_scores": retrieval_scores,
    }


def vector_search(
    query: str,
    top_k: int | None = None,
    *,
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Return dense results using Chroma rank order."""
    if not query.strip():
        return []
    resources = load_resources(config)
    limit = top_k or config.vector_top_k
    response = resources.chroma_collection.query(
        query_embeddings=[resources.embedding_model.embed_query(query)],
        n_results=min(limit, len(resources.segments_by_id)),
        include=["distances"],
    )
    ids = response["ids"][0]
    distances = (response.get("distances") or [[]])[0]
    return [
        _result(
            resources.segments_by_id[chunk_id],
            {
                "vector_rank": float(rank),
                "vector_distance": float(distances[rank - 1]),
            },
        )
        for rank, chunk_id in enumerate(ids, start=1)
    ]


def bm25_search(
    query: str,
    top_k: int | None = None,
    *,
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Return lexical results using BM25 rank order."""
    tokens = _bm25_tokens(query)
    if not tokens:
        return []
    resources = load_resources(config)
    scores = resources.bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    ranked = [index for index in ranked if float(scores[index]) > 0][
        : top_k or config.bm25_top_k
    ]
    return [
        _result(
            resources.segments_by_id[resources.bm25_ids[index]],
            {"bm25_rank": float(rank), "bm25_score": float(scores[index])},
        )
        for rank, index in enumerate(ranked, start=1)
    ]


def _deduplicate_by_id(
    results: Sequence[RetrievedDocument],
) -> list[RetrievedDocument]:
    unique: dict[str, RetrievedDocument] = {}
    for result in results:
        chunk_id = result["chunk_id"]
        if chunk_id not in unique:
            unique[chunk_id] = _copy_result(result)
        else:
            _merge_scores(unique[chunk_id], result)
    return list(unique.values())


def deduplicate_results(
    results: Sequence[RetrievedDocument],
    *,
    intent: str = "",
) -> list[RetrievedDocument]:
    """Remove ID, exact-text, and near-text duplicates with intent-aware preference."""
    candidates = _deduplicate_by_id(results)
    kept: list[RetrievedDocument] = []
    # ponytail: O(n²) is simpler and bounded by the 20-candidate expansion limit.
    for candidate in candidates:
        normalized = _normalized_text(candidate["text"])
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(kept)
                if _are_duplicates(candidate, existing, normalized)
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(_copy_result(candidate))
            continue
        existing = kept[duplicate_index]
        if _preference_key(candidate, intent) > _preference_key(existing, intent):
            replacement = _copy_result(candidate)
            _merge_scores(replacement, existing)
            kept[duplicate_index] = replacement
        else:
            _merge_scores(existing, candidate)
    return kept


def reciprocal_rank_fusion(
    vector_results: Sequence[RetrievedDocument],
    bm25_results: Sequence[RetrievedDocument],
    *,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
    limit: int | None = None,
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Fuse dense and lexical ranks without combining their raw scores."""
    fused = _rank_fuse(
        [_deduplicate_by_id(vector_results), _deduplicate_by_id(bm25_results)],
        [vector_weight, bm25_weight],
        score_name="rrf_score",
    )
    return deduplicate_results(fused)[: limit or config.fused_top_k]


def retrieve_multiple_queries(
    standalone_query: str,
    search_queries: Sequence[str] | None = None,
    *,
    entities: Sequence[str] = (),
    preferred_manuals: Sequence[str] = (),
    intent: str = "",
    limit: int | None = None,
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Fuse hybrid results from the original question and query variations."""
    queries = _unique_queries([standalone_query, *(search_queries or [])])[
        : config.max_search_queries
    ]
    if not queries:
        return []

    per_query = [
        reciprocal_rank_fusion(
            vector_search(query, config.vector_top_k, config=config),
            bm25_search(query, config.bm25_top_k, config=config),
            config=config,
        )
        for query in queries
    ]
    weights = [ORIGINAL_QUERY_WEIGHT, *([1.0] * (len(per_query) - 1))]
    fused = _rank_fuse(per_query, weights, score_name="multi_query_rrf_score")
    for result in fused:
        scores = result["retrieval_scores"]
        scores["entity_bonus"] = _entity_bonus(result, entities)
        scores["heading_bonus"] = _heading_bonus(result, standalone_query)
        scores["manual_preference_bonus"] = _manual_bonus(result, preferred_manuals)
        scores["final_score"] = (
            scores["multi_query_rrf_score"]
            + scores["entity_bonus"]
            + scores["heading_bonus"]
            + scores["manual_preference_bonus"]
        )
    fused.sort(key=lambda result: result["retrieval_scores"]["final_score"], reverse=True)
    return deduplicate_results(fused, intent=intent or standalone_query)[
        : limit or config.fused_top_k
    ]


def retrieve_documents(
    query: str,
    limit: int | None = None,
    *,
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Retrieve candidates for a single standalone query."""
    return retrieve_multiple_queries(query, limit=limit, config=config)


def _rank_fuse(
    result_lists: Sequence[Sequence[RetrievedDocument]],
    weights: Sequence[float],
    *,
    score_name: str,
) -> list[RetrievedDocument]:
    fused: dict[str, RetrievedDocument] = {}
    totals: dict[str, float] = {}
    for results, weight in zip(result_lists, weights, strict=True):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = _copy_result(result)
            else:
                existing = fused[chunk_id]["retrieval_scores"]
                for key, value in result["retrieval_scores"].items():
                    previous = existing.get(key, value)
                    existing[key] = (
                        min(previous, value)
                        if key.endswith(("_rank", "_distance"))
                        else max(previous, value)
                    )
            totals[chunk_id] = totals.get(chunk_id, 0.0) + weight / (RRF_K + rank)
    for chunk_id, score in totals.items():
        fused[chunk_id]["retrieval_scores"][score_name] = score
    return sorted(
        fused.values(),
        key=lambda result: result["retrieval_scores"][score_name],
        reverse=True,
    )


def _copy_result(result: RetrievedDocument) -> RetrievedDocument:
    return {
        "chunk_id": result["chunk_id"],
        "text": result["text"],
        "content_type": result["content_type"],
        "metadata": dict(result["metadata"]),
        "retrieval_scores": dict(result["retrieval_scores"]),
    }


def _merge_scores(target: RetrievedDocument, source: RetrievedDocument) -> None:
    existing = target["retrieval_scores"]
    for key, value in source["retrieval_scores"].items():
        previous = existing.get(key, value)
        existing[key] = (
            min(previous, value)
            if key.endswith(("_rank", "_distance"))
            else max(previous, value)
        )
    aspects = list(
        dict.fromkeys(
            [
                *target["metadata"].get("aspects", []),
                *source["metadata"].get("aspects", []),
            ]
        )
    )
    if aspects:
        target["metadata"]["aspects"] = aspects


def _normalized_text(text: str) -> str:
    return " ".join(_bm25_tokens(text))


def _text_overlap(left: str, right: str) -> float:
    left_tokens = _bm25_tokens(left)
    right_tokens = _bm25_tokens(right)
    if min(len(left_tokens), len(right_tokens)) < 6:
        return 0.0
    left_shingles = set(zip(left_tokens, left_tokens[1:], left_tokens[2:]))
    right_shingles = set(zip(right_tokens, right_tokens[1:], right_tokens[2:]))
    denominator = min(len(left_shingles), len(right_shingles))
    return len(left_shingles & right_shingles) / denominator if denominator else 0.0


def _are_duplicates(
    candidate: RetrievedDocument,
    existing: RetrievedDocument,
    candidate_normalized: str,
) -> bool:
    """Apply near-text dedup only when it cannot discard complementary context."""
    if candidate_normalized == _normalized_text(existing["text"]):
        return True
    candidate_aspects = set(candidate["metadata"].get("aspects", []))
    existing_aspects = set(existing["metadata"].get("aspects", []))
    if candidate_aspects and existing_aspects and candidate_aspects.isdisjoint(existing_aspects):
        return False
    candidate_evidence = candidate["metadata"].get("evidence_id")
    existing_evidence = existing["metadata"].get("evidence_id")
    if candidate_evidence and existing_evidence and candidate_evidence != existing_evidence:
        return False
    if candidate["content_type"] == "table" or existing["content_type"] == "table":
        return False
    levels = {
        candidate["metadata"].get("chunk_level"),
        existing["metadata"].get("chunk_level"),
    }
    if levels == {"evidence", "segment"}:
        return False
    return _text_overlap(candidate["text"], existing["text"]) >= 0.85


def _preference_key(result: RetrievedDocument, intent: str) -> tuple[float, ...]:
    metadata = result["metadata"]
    content_type = result["content_type"]
    intent_text = intent.casefold()
    score_values = [
        result["retrieval_scores"][name]
        for name in (
            "reranker_score",
            "final_score",
            "multi_query_rrf_score",
            "rrf_score",
        )
        if name in result["retrieval_scores"]
    ]
    score = max(score_values, default=0.0)
    if re.search(r"\b(?:procedure|steps?|how[_ ]?to|how do)\b", intent_text):
        step_count = len(re.findall(r"(?m)^\s*\d{1,3}[.)]\s*", result["text"]))
        return (3.0, float(step_count), float(content_type == "procedure"), len(result["text"]), score)
    if re.search(r"\b(?:field|table|column|button|configuration)\b", intent_text):
        structured = content_type in {"table", "field_definition"} and bool(
            metadata.get("table_rows")
        )
        return (2.0, float(structured), score, float(metadata.get("chunk_level") == "evidence"))
    specificity = float(metadata.get("chunk_level") == "evidence") + float(content_type != "text")
    return (1.0, score, specificity, -float(len(result["text"])))


def _unique_queries(queries: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = " ".join(query.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def _entity_bonus(result: RetrievedDocument, entities: Sequence[str]) -> float:
    metadata = result["metadata"]
    searchable = " ".join(
        [result["text"], str(metadata.get("chapter", "")), str(metadata.get("section", ""))]
    ).casefold()
    normalized = [entity.strip().casefold() for entity in entities if entity.strip()]
    matches = sum(1 for entity in normalized if entity in searchable)
    return min(0.009, matches * 0.003)


def _heading_bonus(result: RetrievedDocument, query: str) -> float:
    metadata = result["metadata"]
    heading = f"{metadata.get('chapter', '')} {metadata.get('section', '')}".casefold()
    normalized_query = " ".join(query.casefold().split())
    if normalized_query and normalized_query in heading:
        return 0.006
    query_terms = {term for term in _bm25_tokens(query) if term not in STOPWORDS}
    if not query_terms:
        return 0.0
    heading_terms = set(_bm25_tokens(heading))
    return 0.004 * len(query_terms & heading_terms) / len(query_terms)


def _manual_bonus(result: RetrievedDocument, preferred_manuals: Sequence[str]) -> float:
    metadata = result["metadata"]
    manual = f"{metadata.get('manual', '')} {metadata.get('source_file', '')}".casefold()
    return (
        0.004
        if any(
            preference.strip() and preference.strip().casefold() in manual
            for preference in preferred_manuals
        )
        else 0.0
    )


def expand_context(
    documents: list[RetrievedDocument],
    *,
    intent: str = "",
    limit: int = 20,
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Add neighbouring RetrievalSegments without loading parent sections."""
    resources = load_resources(config)
    seeds = deduplicate_results(documents, intent=intent)[: max(1, limit // 2)]
    expanded = [_copy_result(seed) for seed in seeds]
    for seed in seeds:
        segment = resources.segments_by_id.get(seed["chunk_id"])
        if not segment:
            continue
        for relation, neighbour_id in (
            ("previous", segment.get("previous_segment_id")),
            ("next", segment.get("next_segment_id")),
        ):
            neighbour = resources.segments_by_id.get(str(neighbour_id))
            if not neighbour:
                continue
            scores = dict(seed["retrieval_scores"])
            scores["context_expansion_rank"] = float(len(expanded) + 1)
            result = _result(neighbour, scores)
            result["metadata"]["context_relation"] = relation
            result["metadata"]["aspects"] = list(seed["metadata"].get("aspects", []))
            expanded.append(result)
    return deduplicate_results(expanded, intent=intent)[:limit]


def rerank_documents(
    query: str,
    documents: list[RetrievedDocument],
    *,
    intent: str = "",
    limit: int | None = None,
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Cross-encode expanded RetrievalSegments, with an RRF-order fallback."""
    candidates = deduplicate_results(documents, intent=intent or query)[:20]
    if config.reranker_model in _disabled_rerankers:
        return _reranker_fallback(candidates, intent or query, limit, config)
    try:
        scores = finite_scores(
            create_reranker(config).predict(
                [(query, result["text"]) for result in candidates],
                show_progress_bar=False,
            ),
            len(candidates),
        )
        for result, score in zip(candidates, scores, strict=True):
            result["retrieval_scores"]["reranker_score"] = score
        candidates.sort(
            key=lambda result: result["retrieval_scores"]["reranker_score"],
            reverse=True,
        )
    except Exception as exc:
        _disabled_rerankers.add(config.reranker_model)
        logger.warning("Cross-encoder unavailable; retaining RRF order: %s", exc)
        return _reranker_fallback(candidates, intent or query, limit, config)
    final = deduplicate_results(candidates, intent=intent or query)
    return final[: limit or config.rerank_top_k]


def resolve_evidence_units(
    documents: list[RetrievedDocument],
    *,
    limit: int | None = None,
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Resolve selected RetrievalSegments and deduplicate complete EvidenceUnits."""
    resources = load_resources(config)
    resolved: dict[str, RetrievedDocument] = {}
    for document in documents:
        evidence_id = str(document["metadata"].get("evidence_id", ""))
        unit = resources.evidence_units_by_id.get(evidence_id)
        if not unit:
            continue
        evidence = _result(unit, dict(document["retrieval_scores"]))
        evidence["metadata"]["aspects"] = list(document["metadata"].get("aspects", []))
        evidence["metadata"]["selected_segment_id"] = document["chunk_id"]
        evidence["metadata"]["context_relation"] = "evidence_unit"
        if evidence_id in resolved:
            _merge_scores(resolved[evidence_id], evidence)
        else:
            resolved[evidence_id] = evidence
    return list(resolved.values())[: limit or config.rerank_top_k]


def _reranker_fallback(
    candidates: list[RetrievedDocument],
    intent: str,
    limit: int | None,
    config: Settings,
) -> list[RetrievedDocument]:
    for result in candidates:
        result["retrieval_scores"]["reranker_fallback"] = 1.0
    return deduplicate_results(candidates, intent=intent)[: limit or config.rerank_top_k]
