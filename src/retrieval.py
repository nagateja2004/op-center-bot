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
from src.cache import get as cache_get, normalized, set as cache_set
from src.embeddings import create_embedding_model, create_reranker, finite_scores
from src.ingest import (
    REPRESENTATION_COLLECTION,
    _bm25_tokens,
    _require_index_schema,
)
from src.schemas import RetrievedDocument


RRF_K = 60
ORIGINAL_QUERY_WEIGHT = 1.15
PREFERRED_MANUAL_QUOTA = 4
HEADING_TOP_K = 8
CONCEPT_TOP_K = 8
logger = logging.getLogger(__name__)
_disabled_rerankers: set[str] = set()
_chroma_client: Any | None = None
STOPWORDS = {
    "a", "an", "and", "are", "for", "how", "in", "is", "of", "the", "to", "what", "with"
}


@dataclass(slots=True)
class RetrievalResources:
    evidence_units_by_id: dict[str, dict[str, Any]]
    segments_by_id: dict[str, dict[str, Any]]
    segments_by_evidence_id: dict[str, tuple[dict[str, Any], ...]]
    representations_by_id: dict[str, dict[str, Any]]
    bm25: Any
    bm25_ids: tuple[str, ...]
    chroma_collection: Any
    representation_collection: Any
    embedding_model: Embeddings
    headings: tuple[dict[str, Any], ...]
    concepts: tuple[dict[str, Any], ...]
    alias_mappings: dict[str, tuple[str, ...]]
    manual_names: tuple[str, ...]
    heading_embeddings: list[list[float]] | None = None


def create_chroma_client(config: Settings = settings) -> Any:
    """Create the configured Chroma client; production uses the HTTP server."""
    if config.chroma_mode == "local":
        return chromadb.PersistentClient(path=str(config.chroma_dir))
    return chromadb.HttpClient(
        host=config.chroma_host,
        port=config.chroma_port,
        ssl=config.chroma_ssl,
    )


def configure_chroma_client(client: Any | None) -> None:
    """Set the process-wide Chroma client created during backend startup."""
    global _chroma_client
    _chroma_client = client
    load_resources.cache_clear()


def _configured_chroma_client(config: Settings) -> Any:
    if _chroma_client is not None:
        return _chroma_client
    if config.chroma_mode == "local":
        return create_chroma_client(config)
    raise RuntimeError("Chroma HTTP client has not been initialized")


@lru_cache(maxsize=1)
def load_resources(config: Settings = settings) -> RetrievalResources:
    """Load EvidenceUnits, RetrievalSegments, BM25, Chroma, and embeddings."""
    _require_index_schema(config)
    evidence_path = config.evidence_units_path
    segments_path = config.retrieval_segments_path
    representations_path = config.search_representations_path
    bm25_path = config.bm25_path
    if not all(
        path.exists()
        for path in (evidence_path, segments_path, representations_path, bm25_path)
    ):
        raise FileNotFoundError("Run `python -m src.ingest` before retrieval")

    evidence_units = json.loads(evidence_path.read_text(encoding="utf-8"))
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    representations = json.loads(representations_path.read_text(encoding="utf-8"))
    evidence_by_id = {str(unit["evidence_id"]): unit for unit in evidence_units}
    segments_by_id = {str(segment["segment_id"]): segment for segment in segments}
    representations_by_id = {
        str(item["representation_id"]): item for item in representations
    }
    segments_by_evidence: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        segments_by_evidence.setdefault(str(segment["evidence_id"]), []).append(segment)
    if any(segment.get("evidence_id") not in evidence_by_id for segment in segments):
        raise ValueError("RetrievalSegments reference missing EvidenceUnits")
    if any(item.get("evidence_id") not in evidence_by_id for item in representations):
        raise ValueError("SearchRepresentations reference missing EvidenceUnits")
    with bm25_path.open("rb") as handle:
        payload = pickle.load(handle)
    bm25_ids = tuple(str(segment_id) for segment_id in payload["segment_ids"])
    client = _configured_chroma_client(config)
    collection = client.get_collection(config.chroma_collection)
    representation_collection = client.get_collection(REPRESENTATION_COLLECTION)

    expected = set(segments_by_id)
    if expected != set(bm25_ids) or expected != set(collection.get(include=[])["ids"]):
        raise ValueError("Retrieval indexes do not match retrieval_segments.json")
    if set(representations_by_id) != set(
        representation_collection.get(include=[])["ids"]
    ):
        raise ValueError(
            "Representation index does not match search_representations.json"
        )
    headings = _load_json_records(config.heading_index_path, "heading")
    concepts = _load_json_records(config.concept_index_path, "concept")
    alias_mappings = _load_alias_mappings(config.alias_config_path)
    return RetrievalResources(
        evidence_units_by_id=evidence_by_id,
        segments_by_id=segments_by_id,
        segments_by_evidence_id={
            evidence_id: tuple(sorted(items, key=lambda item: item["segment_index"]))
            for evidence_id, items in segments_by_evidence.items()
        },
        representations_by_id=representations_by_id,
        bm25=payload["bm25"],
        bm25_ids=bm25_ids,
        chroma_collection=collection,
        representation_collection=representation_collection,
        embedding_model=create_embedding_model(config),
        headings=tuple(headings),
        concepts=tuple(concepts),
        alias_mappings=alias_mappings,
        manual_names=tuple(sorted({str(item["metadata"].get("manual", "")) for item in segments})),
    )


def _load_json_records(path: Any, label: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise ValueError("root must be a list of objects")
        return value
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not load %s index from %s: %s", label, path, exc)
        return []


def _load_alias_mappings(path: Any) -> dict[str, tuple[str, ...]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("root must be an object")
        mappings: dict[str, tuple[str, ...]] = {}
        for canonical, entry in value.items():
            aliases = entry.get("aliases", []) if isinstance(entry, dict) else []
            if not isinstance(canonical, str) or not isinstance(aliases, list):
                raise ValueError("concept aliases must be string lists")
            mappings[canonical] = tuple(
                str(alias) for alias in aliases if isinstance(alias, str)
            )
        return mappings
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not load Opcenter aliases from %s: %s", path, exc)
        return {}


def _result(
    record: dict[str, Any], retrieval_scores: dict[str, float]
) -> RetrievedDocument:
    if "representation_id" in record:
        metadata = {
            **record["metadata"],
            "representation_id": record["representation_id"],
            "representation_type": record["representation_type"],
            "evidence_id": record["evidence_id"],
            "embedding_token_count": record["embedding_token_count"],
            "chunk_level": "representation",
        }
        return {
            "chunk_id": str(record["representation_id"]),
            "text": str(record["text"]),
            "content_type": str(record["representation_type"]),
            "metadata": metadata,
            "retrieval_scores": retrieval_scores,
        }
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
            "effective_embedding_limit": record.get("effective_embedding_limit"),
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
    embedding_key = {"query": normalized(query), "model": config.embedding_model}
    embedding = cache_get("embedding", embedding_key)
    if embedding is None:
        embedding = resources.embedding_model.embed_query(query)
        cache_set("embedding", embedding_key, embedding, ttl=3600)
    response = resources.chroma_collection.query(
        query_embeddings=[embedding],
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


def representation_search(
    query: str,
    top_k: int | None = None,
    *,
    preferred_manuals: Sequence[str] = (),
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Return deterministic non-body representation vectors by rank."""
    if not query.strip():
        return []
    resources = load_resources(config)
    limit = top_k or config.vector_top_k
    where: dict[str, Any] | None = None
    if preferred_manuals:
        manuals = _matching_manual_names(resources, preferred_manuals)
        if not manuals:
            return []
        where = {"manual": {"$in": manuals}}
    response = resources.representation_collection.query(
        query_embeddings=[resources.embedding_model.embed_query(query)],
        n_results=min(limit, len(resources.representations_by_id)),
        where=where,
        include=["distances"],
    )
    ids = response["ids"][0]
    distances = (response.get("distances") or [[]])[0]
    return [
        _result(
            resources.representations_by_id[representation_id],
            {
                "representation_rank": float(rank),
                "representation_distance": float(distances[rank - 1]),
            },
        )
        for rank, representation_id in enumerate(ids, start=1)
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


def heading_search(
    query: str,
    top_k: int = HEADING_TOP_K,
    *,
    preferred_manuals: Sequence[str] = (),
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Rank actual body headings by exact, alternate, fuzzy, then semantic match."""
    if not query.strip():
        return []
    resources = load_resources(config)
    query_keys = _heading_query_keys(query)
    ranked: list[tuple[int, float, dict[str, Any]]] = []
    for heading in resources.headings:
        if heading.get("is_toc") or not _manual_matches(heading, preferred_manuals):
            continue
        keys = {
            _normalize_lookup(str(heading.get("normalized_heading", ""))),
            *(_normalize_lookup(str(key)) for key in heading.get("alternate_keys", [])),
        }
        keys.discard("")
        exact = bool(keys.intersection(query_keys))
        fuzzy = max(
            (_token_similarity(query_key, key) for query_key in query_keys for key in keys),
            default=0.0,
        )
        if exact or fuzzy >= 0.5:
            ranked.append((2 if exact else 1, 1.0 if exact else fuzzy, heading))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not ranked or (ranked[0][0] < 2 and ranked[0][1] < 0.65):
        ranked = _semantic_heading_matches(query, resources, preferred_manuals, ranked)
    results = _linked_results(
        ranked,
        top_k,
        resources,
        lambda tier, score, record, rank: {
            "heading_rank": float(rank),
            "heading_exact": float(tier == 2),
            "heading_fuzzy": float(tier == 1),
            "heading_score": float(score),
        },
        match_key="matched_heading",
        name_key="original_heading",
        preferred_manuals=preferred_manuals,
        relevance_values=[query],
    )
    logger.info(
        "heading_matches query_length=%d matches=%s",
        len(query),
        [result["metadata"].get("matched_heading") for result in results[:5]],
    )
    return results


def concept_search(
    query: str,
    top_k: int = CONCEPT_TOP_K,
    *,
    entities: Sequence[str] = (),
    aliases: Sequence[str] = (),
    preferred_manuals: Sequence[str] = (),
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Map canonical terms and aliases to concept-linked RetrievalSegments."""
    if not query.strip() and not entities and not aliases:
        return []
    resources = load_resources(config)
    scoped_values = [*entities, *aliases]
    scoped_inputs = {
        key
        for value in scoped_values
        if str(value).strip()
        for key in _lookup_keys(str(value))
    }
    inputs = {
        key
        for value in [query, *scoped_values]
        if str(value).strip()
        for key in _lookup_keys(str(value))
    }
    query_key = _normalize_lookup(query)
    target_context = _normalize_lookup(" ".join(scoped_values)) if scoped_values else query_key
    target_inputs = scoped_inputs or inputs
    targets: set[str] = set()
    alias_targets: set[str] = set()
    alias_target_weights: dict[str, int] = {}
    alias_component_weights: dict[str, int] = {}
    matched_aliases: set[str] = set()
    available_concepts = {
        key
        for concept in resources.concepts
        for key in _lookup_keys(str(concept.get("canonical_name", "")))
    }
    for canonical, curated_aliases in resources.alias_mappings.items():
        canonical_key = _normalize_lookup(canonical)
        alias_keys = {_normalize_lookup(alias) for alias in curated_aliases}
        alias_hits = {
            alias for alias in alias_keys
            if alias and (alias in target_context or alias in target_inputs)
        }
        if canonical_key in target_inputs or alias_hits:
            targets.update(_lookup_keys(canonical))
            if alias_hits:
                canonical_keys = _lookup_keys(canonical)
                alias_targets.update(canonical_keys)
                specificity = max(len(alias.split()) for alias in alias_hits)
                for key in canonical_keys:
                    alias_target_weights[key] = max(
                        alias_target_weights.get(key, 0), specificity
                    )
                    words = key.split()
                    if words and not canonical_keys.intersection(available_concepts):
                        alias_component_weights[words[0]] = max(
                            alias_component_weights.get(words[0], 0), specificity
                        )
                        alias_component_weights[words[-1]] = max(
                            alias_component_weights.get(words[-1], 0), specificity + 1
                        )
                matched_aliases.update(alias_hits)

    ranked: list[tuple[int, float, dict[str, Any]]] = []
    for concept in resources.concepts:
        if not _manual_matches(concept, preferred_manuals):
            continue
        keys = {
            key
            for value in [concept.get("canonical_name", ""), *concept.get("aliases", [])]
            for key in _lookup_keys(str(value))
        }
        keys.discard("")
        alias_target_exact = bool(keys.intersection(alias_targets))
        alias_component_exact = bool(keys.intersection(alias_component_weights))
        target_exact = bool(keys.intersection(targets))
        query_exact = any(
            len(key.split()) > 1 and key in query_key for key in keys
        )
        input_exact = bool(keys.intersection(inputs))
        target_similarity = max(
            (_token_similarity(value, key) for value in targets for key in keys),
            default=0.0,
        )
        alias_target_similarity = max(
            (_token_similarity(value, key) for value in alias_targets for key in keys),
            default=0.0,
        )
        alias_weight = max(
            (
                weight
                for target, weight in alias_target_weights.items()
                if any(_token_similarity(target, key) >= 0.5 for key in keys)
            ),
            default=0,
        )
        component_weight = max(
            (alias_component_weights[key] for key in keys if key in alias_component_weights),
            default=0,
        )
        similarity = max(
            (_token_similarity(value, key) for value in inputs for key in keys),
            default=0.0,
        )
        if target_exact or query_exact or input_exact or target_similarity >= 0.5 or similarity >= 0.5:
            if alias_target_exact:
                tier, score = 20 + min(alias_weight, 5), 1.0
            elif alias_component_exact:
                tier, score = 15 + min(component_weight, 5), 0.99
            elif alias_target_similarity >= 0.5:
                tier, score = 10 + min(alias_weight, 5), alias_target_similarity
            elif target_exact:
                tier, score = 4, 0.98
            elif query_exact:
                tier, score = 3, 0.96
            elif target_similarity >= 0.5:
                tier, score = 2, target_similarity
            elif input_exact:
                tier, score = 2, 0.9
            else:
                tier, score = 1, similarity
            ranked.append((tier, score, concept))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    results = _linked_results(
        ranked,
        top_k,
        resources,
        lambda tier, score, record, rank: {
            "concept_rank": float(rank),
            "concept_exact": float(score >= 0.96),
            "alias_match": float(
                tier >= 5
                or bool(
                    matched_aliases.intersection(
                        {
                            _normalize_lookup(str(record.get("canonical_name", ""))),
                            *(
                                _normalize_lookup(str(alias))
                                for alias in record.get("aliases", [])
                            ),
                        }
                    )
                )
            ),
            "concept_score": float(score),
        },
        match_key="matched_concept",
        name_key="canonical_name",
        preferred_manuals=preferred_manuals,
        relevance_values=[query, *targets],
    )
    logger.info(
        "concept_matches query_length=%d matches=%s",
        len(query),
        [result["metadata"].get("matched_concept") for result in results[:5]],
    )
    return results


def _linked_results(
    ranked: Sequence[tuple[int, float, dict[str, Any]]],
    top_k: int,
    resources: RetrievalResources,
    scores: Any,
    *,
    match_key: str,
    name_key: str,
    preferred_manuals: Sequence[str] = (),
    relevance_values: Sequence[str] = (),
) -> list[RetrievedDocument]:
    results: list[RetrievedDocument] = []
    seen: set[str] = set()
    prepared: list[tuple[int, int, float, dict[str, Any], list[str]]] = []
    for rank, (tier, score, record) in enumerate(ranked, start=1):
        evidence_ids = list(record.get("evidence_ids", record.get("related_evidence_ids", [])))
        evidence_queries = [*relevance_values, " ".join(relevance_values)]
        evidence_ids.sort(
            key=lambda evidence_id: max(
                (
                    _token_similarity(
                        value,
                        " ".join(
                            [
                                str(
                                    resources.evidence_units_by_id.get(str(evidence_id), {})
                                    .get("metadata", {})
                                    .get("section", "")
                                ),
                                str(
                                    resources.evidence_units_by_id.get(str(evidence_id), {})
                                    .get("text", "")
                                ),
                            ]
                        ),
                    )
                    for value in evidence_queries
                ),
                default=0.0,
            ),
            reverse=True,
        )
        if preferred_manuals:
            evidence_ids = [
                evidence_id
                for evidence_id in evidence_ids
                if _manual_matches(
                    resources.evidence_units_by_id.get(str(evidence_id), {}).get("metadata", {}),
                    preferred_manuals,
                )
            ]
        prepared.append((rank, tier, score, record, evidence_ids))
    depth = 0
    while len(results) < top_k:
        added = False
        for rank, tier, score, record, evidence_ids in prepared[:top_k]:
            if depth >= len(evidence_ids):
                continue
            evidence_id = evidence_ids[depth]
            for segment in resources.segments_by_evidence_id.get(str(evidence_id), ()):
                if not _manual_matches(segment.get("metadata", {}), preferred_manuals):
                    continue
                segment_id = str(segment["segment_id"])
                if segment_id in seen:
                    continue
                result = _result(segment, scores(tier, score, record, rank))
                result["metadata"][match_key] = record.get(name_key, "")
                results.append(result)
                seen.add(segment_id)
                added = True
                if len(results) >= top_k:
                    return results
                break
        if not added:
            return results
        depth += 1
    return results


def _heading_query_keys(query: str) -> set[str]:
    normalized = _normalize_lookup(query)
    keys = {normalized}
    stripped = re.sub(
        r"^(?:(?:please|tell me about|tell me|explain|describe|define|defining|what is|what are)\s+)+",
        "",
        normalized,
    ).strip()
    if stripped:
        keys.add(stripped)
    if stripped.startswith("defining "):
        keys.add(stripped.removeprefix("defining "))
    return keys


def _normalize_lookup(value: str) -> str:
    normalized = value.casefold().replace("modelling", "modeling")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _lookup_keys(value: str) -> set[str]:
    normalized = _normalize_lookup(value)
    keys = {normalized} if normalized else set()
    words = normalized.split()
    if words and words[-1] in {
        "rules", "fields", "models", "controls", "permissions", "roles",
        "resources", "containers", "patterns", "events", "steps", "parts",
    }:
        keys.add(" ".join([*words[:-1], words[-1][:-1]]))
    return keys


def _token_similarity(left: str, right: str) -> float:
    left_terms = {term for term in _bm25_tokens(left) if term not in STOPWORDS}
    right_terms = {term for term in _bm25_tokens(right) if term not in STOPWORDS}
    if not left_terms or not right_terms:
        return 0.0
    overlap = len(left_terms & right_terms)
    score = overlap / min(len(left_terms), len(right_terms))
    return score * (0.7 if min(len(left_terms), len(right_terms)) == 1 else 1.0)


def _semantic_heading_matches(
    query: str,
    resources: RetrievalResources,
    preferred_manuals: Sequence[str],
    existing: list[tuple[int, float, dict[str, Any]]],
) -> list[tuple[int, float, dict[str, Any]]]:
    headings = [heading for heading in resources.headings if not heading.get("is_toc")]
    if resources.heading_embeddings is None:
        resources.heading_embeddings = resources.embedding_model.embed_documents(
            [str(heading.get("original_heading", "")) for heading in headings]
        )
    query_vector = resources.embedding_model.embed_query(query)
    semantic = [
        (0, sum(left * right for left, right in zip(query_vector, vector)), heading)
        for heading, vector in zip(headings, resources.heading_embeddings, strict=True)
        if _manual_matches(heading, preferred_manuals)
    ]
    semantic.sort(key=lambda item: item[1], reverse=True)
    existing_keys = {
        (str(record.get("source_file", "")), str(record.get("section", "")))
        for _, _, record in existing
    }
    existing.extend(
        item
        for item in semantic[:HEADING_TOP_K]
        if (str(item[2].get("source_file", "")), str(item[2].get("section", "")))
        not in existing_keys
    )
    return sorted(existing, key=lambda item: (item[0], item[1]), reverse=True)


def _manual_matches(record: dict[str, Any], preferences: Sequence[str]) -> bool:
    if not preferences:
        return True
    searchable = " ".join(
        [
            str(record.get("manual", "")),
            str(record.get("source_file", "")),
            " ".join(str(value) for value in record.get("manuals", [])),
        ]
    ).casefold()
    return any(preference.strip().casefold() in searchable for preference in preferences if preference.strip())


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


def deduplicate_exact_results(
    results: Sequence[RetrievedDocument],
) -> list[RetrievedDocument]:
    """Remove only duplicate IDs and exact normalized text before reranking."""
    kept: list[RetrievedDocument] = []
    by_text: dict[str, int] = {}
    for candidate in _deduplicate_by_id(results):
        normalized = _normalized_text(candidate["text"])
        if normalized not in by_text:
            by_text[normalized] = len(kept)
            kept.append(_copy_result(candidate))
        else:
            _merge_scores(kept[by_text[normalized]], candidate)
    return kept


def limit_candidates_per_evidence(
    results: Sequence[RetrievedDocument], *, maximum: int = 2
) -> list[RetrievedDocument]:
    """Keep rank order while preferring representation diversity per parent."""
    grouped: dict[str, list[RetrievedDocument]] = {}
    order: list[str] = []
    for result in results:
        evidence_id = str(result["metadata"].get("evidence_id", ""))
        if not evidence_id:
            continue
        if evidence_id not in grouped:
            grouped[evidence_id] = []
            order.append(evidence_id)
        grouped[evidence_id].append(result)
    selected_ids: set[str] = set()
    for evidence_id in order:
        candidates = grouped[evidence_id]
        chosen = [candidates[0]]
        first_type = str(
            candidates[0]["metadata"].get("representation_type", "body")
        )
        different = next(
            (
                candidate
                for candidate in candidates[1:]
                if str(candidate["metadata"].get("representation_type", "body"))
                != first_type
            ),
            None,
        )
        if maximum > 1 and len(candidates) > 1:
            chosen.append(different or candidates[1])
        selected_ids.update(candidate["chunk_id"] for candidate in chosen[:maximum])
    return [
        _copy_result(result)
        for result in results
        if not result["metadata"].get("evidence_id")
        or result["chunk_id"] in selected_ids
    ]


def prepare_rerank_candidates(
    results: Sequence[RetrievedDocument], *, limit: int = 20
) -> list[RetrievedDocument]:
    return limit_candidates_per_evidence(results)[:limit]


def deduplicate_results(
    results: Sequence[RetrievedDocument],
    *,
    intent: str = "",
) -> list[RetrievedDocument]:
    """Compatibility wrapper for the exact-only deduplication policy."""
    del intent
    return deduplicate_exact_results(results)


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
    return deduplicate_exact_results(fused)[: limit or config.fused_top_k]


def retrieve_multiple_queries(
    standalone_query: str,
    search_queries: Sequence[str] | None = None,
    *,
    entities: Sequence[str] = (),
    aliases: Sequence[str] = (),
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

    cache_key = {
        "question": normalized(standalone_query), "queries": [normalized(query) for query in queries],
        "entities": list(entities), "aliases": list(aliases), "manuals": list(preferred_manuals),
        "intent": intent, "index": _index_version(config),
        "config": [config.vector_top_k, config.bm25_top_k, config.fused_top_k, config.max_search_queries],
    }
    cached = cache_get("retrieval", cache_key)
    if cached is not None:
        return cached
    per_query = [
        _retrieve_query_paths(
            query,
            entities=entities,
            aliases=aliases,
            preferred_manuals=preferred_manuals,
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
    results = deduplicate_exact_results(fused)[: limit or config.fused_top_k]
    cache_set("retrieval", cache_key, results)
    return results


def _index_version(config: Settings) -> str:
    try:
        return str(json.loads((config.indexes_dir / "manifest.json").read_text()).get("generated_at", ""))
    except Exception:
        return "unknown"


def _retrieve_query_paths(
    query: str,
    *,
    entities: Sequence[str],
    aliases: Sequence[str],
    preferred_manuals: Sequence[str],
    config: Settings,
) -> list[RetrievedDocument]:
    vector = vector_search(query, config.vector_top_k, config=config)
    representations = representation_search(query, config.vector_top_k, config=config)
    bm25 = bm25_search(query, config.bm25_top_k, config=config)
    headings = heading_search(query, config=config)
    concepts = concept_search(
        query,
        entities=entities,
        aliases=aliases,
        config=config,
    )
    preferred_vector = _preferred_vector_search(query, preferred_manuals, config)
    preferred_representations = representation_search(
        query,
        PREFERRED_MANUAL_QUOTA,
        preferred_manuals=preferred_manuals,
        config=config,
    ) if preferred_manuals else []
    preferred_bm25 = _preferred_bm25_search(query, preferred_manuals, config)
    preferred_headings = heading_search(
        query,
        top_k=PREFERRED_MANUAL_QUOTA,
        preferred_manuals=preferred_manuals,
        config=config,
    ) if preferred_manuals else []
    preferred_concepts = concept_search(
        query,
        top_k=PREFERRED_MANUAL_QUOTA,
        entities=entities,
        aliases=aliases,
        preferred_manuals=preferred_manuals,
        config=config,
    ) if preferred_manuals else []
    exact_headings = [result for result in headings if result["retrieval_scores"].get("heading_exact")]
    supporting_headings = [result for result in headings if not result["retrieval_scores"].get("heading_exact")]
    preferred_exact_headings = [
        result for result in preferred_headings if result["retrieval_scores"].get("heading_exact")
    ]
    preferred_supporting_headings = [
        result for result in preferred_headings if not result["retrieval_scores"].get("heading_exact")
    ]
    fused = _rank_fuse(
        [
            exact_headings,
            concepts,
            bm25,
            preferred_exact_headings,
            supporting_headings,
            preferred_concepts,
            preferred_bm25,
            vector,
            representations,
            preferred_vector,
            preferred_representations,
            preferred_supporting_headings,
        ],
        [2.2, 1.35, 1.2, 1.1, 0.7, 0.85, 0.75, 1.0, 1.0, 0.65, 0.65, 0.45],
        score_name="rrf_score",
    )
    logger.info(
        "query_retrieval query_length=%d preferred=%s body=%s representations=%s bm25=%s fused=%s",
        len(query),
        list(preferred_manuals),
        _sections(vector[:5]),
        _sections(representations[:5]),
        _sections(bm25[:5]),
        _sections(fused[:8]),
    )
    return fused[: config.fused_top_k]


def _preferred_vector_search(
    query: str, preferences: Sequence[str], config: Settings
) -> list[RetrievedDocument]:
    if not preferences:
        return []
    resources = load_resources(config)
    results: list[RetrievedDocument] = []
    for manual in _matching_manual_names(resources, preferences):
        response = resources.chroma_collection.query(
            query_embeddings=[resources.embedding_model.embed_query(query)],
            n_results=min(PREFERRED_MANUAL_QUOTA, len(resources.segments_by_id)),
            where={"manual": {"$eq": manual}},
            include=["distances"],
        )
        ids = response["ids"][0]
        distances = (response.get("distances") or [[]])[0]
        results.extend(
            _result(
                resources.segments_by_id[segment_id],
                {
                    "preferred_vector_rank": float(rank),
                    "preferred_vector_distance": float(distances[rank - 1]),
                },
            )
            for rank, segment_id in enumerate(ids, start=1)
        )
    return _deduplicate_by_id(results)


def _preferred_bm25_search(
    query: str, preferences: Sequence[str], config: Settings
) -> list[RetrievedDocument]:
    tokens = _bm25_tokens(query)
    if not tokens or not preferences:
        return []
    resources = load_resources(config)
    manuals = set(_matching_manual_names(resources, preferences))
    scores = resources.bm25.get_scores(tokens)
    ranked = sorted(
        (
            index
            for index, segment_id in enumerate(resources.bm25_ids)
            if str(resources.segments_by_id[segment_id]["metadata"].get("manual", "")) in manuals
            and float(scores[index]) > 0
        ),
        key=scores.__getitem__,
        reverse=True,
    )[:PREFERRED_MANUAL_QUOTA]
    return [
        _result(
            resources.segments_by_id[resources.bm25_ids[index]],
            {"preferred_bm25_rank": float(rank), "preferred_bm25_score": float(scores[index])},
        )
        for rank, index in enumerate(ranked, start=1)
    ]


def _matching_manual_names(
    resources: RetrievalResources, preferences: Sequence[str]
) -> list[str]:
    return [
        manual
        for manual in resources.manual_names
        if any(preference.strip().casefold() in manual.casefold() for preference in preferences if preference.strip())
    ]


def _sections(results: Sequence[RetrievedDocument]) -> list[str]:
    return list(
        dict.fromkeys(
            str(result["metadata"].get("section", ""))
            for result in results
            if result["metadata"].get("section")
        )
    )


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
    for key in (
        "aspects",
        "matched_representation_types",
        "selected_candidate_ids",
    ):
        values = list(
            dict.fromkeys(
                [
                    *target["metadata"].get(key, []),
                    *source["metadata"].get(key, []),
                ]
            )
        )
        if values:
            target["metadata"][key] = values
    for key in ("selected_segment_id", "selected_representation_id"):
        if key not in target["metadata"] and source["metadata"].get(key):
            target["metadata"][key] = source["metadata"][key]


def _normalized_text(text: str) -> str:
    return " ".join(_bm25_tokens(text))


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
    return min(0.0009, matches * 0.0003)


def _heading_bonus(result: RetrievedDocument, query: str) -> float:
    del result, query
    return 0.0


def _manual_bonus(result: RetrievedDocument, preferred_manuals: Sequence[str]) -> float:
    metadata = result["metadata"]
    manual = f"{metadata.get('manual', '')} {metadata.get('source_file', '')}".casefold()
    return (
        0.0004
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
    seeds = documents[: max(1, limit // 2)]
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
    return limit_candidates_per_evidence(_deduplicate_by_id(expanded))[:limit]


def rerank_documents(
    query: str,
    documents: list[RetrievedDocument],
    *,
    intent: str = "",
    limit: int | None = None,
    config: Settings = settings,
) -> list[RetrievedDocument]:
    """Cross-encode expanded RetrievalSegments, with an RRF-order fallback."""
    candidates = prepare_rerank_candidates(documents, limit=20)
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
    return candidates[: limit or config.rerank_top_k]


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
        evidence["metadata"]["selected_candidate_ids"] = [document["chunk_id"]]
        representation_type = document["metadata"].get("representation_type")
        evidence["metadata"]["matched_representation_types"] = (
            [str(representation_type)] if representation_type else []
        )
        if document["metadata"].get("chunk_level") == "representation":
            evidence["metadata"]["selected_representation_id"] = document["chunk_id"]
        else:
            evidence["metadata"]["selected_segment_id"] = document["chunk_id"]
        for key in ("matched_heading", "matched_concept"):
            if document["metadata"].get(key):
                evidence["metadata"][key] = document["metadata"][key]
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
    return candidates[: limit or config.rerank_top_k]
