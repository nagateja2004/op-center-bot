"""LangGraph nodes for the evidence-gated Opcenter RAG workflow."""

from __future__ import annotations

from functools import lru_cache
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import settings
from src.llm import GroqRequestError, call_llm, call_structured
from src.prompts import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_VERIFICATION_PROMPT,
    DIAGRAM_GENERATION_PROMPT,
    EVIDENCE_GRADING_PROMPT,
    QUERY_BROADENING_PROMPT,
    QUERY_PLANNING_PROMPT,
    SUPPORTED_OUTPUT_TYPES,
)
from src.retrieval import (
    deduplicate_results,
    expand_context as expand_retrieval_context,
    rerank_documents as cross_encoder_rerank,
    resolve_evidence_units,
    retrieve_multiple_queries,
)
from src.schemas import (
    EvidenceGrade,
    QueryPlan,
    RAGState,
    RetrievedDocument,
    SourceInfo,
)


DIAGRAM_RE = re.compile(r"\b(?:hierarch\w*|process|workflow|architecture|relationship)\b", re.I)
OPCENTER_TERMS = {
    "opcenter",
    "factory",
    "resource",
    "container",
    "modeling",
    "shop floor",
    "portal studio",
    "cdo",
    "clf",
    "sampling",
    "lot",
    "manufacturing",
    "workflow",
}
CITATION_RE = re.compile(r"\[S(\d+)\]")
GROUPED_CITATION_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)+)\]", re.I)
MAX_EVIDENCE_SOURCES = 8
GRADER_EXCERPT_CHARS = 600
ANSWER_EXCERPT_CHARS = 500
TRUNCATION_MARKER = "[Evidence truncated at a line boundary.]"
TEMPORARY_LLM_MESSAGE = (
    "The answer service is temporarily unavailable or has reached its request quota. "
    "Please try again shortly."
)
SAMPLING_REQUIRED_ASPECTS = [
    "Current Spec",
    "Sampling Plan",
    "Sample Tests",
    "sampling status",
    "failure movement rule",
    "Move transaction",
]
SAMPLING_ALLOWED_ENTITIES = [
    "Current Spec",
    "Sampling Plan",
    "Sample Tests",
    "Container",
    "Sampling Status",
    "Failure Movement Rule",
    "Move Transaction",
    "Next Workflow Step",
    "Movement Blocked",
]
SAMPLING_REJECTED_NODES = {
    "shop", "floor", "modeling", "using", "explanation", "procedure", "checks"
}


def understand_question(state: RAGState) -> RAGState:
    messages = list(state.get("messages", []))[-6:]
    question = _latest_user_text(messages)
    prompt = QUERY_PLANNING_PROMPT.format(
        question=question,
        conversation=_format_messages(_relevant_messages(messages, question)),
        manual_names=" | ".join(_available_manual_names()) or "none",
        supported_output_types=SUPPORTED_OUTPUT_TYPES,
    )
    try:
        plan = call_structured(
            _trim_to_token_budget(prompt, settings.planner_input_token_budget),
            QueryPlan,
            task="planner",
        )
    except GroqRequestError:
        plan = _deterministic_plan(question)
    standalone = plan.standalone_question.strip() or question
    required_output = _required_output(standalone, plan.required_output, plan.intent)
    aspects = _required_aspects(standalone, plan.required_aspects)
    required_manuals = _required_manuals(standalone)
    entities = list(plan.entities)
    if _is_sampling_movement_question(standalone):
        entities = _unique_text([*entities, *SAMPLING_ALLOWED_ENTITIES])
    aspect_queries = {
        aspect: _aspect_queries(
            aspect,
            [],
            plan.entities,
        )
        for aspect in aspects
    }
    queries = _ensure_queries(standalone, plan.search_queries, plan.entities)
    return {
        "standalone_question": standalone,
        "intent": plan.intent,
        "complexity": "multi_aspect" if len(aspects) > 1 else "single_topic",
        "required_aspects": aspects,
        "aspect_queries": aspect_queries,
        "required_output": required_output,
        "entities": entities,
        "search_queries": queries,
        "manual_filters": {"manuals": plan.preferred_manuals},
        "required_manuals": required_manuals,
        "needs_diagram": (
            plan.needs_diagram
            or "diagram" in required_output
            or bool(DIAGRAM_RE.search(standalone))
        ),
        "retry_count": state.get("retry_count", 0),
    }


def retrieve_documents(state: RAGState) -> RAGState:
    aspects = state.get("required_aspects") or [state["standalone_question"]]
    missing = set(state.get("missing_aspects", []))
    retrying = state.get("retry_count", 0) > 0 and bool(missing)
    aspect_documents = {
        aspect: [_copy_with_aspect(document, aspect) for document in documents]
        for aspect, documents in state.get("aspect_documents", {}).items()
        if retrying and aspect not in missing
    }
    for aspect in aspects:
        if retrying and aspect not in missing:
            continue
        documents = retrieve_multiple_queries(
            aspect,
            state.get("aspect_queries", {}).get(aspect, [aspect]),
            entities=state.get("entities", []),
            preferred_manuals=_manual_preferences(state.get("manual_filters", {})),
            intent=f"{state.get('intent', '')} {aspect}",
        )
        aspect_documents[aspect] = [
            _copy_with_aspect(document, aspect) for document in documents
        ]
    return {
        "aspect_documents": aspect_documents,
        "retrieved_docs": _merge_aspect_documents(aspect_documents, limit=60),
    }


def expand_context(state: RAGState) -> RAGState:
    aspect_documents = {
        aspect: [
            _copy_with_aspect(document, aspect)
            for document in expand_retrieval_context(
                documents,
                intent=f"{state.get('intent', '')} {aspect}",
                limit=20,
            )
        ]
        for aspect, documents in state.get("aspect_documents", {}).items()
    }
    return {
        "aspect_documents": aspect_documents,
        "expanded_docs": _merge_aspect_documents(aspect_documents, limit=60),
    }


def rerank_documents(state: RAGState) -> RAGState:
    aspect_documents = {
        aspect: [
            _copy_with_aspect(document, aspect)
            for document in resolve_evidence_units(
                cross_encoder_rerank(
                    aspect,
                    documents,
                    intent=f"{state.get('intent', '')} {aspect}",
                    limit=6,
                ),
                limit=3,
            )
        ]
        for aspect, documents in state.get("aspect_documents", {}).items()
    }
    documents = _merge_aspect_documents(aspect_documents, limit=10)
    return {"aspect_documents": aspect_documents, "reranked_docs": documents}


def grade_evidence(state: RAGState) -> RAGState:
    coverage: dict[str, str] = {}
    reasons: list[str] = []
    missing_concepts: list[str] = []
    aspects = state.get("required_aspects") or [state["standalone_question"]]
    for aspect in aspects:
        documents = state.get("aspect_documents", {}).get(aspect, [])
        if not documents and len(aspects) == 1:
            documents = state.get("reranked_docs", [])
        if not documents:
            status = (
                "retry"
                if state.get("retry_count", 0) < settings.max_retries
                else "in_scope_insufficient"
            )
            coverage[aspect] = status
            reasons.append(f"{aspect}: no manual evidence was retrieved")
            continue
        try:
            grade = call_structured(
                _trim_to_token_budget(
                    EVIDENCE_GRADING_PROMPT.format(
                        standalone_question=state["standalone_question"],
                        required_aspects=" | ".join(aspects),
                        aspect=aspect,
                        evidence=_format_grader_summaries(documents, aspect),
                    ),
                    settings.grader_input_token_budget,
                ),
                EvidenceGrade,
                task="grader",
                evidence_count=min(2, len(documents)),
            )
        except GroqRequestError:
            grade = _heuristic_grade(state, aspect, documents)
        status = grade.status
        if status == "retry" and state.get("retry_count", 0) >= settings.max_retries:
            status = "in_scope_insufficient"
        if status == "out_of_scope" and _looks_in_scope(
            state["standalone_question"], state.get("entities", [])
        ):
            status = "in_scope_insufficient"
        coverage[aspect] = status
        reasons.append(f"{aspect}: {grade.reason}")
        missing_concepts.extend(grade.missing_concepts)
    missing_aspects = [
        aspect for aspect, status in coverage.items() if status != "sufficient"
    ]
    statuses = set(coverage.values())
    if statuses == {"out_of_scope"}:
        overall = "out_of_scope"
    elif "retry" in statuses and state.get("retry_count", 0) < settings.max_retries:
        overall = "retry"
    elif "sufficient" in statuses or "partial" in statuses:
        overall = "partial" if missing_aspects else "sufficient"
    else:
        overall = "in_scope_insufficient"
    required_manuals = state.get("required_manuals", []) or _required_manuals(
        state["standalone_question"]
    )
    manual_coverage = _manual_coverage(state, required_manuals, coverage)
    missing_manuals = [manual for manual, supported in manual_coverage.items() if not supported]
    if missing_manuals:
        missing_aspects.extend(f"{manual} manual evidence" for manual in missing_manuals)
        reasons.append(f"missing required manual evidence: {', '.join(missing_manuals)}")
        if state.get("retry_count", 0) < settings.max_retries:
            overall = "retry"
        elif any(status in {"sufficient", "partial"} for status in coverage.values()):
            overall = "partial"
    return {
        "evidence_status": overall,
        "evidence_reason": "; ".join(reasons),
        "missing_concepts": list(dict.fromkeys(missing_concepts)),
        "coverage": coverage,
        "missing_aspects": missing_aspects,
        "manual_coverage": manual_coverage,
    }


def broaden_query(state: RAGState) -> RAGState:
    retry_count = state.get("retry_count", 0)
    if retry_count >= settings.max_retries:
        return {
            "evidence_status": "in_scope_insufficient",
            "evidence_reason": "The single broader retrieval attempt was already used.",
            "manual_filters": {},
        }
    sections = _section_names(
        state.get("reranked_docs", []) or state.get("expanded_docs", [])
    )
    retry_aspects = [
        aspect
        for aspect, status in state.get("coverage", {}).items()
        if status == "retry"
    ] or state.get("missing_aspects", [])
    prompt = _trim_to_token_budget(
        QUERY_BROADENING_PROMPT.format(
            standalone_question=state["standalone_question"],
            missing_aspects=" | ".join(retry_aspects) or "none",
            previous_queries=state.get("aspect_queries", {}),
            section_names=" | ".join(sections[:4]) or "none",
        ),
        settings.query_broadening_input_token_budget,
    )
    try:
        broadened = _plain_lines(
            _message_content(call_llm(prompt, task="query_broadening"))
        )
    except GroqRequestError:
        broadened = []
    aspect_queries = dict(state.get("aspect_queries", {}))
    for aspect in retry_aspects:
        additions = [
            *broadened,
            f"{aspect} Opcenter related sections",
            *(f"{aspect} {section}" for section in sections[:2]),
        ]
        aspect_queries[aspect] = _aspect_queries(
            aspect, additions, state.get("entities", [])
        )
    return {
        "aspect_queries": aspect_queries,
        "search_queries": _ensure_queries(
            state["standalone_question"],
            [query for queries in aspect_queries.values() for query in queries],
            state.get("entities", []),
        ),
        "manual_filters": {},
        "missing_aspects": retry_aspects,
        "retry_count": retry_count + 1,
    }


def generate_answer(state: RAGState) -> RAGState:
    documents = state.get("reranked_docs", [])
    if state.get("evidence_status") not in {"sufficient", "partial"} or not documents:
        return generate_fallback(state)
    coverage = state.get("coverage", {})
    supported_aspects = [
        aspect for aspect, status in coverage.items() if status in {"sufficient", "partial"}
    ] or state.get("required_aspects", [])
    required_manuals = state.get("required_manuals", []) or _required_manuals(
        state["standalone_question"]
    )
    documents = _select_answer_documents(
        documents, supported_aspects, required_manuals=required_manuals
    )
    required_output = state.get("required_output", [])
    evidence = _format_answer_evidence(
        documents,
        state["standalone_question"],
        required_output,
        max_chars=max(1_000, settings.answer_input_token_budget * 4 - 1_400),
    )
    prompt = _trim_to_token_budget(
        ANSWER_GENERATION_PROMPT.format(
            standalone_question=state["standalone_question"],
            required_output=", ".join(
                _answer_output_labels(required_output, documents, required_manuals)
            ),
            supported_aspects=" | ".join(supported_aspects) or "none",
            missing_aspects=" | ".join(state.get("missing_aspects", [])) or "none",
            evidence=evidence,
            answer_structure="\n".join(
                f"- {heading}" for heading in _answer_structure(state["standalone_question"], required_output)
            ),
        ),
        settings.answer_input_token_budget,
    )
    try:
        answer = _remove_invalid_citations(
            _normalize_citations(
                _message_content(
                    call_llm(prompt, task="answer", evidence_count=len(documents))
                ).strip()
            ),
            len(documents),
        )
    except GroqRequestError:
        return {
            "answer": TEMPORARY_LLM_MESSAGE,
            "sources": [],
            "reranked_docs": documents,
            "grounded": False,
            "llm_error_role": "answer",
        }
    answer = _ensure_release_warning(answer, documents)
    return {
        "answer": answer,
        "sources": _cited_sources(answer, documents),
        "reranked_docs": documents,
    }


def verify_answer(state: RAGState) -> RAGState:
    documents = state.get("reranked_docs", [])
    draft = state.get("answer", "")
    if state.get("llm_error_role") == "answer":
        return {
            "answer": draft or TEMPORARY_LLM_MESSAGE,
            "sources": [],
            "grounded": False,
            "diagram_supported": False,
            "messages": [AIMessage(content=draft or TEMPORARY_LLM_MESSAGE)],
        }
    cited_numbers = _citation_numbers(draft, len(documents))
    cited_evidence = _format_cited_evidence(
        documents,
        cited_numbers,
        state.get("standalone_question", ""),
        max_chars=max(500, settings.verifier_input_token_budget * 4 - len(draft) - 900),
    )
    prompt = _trim_to_token_budget(
        ANSWER_VERIFICATION_PROMPT.format(
            required_aspects=" | ".join(state.get("required_aspects", [])),
            answer_structure=" | ".join(
                _answer_structure(
                    state.get("standalone_question", ""), state.get("required_output", [])
                )
            ),
            answer=draft,
            evidence=cited_evidence,
        ),
        settings.verifier_input_token_budget,
    )
    try:
        corrected = _normalize_citations(
            _message_content(
                call_llm(prompt, task="verifier", evidence_count=len(cited_numbers))
            ).strip()
        )
    except GroqRequestError:
        corrected = _normalize_citations(draft)
    invalid_citations = any(
        not 1 <= int(number) <= len(documents)
        for number in CITATION_RE.findall(corrected)
    )
    answer = _remove_invalid_citations(corrected, len(documents)).strip()
    sources = _cited_sources(answer, documents)
    answer = _sanitize_cross_manual_label(answer, sources)
    citation_ids = set(CITATION_RE.findall(answer))
    source_ids = {source.source_id[1:] for source in sources}
    grounded = (
        not invalid_citations
        and citation_ids == source_ids
        and bool(citation_ids)
        and bool(answer)
    )
    return {
        "answer": answer,
        "sources": sources,
        "grounded": grounded,
        "unsupported_claims": [],
        "missing_aspects": state.get("missing_aspects", []),
        "diagram_supported": grounded and state.get("needs_diagram", False),
        "messages": [AIMessage(content=answer)],
    }


def generate_diagram(state: RAGState) -> RAGState:
    question = state.get("standalone_question", "")
    documents = state.get("reranked_docs", [])
    if not (
        state.get("allow_diagrams", True)
        and state.get("needs_diagram")
        and state.get("grounded")
        and documents
    ):
        return {"diagram_dot": None}
    verified_answer = state.get("answer", "")
    cited_numbers = _citation_numbers(verified_answer, len(documents))
    source_ids = [f"S{number}" for number in cited_numbers]
    cited_documents = [documents[number - 1] for number in cited_numbers]
    support_text = " ".join(
        f"{document['metadata'].get('section', '')} {document['text']}"
        for document in cited_documents
    )
    diagram_type = _diagram_type(question)
    entities = _verified_entities(
        state.get("entities", []), verified_answer, question, support_text
    )
    relationships = _verified_relationships(verified_answer)
    sampling_decision = _is_sampling_movement_question(question)
    if sampling_decision:
        relationships = [relationship[:180] for relationship in relationships[:5]]
    decisions = _verified_decisions(relationships)
    outcomes = _verified_outcomes(question, verified_answer, support_text)
    if sampling_decision and not {"Sampling Plan", "Sample Tests"}.issubset(entities):
        return {"diagram_dot": None}
    prompt = _trim_to_token_budget(
        DIAGRAM_GENERATION_PROMPT.format(
            diagram_type=diagram_type,
            entities=" | ".join(entities),
            relationships="\n".join(relationships),
            decisions="\n".join(decisions) or "none",
            outcomes=" | ".join(outcomes) or "none",
            source_ids=" | ".join(source_ids),
            diagram_rules=_diagram_rules(sampling_decision),
        ),
        settings.diagram_input_token_budget,
    )
    direction = "TB" if re.search(r"\bhierarch\w*\b", question, re.I) else "LR"
    try:
        dot = _message_content(
            call_llm(prompt, task="diagram", evidence_count=len(source_ids))
        )
    except GroqRequestError:
        return {"diagram_dot": None}
    diagram = _validated_dot(
        dot,
        direction,
        allowed_entities=set(entities) if sampling_decision else None,
        required_entities={
            "Sampling Plan", "Sample Tests", "Sampling Status", "Failure Movement Rule"
        } if sampling_decision else set(),
        source_ids=set(source_ids),
        decision_diagram=sampling_decision,
    )
    return {"diagram_dot": diagram}


def generate_fallback(state: RAGState) -> RAGState:
    if state.get("evidence_status") == "out_of_scope":
        answer = (
            "This assistant answers questions about the supplied Opcenter manuals. "
            "Your question appears unrelated to those manuals."
        )
    else:
        answer = (
            "This is an Opcenter-related question, but the supplied manuals do not "
            "provide enough evidence to answer it reliably."
        )
    return {
        "answer": answer,
        "sources": [],
        "grounded": False,
        "unsupported_claims": [],
        "diagram_dot": None,
        "messages": [AIMessage(content=answer)],
    }


def _latest_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) or _message_role(message) in {"user", "human"}:
            return _message_content(message)
    return _message_content(messages[-1]) if messages else ""


def _format_messages(messages: list[Any]) -> str:
    return "\n".join(
        f"{_message_role(message)}: {_message_content(message)}" for message in messages
    )


def _deterministic_plan(question: str) -> QueryPlan:
    lowered = question.casefold()
    if re.search(r"\b(?:how do|how to|steps?|procedure)\b", lowered):
        intent = "procedure"
    elif re.search(r"\b(?:compare|difference|versus|vs\.?\b)", lowered):
        intent = "comparison"
    elif re.search(r"\b(?:cannot|failed?|issue|problem|stuck)\b", lowered):
        intent = "troubleshooting"
    else:
        intent = "explanation"
    entities = _unique_text(
        [
            *re.findall(r"\b(?:[A-Z]{2,}|[A-Z][A-Za-z]+)\b", question),
            *(term.upper() if term in {"cdo", "clf"} else term.title() for term in OPCENTER_TERMS if term in lowered),
        ]
    )[:8]
    outputs = _required_output(question, [], intent)
    return QueryPlan(
        standalone_question=question,
        intent=intent,
        required_aspects=[question],
        required_output=outputs,
        entities=entities,
        search_queries=[question],
        needs_diagram="diagram" in outputs,
    )


def _heuristic_grade(
    state: RAGState,
    aspect: str,
    documents: list[RetrievedDocument],
) -> EvidenceGrade:
    if not _looks_in_scope(state["standalone_question"], state.get("entities", [])):
        return EvidenceGrade(status="out_of_scope", reason="The request is unrelated to Opcenter.")
    terms = set(_content_terms(f"{state['standalone_question']} {aspect}"))
    evidence_text = " ".join(
        f"{document['metadata'].get('section', '')} {document['text'][:600]}"
        for document in documents[:2]
    )
    overlap = terms & set(_content_terms(evidence_text))
    if len(overlap) >= min(2, max(1, len(terms))):
        return EvidenceGrade(status="sufficient", reason="Retrieved evidence matches the assigned aspect.")
    if overlap:
        return EvidenceGrade(status="partial", reason="Retrieved evidence partially matches the assigned aspect.")
    status = "retry" if state.get("retry_count", 0) < settings.max_retries else "in_scope_insufficient"
    return EvidenceGrade(status=status, reason="Retrieved evidence does not match the assigned aspect closely enough.")


def _relevant_messages(messages: list[Any], question: str) -> list[Any]:
    """Keep up to six recent history messages, preferring lexical relevance."""
    history = list(messages)
    if history and _message_content(history[-1]).strip() == question.strip():
        history.pop()
    history = [message for message in history[-12:] if _message_content(message).strip()]
    if len(history) <= 6:
        return history
    terms = set(_content_terms(question))
    ranked = sorted(
        enumerate(history),
        key=lambda item: (
            len(terms & set(_content_terms(_message_content(item[1])))),
            item[0],
        ),
        reverse=True,
    )
    chosen = {index for index, _ in ranked[:6]}
    return [message for index, message in enumerate(history) if index in chosen]


@lru_cache(maxsize=1)
def _available_manual_names() -> tuple[str, ...]:
    try:
        manifest = json.loads(
            (settings.indexes_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return ()
    names = [
        str(entry.get("manual", "")).strip()
        for entry in manifest.get("manuals", {}).values()
        if isinstance(entry, dict)
    ]
    return tuple(dict.fromkeys(name for name in names if name))


def _estimated_tokens(text: str) -> int:
    """Use one deterministic conservative approximation across Groq tokenizers."""
    return (len(text) + 3) // 4


def _trim_to_token_budget(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        raise ValueError("Input-token budget must be positive")
    max_chars = token_budget * 4
    return text if len(text) <= max_chars else _truncate_evidence(text, max_chars)


def _message_role(message: Any) -> str:
    if isinstance(message, BaseMessage):
        return message.type
    return str(message.get("role", "unknown")) if isinstance(message, dict) else "unknown"


def _message_content(message: Any) -> str:
    if isinstance(message, BaseMessage):
        return str(message.content)
    return str(message.get("content", "")) if isinstance(message, dict) else str(message)


def _ensure_queries(
    standalone: str, queries: list[str], entities: list[str]
) -> list[str]:
    candidates = [standalone, *queries]
    candidates.extend(f"Opcenter {entity} related sections" for entity in entities[:2])
    candidates.append(f"{standalone} Opcenter manual")
    unique: list[str] = []
    seen: set[str] = set()
    for query in candidates:
        cleaned = " ".join(query.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
        if len(unique) == settings.max_search_queries:
            break
    return unique


def _required_aspects(question: str, aspects: list[str]) -> list[str]:
    if _is_sampling_movement_question(question):
        return list(SAMPLING_REQUIRED_ASPECTS)
    unique = list(
        dict.fromkeys(
            cleaned
            for aspect in aspects
            if (cleaned := " ".join(aspect.split()))
        )
    )[:6]
    return unique or [question]


def _aspect_queries(aspect: str, queries: list[str], entities: list[str]) -> list[str]:
    candidates = [aspect, *queries]
    candidates.extend(f"{aspect} {entity}" for entity in entities[:2])
    return _unique_text(candidates)[:3]


def _required_output(question: str, outputs: list[str], intent: str) -> list[str]:
    text = f"{question} {intent}".casefold()
    required = list(outputs)
    if not required:
        required.append("explanation")
    if re.search(r"\b(?:cannot|can't|fails?|problem|issue|stuck|troubleshoot)\b", text):
        required.extend(["likely_reasons", "checks"])
    if re.search(r"\b(?:compare|comparison|difference|versus|vs\.? )\b", text):
        required.append("comparison_table")
    if re.search(r"\b(?:steps?|procedure|how do|how to)\b", text):
        required.append("procedure")
    if DIAGRAM_RE.search(text) or re.search(r"\bmodeled\b.*\bbecomes? usable\b", text):
        required.append("diagram")
    if re.search(r"\b(?:cross[- ]manual|modeling and shop floor|across manuals)\b", text):
        required.append("cross_manual_synthesis")
    if _is_sampling_movement_question(question):
        required = [item for item in required if item != "procedure"]
        required.extend(["explanation", "likely_reasons", "checks", "diagram"])
    return list(dict.fromkeys(required))


def _is_sampling_movement_question(question: str) -> bool:
    text = question.casefold()
    return "sampling" in text and "spec" in text and any(
        term in text for term in ("move transaction", "movement", "next workflow")
    )


def _required_manuals(question: str) -> list[str]:
    text = question.casefold()
    return ["Modeling", "Shop Floor"] if "modeling" in text and "shop floor" in text else []


def _manual_family(manual: str) -> str:
    lowered = manual.casefold()
    if "modeling" in lowered:
        return "Modeling"
    if "shop floor" in lowered:
        return "Shop Floor"
    return manual


def _manual_coverage(
    state: RAGState,
    required_manuals: list[str],
    coverage: dict[str, str] | None = None,
) -> dict[str, bool]:
    aspect_documents = state.get("aspect_documents", {})
    available = {
        _manual_family(str(document["metadata"].get("manual", "")))
        for aspect, documents in aspect_documents.items()
        if coverage is None or coverage.get(aspect) in {"sufficient", "partial"}
        for document in documents
    }
    if not aspect_documents:
        available = {
            _manual_family(str(document["metadata"].get("manual", "")))
            for document in state.get("reranked_docs", [])
        }
    return {manual: manual in available for manual in required_manuals}


def _answer_output_labels(
    required_output: list[str],
    documents: list[RetrievedDocument],
    required_manuals: list[str],
) -> list[str]:
    labels = {
        "explanation": "direct explanation",
        "procedure": "ordered task steps",
        "likely_reasons": "likely reasons movement is blocked",
        "checks": "what to check",
        "comparison_table": "comparison table",
        "diagram": "decision diagram",
        "cross_manual_synthesis": "configuration and runtime relationship across cited manuals",
    }
    present_manuals = {
        _manual_family(str(document["metadata"].get("manual", "")))
        for document in documents
    }
    return [
        labels[item]
        for item in required_output
        if item in labels
        and not (
            item == "cross_manual_synthesis"
            and required_manuals
            and not set(required_manuals).issubset(present_manuals)
        )
    ]


def _answer_structure(question: str, required_output: list[str]) -> list[str]:
    if not (
        _is_sampling_movement_question(question)
        or any(item in required_output for item in ("likely_reasons", "checks"))
    ):
        return ["Direct explanation"]
    headings = [
        "Direct explanation",
        "Configuration relationship",
        "Runtime behavior",
        "Likely reasons movement is blocked",
        "What to check",
    ]
    if "diagram" in required_output:
        headings.append("Decision diagram")
    headings.append("Release note when cited releases differ")
    return headings


def _unique_text(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(value.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def _plain_lines(text: str) -> list[str]:
    return _unique_text(
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
        for line in text.splitlines()
    )[:3]


def _copy_with_aspect(document: RetrievedDocument, aspect: str) -> RetrievedDocument:
    copied: RetrievedDocument = {
        "chunk_id": document["chunk_id"],
        "text": document["text"],
        "content_type": document["content_type"],
        "metadata": dict(document["metadata"]),
        "retrieval_scores": dict(document["retrieval_scores"]),
    }
    copied["metadata"]["aspects"] = list(
        dict.fromkeys([*copied["metadata"].get("aspects", []), aspect])
    )
    return copied


def _merge_aspect_documents(
    aspect_documents: dict[str, list[RetrievedDocument]], *, limit: int
) -> list[RetrievedDocument]:
    """Round-robin aspect evidence so one topic cannot crowd out the others."""
    merged: list[RetrievedDocument] = []
    depth = 0
    while len(merged) < limit:
        added = False
        for aspect, documents in aspect_documents.items():
            if depth >= len(documents):
                continue
            merged.append(_copy_with_aspect(documents[depth], aspect))
            merged = deduplicate_results(merged, intent=aspect)
            added = True
            if len(merged) >= limit:
                break
        if not added:
            break
        depth += 1
    return merged[:limit]


def _manual_preferences(filters: dict[str, str | list[str]]) -> list[str]:
    values: list[str] = []
    for value in filters.values():
        values.extend(value if isinstance(value, list) else [value])
    return [value for value in values if value]


def _content_terms(text: str) -> list[str]:
    return [
        term
        for term in re.findall(r"[a-z0-9_]+", text.casefold())
        if len(term) > 2 and term not in {"and", "the", "with", "from", "that", "this"}
    ]


def _relevant_excerpt(
    text: str, query: str, *, minimum: int = 300, maximum: int = 600
) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= maximum:
        return cleaned
    terms = set(_content_terms(query))
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\s*\|\s*", cleaned)
        if part.strip()
    ]
    ranked = sorted(
        enumerate(parts),
        key=lambda item: (len(terms & set(_content_terms(item[1]))), -item[0]),
        reverse=True,
    )
    chosen: list[tuple[int, str]] = []
    used = 0
    for index, part in ranked:
        if chosen and used >= minimum and not (terms & set(_content_terms(part))):
            continue
        addition = len(part) + bool(chosen)
        if used + addition > maximum:
            continue
        chosen.append((index, part))
        used += addition
        if used >= maximum - 80:
            break
    excerpt = " ".join(part for _, part in sorted(chosen))
    return excerpt or cleaned[:maximum].rsplit(" ", 1)[0]


def _table_excerpt(rows: list[list[Any]], query: str, *, max_rows: int = 6) -> str:
    if not rows:
        return ""
    header = [str(cell) for cell in rows[0]]
    terms = set(_content_terms(query))
    candidates = sorted(
        enumerate(rows[1:]),
        key=lambda item: (
            len(terms & set(_content_terms(" ".join(str(cell) for cell in item[1])))),
            -item[0],
        ),
        reverse=True,
    )[:max_rows]
    selected = [row for _, row in sorted(candidates)]
    width = max([len(header), *(len(row) for row in selected), 1])
    padded_header = header + [""] * (width - len(header))
    lines = ["| " + " | ".join(padded_header) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in selected:
        cells = [str(cell).replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(cells + [""] * (width - len(cells))) + " |")
    return "\n".join(lines)


def _grader_text(document: RetrievedDocument, aspect: str) -> str:
    rows = document["metadata"].get("table_rows")
    source = _table_excerpt(rows, aspect, max_rows=3) if rows else document["text"]
    return _relevant_excerpt(source, aspect, maximum=GRADER_EXCERPT_CHARS)


def _format_grader_summaries(
    documents: list[RetrievedDocument], aspect: str
) -> str:
    summaries: list[str] = []
    for index, document in enumerate(_unique_evidence_documents(documents)[:2], start=1):
        metadata = document["metadata"]
        scores = " ".join(
            f"{name}={value:.4g}"
            for name, value in document["retrieval_scores"].items()
            if name == "final_score"
            or any(key in name for key in ("vector", "bm25", "rrf", "reranker"))
        ) or "unavailable"
        summaries.append(
            "\n".join(
                [
                    f"Source ID: S{index}",
                    f"Evidence ID: {_evidence_id(document)}",
                    f"Manual/release: {metadata.get('manual', 'Manual')} | {metadata.get('release', '')}",
                    f"Section: {metadata.get('section', '')}",
                    f"Content type: {document['content_type']}",
                    f"Assigned aspect: {aspect}",
                    f"Scores: {scores}",
                    f"Excerpt: {_grader_text(document, aspect)}",
                ]
            )
        )
    return "\n\n".join(summaries) or "No evidence summary."


def _evidence_id(document: RetrievedDocument) -> str:
    return str(document["metadata"].get("evidence_id") or document["chunk_id"])


def _unique_evidence_documents(
    documents: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    unique: dict[str, RetrievedDocument] = {}
    for document in documents:
        unique.setdefault(_evidence_id(document), document)
    return list(unique.values())


def _select_answer_documents(
    documents: list[RetrievedDocument],
    supported_aspects: list[str],
    *,
    required_manuals: list[str] | None = None,
) -> list[RetrievedDocument]:
    unique = _unique_evidence_documents(documents)
    selected: list[RetrievedDocument] = []
    for aspect in supported_aspects:
        match = next(
            (
                document
                for document in unique
                if document not in selected
                and aspect in document["metadata"].get("aspects", [])
            ),
            None,
        )
        if match:
            selected.append(match)
    for manual in required_manuals or []:
        match = next(
            (
                document
                for document in unique
                if document not in selected
                and _manual_family(str(document["metadata"].get("manual", ""))) == manual
            ),
            None,
        )
        if match:
            selected.append(match)
    for document in unique:
        if document not in selected:
            selected.append(document)
        if len(selected) == 10:
            break
    return selected[:10]


def _answer_content(
    document: RetrievedDocument, query: str, include_complete_procedure: bool
) -> str:
    rows = document["metadata"].get("table_rows")
    if rows:
        return _table_excerpt(rows, query)
    if include_complete_procedure and document["content_type"] == "procedure":
        return document["text"]
    return _relevant_excerpt(
        document["text"], query, minimum=500, maximum=ANSWER_EXCERPT_CHARS
    )


def _format_answer_evidence(
    documents: list[RetrievedDocument],
    query: str,
    required_output: list[str],
    *,
    max_chars: int,
) -> str:
    complete_procedures = "procedure" in required_output
    entries: list[tuple[str, str, bool]] = []
    procedure_count = 0
    for index, document in enumerate(documents[:10], start=1):
        metadata = document["metadata"]
        protected = (
            complete_procedures
            and document["content_type"] == "procedure"
            and procedure_count < 2
        )
        procedure_count += int(protected)
        header = (
            f"[S{index}] id={_evidence_id(document)} | "
            f"manual={metadata.get('manual', 'Manual')} | release={metadata.get('release', '')} | "
            f"section={metadata.get('section', '')} | type={document['content_type']}"
        )
        entries.append((header, _answer_content(document, query, protected), protected))
    if not entries:
        return "No manual evidence retrieved."
    header_chars = sum(len(header) + 2 for header, _, _ in entries)
    available = max(120 * len(entries), max_chars - header_chars)
    protected_entries = [entry for entry in entries if entry[2]]
    flexible = sum(not protected for _, _, protected in entries)
    protected_room = max(120 * len(protected_entries), available - 120 * flexible)
    protected_total = sum(len(content) for _, content, _ in protected_entries)
    protected_limit = (
        None
        if protected_total <= protected_room
        else max(120, protected_room // max(1, len(protected_entries)))
    )
    protected_chars = sum(
        min(len(content), protected_limit or len(content))
        for _, content, _ in protected_entries
    )
    remaining = max(120 * flexible, available - protected_chars)
    per_flexible = max(120, remaining // max(1, flexible))
    rendered = [
        f"{header}\n{_truncate_evidence(content, protected_limit) if protected_limit else content}"
        if protected
        else f"{header}\n{_truncate_evidence(content, per_flexible)}"
        for header, content, protected in entries
    ]
    return "\n\n".join(rendered)


def _format_cited_evidence(
    documents: list[RetrievedDocument],
    cited_numbers: list[int],
    query: str,
    *,
    max_chars: int,
) -> str:
    if not cited_numbers:
        return "No cited EvidenceUnits."
    per_source = max(160, max_chars // len(cited_numbers) - 160)
    entries: list[str] = []
    for number in cited_numbers:
        document = documents[number - 1]
        metadata = document["metadata"]
        content = _answer_content(document, query, False)
        entries.append(
            f"[S{number}] evidence_id={_evidence_id(document)} | manual={metadata.get('manual', 'Manual')} | "
            f"section={metadata.get('section', '')} | type={document['content_type']}\n"
            f"{_truncate_evidence(content, per_source)}"
        )
    return _truncate_evidence("\n\n".join(entries), max_chars)


def _verified_relationships(answer: str) -> list[str]:
    return [
        _truncate_evidence(" ".join(part.split()), 280)
        for part in re.split(r"(?<=[.!?])\s+|\n+", answer)
        if CITATION_RE.search(part)
    ][:8]


def _verified_entities(
    entities: list[str],
    answer: str,
    question: str = "",
    support_text: str = "",
) -> list[str]:
    if _is_sampling_movement_question(question):
        verified_text = f"{answer} {support_text}".casefold()
        return [
            entity for entity in SAMPLING_ALLOWED_ENTITIES
            if entity.casefold() in verified_text
        ]
    extracted = re.findall(
        r"\b(?:[A-Z]{2,}|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b", answer
    )
    return _unique_text([*entities, *extracted])[:10]


def _verified_decisions(relationships: list[str]) -> list[str]:
    return [
        relationship
        for relationship in relationships
        if re.search(r"\b(?:if|whether|status|pass|fail|in process|blocked|rule)\b", relationship, re.I)
    ][:6]


def _verified_outcomes(question: str, answer: str, support_text: str) -> list[str]:
    combined = f"{answer} {support_text}".casefold()
    if _is_sampling_movement_question(question):
        return [
            outcome
            for outcome in ("Next Workflow Step", "Movement Blocked")
            if outcome.casefold() in combined
        ]
    return []


def _diagram_rules(sampling_decision: bool) -> str:
    if not sampling_decision:
        return "Use boxes for objects/actions and diamonds only for verified conditions."
    return (
        "Use diamonds for Sampling Status and Failure Movement Rule; boxes for objects, actions, and outcomes. "
        "Decision edges use Yes, No, Pass, Fail, or In Process. Start Container -> Current Spec. "
        "Only Move Transaction -> Next Workflow Step advances the container. Failures end at Movement Blocked."
    )


def _diagram_type(question: str) -> str:
    if _is_sampling_movement_question(question):
        return "decision"
    if re.search(r"\bhierarch\w*\b", question, re.I):
        return "hierarchy"
    if re.search(r"\b(?:process|workflow|configuration|runtime)\b", question, re.I):
        return "process"
    return "relationship"


def _format_evidence(documents: list[RetrievedDocument]) -> str:
    """Compatibility formatter; runtime nodes use role-specific evidence views."""
    return _format_answer_evidence(
        documents[:MAX_EVIDENCE_SOURCES], "", [], max_chars=12_000
    )


def _truncate_evidence(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    available = limit - len(TRUNCATION_MARKER) - 1
    lines: list[str] = []
    used = 0
    for line in text.splitlines():
        needed = len(line) + (1 if lines else 0)
        if used + needed > available:
            break
        lines.append(line)
        used += needed
    if not lines:
        prefix = text[:available].rsplit(" ", 1)[0].rstrip()
        lines = [prefix] if prefix else []
    truncated = "\n".join(lines)
    return f"{truncated}\n{TRUNCATION_MARKER}".strip()


def _section_names(documents: list[RetrievedDocument]) -> list[str]:
    return list(
        dict.fromkeys(
            str(document["metadata"].get("section", ""))
            for document in documents
            if document["metadata"].get("section")
        )
    )


def _looks_in_scope(question: str, entities: list[str]) -> bool:
    text = f"{question} {' '.join(entities)}".casefold()
    return any(term in text for term in OPCENTER_TERMS)


def _remove_invalid_citations(answer: str, evidence_count: int) -> str:
    return CITATION_RE.sub(
        lambda match: match.group(0) if 1 <= int(match.group(1)) <= evidence_count else "",
        answer,
    )


def _normalize_citations(answer: str) -> str:
    """Convert grouped citations like [S1, S3] into independently valid IDs."""
    return GROUPED_CITATION_RE.sub(
        lambda match: " ".join(
            f"[{source_id.upper()}]"
            for source_id in re.findall(r"S\d+", match.group(1), re.I)
        ),
        answer,
    )


def _citation_numbers(answer: str, evidence_count: int) -> list[int]:
    return list(
        dict.fromkeys(
            number
            for number in (int(match) for match in CITATION_RE.findall(answer))
            if 1 <= number <= evidence_count
        )
    )


def _cited_sources(answer: str, documents: list[RetrievedDocument]) -> list[SourceInfo]:
    sources: list[SourceInfo] = []
    for number in _citation_numbers(answer, len(documents)):
        metadata = documents[number - 1]["metadata"]
        pdf_page = _safe_pdf_page(metadata)
        if pdf_page is None:
            continue
        printed_page = _safe_printed_page(metadata)
        manual = str(metadata.get("manual", metadata.get("source_file", "Manual")))
        sources.append(
            SourceInfo(
                source_id=f"S{number}",
                source=manual,
                page=printed_page,
                manual=manual,
                source_file=metadata.get("source_file"),
                chapter=metadata.get("chapter"),
                section=metadata.get("section"),
                release=metadata.get("release"),
                printed_page=printed_page,
                pdf_page=pdf_page,
                content_type=documents[number - 1]["content_type"],
            )
        )
    return sources


def _safe_printed_page(metadata: dict[str, Any]) -> str:
    page = str(metadata.get("printed_page") or "").strip()
    return page if page and not page.endswith("-") else "Not listed"


def _safe_pdf_page(metadata: dict[str, Any]) -> int | None:
    try:
        page = int(metadata.get("pdf_page"))
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def _ensure_release_warning(
    answer: str, documents: list[RetrievedDocument]
) -> str:
    releases: dict[str, int] = {}
    for index, document in enumerate(documents, start=1):
        release = str(document["metadata"].get("release") or "").strip()
        if release and release not in releases:
            releases[release] = index
    if len(releases) < 2 or "release warning" in answer.casefold():
        return answer
    labels = ", ".join(releases)
    citations = " ".join(f"[S{index}]" for index in releases.values())
    return f"{answer.rstrip()}\n\n**Release warning:** The cited evidence spans {labels}. {citations}"


def _sanitize_cross_manual_label(answer: str, sources: list[SourceInfo]) -> str:
    manuals = {_manual_family(source.manual or source.source) for source in sources}
    if len(manuals) >= 2:
        return answer
    answer = re.sub(r"cross[- ]manual(?:\s+(?:synthesis|view))?", "supported evidence", answer, flags=re.I)
    return re.sub(r"across\s+(?:the\s+)?(?:cited\s+)?manuals", "from the cited manual", answer, flags=re.I)


def _validated_dot(
    dot: str,
    direction: str,
    *,
    allowed_entities: set[str] | None = None,
    required_entities: set[str] | None = None,
    source_ids: set[str] | None = None,
    decision_diagram: bool = False,
) -> str | None:
    dot = re.sub(r"^```(?:dot|graphviz)?\s*|\s*```$", "", dot.strip(), flags=re.I)
    if (
        not re.match(r"^digraph\b", dot, re.I)
        or not _balanced_delimiters(dot)
        or re.search(r"\b(?:image|shapefile|href|url)\s*=", dot, re.I)
    ):
        return None
    node_names = set(
        re.findall(r'(?m)^\s*("[^"\n]+"|[A-Za-z_][\w.-]*)\s*\[', dot)
    )
    node_names.update(
        re.findall(r'(?:->|--)\s*("[^"\n]+"|[A-Za-z_][\w.-]*)', dot)
    )
    node_names -= {"graph", "node", "edge", "digraph"}
    if not 3 <= len(node_names) <= 10:
        return None
    if decision_diagram and not _valid_sampling_decision_dot(
        dot,
        allowed_entities or set(),
        required_entities or set(),
        source_ids or set(),
    ):
        return None
    if re.search(r"rankdir\s*=", dot, re.I):
        dot = re.sub(r"rankdir\s*=\s*[A-Za-z]+", f"rankdir={direction}", dot, count=1, flags=re.I)
    else:
        dot = dot.replace("{", f"{{\n  rankdir={direction};", 1)
    return dot


def _valid_sampling_decision_dot(
    dot: str,
    allowed_entities: set[str],
    required_entities: set[str],
    source_ids: set[str],
) -> bool:
    declarations = re.findall(
        r'(?m)^\s*("[^"\n]+"|[A-Za-z_][\w.-]*)\s*\[(.*?)\]\s*;', dot
    )
    if not declarations or "sampling failed" in dot.casefold():
        return False
    id_to_label: dict[str, str] = {}
    id_to_shape: dict[str, str] = {}
    for node_id, attributes in declarations:
        label_match = re.search(r'label\s*=\s*"([^"]+)"', attributes, re.I)
        if not label_match:
            return False
        raw_label = label_match.group(1)
        citations = {f"S{number}" for number in CITATION_RE.findall(raw_label)}
        if not citations or not citations.issubset(source_ids):
            return False
        label = CITATION_RE.sub("", raw_label).strip().rstrip("?").strip()
        if label.casefold() in SAMPLING_REJECTED_NODES or any(
            word in SAMPLING_REJECTED_NODES for word in _content_terms(label)
        ):
            return False
        if label not in allowed_entities:
            return False
        shape_match = re.search(r"shape\s*=\s*([A-Za-z]+)", attributes, re.I)
        id_to_label[node_id] = label
        id_to_shape[node_id] = shape_match.group(1).casefold() if shape_match else ""
    labels = set(id_to_label.values())
    if not required_entities.issubset(labels) or not {
        "Container", "Current Spec", "Move Transaction", "Next Workflow Step", "Movement Blocked"
    }.issubset(labels):
        return False
    decision_labels = {"Sampling Status", "Failure Movement Rule"} & labels
    if any(id_to_shape[node_id] != "diamond" for node_id, label in id_to_label.items() if label in decision_labels):
        return False
    if any(
        id_to_shape[node_id] not in {"box", "rect", "rectangle"}
        for node_id, label in id_to_label.items()
        if label not in decision_labels
    ):
        return False
    edges = re.findall(
        r'(?m)^\s*("[^"\n]+"|[A-Za-z_][\w.-]*)\s*->\s*("[^"\n]+"|[A-Za-z_][\w.-]*)\s*(?:\[(.*?)\])?\s*;',
        dot,
    )
    if not edges:
        return False
    connected = {node for left, right, _ in edges for node in (left, right)}
    if set(id_to_label) != connected:
        return False
    pairs = {(id_to_label[left], id_to_label[right]) for left, right, _ in edges}
    if ("Container", "Current Spec") not in pairs or ("Move Transaction", "Next Workflow Step") not in pairs:
        return False
    if ("Current Spec", "Next Workflow Step") in pairs:
        return False
    allowed_pairs = {
        ("Container", "Current Spec"),
        ("Current Spec", "Sampling Plan"),
        ("Sampling Plan", "Sample Tests"),
        ("Sample Tests", "Sampling Status"),
        ("Sampling Status", "Move Transaction"),
        ("Sampling Status", "Failure Movement Rule"),
        ("Sampling Status", "Movement Blocked"),
        ("Failure Movement Rule", "Move Transaction"),
        ("Failure Movement Rule", "Movement Blocked"),
        ("Move Transaction", "Next Workflow Step"),
    }
    if not pairs.issubset(allowed_pairs):
        return False
    for left, _, attributes in edges:
        if id_to_label[left] not in decision_labels:
            continue
        label_match = re.search(r'label\s*=\s*"([^"]+)"', attributes, re.I)
        if not label_match or label_match.group(1).casefold() not in {
            "yes", "no", "pass", "fail", "in process"
        }:
            return False
    return True


def _balanced_delimiters(dot: str) -> bool:
    pairs = {"}": "{", "]": "["}
    stack: list[str] = []
    quoted = False
    escaped = False
    for character in dot:
        if escaped:
            escaped = False
            continue
        if character == "\\" and quoted:
            escaped = True
            continue
        if character == '"':
            quoted = not quoted
            continue
        if quoted:
            continue
        if character in "{[":
            stack.append(character)
        elif character in pairs and (not stack or stack.pop() != pairs[character]):
            return False
    return not quoted and not stack
