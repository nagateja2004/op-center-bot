"""LangGraph nodes for the evidence-gated Opcenter RAG workflow."""

from __future__ import annotations

import asyncio
from functools import lru_cache
import inspect
import json
import logging
import math
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.compression import compress_evidence
from src.config import ROOT_DIR, settings
from src.llm import GroqRequestError, call_llm, call_structured
from src.inference import embedding_gate, reranker_gate
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
EXPLICIT_DIAGRAM_RE = re.compile(
    r"\b(?:diagrams?|flow[ -]?charts?|visuali[sz]e|draw|relationship map|tree diagram)\b",
    re.I,
)
MAX_DIAGRAM_DOT_LENGTH = 20_000
logger = logging.getLogger(__name__)
ALIAS_CONFIG_PATH = ROOT_DIR / "config" / "opcenter_aliases.json"
DOMAIN_CONCEPTS = {
    "Opcenter Execution Core": ("opcenter execution core", "execution core", "opcenter"),
    "Electronic Signatures": (
        "electronic signatures", "electronic signature", "esignatures", "esignature",
        "e-signatures", "e-signature", "esig requirements", "esig requirement", "esig",
    ),
    "Physical Modeling Sequence": (
        "physical modeling sequence", "physical modelling sequence",
        "physical modeling hierarchy", "physical modelling hierarchy",
        "hierarchy of physical modeling", "hierarchy of physical modelling",
    ),
    "Factory Hierarchy": ("factory hierarchy", "company-to-machine hierarchy"),
    "Information Model": ("information model",),
    "Physical Model": ("physical model", "physical modeling", "physical modelling"),
    "Process Model": ("process model",),
    "Execution Model": ("execution model",),
    "Enterprise": ("enterprise", "company"),
    "Factory": ("factory",),
    "Location": ("location",),
    "Resource": ("resource", "machine", "equipment"),
    "Workflow": ("workflow",),
    "Spec": ("manufacturing step", "workflow step", "spec"),
    "CDO": ("configurable data object", "cdo"),
    "CLF": ("configurable logic flow", "clf"),
    "Container": ("container",),
    "Shop Floor": ("shop floor",),
    "Portal Studio": ("portal studio",),
    "Designer": ("designer",),
    "Sampling": ("sampling",),
    "Manufacturing": ("manufacturing",),
    "Lot": ("lot",),
}
OPCENTER_TERMS = {
    alias for aliases in DOMAIN_CONCEPTS.values() for alias in aliases
}
CLEARLY_UNRELATED_RE = re.compile(
    r"\b(?:bake|cake|weather|forecast|poem|ocean|stock price|sports score|"
    r"football|cricket|movie review|restaurant|vacation|capital of|president of)\b",
    re.I,
)
CITATION_RE = re.compile(r"\[S(\d+)\]")
GROUPED_CITATION_RE = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)+)\]", re.I)
DECORATIVE_CITATION_RE = re.compile(
    r"【\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*】", re.I
)
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


def detect_explicit_diagram_request(question: str) -> tuple[bool, str]:
    """Detect an explicit diagram request without relying on the planner model."""
    requested = bool(EXPLICIT_DIAGRAM_RE.search(question))
    if not requested:
        return False, "auto"
    if re.search(r"\bdecision(?:[ -]+flow)?(?:[ -]+diagram|[ -]+flow[ -]?chart)\b", question, re.I):
        return True, "decision"
    if re.search(r"\b(?:flow[ -]?chart|process(?:[ -]+flow)?[ -]+diagram|process[ -]+flow)\b", question, re.I):
        return True, "process"
    if re.search(r"\b(?:hierarch\w*[ -]+diagram|tree[ -]+diagram)\b", question, re.I):
        return True, "hierarchy"
    if re.search(r"\brelationship(?:[ -]+diagram|[ -]+map)\b", question, re.I):
        return True, "relationship"
    if re.search(r"\barchitecture[ -]+diagram\b", question, re.I):
        return True, "architecture"
    for diagram_type in ("decision", "process", "hierarchy", "relationship", "architecture"):
        if re.search(rf"\b{diagram_type}\b", question, re.I):
            return True, diagram_type
    return True, "auto"


def basic_chat_response(question: str) -> str | None:
    """Return a deterministic response for short conversational messages."""
    normalized = " ".join(re.sub(r"[^a-z0-9\s]", " ", question.casefold()).split())
    if normalized in {
        "hi", "hii", "hello", "hellow", "hey", "hi there", "hello there",
        "good morning", "good afternoon", "good evening",
    }:
        return (
            "Hello! Ask me anything about the indexed Opcenter manuals. I can "
            "explain concepts, provide steps, cite sources, and create diagrams."
        )
    if normalized in {"thanks", "thank you", "thankyou", "thx"}:
        return "You're welcome! Ask another Opcenter question whenever you're ready."
    if normalized in {"bye", "goodbye", "see you", "see you later"}:
        return "Goodbye! Come back whenever you need help with Opcenter."
    if normalized in {"how are you", "how are you doing"}:
        return "I'm ready to help with questions about your indexed Opcenter manuals."
    if normalized in {"help", "what can you do", "who are you"}:
        return (
            "I answer questions from the indexed Opcenter manuals. I can explain "
            "concepts, list requested steps, cite manual sources, and generate useful diagrams."
        )
    return None


async def aunderstand_question(state: RAGState) -> RAGState:
    messages = list(state.get("messages", []))[-6:]
    question = _latest_user_text(messages)
    conversational_answer = basic_chat_response(question)
    if conversational_answer:
        return {
            "basic_chat": True,
            "standalone_question": question,
            "answer": conversational_answer,
            "sources": [],
            "grounded": False,
            "retrieved_docs": [],
            "expanded_docs": [],
            "reranked_docs": [],
            "aspect_documents": {},
            "compressed_views": {},
            "diagram_requested": False,
            "diagram_useful": False,
            "diagram_dot": "",
            "diagram_error": "",
            "diagram_generated": False,
            "retry_count": 0,
        }
    diagram_requested, requested_diagram_type = detect_explicit_diagram_request(question)
    diagram_override = state.get("diagram_type_override", "auto")
    if diagram_override != "auto":
        diagram_requested, requested_diagram_type = True, diagram_override
    diagram_enabled = state.get("diagram_enabled", state.get("allow_diagrams", True))
    if _simple_direct_question(question, messages):
        plan = _deterministic_plan(question)
    else:
        prompt = QUERY_PLANNING_PROMPT.format(
            question=question,
            conversation=_format_messages(_relevant_messages(messages, question)),
            manual_names=" | ".join(_available_manual_names()) or "none",
            supported_output_types=SUPPORTED_OUTPUT_TYPES,
        )
        try:
            plan = await _await_maybe(call_structured(
                _trim_to_token_budget(prompt, settings.planner_input_token_budget),
                QueryPlan,
                task="planner",
            ))
        except GroqRequestError:
            plan = _deterministic_plan(question)
    standalone = plan.standalone_question.strip() or question
    domain = _merge_plan_domain(
        question,
        _domain_context(f"{question} {standalone}"),
        plan,
    )
    required_output = _required_output(standalone, plan.required_output, plan.intent)
    diagram_useful = bool(
        plan.needs_diagram
        or "diagram" in required_output
        or DIAGRAM_RE.search(standalone)
    )
    aspects = _required_aspects(standalone, plan.required_aspects)
    required_manuals = _required_manuals(standalone)
    entities = _unique_text([*plan.entities, *domain["canonical_terms"]])
    if _is_sampling_movement_question(standalone):
        entities = _unique_text([*entities, *SAMPLING_ALLOWED_ENTITIES])
    concept_queries = _unique_text([
        *plan.exact_phrases,
        *domain["canonical_terms"],
        *domain["aliases"],
    ])
    aspect_queries = {
        aspect: _aspect_queries(
            aspect,
            plan.search_queries,
            concept_queries,
            multiple_aspects=len(aspects) > 1,
        )
        for aspect in aspects
    }
    queries = _ensure_queries(standalone, plan.search_queries, entities)
    preferred_manuals = _unique_text([
        *required_manuals,
        *domain["manual_hints"],
        *plan.preferred_manuals,
    ])
    logger.info(
        "question_domain status=%s term_count=%d canonical_count=%d alias_count=%d "
        "aspect_count=%d manuals=%s",
        domain["domain_status"], len(domain["domain_terms"]),
        len(domain["canonical_terms"]), len(domain["aliases"]), len(aspect_queries),
        preferred_manuals,
    )
    return {
        "basic_chat": False,
        "standalone_question": standalone,
        "exact_phrases": list(plan.exact_phrases),
        **domain,
        "intent": plan.intent,
        "complexity": "multi_aspect" if len(aspects) > 1 else "single_topic",
        "required_aspects": aspects,
        "aspect_queries": aspect_queries,
        "required_output": required_output,
        "entities": entities,
        "search_queries": queries,
        "manual_filters": {"manuals": preferred_manuals},
        "required_manuals": required_manuals,
        "needs_diagram": diagram_requested or diagram_useful,
        "diagram_requested": diagram_requested,
        "requested_diagram_type": requested_diagram_type,
        "diagram_enabled": bool(diagram_enabled),
        "diagram_useful": diagram_useful,
        "retry_count": 0,
        "missing_aspects": [],
        "missing_concepts": [],
        "partial_aspects": [],
        "coverage": {},
        "manual_coverage": {},
        "aspect_documents": {},
        "compressed_views": {},
        "retrieved_docs": [],
        "expanded_docs": [],
        "reranked_docs": [],
        "evidence_reason": "",
        "answer": "",
        "sources": [],
        "grounded": False,
        "unsupported_claims": [],
        "diagram_dot": "",
        "diagram_error": "",
        "diagram_generated": False,
        "diagram_supported": False,
        "llm_error_role": "",
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
        queries = state.get("aspect_queries", {}).get(aspect, [aspect])
        term_queries = [
            query for query in queries
            if query.casefold() != state["standalone_question"].casefold()
        ]
        retrieval_entities = _relevant_retrieval_terms(
            aspect, term_queries, state.get("entities", [])
        )
        retrieval_aliases = _relevant_retrieval_terms(
            aspect, term_queries, state.get("aliases", [])
        )
        documents = retrieve_multiple_queries(
            standalone_query=state["standalone_question"],
            search_queries=queries,
            entities=retrieval_entities,
            aliases=retrieval_aliases,
            preferred_manuals=_manual_preferences(state.get("manual_filters", {})),
            intent=f"{state.get('intent', '')} {aspect}",
        )
        logger.info(
            "aspect_retrieval query_count=%d headings=%s concepts=%s preferred=%s "
            "fused=%s evidence_ids=%s",
            len(queries),
            _matched_values(documents, "matched_heading"),
            _matched_values(documents, "matched_concept"),
            _manual_preferences(state.get("manual_filters", {})),
            _section_names(documents[:8]),
            _evidence_ids(documents[:8]),
        )
        aspect_documents[aspect] = [
            _copy_with_aspect(document, aspect) for document in documents
        ]
    return {
        "aspect_documents": aspect_documents,
        "retrieved_docs": _merge_aspect_documents(aspect_documents, limit=60),
    }


async def aretrieve_documents(state: RAGState) -> RAGState:
    async with embedding_gate.slot():
        return await asyncio.to_thread(retrieve_documents, state)


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
    standalone = state["standalone_question"]
    aspect_documents: dict[str, list[RetrievedDocument]] = {}
    compressed_views = {}
    complete_procedure = "procedure" in state.get("required_output", [])
    for aspect, documents in state.get("aspect_documents", {}).items():
        resolved = resolve_evidence_units(
            cross_encoder_rerank(
                f"{standalone}\nRequired aspect: {aspect}",
                documents,
                intent=f"{state.get('intent', '')} {aspect}",
                limit=6,
            ),
            limit=3,
        )
        compressed = [
            _with_compressed_view(
                _copy_with_aspect(document, aspect),
                aspect,
                standalone,
                state.get("canonical_terms", []),
                include_complete_procedure=complete_procedure,
            )
            for document in resolved
        ]
        aspect_documents[aspect] = compressed
        compressed_views[aspect] = [
            document["metadata"]["compressed_views"][aspect]
            for document in compressed
        ]
    documents = _merge_aspect_documents(
        aspect_documents, limit=MAX_EVIDENCE_SOURCES
    )
    for aspect, aspect_results in aspect_documents.items():
        logger.info(
            "aspect_rerank sections=%s evidence_ids=%s",
            _section_names(aspect_results),
            _evidence_ids(aspect_results),
        )
    return {
        "aspect_documents": aspect_documents,
        "compressed_views": compressed_views,
        "reranked_docs": documents,
    }


async def arerank_documents(state: RAGState) -> RAGState:
    async with reranker_gate.slot():
        return await asyncio.to_thread(rerank_documents, state)


async def agrade_evidence(state: RAGState) -> RAGState:
    coverage: dict[str, str] = {}
    reasons: list[str] = []
    missing_concepts: list[str] = []
    aspects = state.get("required_aspects") or [state["standalone_question"]]
    domain_status = state.get("domain_status") or _domain_context(
        state["standalone_question"]
    )["domain_status"]
    if domain_status == "out_of_scope":
        return {
            "evidence_status": "out_of_scope",
            "evidence_reason": "The request is clearly unrelated to Opcenter.",
            "missing_concepts": [],
            "coverage": {aspect: "out_of_scope" for aspect in aspects},
            "missing_aspects": list(aspects),
            "partial_aspects": [],
            "manual_coverage": {},
        }
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
            grade = await _await_maybe(call_structured(
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
            ))
        except GroqRequestError:
            grade = _heuristic_grade(state, aspect, documents)
        status = grade.status
        if status == "retry" and state.get("retry_count", 0) >= settings.max_retries:
            status = "in_scope_insufficient"
        if status == "out_of_scope" and domain_status == "in_scope":
            status = (
                "retry"
                if state.get("retry_count", 0) < settings.max_retries
                else "in_scope_insufficient"
            )
        coverage[aspect] = status
        reasons.append(f"{aspect}: {grade.reason}")
        missing_concepts.extend(grade.missing_concepts)
    partial_aspects = [
        aspect for aspect, status in coverage.items() if status == "partial"
    ]
    missing_aspects = [
        aspect
        for aspect, status in coverage.items()
        if status in {"retry", "in_scope_insufficient", "out_of_scope"}
    ]
    statuses = set(coverage.values())
    if statuses == {"sufficient"}:
        overall = "sufficient"
    elif "retry" in statuses and state.get("retry_count", 0) < settings.max_retries:
        overall = "retry"
    elif "sufficient" in statuses or "partial" in statuses:
        overall = "partial"
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
        "partial_aspects": partial_aspects,
        "manual_coverage": manual_coverage,
    }


async def abroaden_query(state: RAGState) -> RAGState:
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
            _message_content(await _await_maybe(call_llm(prompt, task="query_broadening")))
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


async def agenerate_answer(state: RAGState) -> RAGState:
    documents = state.get("reranked_docs", [])
    if state.get("evidence_status") not in {"sufficient", "partial"} or not documents:
        return generate_fallback(state)
    coverage = state.get("coverage", {})
    supported_aspects = [
        aspect for aspect, status in coverage.items() if status == "sufficient"
    ]
    partial_aspects = state.get("partial_aspects", []) or [
        aspect for aspect, status in coverage.items() if status == "partial"
    ]
    answerable_aspects = _unique_text([*supported_aspects, *partial_aspects]) or state.get(
        "required_aspects", []
    )
    required_manuals = state.get("required_manuals", []) or _required_manuals(
        state["standalone_question"]
    )
    documents = _select_answer_documents(
        documents, answerable_aspects, required_manuals=required_manuals
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
            partial_aspects=" | ".join(partial_aspects) or "none",
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
                    await _await_maybe(
                        call_llm(prompt, task="answer", evidence_count=len(documents))
                    )
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


async def averify_answer(state: RAGState) -> RAGState:
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
            standalone_question=state.get("standalone_question", ""),
            required_aspects=" | ".join(state.get("required_aspects", [])),
            partial_aspects=" | ".join(state.get("partial_aspects", [])) or "none",
            missing_aspects=" | ".join(state.get("missing_aspects", [])) or "none",
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
                await _await_maybe(
                    call_llm(prompt, task="verifier", evidence_count=len(cited_numbers))
                )
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
        "diagram_supported": grounded and (
            state.get("diagram_requested", False)
            or state.get("diagram_useful", state.get("needs_diagram", False))
        ),
        "messages": [AIMessage(content=answer)],
    }


def _diagram_result(dot: str = "", error: str = "") -> RAGState:
    return {
        "diagram_dot": dot,
        "diagram_error": error,
        "diagram_generated": bool(dot),
    }


async def agenerate_diagram(state: RAGState) -> RAGState:
    question = state.get("standalone_question", "")
    documents = state.get("reranked_docs", [])
    enabled = state.get("diagram_enabled", state.get("allow_diagrams", True))
    useful = state.get("diagram_useful", state.get("needs_diagram", False))
    if not (
        enabled
        and (state.get("diagram_requested", False) or useful)
        and state.get("grounded")
        and documents
    ):
        return _diagram_result()
    verified_answer = state.get("answer", "")
    cited_numbers = _citation_numbers(verified_answer, len(documents))
    if not cited_numbers:
        return _diagram_result(error="no_cited_evidence")
    source_ids = [f"S{number}" for number in cited_numbers]
    cited_documents = [documents[number - 1] for number in cited_numbers]
    support_text = " ".join(
        f"{document['metadata'].get('section', '')} {document['text']}"
        for document in cited_documents
    )
    requested_type = state.get("requested_diagram_type", "auto")
    diagram_type = requested_type if requested_type != "auto" else _diagram_type(question)
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
        return _diagram_result(error="insufficient_verified_evidence")
    coverage = state.get("coverage", {})
    relevant_aspects = [
        aspect
        for aspect in state.get("required_aspects", [])
        if not coverage or coverage.get(aspect) in {"sufficient", "partial"}
    ]
    cited_evidence = _format_cited_evidence(
        documents,
        cited_numbers,
        question,
        max_chars=max(800, settings.diagram_input_token_budget * 2),
    )
    prompt = _trim_to_token_budget(
        DIAGRAM_GENERATION_PROMPT.format(
            standalone_question=question,
            diagram_type=diagram_type,
            verified_answer=verified_answer,
            evidence=cited_evidence,
            relevant_aspects=" | ".join(relevant_aspects) or "none",
            entities=" | ".join(entities),
            relationships="\n".join(relationships),
            decisions="\n".join(decisions) or "none",
            outcomes=" | ".join(outcomes) or "none",
            source_ids=" | ".join(source_ids),
            diagram_rules=_diagram_rules(sampling_decision),
        ),
        settings.diagram_input_token_budget,
    )
    direction = "TB" if diagram_type in {"decision", "hierarchy"} else "LR"
    try:
        dot = _message_content(
            await _await_maybe(
                call_llm(prompt, task="diagram", evidence_count=len(source_ids))
            )
        )
    except GroqRequestError as exc:
        logger.info(
            "diagram_result requested=%s enabled=%s type=%s output_length=0 valid=false failure=%s",
            state.get("diagram_requested", False), enabled, diagram_type, exc.kind,
        )
        return _diagram_result(error=exc.kind)
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
    if not diagram and not sampling_decision:
        fallback = _cited_step_diagram(verified_answer, direction, set(source_ids))
        fallback_direction = direction
        if not fallback:
            fallback = _cited_ascii_hierarchy_diagram(
                verified_answer, set(source_ids)
            )
            fallback_direction = "TB"
        diagram = _validated_dot(
            fallback, fallback_direction, source_ids=set(source_ids)
        )
    failure = "" if diagram else (
        "no_diagram" if dot.strip() == "NO_DIAGRAM" else "invalid_dot"
    )
    logger.info(
        "diagram_result requested=%s enabled=%s type=%s output_length=%s valid=%s failure=%s",
        state.get("diagram_requested", False), enabled, diagram_type, len(dot),
        bool(diagram), failure or "none",
    )
    return _diagram_result(diagram or "", failure)


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
        "diagram_dot": "",
        "diagram_error": "",
        "diagram_generated": False,
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


def _simple_direct_question(question: str, messages: list[BaseMessage]) -> bool:
    text = question.strip()
    return (
        len(messages) <= 1
        and len(text.split()) <= 14
        and bool(re.match(r"^(?:what is|define|explain)\b", text, re.I))
        and not bool(
            re.search(
                r"\b(?:and|or|compare|difference|how|why|across|hierarch\w*|workflow|process|relationship)\b",
                text,
                re.I,
            )
        )
    )


def _deterministic_plan(question: str) -> QueryPlan:
    lowered = question.casefold()
    domain = _domain_context(question)
    if re.search(r"\b(?:how do|how to|procedure|steps?|step-by-step)\b", lowered):
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
            *domain["canonical_terms"],
        ]
    )[:8]
    outputs = _required_output(question, [], intent)
    return QueryPlan(
        standalone_question=question,
        intent=intent,
        complexity="multi_aspect" if "?" in question.rstrip("?") else "single_topic",
        required_aspects=_fallback_aspects(question, domain["canonical_terms"]),
        required_output=outputs,
        entities=entities,
        search_queries=_unique_text([question, *domain["canonical_terms"]])[:4],
        exact_phrases=domain["aliases"],
        canonical_terms=domain["canonical_terms"],
        aliases=domain["aliases"],
        manual_hints=domain["manual_hints"],
        preferred_manuals=domain["manual_hints"],
        needs_diagram="diagram" in outputs,
    )


def _fallback_aspects(question: str, canonical_terms: list[str]) -> list[str]:
    concepts = set(canonical_terms)
    field_aspects = [
        concept
        for concept in ("Scalar Field", "List Field", "Validate Event")
        if concept in concepts
    ]
    if len(field_aspects) > 1:
        return field_aspects
    if "Portal Studio Control" in concepts and (
        concepts.intersection({"Role", "Permission", "Security Server", "SSL"})
        or re.search(r"\bsecurity\b", question, re.I)
    ):
        return ["Portal Studio controls", "security configuration"]
    if {"Spec", "Resource Group", "Resource"}.issubset(concepts):
        return ["Spec, Resource Group, and Resource relationship"]
    if {"Role", "Permission", "Security Server"}.issubset(concepts):
        return [
            "Role and Permission configuration",
            "Security Server and SSL configuration",
        ]
    if {"Role", "Permission"}.issubset(concepts):
        return ["Role and Permission configuration"]
    if "Numbering Rule" in concepts:
        return ["Numbering Rule and Container identifiers"]
    if "Validate Event" in concepts:
        return ["Validate Event and field value acceptance"]
    if "Factory Hierarchy" in concepts:
        return ["Physical Modeling Sequence and Factory Hierarchy"]
    if "Recipe Pattern" in concepts:
        return ["Recipe Pattern"]
    if "Portal Studio Control" in concepts:
        return ["Portal Studio controls"]
    return [question]


def _heuristic_grade(
    state: RAGState,
    aspect: str,
    documents: list[RetrievedDocument],
) -> EvidenceGrade:
    if not _looks_in_scope(state["standalone_question"], state.get("entities", [])):
        return EvidenceGrade(status="out_of_scope", reason="The request is unrelated to Opcenter.")
    weak_status = (
        "retry"
        if state.get("retry_count", 0) < settings.max_retries
        else "in_scope_insufficient"
    )
    question = state["standalone_question"]
    aspect_text = " ".join(aspect.casefold().split())
    aspect_terms = set(_content_terms(aspect))
    sections = " ".join(
        str(document["metadata"].get("section", "")) for document in documents
    ).casefold()
    evidence = " ".join(document["text"][:800] for document in documents).casefold()
    combined = f"{sections} {evidence}"
    evidence_terms = set(_content_terms(combined))
    section_overlap = aspect_terms & set(_content_terms(sections))
    evidence_overlap = aspect_terms & evidence_terms
    content_types = {document["content_type"] for document in documents}
    relevant_canonical = [
        term
        for term in state.get("canonical_terms", [])
        if aspect_terms & set(_content_terms(term))
        or (
            "security" in aspect_terms
            and term in {"Role", "Permission", "Employee", "Security Server", "SSL"}
        )
    ]
    canonical_hits = [
        term
        for term in relevant_canonical
        if " ".join(term.casefold().split()) in combined
    ]
    exact_aspect = aspect_text in combined
    event_requested = "event" in aspect_terms
    procedure_requested = bool(
        re.search(r"\b(?:how do|how to|procedure|steps?|step-by-step)\b", question, re.I)
    )
    definition_requested = bool(
        re.search(r"\b(?:what is|what are|define|definition|difference)\b", question, re.I)
    )
    field_or_table_requested = bool(aspect_terms & {"field", "fields", "table"})
    definition_language = bool(
        re.search(r"\b(?:means|represents|is defined as|defines|consists of|refers to)\b", evidence)
    )

    if event_requested:
        if not exact_aspect or "event" not in evidence_terms:
            return EvidenceGrade(
                status=weak_status,
                reason="Evidence is related but does not directly establish the requested event.",
            )
        if section_overlap and (definition_language or "event" in sections):
            return EvidenceGrade(status="sufficient", reason="Event-specific evidence directly matches the aspect.")
        return EvidenceGrade(status="partial", reason="The event is identified but its requested behavior is incomplete.")

    if "control" in aspect_terms and "control" not in evidence_terms:
        return EvidenceGrade(
            status=weak_status,
            reason="Page or web-part evidence does not directly define the requested controls.",
        )

    if "list field" in aspect_text and "object reference" in combined and not re.search(
        r"\b(?:all list fields|list field types|scalar list)\b", combined
    ):
        return EvidenceGrade(
            status="partial",
            reason="Object-reference list-field evidence cannot define all list-field types.",
        )

    if procedure_requested:
        if "procedure" in content_types and evidence_overlap:
            return EvidenceGrade(status="sufficient", reason="Procedure evidence directly matches the aspect.")
        return EvidenceGrade(
            status=weak_status,
            reason="Related evidence does not provide the requested procedure.",
        )

    if field_or_table_requested and content_types.intersection({"field_definition", "table"}):
        if exact_aspect or section_overlap or canonical_hits:
            return EvidenceGrade(status="sufficient", reason="Structured field evidence directly matches the aspect.")

    if definition_requested:
        if (exact_aspect or canonical_hits) and definition_language and section_overlap:
            return EvidenceGrade(status="sufficient", reason="Definition evidence directly matches the aspect.")
        if exact_aspect or canonical_hits or len(evidence_overlap) >= max(1, len(aspect_terms) - 1):
            return EvidenceGrade(
                status="partial",
                reason="The concept is present, but the requested definition or distinction is incomplete.",
            )
        return EvidenceGrade(
            status=weak_status,
            reason="Evidence is related but does not define the assigned aspect.",
        )

    if exact_aspect and section_overlap and len(evidence_overlap) >= max(1, len(aspect_terms) // 2):
        return EvidenceGrade(status="sufficient", reason="Section and aspect-specific evidence directly match.")
    if "security" in aspect_terms and canonical_hits:
        return EvidenceGrade(
            status="partial",
            reason="A security component is supported, but the broader security model is incomplete.",
        )
    if exact_aspect or canonical_hits:
        return EvidenceGrade(status="partial", reason="Evidence directly supports only part of the aspect.")
    return EvidenceGrade(
        status=weak_status,
        reason="Evidence is related but does not directly answer the assigned aspect.",
    )


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
    text = question.casefold()
    model_types = bool(re.search(r"\b(?:types? of models?|models? (?:are )?used)\b", text))
    hierarchy = bool(re.search(r"\b(?:hierarch\w*|sequence|structure)\b", text))
    if model_types and hierarchy:
        return [
            "model types",
            "relationship between models",
            "physical modeling hierarchy",
        ]
    if model_types:
        return ["model types", "relationship between models"]
    if hierarchy and re.search(r"\bphysical modell?(?:ing)?\b", text):
        return ["physical modeling hierarchy"]
    if _contains_alias(question, "Electronic Signatures") and (
        not aspects or all("electronic signature" in aspect.casefold() for aspect in aspects)
    ):
        return ["Electronic Signatures"]
    unique = list(
        dict.fromkeys(
            cleaned
            for aspect in aspects
            if (cleaned := " ".join(aspect.split()))
        )
    )[:6]
    return unique or [question]


def _aspect_queries(
    aspect: str,
    planner_queries: list[str],
    canonical_terms: list[str],
    *,
    multiple_aspects: bool = False,
) -> list[str]:
    aspect_terms = set(_content_terms(aspect))
    specific_terms = aspect_terms - {
        "configuration", "event", "events", "field", "fields", "model", "models"
    }
    match_terms = specific_terms or aspect_terms

    def related(values: list[str], *, require_overlap: bool) -> list[str]:
        ranked = sorted(
            enumerate(values),
            key=lambda item: (
                len(match_terms & set(_content_terms(item[1]))),
                -item[0],
            ),
            reverse=True,
        )
        return [
            value
            for _, value in ranked
            if not require_overlap or match_terms & set(_content_terms(value))
        ]

    candidates = [
        aspect,
        *related(planner_queries, require_overlap=multiple_aspects),
        *related(canonical_terms, require_overlap=multiple_aspects),
    ]
    return _unique_text(candidates)[:4]


def _required_output(question: str, outputs: list[str], intent: str) -> list[str]:
    text = f"{question} {intent}".casefold()
    required = list(outputs)
    if not required:
        required.append("explanation")
    if re.search(r"\b(?:cannot|can't|fails?|problem|issue|stuck|troubleshoot)\b", text):
        required.extend(["likely_reasons", "checks"])
    if re.search(r"\b(?:compare|comparison|difference|versus|vs\.? )\b", text):
        required.append("comparison_table")
    if re.search(r"\b(?:procedure|how do|how to|steps?|step-by-step)\b", text):
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
    required = []
    if "modeling" in text and "shop floor" in text:
        required.extend(["Modeling", "Shop Floor"])
    if re.search(r"\b(?:execution electronics|opcenter electronics|ocexel)\b", text):
        required.append("Execution Electronics")
    if re.search(r"\b(?:execution discrete|opcenter discrete|ex[ -]?ds|exds)\b", text):
        required.append("Execution Discrete")
    return list(dict.fromkeys(required))


def _manual_family(manual: str) -> str:
    lowered = manual.casefold()
    if "execution electronics" in lowered or "ocexel" in lowered:
        return "Execution Electronics"
    if "execution discrete" in lowered or re.search(r"\bex[ -]?ds\b", lowered):
        return "Execution Discrete"
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
    if "comparison_table" in required_output:
        return ["Direct answer", "Comparison table", "Key differences"]
    if not (
        _is_sampling_movement_question(question)
        or any(item in required_output for item in ("likely_reasons", "checks"))
    ):
        if re.search(r"\b(?:hierarch\w*|relationship|architecture)\b", question, re.I):
            return ["Direct explanation", "Key entities", "Supported relationships"]
        return (
            ["Prerequisites when supported", "Numbered steps", "Expected result when supported"]
            if "procedure" in required_output
            else ["Direct explanation"]
        )
    headings = [
        "Direct explanation",
        "Configuration relationship",
        "Runtime behavior",
        "Likely reasons movement is blocked",
        "What to check",
    ]
    if "procedure" in required_output:
        headings[1:1] = ["Prerequisites when supported", "Numbered steps"]
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
    copied["metadata"]["compressed_views"] = dict(
        copied["metadata"].get("compressed_views", {})
    )
    return copied


def _with_compressed_view(
    document: RetrievedDocument,
    aspect: str,
    question: str,
    canonical_terms: list[str],
    *,
    include_complete_procedure: bool,
) -> RetrievedDocument:
    copied = _copy_with_aspect(document, aspect)
    copied["metadata"]["compressed_views"][aspect] = compress_evidence(
        copied,
        aspect,
        question,
        canonical_terms=canonical_terms,
        include_complete_procedure=include_complete_procedure,
        max_characters=700,
    )
    return copied


def _merge_document_details(
    existing: RetrievedDocument, candidate: RetrievedDocument
) -> None:
    for key in ("aspects", "matched_representation_types", "selected_candidate_ids"):
        existing["metadata"][key] = list(
            dict.fromkeys(
                [
                    *existing["metadata"].get(key, []),
                    *candidate["metadata"].get(key, []),
                ]
            )
        )
    existing_views = existing["metadata"].setdefault("compressed_views", {})
    existing_views.update(candidate["metadata"].get("compressed_views", {}))
    for name, value in candidate["retrieval_scores"].items():
        current = existing["retrieval_scores"].get(name)
        if current is None:
            existing["retrieval_scores"][name] = value
        elif "rank" in name or "distance" in name:
            existing["retrieval_scores"][name] = min(current, value)
        else:
            existing["retrieval_scores"][name] = max(current, value)


def _merge_aspect_documents(
    aspect_documents: dict[str, list[RetrievedDocument]], *, limit: int
) -> list[RetrievedDocument]:
    """Round-robin aspect evidence so one topic cannot crowd out the others."""
    merged: list[RetrievedDocument] = []
    by_evidence_id: dict[str, RetrievedDocument] = {}
    depth = 0
    while len(merged) < limit:
        examined = False
        for aspect, documents in aspect_documents.items():
            if depth >= len(documents):
                continue
            examined = True
            candidate = _copy_with_aspect(documents[depth], aspect)
            evidence_id = _evidence_id(candidate)
            if evidence_id in by_evidence_id:
                _merge_document_details(by_evidence_id[evidence_id], candidate)
            else:
                merged.append(candidate)
                by_evidence_id[evidence_id] = candidate
            if len(merged) >= limit:
                break
        if not examined:
            break
        depth += 1
    return merged[:limit]


def _manual_preferences(filters: dict[str, str | list[str]]) -> list[str]:
    values: list[str] = []
    for value in filters.values():
        values.extend(value if isinstance(value, list) else [value])
    return [value for value in values if value]


def _relevant_retrieval_terms(
    aspect: str, queries: list[str], terms: list[str]
) -> list[str]:
    reference = set(_content_terms(" ".join([aspect, *queries])))
    ranked = sorted(
        (
            (len(reference.intersection(_content_terms(term))), index, term)
            for index, term in enumerate(terms)
            if term.strip()
        ),
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    )
    return [term for overlap, _, term in ranked if overlap][:8]


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
    view = document["metadata"].get("compressed_views", {}).get(aspect)
    if view is None:
        view = compress_evidence(
            document,
            aspect,
            aspect,
            max_characters=GRADER_EXCERPT_CHARS,
        )
    return _truncate_evidence(
        str(view.get("compressed_text", "")), GRADER_EXCERPT_CHARS
    )


def _format_grader_summaries(
    documents: list[RetrievedDocument], aspect: str
) -> str:
    summaries: list[str] = []
    aspect_terms = set(_content_terms(aspect))
    ranked = sorted(
        enumerate(_unique_evidence_documents(documents)),
        key=lambda item: (
            len(aspect_terms & set(_content_terms(
                f"{item[1]['metadata'].get('section', '')} {item[1]['text']}"
            ))),
            -item[0],
        ),
        reverse=True,
    )
    for index, (_, document) in enumerate(ranked[:2], start=1):
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
        if len(selected) == MAX_EVIDENCE_SOURCES:
            break
    return selected[:MAX_EVIDENCE_SOURCES]


def _answer_content(
    document: RetrievedDocument, query: str, include_complete_procedure: bool
) -> str:
    if include_complete_procedure and document["content_type"] == "procedure":
        return compress_evidence(
            document,
            query,
            query,
            include_complete_procedure=True,
            max_characters=700,
        )["compressed_text"]
    views = list(document["metadata"].get("compressed_views", {}).values())
    query_terms = set(_content_terms(query))
    if views:
        view = max(
            views,
            key=lambda item: len(
                query_terms
                & set(
                    _content_terms(
                        f"{item.get('aspect', '')} {item.get('compressed_text', '')}"
                    )
                )
            ),
        )
    else:
        view = compress_evidence(
            document,
            query,
            query,
            include_complete_procedure=include_complete_procedure,
            max_characters=700,
        )
    return str(view.get("compressed_text", ""))


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
    for index, document in enumerate(documents[:MAX_EVIDENCE_SOURCES], start=1):
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
    per_source = max(320, max_chars // len(cited_numbers) - 160)
    entries: list[str] = []
    for number in cited_numbers:
        document = documents[number - 1]
        metadata = document["metadata"]
        compressed = _answer_content(document, query, False)
        original = _relevant_excerpt(
            document["text"], query, minimum=180, maximum=max(220, per_source // 2)
        )
        content = f"Compressed view:\n{compressed}\n\nOriginal context:\n{original}"
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


def _cited_step_diagram(answer: str, direction: str, source_ids: set[str]) -> str:
    """Turn an answer's cited numbered steps into a safe visual fallback."""
    steps: list[tuple[str, list[str]]] = []
    for match in re.finditer(
        r"(?ms)^\s*\d+[.)]\s+\*\*(?P<title>[^*\n]+)\*\*(?P<body>.*?)(?=^\s*\d+[.)]\s+\*\*|\Z)",
        answer,
    ):
        citations = list(
            dict.fromkeys(
                f"S{number}" for number in CITATION_RE.findall(match.group(0))
                if f"S{number}" in source_ids
            )
        )
        if citations:
            title = re.sub(r"\s+", " ", match.group("title")).strip(" -–—")
            steps.append((title, citations))
    if len(steps) < 2:
        return ""

    lines = [
        "digraph G {",
        f"  rankdir={direction};",
        '  graph [bgcolor="#0E1117", pad="0.3", nodesep="0.45", ranksep="0.55"];',
        '  node [shape=box, style="rounded,filled", fillcolor="#1F2937", color="#60A5FA", fontcolor="#F9FAFB", fontname="Arial", margin="0.18,0.12"];',
        '  edge [color="#94A3B8", penwidth=1.5, arrowsize=0.8];',
    ]
    for index, (title, citations) in enumerate(steps, start=1):
        cited = " ".join(f"[{citation}]" for citation in citations)
        label = f"{title}\\n{cited}".replace('"', r'\"')
        lines.append(f'  step{index} [label="{label}"];')
    lines.extend(
        f"  step{index} -> step{index + 1};"
        for index in range(1, len(steps))
    )
    lines.append("}")
    return "\n".join(lines)


def _cited_ascii_hierarchy_diagram(answer: str, source_ids: set[str]) -> str:
    """Convert a cited Markdown tree into boxes when the diagram model fails."""
    blocks = list(re.finditer(r"```[^\n]*\n(?P<tree>.*?)```", answer, re.S))
    for block in reversed(blocks):
        tree = block.group("tree")
        if not re.search(r"[├└][─-]", tree):
            continue
        nearby = answer[block.end():block.end() + 1_200]
        citations = list(
            dict.fromkeys(
                f"S{number}" for number in CITATION_RE.findall(nearby)
                if f"S{number}" in source_ids
            )
        )
        if not citations:
            continue

        nodes: list[tuple[str, int]] = []
        for line in tree.splitlines():
            branch = re.search(r"[├└][─-]\s*", line)
            if branch:
                title = line[branch.end():].strip()
                depth = math.ceil(len(line[:branch.start()].expandtabs(4)) / 4) + 1
            else:
                title = line.strip()
                depth = 0
            if title and not re.fullmatch(r"[│\s]+", title):
                nodes.append((title, depth))
        if len(nodes) < 2:
            continue

        cited = " ".join(f"[{citation}]" for citation in citations)
        lines = [
            "digraph G {",
            "  rankdir=TB;",
            '  graph [bgcolor="#0E1117", pad="0.3", nodesep="0.45", ranksep="0.55"];',
            '  node [shape=box, style="rounded,filled", fillcolor="#1F2937", color="#60A5FA", fontcolor="#F9FAFB", fontname="Arial", margin="0.18,0.12"];',
            '  edge [color="#94A3B8", penwidth=1.5, arrowsize=0.8];',
        ]
        parents: dict[int, str] = {}
        previous_depth = 0
        for index, (title, depth) in enumerate(nodes, start=1):
            depth = min(depth, previous_depth + 1)
            node_id = f"node{index}"
            label = f"{title}\\n{cited}".replace('"', r'\"')
            lines.append(f'  {node_id} [label="{label}"];')
            if depth and depth - 1 in parents:
                lines.append(f"  {parents[depth - 1]} -> {node_id};")
            parents[depth] = node_id
            parents = {level: parent for level, parent in parents.items() if level <= depth}
            previous_depth = depth
        lines.append("}")
        return "\n".join(lines)
    return ""


def _verified_entities(
    entities: list[str],
    answer: str,
    question: str = "",
    support_text: str = "",
) -> list[str]:
    verified_text = f"{answer} {support_text}".casefold()
    if _is_sampling_movement_question(question):
        return [
            entity for entity in SAMPLING_ALLOWED_ENTITIES
            if entity.casefold() in verified_text
        ]
    extracted = re.findall(
        r"\b(?:[A-Z]{2,}|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\b", answer
    )
    supported_entities = [
        entity for entity in entities if entity.casefold() in verified_text
    ]
    return _unique_text([*supported_entities, *extracted])[:10]


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


def _matched_values(documents: list[RetrievedDocument], key: str) -> list[str]:
    return list(
        dict.fromkeys(
            str(document["metadata"].get(key, ""))
            for document in documents
            if document["metadata"].get(key)
        )
    )[:8]


def _evidence_ids(documents: list[RetrievedDocument]) -> list[str]:
    return list(
        dict.fromkeys(
            str(document["metadata"].get("evidence_id", ""))
            for document in documents
            if document["metadata"].get("evidence_id")
        )
    )


def _looks_in_scope(question: str, entities: list[str]) -> bool:
    return _domain_context(f"{question} {' '.join(entities)}")["domain_status"] == "in_scope"


def _contains_alias(text: str, canonical: str) -> bool:
    lowered = text.casefold()
    entry = _concept_catalog().get(canonical, {})
    return any(
        re.search(rf"(?<!\w){re.escape(alias.casefold())}(?!\w)", lowered)
        for alias in entry.get("aliases", [])
    )


@lru_cache(maxsize=1)
def _concept_catalog() -> dict[str, dict[str, list[str]]]:
    catalog = {
        canonical: {
            "aliases": _unique_text([canonical, *aliases]),
            "manual_hints": [],
        }
        for canonical, aliases in DOMAIN_CONCEPTS.items()
    }
    try:
        raw = json.loads(ALIAS_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("root must be an object")
        for canonical, entry in raw.items():
            if not isinstance(canonical, str) or not isinstance(entry, dict):
                raise ValueError("concept entries must be objects")
            aliases = entry.get("aliases", [])
            hints = entry.get("manual_hints", [])
            if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
                raise ValueError(f"{canonical}.aliases must be a string list")
            if not isinstance(hints, list) or not all(isinstance(item, str) for item in hints):
                raise ValueError(f"{canonical}.manual_hints must be a string list")
            current = catalog.setdefault(canonical, {"aliases": [], "manual_hints": []})
            current["aliases"] = _unique_text([canonical, *current["aliases"], *aliases])
            current["manual_hints"] = _unique_text([*current["manual_hints"], *hints])
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not load Opcenter aliases from %s: %s", ALIAS_CONFIG_PATH, exc)
    return catalog


def _catalog_details(canonical_terms: list[str]) -> tuple[list[str], list[str], list[str]]:
    by_name = {name.casefold(): (name, entry) for name, entry in _concept_catalog().items()}
    known: list[str] = []
    aliases: list[str] = []
    manual_hints: list[str] = []
    for term in canonical_terms:
        match = by_name.get(term.casefold())
        if not match:
            continue
        canonical, entry = match
        known.append(canonical)
        aliases.extend(entry["aliases"])
        manual_hints.extend(entry["manual_hints"])
    return _unique_text(known), _unique_text(aliases), _unique_text(manual_hints)


def _merge_plan_domain(question: str, domain: dict[str, Any], plan: QueryPlan) -> dict[str, Any]:
    canonical_terms = _unique_text([*plan.canonical_terms, *domain["canonical_terms"]])
    known, catalog_aliases, catalog_hints = _catalog_details(canonical_terms)
    aliases = _unique_text([*plan.aliases, *domain["aliases"], *catalog_aliases])
    manual_hints = _unique_text([
        *plan.manual_hints,
        *domain["manual_hints"],
        *catalog_hints,
    ])
    if re.search(r"\b(?:architecture|overview|introduction|getting started)\b", question, re.I):
        manual_hints = _unique_text([*manual_hints, "Getting Started"])
    return {
        **domain,
        "domain_status": "in_scope" if known else domain["domain_status"],
        "canonical_terms": canonical_terms,
        "aliases": aliases,
        "manual_hints": manual_hints,
    }


def _domain_context(question: str) -> dict[str, Any]:
    lowered = question.casefold()
    matches = [
        (canonical, alias)
        for canonical, entry in _concept_catalog().items()
        for alias in entry["aliases"]
        if re.search(rf"(?<!\w){re.escape(alias.casefold())}(?!\w)", lowered)
    ]
    canonical_terms = _unique_text([canonical for canonical, _ in matches])
    aliases = _unique_text([alias for _, alias in matches])
    if (
        re.search(r"\b(?:unique|automatic\w*)\b.*\b(?:numbers?|identifiers?)\b", lowered)
        and "container" in lowered
    ):
        canonical_terms = _unique_text([*canonical_terms, "Numbering Rule", "Container"])
    if re.search(r"\bsecur(?:ity|e)\b.*\b(?:configur\w*|model|access)\b", lowered):
        canonical_terms = _unique_text([
            *canonical_terms,
            "Role",
            "Permission",
            "Security Server",
            "SSL",
        ])
    if re.search(r"\b(?:types? of models?|models? (?:are )?used)\b", lowered) and any(
        term in canonical_terms for term in ("Opcenter Execution Core", "Physical Model")
    ):
        canonical_terms = _unique_text([
            *canonical_terms,
            "Information Model",
            "Physical Model",
            "Process Model",
            "Execution Model",
        ])
    if re.search(
        r"\b(?:physical modell?(?:ing)? (?:hierarch\w*|sequence|structure)|"
        r"(?:hierarch\w*|sequence|structure) of physical modell?(?:ing)?)\b",
        lowered,
    ):
        canonical_terms = _unique_text([
            *canonical_terms,
            "Physical Modeling Sequence",
            "Factory Hierarchy",
            "Enterprise",
            "Factory",
            "Location",
            "Resource",
        ])
    known, _, catalog_hints = _catalog_details(canonical_terms)
    canonical_terms = _unique_text([*canonical_terms, *known])
    clearly_unrelated = bool(CLEARLY_UNRELATED_RE.search(question)) and not canonical_terms
    modeling_concepts = {
        "Electronic Signatures", "Physical Modeling Sequence", "Factory Hierarchy",
        "Physical Model", "Factory", "Location", "Resource", "Workflow", "Spec",
    }
    manual_hints = list(catalog_hints)
    if modeling_concepts.intersection(canonical_terms):
        manual_hints.append("Modeling")
    if re.search(r"\b(?:move transaction|container transaction|collect sampling data)\b", lowered):
        manual_hints.append("Shop Floor")
    if re.search(r"\b(?:installation components?|security server|ssl|tls)\b", lowered):
        manual_hints.append("Installation")
    if re.search(r"\b(?:portal studio|web parts?|portal controls?)\b", lowered):
        manual_hints.append("Portal Studio")
    manual_hints = _unique_text(manual_hints)
    return {
        "domain_status": "out_of_scope" if clearly_unrelated else "in_scope",
        "domain_terms": aliases,
        "canonical_terms": canonical_terms,
        "aliases": aliases,
        "manual_hints": manual_hints,
    }


def _remove_invalid_citations(answer: str, evidence_count: int) -> str:
    return CITATION_RE.sub(
        lambda match: match.group(0) if 1 <= int(match.group(1)) <= evidence_count else "",
        answer,
    )


def _normalize_citations(answer: str) -> str:
    """Convert grouped citations like [S1, S3] into independently valid IDs."""
    answer = DECORATIVE_CITATION_RE.sub(lambda match: f"[{match.group(1)}]", answer)
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
        not dot
        or len(dot) > MAX_DIAGRAM_DOT_LENGTH
        or dot.casefold() == "no_diagram"
        or "```" in dot
        or not re.match(r"^(?:di)?graph\b", dot, re.I)
        or not _balanced_delimiters(dot)
        or not re.search(r"(?:->|--)", dot)
        or re.search(r"<[^>]+>|\b(?:image|shapefile|href|url)\s*=", dot, re.I)
    ):
        return None
    declarations = [
        (name, attributes)
        for name, attributes in re.findall(
            r'(?:^|(?<=[;{]))\s*("[^"\n]+"|[A-Za-z_][\w.-]*)\s*\[(.*?)\]\s*;',
            dot,
            re.M,
        )
        if name.strip('"').casefold() not in {"graph", "node", "edge", "digraph"}
    ]
    if len(declarations) < 2:
        return None
    node_names = {name for name, _ in declarations}
    node_names.update(
        re.findall(r'(?:->|--)\s*("[^"\n]+"|[A-Za-z_][\w.-]*)', dot)
    )
    node_names -= {"graph", "node", "edge", "digraph"}
    if not 2 <= len(node_names) <= 20:
        return None
    if source_ids:
        for _, attributes in declarations:
            label_match = re.search(r'label\s*=\s*"([^"]+)"', attributes, re.I)
            citations = set(CITATION_RE.findall(label_match.group(1))) if label_match else set()
            if not citations or not {f"S{number}" for number in citations}.issubset(source_ids):
                return None
        all_citations = {f"S{number}" for number in CITATION_RE.findall(dot)}
        if not all_citations.issubset(source_ids):
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
    if not re.search(r"\bgraph\s*\[\s*bgcolor\s*=", dot, re.I):
        dot = dot.replace(
            "{",
            "{\n  graph [bgcolor=\"transparent\", pad=0.4, nodesep=0.5, ranksep=0.7];"
            "\n  node [shape=box, style=\"rounded,filled\", fillcolor=\"#E8F1FF\","
            " color=\"#2563EB\", fontname=\"Arial\", fontsize=11];"
            "\n  edge [color=\"#64748B\", fontname=\"Arial\", fontsize=10, arrowsize=0.8];",
            1,
        )
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
        if node_id.strip('"').casefold() in {"graph", "node", "edge"}:
            continue
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
        edge_label = CITATION_RE.sub("", label_match.group(1)).strip().casefold() if label_match else ""
        if edge_label not in {
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


async def _await_maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


# Compatibility entry points for direct local calls and the existing unit tests.
# Production LangGraph wiring uses the async variants above.
def understand_question(state: RAGState) -> RAGState:
    return asyncio.run(aunderstand_question(state))


def grade_evidence(state: RAGState) -> RAGState:
    return asyncio.run(agrade_evidence(state))


def broaden_query(state: RAGState) -> RAGState:
    return asyncio.run(abroaden_query(state))


def generate_answer(state: RAGState) -> RAGState:
    return asyncio.run(agenerate_answer(state))


def verify_answer(state: RAGState) -> RAGState:
    return asyncio.run(averify_answer(state))


def generate_diagram(state: RAGState) -> RAGState:
    return asyncio.run(agenerate_diagram(state))
