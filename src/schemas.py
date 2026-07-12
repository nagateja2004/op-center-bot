"""Shared data structures and LangGraph state."""

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


@dataclass(slots=True)
class DocumentChunk:
    id: str
    text: str
    source: str
    page: int
    chapter: str | None = None
    section: str | None = None
    release: str | None = None
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievedDocument(TypedDict):
    chunk_id: str
    text: str
    content_type: str
    metadata: dict[str, Any]
    retrieval_scores: dict[str, float]


class EvidenceUnit(TypedDict):
    evidence_id: str
    text: str
    content_type: str
    metadata: dict[str, Any]
    token_count: int
    structured_table: dict[str, Any] | None
    procedure_steps: list[str]
    annotations: list[dict[str, str]]


class RetrievalSegment(TypedDict):
    segment_id: str
    evidence_id: str
    searchable_text: str
    content_type: str
    metadata: dict[str, Any]
    segment_index: int
    previous_segment_id: str | None
    next_segment_id: str | None
    word_count: int
    embedding_token_count: int


class SourceInfo(BaseModel):
    source_id: str
    source: str
    page: str | int | None = None
    manual: str | None = None
    source_file: str | None = None
    chapter: str | None = None
    section: str | None = None
    release: str | None = None
    printed_page: str | None = None
    pdf_page: int | None = None
    content_type: str | None = None


class QueryPlan(BaseModel):
    standalone_question: str
    intent: str
    complexity: Literal["single_topic", "multi_aspect"] = "single_topic"
    required_aspects: list[str] = Field(default_factory=list)
    required_output: list[
        Literal[
            "explanation",
            "procedure",
            "likely_reasons",
            "checks",
            "comparison_table",
            "diagram",
            "cross_manual_synthesis",
        ]
    ] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    preferred_manuals: list[str] = Field(default_factory=list)
    needs_diagram: bool = False


class EvidenceGrade(BaseModel):
    status: Literal[
        "sufficient", "partial", "retry", "in_scope_insufficient", "out_of_scope"
    ]
    reason: str
    missing_concepts: list[str] = Field(default_factory=list)


class RAGState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    standalone_question: str
    intent: str
    complexity: Literal["single_topic", "multi_aspect"]
    required_aspects: list[str]
    aspect_queries: dict[str, list[str]]
    required_output: list[str]
    entities: list[str]
    search_queries: list[str]
    manual_filters: dict[str, str | list[str]]
    required_manuals: list[str]
    needs_diagram: bool
    allow_diagrams: bool
    retrieved_docs: list[RetrievedDocument]
    expanded_docs: list[RetrievedDocument]
    reranked_docs: list[RetrievedDocument]
    aspect_documents: dict[str, list[RetrievedDocument]]
    retry_count: int
    evidence_status: Literal[
        "sufficient", "partial", "retry", "in_scope_insufficient", "out_of_scope"
    ]
    evidence_reason: str
    missing_concepts: list[str]
    coverage: dict[str, str]
    missing_aspects: list[str]
    manual_coverage: dict[str, bool]
    answer: str
    sources: list[SourceInfo]
    grounded: bool
    unsupported_claims: list[str]
    diagram_supported: bool
    diagram_dot: str | None
    llm_error_role: str
