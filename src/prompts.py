"""Minimal prompts for each LLM role."""


SUPPORTED_OUTPUT_TYPES = (
    "explanation, procedure, likely_reasons, checks, comparison_table, "
    "diagram, cross_manual_synthesis"
)


QUERY_PLANNING_PROMPT = """Plan Opcenter manual retrieval only. Never answer the question and never classify it as out of scope.
Current question: {question}
Relevant recent messages:
{conversation}
Available manuals: {manual_names}
Supported output types: {supported_output_types}
Return one flat QueryPlan. Resolve follow-ups from recent conversation while preserving the user's original meaning. Split compound questions into 1-8 independent required_aspects. When the user explicitly lists objects for a comparison, preserve every listed object as its own required aspect. Detect exact manual phrases/headings in exact_phrases. Map indirect wording to canonical Opcenter terminology in canonical_terms and record matched wording in aliases. Generate 1-3 strong search_queries per aspect. Distinguish configuration concepts from runtime behavior. Suggest manual_hints and preferred_manuals as soft preferences only; do not exclude other manuals.

Examples:
- "How are unique numbers assigned to containers?" -> Numbering Rule, Container, prefix, sequence, suffix.
- "What runs when a field value changes?" -> Field Event, Validate Event, CLF.
- "How does the system decide which machine is valid?" -> Spec, Resource Group, Resource.
- "What is the hierarchy of physical modelling?" -> Physical Modeling Sequence, Factory Hierarchy, Enterprise, Factory, Location, Resource.
- "What are Portal Studio controls and how is security configured?" -> two primary aspects: Portal Studio controls; security configuration."""


QUERY_BROADENING_PROMPT = """Return 1-3 broader Opcenter search queries, one per line.
Question: {standalone_question}
Missing aspects: {missing_aspects}
Queries tried: {previous_queries}
Related sections: {section_names}
No answer, JSON, analysis, or Markdown bullets."""


EVIDENCE_GRADING_PROMPT = """Grade only the assigned aspect from the supplied manual evidence; do not answer.
Question: {standalone_question}
Required aspects: {required_aspects}
Assigned aspect: {aspect}
Relationship requirement: {relationship_requirement}
Evidence summaries:
{evidence}
Return one flat EvidenceGrade with one short reason and at most six missing_concepts.
- sufficient: evidence directly establishes the requested definition, behavior, procedure, relationship, condition, or result.
- partial: evidence directly supports part of this aspect, but an important requested part is missing.
- retry: evidence is related but does not directly answer this aspect, or better manual evidence likely exists.
- in_scope_insufficient: one broader retrieval was already used and evidence still does not directly answer this aspect.
- out_of_scope: only for a question clearly unrelated to Opcenter manuals.
Do not infer direct support from related capability: CLF input validation does not define the Validate field event; support for scalar and list fields does not define their difference; pages containing web parts do not define Portal Studio controls; SSL mention does not explain the role/permission security model; an Object-reference list-field section cannot define every list-field type."""


ANSWER_GENERATION_PROMPT = """Answer only from the supplied EvidenceUnits with [S#] citations.
Question: {standalone_question}
Required output: {required_output}
Supported aspects: {supported_aspects}
Partial aspects: {partial_aspects}
Missing aspects: {missing_aspects}
EvidenceUnits:
{evidence}
Relationship requirements: {relationship_requirements}
Use this structure when requested:
{answer_structure}
Explain supported facts normally. Clearly label incomplete aspects and never invent the missing part.
For a requested procedure, preserve source order and use numbered steps. Include prerequisites and expected results only when supported; never invent omitted actions.
Do not turn a general related statement into a formal definition. Label unsupported aspects as not established by the retrieved evidence; never claim they are not covered by the manuals unless that is explicitly established. Keep Setup (the modeled machine configuration) distinct from the Resource Setup transaction. Do not infer availability from a status-code name; use its configured Availability value.
Do not repeat the same fact under reasons and checks.
A troubleshooting checklist is not a procedure. Preserve exact terms, relevant
table rows, page-specific scope, and configuration/runtime distinctions. Never invent a named object or rule requested by the user when that exact mechanism is absent from evidence. Do not treat counters or report colors as causes unless evidence explicitly does so. Do not emit DOT. Return answer text only."""


ANSWER_VERIFICATION_PROMPT = """Correct the draft using only its cited EvidenceUnits.
Question: {standalone_question}
Required aspects: {required_aspects}
Partial aspects: {partial_aspects}
Missing aspects: {missing_aspects}
Relationship requirements: {relationship_requirements}
Required structure: {answer_structure}
Draft answer:
{answer}
Cited EvidenceUnits:
{evidence}
Check every requested aspect. Distinguish a definition from a related capability, an event from a CLF attached to it, and configuration-time from runtime behavior. Do not generalize an object-specific table or procedure. Remove unsupported claims and invalid citations; retain clearly labeled partial answers and keep reasons/checks non-duplicative. Describe missing support as not established by the retrieved evidence, not as absent from the manuals. Keep Setup distinct from the Resource Setup transaction, and use the configured Availability value rather than assuming a status code must be named Up.
Do not invent an object or rule named only by the question; if absent, say so and use the evidenced mechanism. For sampling: failed movement uses the Sample Test allow-move option; Switching Rules change inspection levels; Sample Rate Counter counts matching containers started or moved, not samples; colors indicate status; keep selection rules page-scoped.
Return corrected text only."""


DIAGRAM_GENERATION_PROMPT = """You generate a focused Graphviz diagram from verified evidence.
Question: {standalone_question}
Diagram type: {diagram_type}
Relevant verified aspects: {relevant_aspects}
Verified final answer:
{verified_answer}
Cited EvidenceUnits only:
{evidence}
Allowed citation IDs: {source_ids}
Verified entities: {entities}
Verified outcomes: {outcomes}
Verified relationships:
{relationships}
Verified decisions:
{decisions}

Rules:
1. Use only relationships and decisions explicitly supported by the supplied evidence.
2. Follow the requested type: hierarchy uses top-to-bottom boxes; relationship uses labeled entity arrows; process uses left-to-right ordered actions; architecture groups components and labels data flow; decision uses diamonds, supported Yes/No or Pass/Fail edge labels, and supported outcomes.
3. Use short labels. Apply rounded filled boxes, light blue fills, blue borders, gray arrows, Arial font, and comfortable node spacing. Do not encode meaning with color alone.
4. Exclude buttons, pages, tabs, dialog boxes, navigation, and click/save/close instructions unless the user explicitly asks for a UI procedure diagram.
5. Do not convert every noun phrase into a node or add vague nodes such as related objects, etc., or in Opcenter Execution.
6. Keep the diagram focused on the question. Every node and important edge must be traceable to an allowed [S#] citation; include citations in node labels and decision-edge labels.
7. Return Graphviz DOT only, without Markdown fences. For decisions start with: digraph G {{ rankdir=TB; node [fontname="Arial"]; edge [fontname="Arial"]; }}
8. If fewer than two meaningful relationships are supported, return exactly NO_DIAGRAM.
9. Treat fields and field values as attributes, not modeling-object containers. Use solid edges only for verified parent-child or sequence relationships and dashed edges for verified optional references.
Additional verified constraints: {diagram_rules}"""
