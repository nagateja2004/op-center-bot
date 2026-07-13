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
Return one flat QueryPlan. Resolve follow-ups from recent conversation while preserving the user's original meaning. Split compound questions into 1-6 independent required_aspects. Detect exact manual phrases/headings in exact_phrases. Map indirect wording to canonical Opcenter terminology in canonical_terms and record matched wording in aliases. Generate 1-3 strong search_queries per aspect. Distinguish configuration concepts from runtime behavior. Suggest manual_hints and preferred_manuals as soft preferences only; do not exclude other manuals.

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
Use this structure when requested:
{answer_structure}
Explain supported facts normally. Clearly label incomplete aspects and never invent the missing part.
Do not turn a general related statement into a formal definition. Label unsupported aspects.
Do not repeat the same fact under reasons and checks.
A troubleshooting checklist is not a procedure. Preserve exact terms, relevant
table rows, and configuration/runtime distinctions. Do not emit DOT. Return answer text only."""


ANSWER_VERIFICATION_PROMPT = """Correct the draft using only its cited EvidenceUnits.
Question: {standalone_question}
Required aspects: {required_aspects}
Partial aspects: {partial_aspects}
Missing aspects: {missing_aspects}
Required structure: {answer_structure}
Draft answer:
{answer}
Cited EvidenceUnits:
{evidence}
Check every requested aspect. Distinguish a definition from a related capability, an event from a CLF attached to it, and configuration-time from runtime behavior. Do not generalize an object-specific table or procedure. Remove unsupported claims and invalid citations; retain clearly labeled partial answers and keep reasons/checks non-duplicative.
Return corrected text only."""


DIAGRAM_GENERATION_PROMPT = """Return Graphviz DOT only from these verified facts.
Diagram type: {diagram_type}
Verified entities: {entities}
Verified outcomes: {outcomes}
Supporting source IDs: {source_ids}
Verified relationships:
{relationships}
Verified decisions:
{decisions}
Rules:
{diagram_rules}
Use 3-10 connected nodes, exact labels, and [S#] citations. Return empty text if insufficient."""
