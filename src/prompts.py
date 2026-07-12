"""Minimal prompts for each LLM role."""


SUPPORTED_OUTPUT_TYPES = (
    "explanation, procedure, likely_reasons, checks, comparison_table, "
    "diagram, cross_manual_synthesis"
)


QUERY_PLANNING_PROMPT = """Plan retrieval; do not answer.
Current question: {question}
Relevant recent messages:
{conversation}
Available manuals: {manual_names}
Supported output types: {supported_output_types}
Resolve follow-ups. Return a flat QueryPlan with 1-6 required aspects and no more
than four search queries."""


QUERY_BROADENING_PROMPT = """Return 1-3 broader Opcenter search queries, one per line.
Question: {standalone_question}
Missing aspects: {missing_aspects}
Queries tried: {previous_queries}
Related sections: {section_names}
No answer, JSON, analysis, or Markdown bullets."""


EVIDENCE_GRADING_PROMPT = """Grade manual support; do not answer the question.
Question: {standalone_question}
Required aspects: {required_aspects}
Assigned aspect: {aspect}
Evidence summaries:
{evidence}
Return one flat EvidenceGrade. status: sufficient, partial, retry,
in_scope_insufficient, or out_of_scope. Weak evidence is not out_of_scope.
Use one short reason and at most six short missing_concepts."""


ANSWER_GENERATION_PROMPT = """Answer only from the supplied EvidenceUnits with [S#] citations.
Question: {standalone_question}
Required output: {required_output}
Supported aspects: {supported_aspects}
Missing aspects: {missing_aspects}
EvidenceUnits:
{evidence}
Use this structure when requested:
{answer_structure}
Label unsupported aspects. Do not repeat the same fact under reasons and checks.
A troubleshooting checklist is not a procedure. Preserve exact terms, relevant
table rows, and configuration/runtime distinctions. Do not emit DOT. Return answer text only."""


ANSWER_VERIFICATION_PROMPT = """Correct the draft using only its cited EvidenceUnits.
Required aspects: {required_aspects}
Required structure: {answer_structure}
Draft answer:
{answer}
Cited EvidenceUnits:
{evidence}
Remove unsupported claims and invalid citations; retain supported partial answers
and keep reasons/checks non-duplicative under the required headings.
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
