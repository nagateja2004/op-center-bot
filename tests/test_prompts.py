from src.prompts import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_VERIFICATION_PROMPT,
    DIAGRAM_GENERATION_PROMPT,
    EVIDENCE_GRADING_PROMPT,
    QUERY_BROADENING_PROMPT,
    QUERY_PLANNING_PROMPT,
    SUPPORTED_OUTPUT_TYPES,
)


def test_planner_prompt_has_canonical_mapping_instructions() -> None:
    assert all(
        field in QUERY_PLANNING_PROMPT
        for field in ("{question}", "{conversation}", "{manual_names}", "{supported_output_types}")
    )
    assert "Evidence" not in QUERY_PLANNING_PROMPT
    assert all(
        instruction in QUERY_PLANNING_PROMPT
        for instruction in (
            "preserving the user's original meaning",
            "1-6 independent required_aspects",
            "exact manual phrases/headings",
            "canonical Opcenter terminology",
            "1-3 strong search_queries per aspect",
            "configuration concepts from runtime behavior",
            "soft preferences only",
            "never classify it as out of scope",
        )
    )
    assert all(
        example in QUERY_PLANNING_PROMPT
        for example in (
            "unique numbers assigned to containers",
            "field value changes",
            "which machine is valid",
            "hierarchy of physical modelling",
            "Portal Studio controls and how is security configured",
        )
    )
    assert "procedure" in SUPPORTED_OUTPUT_TYPES


def test_grader_prompt_requests_flat_summary_fields() -> None:
    assert "{required_aspects}" in EVIDENCE_GRADING_PROMPT
    assert "{aspect}" in EVIDENCE_GRADING_PROMPT
    assert "{evidence}" in EVIDENCE_GRADING_PROMPT
    assert "conversation" not in EVIDENCE_GRADING_PROMPT.casefold()
    assert "flat EvidenceGrade" in EVIDENCE_GRADING_PROMPT
    assert all(
        example in EVIDENCE_GRADING_PROMPT
        for example in (
            "CLF input validation does not define the Validate field event",
            "support for scalar and list fields does not define their difference",
            "pages containing web parts do not define Portal Studio controls",
            "SSL mention does not explain the role/permission security model",
            "Object-reference list-field section cannot define every list-field type",
        )
    )


def test_answer_and_verifier_prompts_use_distinct_evidence_boundaries() -> None:
    assert "Supported aspects" in ANSWER_GENERATION_PROMPT
    assert "Partial aspects" in ANSWER_GENERATION_PROMPT
    assert "Missing aspects" in ANSWER_GENERATION_PROMPT
    assert "EvidenceUnits" in ANSWER_GENERATION_PROMPT
    assert "cited EvidenceUnits" in ANSWER_VERIFICATION_PROMPT
    assert "{standalone_question}" in ANSWER_VERIFICATION_PROMPT
    assert "configuration-time from runtime behavior" in ANSWER_VERIFICATION_PROMPT


def test_diagram_prompt_accepts_only_verified_facts() -> None:
    assert all(
        field in DIAGRAM_GENERATION_PROMPT
        for field in (
            "{diagram_type}", "{entities}", "{relationships}", "{decisions}",
            "{outcomes}", "{source_ids}", "{diagram_rules}",
        )
    )
    assert "{evidence}" not in DIAGRAM_GENERATION_PROMPT
    assert "{standalone_question}" not in DIAGRAM_GENERATION_PROMPT


def test_generation_prompts_request_plain_text_only() -> None:
    assert "Return answer text only" in ANSWER_GENERATION_PROMPT
    assert "Return corrected text only" in ANSWER_VERIFICATION_PROMPT
    assert "Return Graphviz DOT only" in DIAGRAM_GENERATION_PROMPT


def test_prompts_remain_concise() -> None:
    assert len(QUERY_PLANNING_PROMPT) < 2_500
    assert len(EVIDENCE_GRADING_PROMPT) < 2_000
    assert len(ANSWER_GENERATION_PROMPT) < 1_200
    assert len(ANSWER_VERIFICATION_PROMPT) < 1_200
    assert all(len(prompt) < 700 for prompt in (
        QUERY_BROADENING_PROMPT,
        DIAGRAM_GENERATION_PROMPT,
    ))
