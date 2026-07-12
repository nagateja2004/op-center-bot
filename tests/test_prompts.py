from src.prompts import (
    ANSWER_GENERATION_PROMPT,
    ANSWER_VERIFICATION_PROMPT,
    DIAGRAM_GENERATION_PROMPT,
    EVIDENCE_GRADING_PROMPT,
    QUERY_BROADENING_PROMPT,
    QUERY_PLANNING_PROMPT,
    SUPPORTED_OUTPUT_TYPES,
)


def test_planner_prompt_has_only_minimal_planning_context() -> None:
    assert all(
        field in QUERY_PLANNING_PROMPT
        for field in ("{question}", "{conversation}", "{manual_names}", "{supported_output_types}")
    )
    assert "Evidence" not in QUERY_PLANNING_PROMPT
    assert "procedure" in SUPPORTED_OUTPUT_TYPES


def test_grader_prompt_requests_flat_summary_fields() -> None:
    assert "{required_aspects}" in EVIDENCE_GRADING_PROMPT
    assert "{aspect}" in EVIDENCE_GRADING_PROMPT
    assert "{evidence}" in EVIDENCE_GRADING_PROMPT
    assert "conversation" not in EVIDENCE_GRADING_PROMPT.casefold()
    assert "flat EvidenceGrade" in EVIDENCE_GRADING_PROMPT


def test_answer_and_verifier_prompts_use_distinct_evidence_boundaries() -> None:
    assert "Supported aspects" in ANSWER_GENERATION_PROMPT
    assert "Missing aspects" in ANSWER_GENERATION_PROMPT
    assert "EvidenceUnits" in ANSWER_GENERATION_PROMPT
    assert "cited EvidenceUnits" in ANSWER_VERIFICATION_PROMPT
    assert "{standalone_question}" not in ANSWER_VERIFICATION_PROMPT


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
    assert all(
        len(prompt) < 700
        for prompt in (
            QUERY_PLANNING_PROMPT,
            QUERY_BROADENING_PROMPT,
            EVIDENCE_GRADING_PROMPT,
            ANSWER_GENERATION_PROMPT,
            ANSWER_VERIFICATION_PROMPT,
            DIAGRAM_GENERATION_PROMPT,
        )
    )
