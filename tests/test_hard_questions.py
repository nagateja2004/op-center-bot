from __future__ import annotations

import re
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
import pytest

import src.nodes as nodes
import src.retrieval as retrieval
from src.graph import graph
from src.schemas import (
    EvidenceGrade,
    QueryPlan,
)


QUESTIONS = {
    "inspection": (
        "An operator says a container cannot continue to the next workflow step "
        "after inspection. Explain the relevant concepts, likely reasons, and checks."
    ),
    "cdo_compare": (
        "Compare Data CDOs and Service CDOs, including purpose, persistence, fields, "
        "events, methods, CLFs, configuration, and usage."
    ),
    "resource": (
        "Explain how a modeled Resource becomes usable in a Shop Floor transaction."
    ),
    "clf": "When a CLF is configured on a CDO event, what is configured and what executes at runtime?",
    "cdo": "What is a CDO, and are all CDO instances maintained in Modeling?",
    "cdo_relationship": (
        "Explain how a CDO, its fields, events, methods, CLFs, and functions are "
        "related. Include a relationship diagram."
    ),
    "sampling_movement": (
        "Using Modeling and Shop Floor evidence, explain how the Current Spec, "
        "Sampling Plan, Sample Tests, sampling status, failure movement rule, and "
        "Move transaction determine whether a Container advances to the Next Workflow "
        "Step or remains Movement Blocked. Include likely reasons, checks, and a decision diagram."
    ),
}


def document(aspect: str) -> dict:
    lowered = aspect.casefold()
    manual = "Opcenter Execution Core Designer User Guide"
    release = "Release 2504+ Rev. 1"
    section = aspect
    text = f"Manual evidence for {aspect}."
    if any(word in lowered for word in ("current spec", "sampling plan", "sample tests", "failure movement")):
        manual = "Opcenter Execution Core Modeling User Guide"
        release = "Release 2510+ Rev. 1"
        section = "Spec sampling configuration"
        text = (
            f"{aspect} configures the Current Spec, Sampling Plan, Sample Tests, and "
            "Failure Movement Rule for the Container and Movement Blocked outcome."
        )
    elif any(word in lowered for word in ("sampling status", "move transaction")):
        manual = "Opcenter Execution Core Shop Floor User Guide"
        release = "Release 2310+ Rev. 1"
        section = "Sampling status and Move transaction"
        text = (
            f"{aspect} is evaluated for the Container at runtime. The Move Transaction "
            "advances it to the Next Workflow Step or leaves Movement Blocked."
        )
    elif any(word in lowered for word in ("inspection", "sampling", "workflow", "checks")):
        manual = "Opcenter Execution Core Shop Floor User Guide"
        release = "Release 2310+ Rev. 1"
        section = "AQL Sampling and container workflow"
        text = "Sampling inspection status and workflow checks for a container."
    elif "modeling" in lowered or "modeled" in lowered:
        manual = "Opcenter Execution Core Modeling User Guide"
        release = "Release 2510+ Rev. 1"
        section = "Defining a Resource"
        text = "A Resource is configured in Modeling before Shop Floor use."
    elif "shop floor" in lowered or "runtime transaction" in lowered:
        manual = "Opcenter Execution Core Shop Floor User Guide"
        release = "Release 2310+ Rev. 1"
        section = "Resource transaction execution"
        text = "A Shop Floor transaction uses the configured Resource at runtime."
    elif "clf" in lowered or "event" in lowered or "runtime" in lowered:
        section = "CDO events and CLFs"
        text = "CLFs are configured on an event; their functions execute sequentially at runtime."
    elif "data cdo" in lowered or "service cdo" in lowered or "comparison" in lowered:
        section = "Data CDOs and Service CDOs"
        text = f"Designer definitions for {aspect}, including configuration and usage."
    elif "cdo" in lowered:
        section = "Configurable Data Objects"
        text = "A CDO is a Configurable Data Object; maintenance depends on its CDO type."
    return {
        "chunk_id": re.sub(r"[^a-z0-9]+", "-", aspect.casefold()).strip("-"),
        "text": text,
        "content_type": "text",
        "metadata": {
            "manual": manual,
            "release": release,
            "chapter": "Manual-backed concepts",
            "section": section,
            "printed_page": "4-12",
            "pdf_page": 120 + len(aspect),
            "chunk_level": "child",
        },
        "retrieval_scores": {"final_score": 1.0},
    }


def plan_for(question: str) -> QueryPlan:
    lowered = question.casefold()
    if "current spec" in lowered and "move transaction" in lowered:
        aspects = list(nodes.SAMPLING_REQUIRED_ASPECTS)
        outputs = ["explanation", "likely_reasons", "checks", "diagram", "cross_manual_synthesis"]
    elif "cannot continue" in lowered:
        aspects = ["sampling inspection concepts", "likely reasons", "operator checks"]
        outputs = ["explanation", "likely_reasons", "checks"]
    elif "data cdos" in lowered:
        aspects = [
            "Data CDO purpose and persistence",
            "Service CDO purpose and persistence",
            "fields and events",
            "methods and CLFs",
            "configuration and usage",
            "Data CDO and Service CDO comparison",
        ]
        outputs = ["explanation", "comparison_table"]
    elif "modeled resource" in lowered:
        aspects = [
            "Modeling Resource configuration",
            "Shop Floor runtime transaction",
            "configuration to runtime relationship",
        ]
        outputs = ["explanation", "cross_manual_synthesis", "diagram"]
    elif "its fields, events, methods" in lowered:
        aspects = ["CDO", "fields", "events", "methods", "CLFs", "functions"]
        outputs = ["explanation", "diagram"]
    elif "clf is configured" in lowered:
        aspects = ["CLF event configuration", "runtime CLF execution", "function order"]
        outputs = ["explanation"]
    else:
        aspects = ["CDO definition", "CDO type-specific maintenance"]
        outputs = ["explanation"]
    return QueryPlan(
        standalone_question=question,
        intent="multi-part manual question",
        complexity="multi_aspect",
        required_aspects=aspects,
        required_output=outputs,
        entities=(list(nodes.SAMPLING_ALLOWED_ENTITIES) if "current spec" in lowered else ["Opcenter"]),
        search_queries=[question],
        needs_diagram="diagram" in outputs,
    )


@pytest.fixture
def hard_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_structured(prompt, schema, *, task, **kwargs):
        prompt = str(prompt)
        question_match = re.search(r"(?:Current question|Question): (.+)", prompt)
        question = question_match.group(1).strip() if question_match else ""
        lowered = question.casefold()
        if schema is QueryPlan:
            return plan_for(question)
        if schema is EvidenceGrade:
            return EvidenceGrade(status="sufficient", reason="Aspect supported")
        raise AssertionError(schema)

    def fake_plain(prompt, *, task, **kwargs):
        prompt = str(prompt)
        question_match = re.search(r"Question: (.+)", prompt)
        question = question_match.group(1).strip() if question_match else ""
        lowered = question.casefold()
        if task == "answer":
            if "current spec" in lowered and "move transaction" in lowered:
                answer = (
                    "**Direct explanation**\nThe Container is evaluated at its Current Spec before movement [S1].\n\n"
                    "**Configuration relationship**\nThe Current Spec links the Sampling Plan and Sample Tests, while the Failure Movement Rule controls failed-sample movement [S2] [S3] [S5].\n\n"
                    "**Runtime behavior**\nSampling Status is evaluated at runtime; the Move Transaction advances the Container to the Next Workflow Step when movement is allowed [S4] [S6].\n\n"
                    "**Likely reasons movement is blocked**\nSampling Status may be In Process, or a Fail result may be blocked by the Failure Movement Rule and leave Movement Blocked [S4] [S5].\n\n"
                    "**What to check**\nCheck the Current Spec links, configured Sampling Plan and Sample Tests, current Sampling Status, Failure Movement Rule, and availability of the Move Transaction [S1] [S2] [S3] [S4] [S5] [S6].\n\n"
                    "**Decision diagram**\nThe diagram summarizes the verified runtime decision path [S1] [S4] [S6]."
                )
            elif "cannot continue" in lowered:
                answer = (
                    "**Concepts:** Sampling inspection status can control container workflow progression [S1].\n\n"
                    "**Likely reasons:** the inspection result or sampling status has not satisfied the next-step condition [S2].\n\n"
                    "**Checks:** verify the container inspection result, sampling status, and next workflow step [S3]."
                )
            elif "data cdos" in lowered:
                answer = (
                    "Data CDOs hold persistent business data and expose configured fields and events [S1] [S3]. "
                    "Service CDOs perform service-oriented behavior through methods and CLFs [S2] [S4]. "
                    "Configuration determines their available behavior and usage [S5].\n\n"
                    "| Area | Data CDO | Service CDO |\n|---|---|---|\n"
                    "| Purpose | Business data | Service behavior |\n"
                    "| Persistence | Persistent data | Operation-oriented |\n"
                    "| Fields/events | Configured fields and events | Service inputs/events |\n"
                    "| Methods/CLFs | Event behavior | Methods and CLFs |\n"
                    "| Configuration/usage | Data modeling | Runtime services | [S6]"
                )
            elif "modeled resource" in lowered:
                answer = (
                    "**Configuration time:** define the Resource in Modeling [S1].\n\n"
                    "**Runtime:** the Shop Floor transaction uses that configured Resource [S2].\n\n"
                    "The relationship carries the modeled definition into transaction execution [S3]."
                )
            elif "its fields, events, methods" in lowered:
                answer = (
                    "A CDO defines fields that hold its data [S1] [S2]. Events expose points where "
                    "configured behavior can run [S3]. Methods use CLFs to perform that behavior, "
                    "and CLFs contain ordered functions [S4] [S5] [S6]."
                )
            elif "clf is configured" in lowered:
                answer = (
                    "**Configuration time:** attach the CLF and its functions to the CDO event [S1].\n\n"
                    "**Runtime:** when the event fires, the CLF functions execute sequentially [S2] [S3]."
                )
            else:
                answer = (
                    "A CDO is a Configurable Data Object [S1]. Whether an instance is maintained in "
                    "Modeling depends on the specific CDO type; this is not true for every CDO [S2]."
                )
            return AIMessage(content=answer)
        if task == "verifier":
            answer = re.search(
                r"Draft answer:\n(.*?)\nCited EvidenceUnits:", prompt, re.S
            ).group(1).strip()
            return AIMessage(content=answer)
        if task == "diagram":
            if "Diagram type: decision" in prompt:
                return AIMessage(
                    content=(
                        'digraph G {\n'
                        'container [label="Container [S1]", shape=box];\n'
                        'spec [label="Current Spec [S1]", shape=box];\n'
                        'plan [label="Sampling Plan [S2]", shape=box];\n'
                        'tests [label="Sample Tests [S3]", shape=box];\n'
                        'status [label="Sampling Status? [S4]", shape=diamond];\n'
                        'rule [label="Failure Movement Rule? [S5]", shape=diamond];\n'
                        'move [label="Move Transaction [S6]", shape=box];\n'
                        'next [label="Next Workflow Step [S6]", shape=box];\n'
                        'blocked [label="Movement Blocked [S5]", shape=box];\n'
                        'container -> spec;\nspec -> plan;\nplan -> tests;\ntests -> status;\n'
                        'status -> move [label="Pass"];\nstatus -> rule [label="Fail"];\n'
                        'status -> blocked [label="In Process"];\n'
                        'rule -> move [label="Yes"];\nrule -> blocked [label="No"];\n'
                        'move -> next;\n}'
                    )
                )
            if "its fields, events, methods" in lowered:
                return AIMessage(
                    content=(
                        '```dot\ndigraph G {\ncdo [label="CDO [S1]"];\n'
                        'fields [label="Fields [S2]"];\nevents [label="Events [S3]"];\n'
                        'methods [label="Methods [S4]"];\nclfs [label="CLFs [S5]"];\n'
                        'functions [label="Functions [S6]"];\ncdo -> fields;\n'
                        'cdo -> events;\nevents -> methods;\nmethods -> clfs;\n'
                        'clfs -> functions;\n}\n```'
                    )
                )
            return AIMessage(
                content=(
                    'digraph G {\ninput [label="Input: Modeled Resource [S1]"];\n'
                    'process [label="Process: Shop Floor transaction [S2]"];\n'
                    'output [label="Output: Runtime use [S3]"];\n'
                    "input -> process;\nprocess -> output;\n}"
                )
            )
        return AIMessage(content="")

    monkeypatch.setattr(nodes, "call_structured", fake_structured)
    monkeypatch.setattr(nodes, "call_llm", fake_plain)
    monkeypatch.setattr(
        nodes,
        "retrieve_multiple_queries",
        lambda standalone_query, search_queries, **kwargs: [document(search_queries[0])],
    )
    monkeypatch.setattr(nodes, "expand_retrieval_context", lambda documents, **kwargs: documents)
    monkeypatch.setattr(
        nodes,
        "cross_encoder_rerank",
        lambda query, documents, **kwargs: documents[:3],
    )
    monkeypatch.setattr(
        nodes,
        "resolve_evidence_units",
        lambda documents, **kwargs: documents[: kwargs["limit"]],
    )


def invoke(question: str) -> dict:
    return graph.invoke(
        {"messages": [HumanMessage(content=question)], "retry_count": 0, "allow_diagrams": True},
        config={"configurable": {"thread_id": str(uuid4())}},
    )


def test_operator_container_inspection_question(hard_pipeline) -> None:
    result = invoke(QUESTIONS["inspection"])
    assert result["evidence_status"] == "sufficient"
    assert all(label in result["answer"] for label in ("Concepts", "Likely reasons", "Checks"))
    assert any("Shop Floor" in source.manual for source in result["sources"])
    assert any("Sampling" in document["metadata"]["section"] for document in result["reranked_docs"])


def test_data_cdo_service_cdo_comparison(hard_pipeline) -> None:
    result = invoke(QUESTIONS["cdo_compare"])
    for term in ("purpose", "persistent", "fields", "events", "methods", "CLFs", "Configuration", "usage"):
        assert term.casefold() in result["answer"].casefold()
    assert result["answer"].rstrip().splitlines()[-1].startswith("| Configuration/usage")
    assert len(result["required_aspects"]) == 6
    assert all(result["coverage"][aspect] == "sufficient" for aspect in result["required_aspects"])
    assert "comparison_table" in result["required_output"]


def test_modeled_resource_to_shop_floor_transaction(hard_pipeline) -> None:
    result = invoke(QUESTIONS["resource"])
    manuals = {source.manual for source in result["sources"]}
    assert any("Modeling" in manual for manual in manuals)
    assert any("Shop Floor" in manual for manual in manuals)
    assert "Configuration time" in result["answer"] and "Runtime" in result["answer"]
    assert "Release warning" in result["answer"]
    assert all(label in result["diagram_dot"] for label in ("Input:", "Process:", "Output:"))


def test_clf_event_configuration_and_runtime(hard_pipeline) -> None:
    result = invoke(QUESTIONS["clf"])
    assert "Configuration time" in result["answer"]
    assert "Runtime" in result["answer"]
    assert "execute sequentially" in result["answer"]
    for source in result["sources"]:
        document = result["reranked_docs"][int(source.source_id[1:]) - 1]
        assert source.pdf_page == document["metadata"]["pdf_page"]


def test_cdo_definition_is_type_specific(hard_pipeline) -> None:
    result = invoke(QUESTIONS["cdo"])
    assert "not true for every CDO" in result["answer"]
    assert "every CDO instance is maintained in Modeling" not in result["answer"]


def test_cdo_relationship_answer_uses_plain_text_calls_and_valid_dot(hard_pipeline) -> None:
    result = invoke(QUESTIONS["cdo_relationship"])

    assert re.search(r"\[S\d+\]", result["answer"])
    assert {source.source_id for source in result["sources"]}
    assert result["diagram_dot"].startswith("digraph")
    assert nodes._balanced_delimiters(result["diagram_dot"])


def test_diagram_failure_preserves_verified_answer_and_sources(
    hard_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = nodes.call_llm

    def fail_diagram(prompt, *, task, **kwargs):
        if task == "diagram":
            raise nodes.GroqRequestError("unavailable_model", "diagram", 404)
        return original(prompt, task=task, **kwargs)

    monkeypatch.setattr(nodes, "call_llm", fail_diagram)

    result = invoke(QUESTIONS["cdo_relationship"])

    assert re.search(r"\[S\d+\]", result["answer"])
    assert result["sources"]
    assert result["grounded"] is True
    assert result["diagram_dot"] is None


def test_sampling_movement_uses_both_manuals_and_valid_decision_diagram(
    hard_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nodes, "retrieve_multiple_queries", retrieval.retrieve_multiple_queries)
    monkeypatch.setattr(nodes, "expand_retrieval_context", retrieval.expand_context)
    monkeypatch.setattr(
        nodes,
        "cross_encoder_rerank",
        lambda query, documents, **kwargs: documents[: kwargs["limit"]],
    )
    monkeypatch.setattr(nodes, "resolve_evidence_units", retrieval.resolve_evidence_units)
    diagram_prompts: list[str] = []
    original_llm = nodes.call_llm

    def capture_diagram_prompt(prompt, *, task, **kwargs):
        if task == "diagram":
            diagram_prompts.append(str(prompt))
        return original_llm(prompt, task=task, **kwargs)

    monkeypatch.setattr(nodes, "call_llm", capture_diagram_prompt)
    validation_calls: list[dict] = []
    original_validate = nodes._validated_dot

    def capture_validation(dot, direction, **kwargs):
        validation_calls.append(kwargs)
        return original_validate(dot, direction, **kwargs)

    monkeypatch.setattr(nodes, "_validated_dot", capture_validation)

    result = invoke(QUESTIONS["sampling_movement"])

    retrieved_manuals = {
        nodes._manual_family(document["metadata"]["manual"])
        for document in result["reranked_docs"]
    }
    cited_manuals = {nodes._manual_family(source.manual) for source in result["sources"]}
    assert retrieved_manuals == cited_manuals == {"Modeling", "Shop Floor"}
    assert result["evidence_status"] == "sufficient"
    assert result["required_aspects"] == nodes.SAMPLING_REQUIRED_ASPECTS
    assert all(result["coverage"][aspect] == "sufficient" for aspect in nodes.SAMPLING_REQUIRED_ASPECTS)
    for heading in (
        "Direct explanation",
        "Configuration relationship",
        "Runtime behavior",
        "Likely reasons movement is blocked",
        "What to check",
        "Decision diagram",
    ):
        assert heading in result["answer"]
    assert not any(
        key in result["answer"]
        for key in ("Cross_manual_synthesis", "required_output", "coverage", "Procedure")
    )
    dot = result["diagram_dot"]
    assert diagram_prompts
    diagram_prompt = diagram_prompts[0]
    assert "Verified decisions:" in diagram_prompt
    assert "Verified outcomes: Next Workflow Step | Movement Blocked" in diagram_prompt
    assert "Direct explanation" not in diagram_prompt and "What to check" not in diagram_prompt
    assert not any(
        f"Verified entities: {generic}" in diagram_prompt
        for generic in nodes.SAMPLING_REJECTED_NODES
    )
    assert validation_calls
    assert dot, validation_calls
    assert nodes._balanced_delimiters(dot)
    assert all(entity in dot for entity in nodes.SAMPLING_ALLOWED_ENTITIES)
    assert not any(
        re.search(rf'label="{generic}(?:\s|\[)', dot, re.I)
        for generic in nodes.SAMPLING_REJECTED_NODES
    )
    assert 'shape=diamond' in dot
    assert 'move -> next' in dot and 'spec -> next' not in dot
    validation = {
        "allowed_entities": set(nodes.SAMPLING_ALLOWED_ENTITIES),
        "required_entities": {
            "Sampling Plan", "Sample Tests", "Sampling Status", "Failure Movement Rule"
        },
        "source_ids": {f"S{number}" for number in range(1, 7)},
        "decision_diagram": True,
    }
    assert nodes._validated_dot(dot, "LR", **validation)
    assert nodes._validated_dot(
        dot.replace("Sampling Plan [S2]", "Checks [S2]"), "LR", **validation
    ) is None
    isolated = dot.replace(
        "}", 'isolated [label="Container [S1]", shape=box];\n}', 1
    )
    assert nodes._validated_dot(isolated, "LR", **validation) is None


def test_v2_update_stream_finishes_before_final_answer_is_read(hard_pipeline) -> None:
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    events = list(
        graph.stream(
            {
                "messages": [HumanMessage(content=QUESTIONS["inspection"])],
                "retry_count": 0,
                "allow_diagrams": True,
            },
            config=config,
            stream_mode="updates",
            version="v2",
        )
    )
    final_state = dict(graph.get_state(config).values)

    assert events
    assert final_state["grounded"] is True
    assert final_state["answer"]
