import pytest

import src.nodes as nodes


@pytest.mark.parametrize(
    ("question", "canonical_terms", "manual_hint", "aspect_term"),
    [
        (
            "How are unique identifiers automatically generated for containers?",
            {"Numbering Rule", "Container"},
            "Modeling",
            "Numbering Rule",
        ),
        (
            "How are unique numbers automatically assigned to containers?",
            {"Numbering Rule", "Container"},
            "Modeling",
            "Numbering Rule",
        ),
        (
            "What happens before a new field value is accepted?",
            {"Validate Event"},
            "Designer",
            "Validate Event",
        ),
        (
            "How is a machine associated with a workflow step?",
            {"Spec", "Resource Group", "Resource"},
            "Modeling",
            "Resource",
        ),
        (
            "Explain the company-to-equipment hierarchy.",
            {"Factory Hierarchy"},
            "Modeling",
            "Factory Hierarchy",
        ),
        (
            "Explain Recipe Patterns.",
            {"Recipe Pattern"},
            "Modeling",
            "Recipe Pattern",
        ),
        (
            "Explain the resource-modeling sequence from resource categories and families.",
            {"Resource Modeling Sequence", "Resource"},
            "Execution Electronics",
            "Resource Modeling Sequence",
        ),
        (
            "What are Portal Studio controls?",
            {"Portal Studio Control"},
            "Portal Studio",
            "Portal Studio controls",
        ),
        (
            "How are roles and permissions configured?",
            {"Role", "Permission"},
            "Modeling",
            "Role and Permission",
        ),
    ],
)
def test_deterministic_planner_maps_indirect_opcenter_wording(
    question, canonical_terms, manual_hint, aspect_term
) -> None:
    plan = nodes._deterministic_plan(question)
    domain = nodes._domain_context(question)

    assert canonical_terms.issubset(plan.canonical_terms)
    assert manual_hint in plan.manual_hints
    assert any(aspect_term.casefold() in aspect.casefold() for aspect in plan.required_aspects)
    assert domain["domain_status"] == "in_scope"


def test_alias_catalog_contains_required_concepts() -> None:
    catalog = nodes._concept_catalog()

    assert {
        "Numbering Rule", "Physical Modeling Sequence", "Resource Modeling Sequence", "Factory Hierarchy",
        "Factory Level", "Inventory Location", "Equipment Resource",
        "Information Model", "Physical Model", "Process Model", "Execution Model",
        "CDO", "CLF", "Scalar Field", "List Field", "Field Event", "Validate Event",
        "Portal Studio Control", "Web Part", "Role", "Permission", "Employee",
        "Resource Family", "Resource Group", "Work Center", "Setup", "Resource Setup",
        "Recipe Matrix", "Resource Status Model",
        "Security Server", "SSL", "Resource", "Resource Group", "Spec", "Workflow",
        "Sampling Plan", "Sample Test", "Move Transaction", "Container", "Recipe",
        "Recipe Pattern",
    }.issubset(catalog)


def test_deterministic_planner_splits_compound_questions() -> None:
    fields = nodes._deterministic_plan(
        "What are scalar fields and list fields? What is the Validate event?"
    )
    portal = nodes._deterministic_plan(
        "What are Portal Studio controls and how is security configured?"
    )

    assert fields.required_aspects == ["Scalar Field", "List Field", "Validate Event"]
    assert {"Scalar Field", "List Field", "Validate Event"}.issubset(
        fields.canonical_terms
    )
    assert portal.required_aspects == ["Portal Studio controls", "security configuration"]


def test_security_configuration_splits_model_and_installation_aspects() -> None:
    plan = nodes._deterministic_plan(
        "How is security configured in Opcenter Execution Core?"
    )

    assert {"Role", "Permission", "Security Server", "SSL"}.issubset(
        plan.canonical_terms
    )
    assert plan.required_aspects == [
        "Role and Permission configuration",
        "Security Server and SSL configuration",
    ]
    assert {"Modeling", "Installation"}.issubset(plan.manual_hints)


def test_malformed_alias_catalog_warns_and_keeps_builtins(
    monkeypatch, tmp_path, caplog
) -> None:
    malformed = tmp_path / "opcenter_aliases.json"
    malformed.write_text("{", encoding="utf-8")
    monkeypatch.setattr(nodes, "ALIAS_CONFIG_PATH", malformed)
    nodes._concept_catalog.cache_clear()

    try:
        catalog = nodes._concept_catalog()
        assert "CDO" in catalog
        assert "Could not load Opcenter aliases" in caplog.text
    finally:
        nodes._concept_catalog.cache_clear()
