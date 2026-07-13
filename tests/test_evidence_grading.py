import src.nodes as nodes


def document(number, section, text, *, content_type="text"):
    return {
        "chunk_id": f"e{number}",
        "text": text,
        "content_type": content_type,
        "metadata": {
            "evidence_id": f"e{number}",
            "manual": "Opcenter Manual",
            "section": section,
            "pdf_page": number,
        },
        "retrieval_scores": {"final_score": 1 / number},
    }


def use_heuristic(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "call_structured",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            nodes.GroqRequestError("rate_limit", "grader", 429)
        ),
    )


def test_related_field_evidence_does_not_directly_support_every_aspect(monkeypatch):
    use_heuristic(monkeypatch)
    aspects = ["Scalar Field", "List Field", "Validate Event"]
    evidence = [
        document(1, "Logic", "CLFs can validate input before logic continues."),
        document(2, "Fields", "Scalar and list fields are supported by CDO fields."),
        document(
            3,
            "Defining List Fields",
            "List fields can contain Object references selected from a configured CDO.",
        ),
    ]

    result = nodes.grade_evidence(
        {
            "standalone_question": (
                "What are scalar fields and list fields? What is the Validate event of a CDO field?"
            ),
            "domain_status": "in_scope",
            "canonical_terms": ["Scalar Field", "List Field", "Validate Event", "CDO"],
            "required_aspects": aspects,
            "aspect_documents": {aspect: evidence for aspect in aspects},
            "retry_count": 0,
        }
    )

    assert result["coverage"]["Scalar Field"] in {"partial", "retry"}
    assert result["coverage"]["List Field"] in {"sufficient", "partial"}
    assert result["coverage"]["Validate Event"] == "retry"
    assert result["evidence_status"] == "retry"
    assert "Validate Event" in result["missing_aspects"]
    assert "List Field" not in result["missing_aspects"]


def test_related_portal_and_ssl_evidence_is_not_sufficient(monkeypatch):
    use_heuristic(monkeypatch)
    aspects = ["Portal Studio Control", "Security configuration"]
    evidence = [
        document(1, "Portal Studio Pages", "Portal Studio pages contain web parts."),
        document(2, "SSL", "SSL encrypts traffic between installed components."),
    ]

    result = nodes.grade_evidence(
        {
            "standalone_question": "What are Portal Studio controls? How is security configured?",
            "domain_status": "in_scope",
            "canonical_terms": ["Portal Studio Control", "Role", "Permission", "SSL"],
            "required_aspects": aspects,
            "aspect_documents": {aspect: evidence for aspect in aspects},
            "retry_count": 0,
        }
    )

    assert result["coverage"]["Portal Studio Control"] == "retry"
    assert result["coverage"]["Security configuration"] in {"partial", "retry"}
    assert result["evidence_status"] == "retry"
    assert "sufficient" not in result["coverage"].values()
