"""Append one human-approved JSON case from a GitHub issue to the golden dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


ALLOWED_STATUSES = {"sufficient", "in_scope_insufficient", "out_of_scope"}
ALLOWED_FIELDS = {
    "id",
    "category",
    "question",
    "context_question",
    "expected_status",
    "expected_terms",
    "required_terms",
    "expected_manuals",
    "expected_diagram",
}


def extract_case(issue_body: str) -> dict[str, Any]:
    for block in reversed(re.findall(r"```json\s*(\{.*?\})\s*```", issue_body, re.S | re.I)):
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and {"id", "question", "expected_status"} <= value.keys():
            return value
    raise ValueError("Issue must contain a JSON case block with id, question, and expected_status")


def validate_case(value: dict[str, Any], existing_ids: set[str]) -> dict[str, Any]:
    unknown = set(value) - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"Unsupported case fields: {', '.join(sorted(unknown))}")
    case_id = str(value.get("id", ""))
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", case_id):
        raise ValueError("id must use 3-64 lowercase letters, numbers, or underscores")
    if case_id in existing_ids:
        raise ValueError(f"Golden case id already exists: {case_id}")
    question = str(value.get("question", "")).strip()
    if not question or len(question) > 1_000:
        raise ValueError("question must contain 1-1000 characters")
    status = str(value.get("expected_status", ""))
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"expected_status must be one of {sorted(ALLOWED_STATUSES)}")
    category = str(value.get("category", "production_failure")).strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", category):
        raise ValueError("category must use lowercase letters, numbers, or underscores")

    normalized: dict[str, Any] = {
        "id": case_id,
        "category": category,
        "question": question,
        "expected_status": status,
    }
    if value.get("context_question") is not None:
        context_question = str(value["context_question"]).strip()
        if not context_question or len(context_question) > 1_000:
            raise ValueError("context_question must contain 1-1000 characters")
        normalized["context_question"] = context_question
    for field in ("expected_terms", "required_terms", "expected_manuals"):
        items = value.get(field, [])
        if not isinstance(items, list) or len(items) > 20:
            raise ValueError(f"{field} must be a list containing at most 20 strings")
        cleaned = [str(item).strip() for item in items]
        if any(not item or len(item) > 200 for item in cleaned):
            raise ValueError(f"{field} contains an empty or oversized value")
        if cleaned or field == "expected_terms":
            normalized[field] = cleaned
    if "expected_diagram" in value:
        if not isinstance(value["expected_diagram"], bool):
            raise ValueError("expected_diagram must be true or false")
        normalized["expected_diagram"] = value["expected_diagram"]
    return normalized


def append_case(dataset_path: Path, issue_body: str, source_issue: str = "") -> dict[str, Any]:
    raw = dataset_path.read_text(encoding="utf-8")
    cases = json.loads(raw)
    if not isinstance(cases, list):
        raise ValueError("Golden dataset must be a JSON array")
    value = validate_case(extract_case(issue_body), {str(case.get("id")) for case in cases})
    if source_issue:
        value["source_issue"] = source_issue
    closing = raw.rfind("]")
    if closing < 0:
        raise ValueError("Golden dataset is missing its closing array bracket")
    prefix = raw[:closing].rstrip()
    separator = "" if prefix.endswith("[") else ","
    compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    dataset_path.write_text(f"{prefix}{separator}\n  {compact}\n]\n", encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("tests/evaluation_questions.json"))
    parser.add_argument("--issue-body", type=Path, required=True)
    parser.add_argument("--source-issue", default="")
    args = parser.parse_args()
    value = append_case(
        args.dataset,
        args.issue_body.read_text(encoding="utf-8"),
        args.source_issue,
    )
    print(f"Added human-approved golden case: {value['id']}")


if __name__ == "__main__":
    main()
