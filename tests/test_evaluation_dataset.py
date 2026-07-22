from collections import Counter
import json
from pathlib import Path


def test_evaluation_question_category_counts() -> None:
    questions = json.loads(Path("tests/evaluation_questions.json").read_text(encoding="utf-8"))
    counts = Counter(question["category"] for question in questions)

    assert counts["direct"] >= 8
    assert counts["indirect"] >= 8
    assert counts["follow_up"] >= 5
    assert counts["procedure"] >= 5
    assert counts["table_field"] >= 5
    assert counts["comparison"] >= 4
    assert counts["unsupported"] >= 3
    assert counts["irrelevant"] >= 3
    assert counts["electronics"] >= 5
    assert counts["discrete"] >= 5
