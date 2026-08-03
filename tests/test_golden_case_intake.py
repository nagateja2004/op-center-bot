import json
from pathlib import Path
import tempfile
import unittest

from scripts.add_golden_case import append_case, extract_case


class GoldenCaseIntakeTests(unittest.TestCase):
    def test_human_approved_json_is_validated_and_appended(self):
        body = """Review complete.
```json
{"id":"production_001","category":"production_failure","question":"What failed?","expected_status":"sufficient","expected_terms":["factory"]}
```"""
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "cases.json"
            dataset.write_text(
                '[\n  {"id":"direct_01","category":"direct","question":"What is a Factory?","expected_status":"sufficient","expected_terms":["factory"]}\n]\n',
                encoding="utf-8",
            )
            added = append_case(dataset, body, "https://github.example/issues/1")
            cases = json.loads(dataset.read_text(encoding="utf-8"))

        self.assertEqual(added["id"], "production_001")
        self.assertEqual(cases[-1]["source_issue"], "https://github.example/issues/1")

    def test_issue_without_valid_json_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_case("No approved case")


if __name__ == "__main__":
    unittest.main()
