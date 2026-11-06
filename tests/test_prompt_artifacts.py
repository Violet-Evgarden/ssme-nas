"""Read-only checks for the published offline prompt artifacts."""
import json
from pathlib import Path
import re
import unittest


class PromptArtifactsTest(unittest.TestCase):
    root = Path(__file__).resolve().parents[1] / "prompts"

    def test_requests_and_label_free_inputs(self):
        for name in ("nb101", "nb201"):
            with self.subTest(space=name):
                raw = (self.root / f"{name}_flash_request.json").read_text(encoding="utf-8")
                request = json.loads(raw)
                self.assertEqual(request["model"], "deepseek-flash")
                self.assertEqual(request["temperature"], 0)
                self.assertEqual(request["max_tokens"], 4096)
                self.assertEqual(request["thinking"], {"type": "disabled"})
                self.assertEqual(request["response_format"], {"type": "json_object"})
                self.assertFalse(request["stream"])
                self.assertEqual([m["role"] for m in request["messages"]], ["system", "user"])
                payload = json.loads(request["messages"][1]["content"])
                fields = set(payload["feature_catalog"])
                self.assertEqual(len(fields), 22)
                self.assertEqual(len(payload["examples"]), 16)
                self.assertTrue(payload["all_examples_are_unlabelled"])
                self.assertTrue(payload["performance_fields_not_supplied"])
                for example in payload["examples"]:
                    self.assertEqual(example["performance"], "NOT_QUERIED_OR_PROVIDED")
                    self.assertEqual(set(example["structural_fields"]), fields)
                self.assertIsNone(re.search(r"sk-[A-Za-z0-9]{16,}|Bearer |/data/Xu|C:\\\\Users|api_key|password", raw))

    def test_response_schemas_match_request_catalogs(self):
        for name in ("nb101", "nb201"):
            with self.subTest(space=name):
                request = json.loads((self.root / f"{name}_flash_request.json").read_text(encoding="utf-8"))
                payload = json.loads(request["messages"][1]["content"])
                schema = json.loads((self.root / f"{name}_response.schema.json").read_text(encoding="utf-8"))
                concepts = schema["properties"]["concepts"]
                self.assertEqual(concepts["minItems"], 8 if name == "nb101" else 12)
                self.assertEqual(concepts["maxItems"], 12)
                features = concepts["items"]["properties"]["features"]
                self.assertEqual(features["minItems"], 1)
                self.assertEqual(features["maxItems"], 3)
                self.assertTrue(features["uniqueItems"])
                self.assertEqual(set(features["items"]["enum"]), set(payload["feature_catalog"]))


if __name__ == "__main__":
    unittest.main()
