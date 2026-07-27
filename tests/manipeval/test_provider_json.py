import unittest

from mea.providers.json_response import ProviderJSONError, extract_json_response
from mea.taskgen.prototype import (
    TaskGenError,
    extract_json_response as extract_taskgen_json_response,
)


class ProviderJSONTests(unittest.TestCase):
    def test_extracts_strict_fenced_and_embedded_objects(self) -> None:
        self.assertEqual(extract_json_response('{"value": 1}'), {"value": 1})
        self.assertEqual(
            extract_json_response('```json\n{"value": 2}\n```'),
            {"value": 2},
        )
        self.assertEqual(
            extract_json_response('result: {"value": 3}'),
            {"value": 3},
        )

    def test_rejects_non_object_or_invalid_response(self) -> None:
        for response in ("[]", "not json"):
            with self.subTest(response=response):
                with self.assertRaises(ProviderJSONError):
                    extract_json_response(response)

    def test_taskgen_wrapper_preserves_legacy_error_contract(self) -> None:
        with self.assertRaises(TaskGenError):
            extract_taskgen_json_response("not json")


if __name__ == "__main__":
    unittest.main()
