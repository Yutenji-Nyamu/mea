import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mea.vqa_perturbations import PERTURBATIONS, build_proxy_images, file_sha256


class VQAPerturbationTests(unittest.TestCase):
    def test_transforms_are_deterministic_and_preserve_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (96, 64), (120, 130, 140)).save(source)
            before = file_sha256(source)
            first = build_proxy_images(source, root / "first", seed=7)
            second = build_proxy_images(source, root / "second", seed=7)
            self.assertEqual(file_sha256(source), before)
            self.assertEqual([item["condition"] for item in first], list(PERTURBATIONS))
            self.assertEqual(
                [item["derived_sha256"] for item in first],
                [item["derived_sha256"] for item in second],
            )
            self.assertEqual(first[0]["derived_sha256"], before)
            self.assertTrue(first[0]["paper_condition_equivalent"])
            self.assertTrue(
                all(not item["paper_condition_equivalent"] for item in first[1:])
            )


if __name__ == "__main__":
    unittest.main()
