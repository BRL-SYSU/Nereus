from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Experiments.external_counting_nereus.dataset import read_image_manifest


class LabelIsolationTests(unittest.TestCase):
    def test_image_manifest_exposes_only_label_free_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "test_images.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "image_id": "sample",
                        "image_path": "/images/sample.jpg",
                        "width": 1920,
                        "height": 1080,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = read_image_manifest(manifest)

        self.assertEqual(
            records,
            [
                {
                    "image_id": "sample",
                    "image_path": "/images/sample.jpg",
                    "width": 1920,
                    "height": 1080,
                }
            ],
        )

    def test_manifest_with_count_or_points_is_rejected(self) -> None:
        for prohibited in ("count", "points_xy"):
            with self.subTest(prohibited=prohibited):
                with tempfile.TemporaryDirectory() as directory:
                    manifest = Path(directory) / "leaky.jsonl"
                    record = {
                        "image_id": "sample",
                        "image_path": "/images/sample.jpg",
                        "width": 1920,
                        "height": 1080,
                        prohibited: 7,
                    }
                    manifest.write_text(
                        json.dumps(record) + "\n", encoding="utf-8"
                    )

                    with self.assertRaisesRegex(
                        ValueError, "label-bearing field"
                    ):
                        read_image_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
