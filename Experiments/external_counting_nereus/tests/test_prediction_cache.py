from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Experiments.external_counting_nereus.prediction_cache import (
    PredictionCache,
)


class PredictionCacheTests(unittest.TestCase):
    def test_completed_record_can_be_resumed(self) -> None:
        record = {
            "image_id": "0003",
            "method": "CountGD",
            "checkpoint_sha256": "abc",
            "text_query": "fish",
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = PredictionCache(Path(directory))
            cache.write(record)

            resumed = cache.read(
                "0003",
                method="CountGD",
                checkpoint_sha256="abc",
                text_query="fish",
            )

        self.assertEqual(resumed, record)

    def test_stale_checkpoint_record_is_not_resumed(self) -> None:
        record = {
            "image_id": "0003",
            "method": "CountGD",
            "checkpoint_sha256": "old",
            "text_query": "fish",
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = PredictionCache(Path(directory))
            cache.write(record)

            resumed = cache.read(
                "0003",
                method="CountGD",
                checkpoint_sha256="new",
                text_query="fish",
            )

        self.assertIsNone(resumed)

    def test_path_traversal_image_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = PredictionCache(Path(directory))
            with self.assertRaisesRegex(ValueError, "unsafe image ID"):
                cache.write(
                    {
                        "image_id": "../escape",
                        "method": "CountGD",
                        "checkpoint_sha256": "abc",
                        "text_query": "fish",
                    }
                )


if __name__ == "__main__":
    unittest.main()
