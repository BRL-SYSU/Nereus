from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Experiments.external_counting_nereus.io_utils import (
    atomic_write_jsonl,
    sha256_file,
)


class AtomicPredictionOutputTests(unittest.TestCase):
    def test_writer_returns_count_and_hash_for_complete_jsonl(self) -> None:
        records = [{"image_id": "a", "pred_count": 1}, {"image_id": "b", "pred_count": 2}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"

            result = atomic_write_jsonl(records, output)
            parsed = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(parsed, records)
            self.assertEqual(result["records"], 2)
            self.assertEqual(result["sha256"], sha256_file(output))
            self.assertFalse(any(Path(directory).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
