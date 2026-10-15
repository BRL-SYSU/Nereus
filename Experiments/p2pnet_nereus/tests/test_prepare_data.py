import unittest
import json

import numpy as np

from Experiments.p2pnet_nereus.config import ExperimentPaths
from Experiments.p2pnet_nereus.prepare_data import (
    audit_records,
    build_split_records,
    load_points,
)


class PrepareDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = ExperimentPaths.from_project_root()
        cls.records = build_split_records(cls.paths)

    def test_fixed_split_sizes_and_disjoint_ids(self) -> None:
        self.assertEqual(
            {name: len(rows) for name, rows in self.records.items()},
            {"train": 3945, "val": 563, "test": 1129},
        )
        train_ids = {row["image_id"] for row in self.records["train"]}
        val_ids = {row["image_id"] for row in self.records["val"]}
        test_ids = {row["image_id"] for row in self.records["test"]}
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)

    def test_points_are_xy_and_count_consistent(self) -> None:
        points = load_points(self.paths.annotation_dir / "0003.json")

        self.assertEqual(points.shape, (86, 2))
        np.testing.assert_array_equal(points[0], np.array([28.0, 120.0]))
        self.assertEqual(points.dtype, np.float32)

    def test_inclusive_source_boundary_is_mapped_to_last_pixel(self) -> None:
        points = load_points(self.paths.annotation_dir / "0111.json")

        self.assertTrue(np.any(np.all(points == [1919.0, 656.0], axis=1)))
        self.assertLess(points[:, 0].max(), 1920)
        self.assertLess(points[:, 1].max(), 1080)

    def test_audit_accepts_all_real_records(self) -> None:
        audit_records(self.records)

    def test_audit_rejects_an_out_of_bounds_point(self) -> None:
        bad_record = {
            "image_id": "bad",
            "image_path": str(self.paths.image_dir / "0003.jpg"),
            "annotation_path": str(self.paths.annotation_dir / "0003.json"),
            "width": 1920,
            "height": 1080,
            "count": 1,
            "points_xy": [[1920.0, 20.0]],
        }

        with self.assertRaisesRegex(ValueError, "out-of-bounds point"):
            audit_records({"train": [bad_record], "val": [], "test": []},
                          expected_sizes=None)

    def test_test_inference_manifest_contains_no_label_fields(self) -> None:
        manifest = self.paths.manifest_dir / "test_images.jsonl"
        records = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]

        self.assertEqual(len(records), 1129)
        self.assertEqual(
            set(records[0]),
            {"image_id", "image_path", "width", "height"},
        )


if __name__ == "__main__":
    unittest.main()
