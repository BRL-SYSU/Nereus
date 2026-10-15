import math
import unittest

import numpy as np

from Experiments.p2pnet_nereus.grid_metrics import (
    GRID_ORDER,
    aggregate_metrics,
    densest_region,
    grid_counts,
)


class GridMetricTests(unittest.TestCase):
    def test_internal_boundaries_go_right_and_down(self) -> None:
        points = np.array(
            [[640.0, 360.0], [1280.0, 720.0]], dtype=np.float32
        )

        counts = grid_counts(points, width=1920, height=1080)

        self.assertEqual(counts["center"], 1)
        self.assertEqual(counts["bottom-right"], 1)
        self.assertEqual(sum(counts.values()), 2)

    def test_densest_tie_is_row_major(self) -> None:
        counts = {name: 0 for name in GRID_ORDER}
        counts["top-center"] = 2
        counts["center-left"] = 2

        self.assertEqual(densest_region(counts), "top-center")

    def test_all_zero_grid_selects_top_left(self) -> None:
        self.assertEqual(
            densest_region({name: 0 for name in GRID_ORDER}), "top-left"
        )

    def test_relative_tolerances_and_zero_count(self) -> None:
        result = aggregate_metrics(
            targets=[
                {"count": 100, "densest": "center"},
                {"count": 0, "densest": "top-left"},
            ],
            predictions=[
                {"count": 105, "densest": "center"},
                {"count": 1, "densest": "top-left"},
            ],
        )

        self.assertEqual(result["mae"], 3.0)
        self.assertAlmostEqual(result["rmse"], math.sqrt(13.0))
        self.assertEqual(result["exact"], 0.0)
        self.assertEqual(result["acc_at_5"], 50.0)
        self.assertEqual(result["acc_at_10"], 50.0)
        self.assertEqual(result["densest_cell"], 100.0)

    def test_grid_rejects_out_of_bounds_predictions(self) -> None:
        with self.assertRaisesRegex(ValueError, "out-of-bounds point"):
            grid_counts(
                np.array([[1920.0, 100.0]], dtype=np.float32),
                width=1920,
                height=1080,
            )


if __name__ == "__main__":
    unittest.main()
