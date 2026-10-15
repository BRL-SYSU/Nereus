from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))

from count_range_eval import (  # noqa: E402
    assign_count_range,
    bootstrap_rows,
    compute_metrics,
    micro_set_f1,
)


def row(gt: int, pred: int, densest_match: bool = True) -> dict:
    return {
        "gt_count": gt,
        "pred_count": pred,
        "valid_prediction": True,
        "densest_cell_gt": "center",
        "densest_cell_pred": "center" if densest_match else "top-left",
        "camouflage_gt_concepts": {"ambiguous", "dominant:ambiguous"},
        "camouflage_pred_concepts": {
            "ambiguous",
            "dominant:ambiguous",
        },
    }


class CountRangeTests(unittest.TestCase):
    def test_exact_required_boundaries(self) -> None:
        expected = {
            0: "0-50",
            50: "0-50",
            51: "51-100",
            100: "51-100",
            101: "101-200",
            200: "101-200",
            201: ">200",
        }
        for count, count_range in expected.items():
            self.assertEqual(assign_count_range(count), count_range)
        with self.assertRaises(ValueError):
            assign_count_range(-1)

    def test_metric_definitions_use_relative_thresholds(self) -> None:
        metrics = compute_metrics([row(10, 10), row(20, 18, False)])
        self.assertEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2.0))
        self.assertEqual(metrics["exact_accuracy_pct"], 50.0)
        self.assertEqual(metrics["acc_at_5_pct"], 50.0)
        self.assertEqual(metrics["acc_at_10_pct"], 100.0)
        self.assertAlmostEqual(metrics["nmae_wape_pct"], 100.0 * 2 / 30)
        self.assertEqual(metrics["bias"], -1.0)
        self.assertEqual(metrics["under_count_rate_pct"], 50.0)
        self.assertEqual(metrics["densest_cell_accuracy_pct"], 50.0)

    def test_micro_set_f1_matches_global_counts(self) -> None:
        targets = [{"a", "b"}, {"c"}]
        predictions = [{"a", "x"}, {"c"}]
        self.assertAlmostEqual(
            micro_set_f1(targets, predictions),
            100.0 * 4 / 6,
        )

    def test_bootstrap_is_reproducible(self) -> None:
        rows = [row(10, 9), row(20, 18), row(30, 35)]
        first = bootstrap_rows(rows, repetitions=50, seed=42)
        second = bootstrap_rows(rows, repetitions=50, seed=42)
        self.assertEqual(first, second)
        self.assertEqual({item["metric"] for item in first}, {
            "mae",
            "nmae_wape_pct",
            "acc_at_10_pct",
            "densest_cell_accuracy_pct",
        })


if __name__ == "__main__":
    unittest.main()
