from __future__ import annotations

import math
import unittest

from Experiments.external_counting_nereus.evaluate import (
    evaluate_records,
    render_table3,
    render_table3_row,
)


def _clip_record(
    image_id: str,
    *,
    count: int,
    densest: str,
) -> dict[str, object]:
    masses = {
        "top-left": 0.0,
        "top-center": 0.0,
        "top-right": 0.0,
        "center-left": 0.0,
        "center": 0.0,
        "center-right": 0.0,
        "bottom-left": 0.0,
        "bottom-center": 0.0,
        "bottom-right": 0.0,
    }
    masses[densest] = float(count)
    return {
        "method": "CLIP-Count",
        "setting": "External pretrained Direct-ZS",
        "image_id": image_id,
        "image_width": 100,
        "image_height": 100,
        "pred_count_float": float(count),
        "pred_count": count,
        "pred_grid_mass": masses,
        "pred_densest_grid_region": densest,
        "text_query": "fish",
        "checkpoint_sha256": "checkpoint",
    }


class ExternalEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = [
            {
                "image_id": "a",
                "width": 100,
                "height": 100,
                "count": 10,
                "points_xy": [[1.0, 1.0]] * 10,
            },
            {
                "image_id": "b",
                "width": 100,
                "height": 100,
                "count": 20,
                "points_xy": [[90.0, 90.0]] * 20,
            },
        ]
        self.predictions = [
            _clip_record("a", count=10, densest="top-left"),
            _clip_record("b", count=18, densest="bottom-right"),
        ]

    def test_metrics_use_relative_five_and_ten_percent_tolerances(self) -> None:
        metrics = evaluate_records(
            self.targets, self.predictions, expected_size=2
        )

        self.assertEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2.0))
        self.assertEqual(metrics["exact"], 50.0)
        self.assertEqual(metrics["acc_at_5"], 50.0)
        self.assertEqual(metrics["acc_at_10"], 100.0)
        self.assertEqual(metrics["densest_cell"], 100.0)

    def test_missing_prediction_id_is_rejected(self) -> None:
        wrong_ids = [
            self.predictions[0],
            _clip_record("c", count=18, densest="bottom-right"),
        ]
        with self.assertRaisesRegex(ValueError, "image ID set mismatch"):
            evaluate_records(self.targets, wrong_ids, expected_size=2)

    def test_table_row_marks_unsupported_tracks_na(self) -> None:
        row = render_table3_row(
            "CLIP-Count",
            "FSC-147 pretrained Direct-ZS, text=fish",
            {
                "mae": 1.0,
                "rmse": math.sqrt(2.0),
                "exact": 50.0,
                "acc_at_5": 50.0,
                "acc_at_10": 100.0,
                "densest_cell": 100.0,
            },
        )

        self.assertEqual(
            row,
            "| CLIP-Count | FSC-147 pretrained Direct-ZS, text=fish | "
            "1.00 | 1.41 | 50.00 | 50.00 | 100.00 | 100.00 | "
            "N/A | N/A | N/A |",
        )

    def test_full_table_contains_requested_track_a_columns(self) -> None:
        table = render_table3(
            [
                render_table3_row(
                    "CLIP-Count",
                    "FSC-147 pretrained Direct-ZS, text=fish",
                    {
                        "mae": 1.0,
                        "rmse": math.sqrt(2.0),
                        "exact": 50.0,
                        "acc_at_5": 50.0,
                        "acc_at_10": 100.0,
                        "densest_cell": 100.0,
                    },
                )
            ]
        )

        self.assertIn("| Backbone | Setting | MAE | RMSE | Exact |", table)
        self.assertIn("| Acc@5 | Acc@10 | Densest Cell |", table)
        self.assertIn("| Camouflage F1 | Track B | Track C |", table)


if __name__ == "__main__":
    unittest.main()
