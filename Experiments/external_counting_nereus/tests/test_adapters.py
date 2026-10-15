from __future__ import annotations

import unittest

import numpy as np

from Experiments.external_counting_nereus.adapters import (
    clip_density_to_prediction,
    countgd_boxes_to_prediction,
    round_nonnegative_count,
    validate_prediction_record,
)


class CountRoundingTests(unittest.TestCase):
    def test_half_values_round_up_and_negative_values_clamp_to_zero(self) -> None:
        self.assertEqual(round_nonnegative_count(-0.2), 0)
        self.assertEqual(round_nonnegative_count(2.49), 2)
        self.assertEqual(round_nonnegative_count(2.50), 3)


class CountGdAdapterTests(unittest.TestCase):
    def test_threshold_is_strict_and_centers_map_to_original_pixels(self) -> None:
        record = countgd_boxes_to_prediction(
            image_id="sample",
            width=300,
            height=150,
            boxes_cxcywh=np.asarray(
                [
                    [0.10, 0.20, 0.05, 0.05],
                    [0.50, 0.50, 0.10, 0.10],
                    [0.90, 0.80, 0.10, 0.10],
                ],
                dtype=np.float32,
            ),
            scores=np.asarray([0.23, 0.23001, 0.90], dtype=np.float32),
            confidence_threshold=0.23,
            checkpoint_sha256="abc",
        )

        self.assertEqual(record["pred_count"], 2)
        np.testing.assert_allclose(
            record["pred_points_xy"],
            [[150.0, 75.0], [270.0, 120.0]],
            rtol=0.0,
            atol=1e-5,
        )
        self.assertEqual(record["pred_grid_counts"]["center"], 1)
        self.assertEqual(record["pred_grid_counts"]["bottom-right"], 1)
        self.assertEqual(record["pred_densest_grid_region"], "center")
        validate_prediction_record(record)


class ClipCountAdapterTests(unittest.TestCase):
    def test_density_mass_drives_integer_count_and_densest_cell(self) -> None:
        density = np.zeros((6, 6), dtype=np.float64)
        density[0:2, 0:2] = 2.0
        density[4:6, 4:6] = 3.0

        record = clip_density_to_prediction(
            image_id="sample",
            width=1920,
            height=1080,
            density_map=density,
            scale_factor=4.0,
            checkpoint_sha256="def",
        )

        self.assertAlmostEqual(record["pred_count_float"], 5.0)
        self.assertEqual(record["pred_count"], 5)
        self.assertAlmostEqual(record["pred_grid_mass"]["top-left"], 2.0)
        self.assertAlmostEqual(record["pred_grid_mass"]["bottom-right"], 3.0)
        self.assertEqual(record["pred_densest_grid_region"], "bottom-right")
        validate_prediction_record(record)


if __name__ == "__main__":
    unittest.main()
