import json
import os
import subprocess
import unittest
from pathlib import Path

from Experiments.p2pnet_nereus.calibrate import (
    evaluate_thresholds,
    select_threshold,
)
from Experiments.p2pnet_nereus.evaluate import (
    evaluate_records,
    render_table3_row,
)
from Experiments.p2pnet_nereus.grid_metrics import GRID_ORDER
from Experiments.p2pnet_nereus.infer import validate_prediction_record
from Experiments.p2pnet_nereus.train import (
    checkpoint_rank,
    ddp_options,
    make_checkpoint,
    validate_training_topology,
    weighted_loss,
)


class TrainingContractTests(unittest.TestCase):
    def test_checkpoint_rank_uses_mae_rmse_exact_then_epoch(self) -> None:
        first = {"mae": 10.0, "rmse": 20.0, "exact": 5.0, "epoch": 20}
        second = {"mae": 10.0, "rmse": 21.0, "exact": 8.0, "epoch": 10}

        self.assertLess(checkpoint_rank(first), checkpoint_rank(second))

    def test_checkpoint_rank_prefers_higher_exact_and_earlier_epoch(self) -> None:
        high_exact = {"mae": 10.0, "rmse": 20.0, "exact": 8.0, "epoch": 20}
        low_exact = {"mae": 10.0, "rmse": 20.0, "exact": 5.0, "epoch": 10}
        earlier = {"mae": 10.0, "rmse": 20.0, "exact": 8.0, "epoch": 10}

        self.assertLess(checkpoint_rank(high_exact), checkpoint_rank(low_exact))
        self.assertLess(checkpoint_rank(earlier), checkpoint_rank(high_exact))

    def test_checkpoint_contains_complete_provenance(self) -> None:
        checkpoint = make_checkpoint(
            model_state={"weight": 1},
            optimizer_state={"step": 2},
            epoch=5,
            config={"seed": 42},
            validation={"mae": 10.0},
            artifact_manifest_sha256="a" * 64,
        )

        self.assertEqual(
            set(checkpoint),
            {
                "model",
                "optimizer",
                "epoch",
                "config",
                "validation",
                "artifact_manifest_sha256",
            },
        )

    def test_checkpoint_rejects_invalid_artifact_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            make_checkpoint(
                model_state={},
                optimizer_state={},
                epoch=0,
                config={},
                validation={},
                artifact_manifest_sha256="not-a-hash",
            )

    def test_weighted_loss_uses_point_loss_key_and_coefficient(self) -> None:
        loss = weighted_loss({"loss_ce": 2.0, "loss_point": 5000.0})

        self.assertAlmostEqual(float(loss), 3.0)

    def test_ddp_allows_official_declared_but_unused_head_layers(self) -> None:
        self.assertTrue(ddp_options()["find_unused_parameters"])

    def test_two_and_four_gpu_topologies_preserve_global_batch_eight(self) -> None:
        self.assertEqual(validate_training_topology(2, 4), 8)
        self.assertEqual(validate_training_topology(4, 2), 8)

    def test_topology_rejects_a_smaller_global_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "global batch size 8"):
            validate_training_topology(2, 2)

    def test_topology_rejects_an_unsupported_world_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "two or four"):
            validate_training_topology(3, 2)


class RunnerContractTests(unittest.TestCase):
    def test_dry_run_reports_the_active_two_gpu_topology(self) -> None:
        runner = Path(__file__).resolve().parents[1] / "run_current_session.sh"
        environment = os.environ.copy()
        environment["P2PNET_DRY_RUN"] = "1"

        result = subprocess.run(
            ["timeout", "1", "bash", str(runner)],
            cwd=runner.parents[2],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "physical_gpus": "0,3",
                "world_size": 2,
                "batch_size_per_gpu": 4,
                "global_batch_size": 8,
            },
        )


class CalibrationContractTests(unittest.TestCase):
    def test_threshold_tie_break_prefers_larger_threshold(self) -> None:
        candidates = [
            {"threshold": 0.45, "mae": 1.0, "rmse": 2.0, "exact": 10.0},
            {"threshold": 0.50, "mae": 1.0, "rmse": 2.0, "exact": 10.0},
        ]

        self.assertEqual(select_threshold(candidates)["threshold"], 0.50)

    def test_threshold_selection_prefers_mae_before_other_metrics(self) -> None:
        candidates = [
            {"threshold": 0.45, "mae": 1.0, "rmse": 3.0, "exact": 0.0},
            {"threshold": 0.50, "mae": 2.0, "rmse": 2.0, "exact": 100.0},
        ]

        self.assertEqual(select_threshold(candidates)["threshold"], 0.45)

    def test_one_candidate_pass_scores_all_nineteen_thresholds(self) -> None:
        candidates = evaluate_thresholds(
            [
                {
                    "image_id": "a",
                    "image_width": 1920,
                    "image_height": 1080,
                    "target_count": 1,
                    "target_densest_grid_region": "top-left",
                    "candidate_points_xy": [[10.0, 10.0]],
                    "candidate_scores": [0.5],
                }
            ]
        )

        self.assertEqual(len(candidates), 19)
        self.assertEqual(select_threshold(candidates)["threshold"], 0.5)


class PredictionRecordTests(unittest.TestCase):
    def test_valid_record_satisfies_count_and_grid_invariants(self) -> None:
        grid = {name: 0 for name in GRID_ORDER}
        grid["top-left"] = 1
        record = {
            "image_id": "0003",
            "pred_count": 1,
            "pred_points_xy": [[10.0, 20.0]],
            "pred_grid_counts": grid,
            "pred_densest_grid_region": "top-left",
            "threshold": 0.5,
        }

        validate_prediction_record(record)

    def test_record_rejects_count_point_length_mismatch(self) -> None:
        grid = {name: 0 for name in GRID_ORDER}
        with self.assertRaisesRegex(ValueError, "point length"):
            validate_prediction_record(
                {
                    "image_id": "0003",
                    "pred_count": 1,
                    "pred_points_xy": [],
                    "pred_grid_counts": grid,
                    "pred_densest_grid_region": "top-left",
                    "threshold": 0.5,
                }
            )

    def test_record_rejects_incorrect_grid_counts(self) -> None:
        grid = {name: 0 for name in GRID_ORDER}
        grid["center"] = 1
        with self.assertRaisesRegex(ValueError, "grid"):
            validate_prediction_record(
                {
                    "image_id": "0003",
                    "pred_count": 1,
                    "pred_points_xy": [[10.0, 20.0]],
                    "pred_grid_counts": grid,
                    "pred_densest_grid_region": "center",
                    "threshold": 0.5,
                }
            )


class EvaluationContractTests(unittest.TestCase):
    def test_fixture_aggregates_six_metrics_and_renders_na_cells(self) -> None:
        targets = [
            {
                "image_id": "a",
                "width": 1920,
                "height": 1080,
                "count": 2,
                "points_xy": [[10.0, 10.0], [700.0, 400.0]],
            },
            {
                "image_id": "b",
                "width": 1920,
                "height": 1080,
                "count": 0,
                "points_xy": [],
            },
        ]
        first_grid = {name: 0 for name in GRID_ORDER}
        first_grid["top-left"] = 1
        second_grid = {name: 0 for name in GRID_ORDER}
        second_grid["bottom-right"] = 1
        predictions = [
            {
                "image_id": "a",
                "pred_count": 1,
                "pred_points_xy": [[10.0, 10.0]],
                "pred_grid_counts": first_grid,
                "pred_densest_grid_region": "top-left",
                "threshold": 0.5,
            },
            {
                "image_id": "b",
                "pred_count": 1,
                "pred_points_xy": [[1900.0, 1000.0]],
                "pred_grid_counts": second_grid,
                "pred_densest_grid_region": "bottom-right",
                "threshold": 0.5,
            },
        ]

        metrics = evaluate_records(targets, predictions, expected_size=2)

        self.assertEqual(
            metrics,
            {
                "mae": 1.0,
                "rmse": 1.0,
                "exact": 0.0,
                "acc_at_5": 0.0,
                "acc_at_10": 0.0,
                "densest_cell": 50.0,
            },
        )
        self.assertIn("| N/A | N/A | N/A |", render_table3_row(metrics))

    def test_evaluation_joins_by_exact_image_id_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "image ID"):
            evaluate_records(
                [
                    {
                        "image_id": "a",
                        "width": 1920,
                        "height": 1080,
                        "count": 0,
                        "points_xy": [],
                    }
                ],
                [],
                expected_size=1,
            )


if __name__ == "__main__":
    unittest.main()
