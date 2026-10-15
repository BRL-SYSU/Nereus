import unittest

import torch

from Experiments.p2pnet_nereus.p2pnet_adapter import (
    build_p2pnet,
    decode_predictions,
)


class P2PNetAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.model = build_p2pnet(training=False)

    def test_build_uses_official_p2pnet_class(self) -> None:
        self.assertEqual(self.model.__class__.__name__, "P2PNet")

    def test_training_criterion_weights_the_emitted_point_loss(self) -> None:
        _, criterion = build_p2pnet(training=True)

        self.assertEqual(criterion.weight_dict["loss_ce"], 1)
        self.assertEqual(criterion.weight_dict["loss_point"], 0.0002)
        self.assertNotIn("loss_points", criterion.weight_dict)

    def test_training_criterion_accepts_an_all_empty_batch(self) -> None:
        _, criterion = build_p2pnet(training=True)
        outputs = {
            "pred_logits": torch.zeros(2, 4, 2),
            "pred_points": torch.zeros(2, 4, 2),
        }
        targets = [
            {
                "point": torch.empty(0, 2),
                "labels": torch.empty(0, dtype=torch.int64),
            },
            {
                "point": torch.empty(0, 2),
                "labels": torch.empty(0, dtype=torch.int64),
            },
        ]

        losses = criterion(outputs, targets)

        self.assertTrue(torch.isfinite(losses["loss_ce"]))
        self.assertEqual(float(losses["loss_point"]), 0.0)

    def test_decode_returns_integer_count_and_xy_points(self) -> None:
        outputs = {
            "pred_logits": torch.tensor(
                [[[0.0, 2.0], [2.0, 0.0]]], dtype=torch.float32
            ),
            "pred_points": torch.tensor(
                [[[10.0, 20.0], [30.0, 40.0]]], dtype=torch.float32
            ),
        }

        decoded = decode_predictions(
            outputs, threshold=0.5, image_sizes=[(1920, 1080)]
        )[0]

        self.assertEqual(decoded["count"], 1)
        self.assertEqual(decoded["points_xy"], [[10.0, 20.0]])
        self.assertIsInstance(decoded["count"], int)

    def test_decode_discards_high_confidence_out_of_frame_point(self) -> None:
        outputs = {
            "pred_logits": torch.tensor(
                [[[0.0, 2.0], [0.0, 2.0]]], dtype=torch.float32
            ),
            "pred_points": torch.tensor(
                [[[10.0, 20.0], [32.0, 10.0]]], dtype=torch.float32
            ),
        }

        decoded = decode_predictions(
            outputs, threshold=0.5, image_sizes=[(32, 32)]
        )[0]

        self.assertEqual(decoded["count"], 1)
        self.assertEqual(decoded["points_xy"], [[10.0, 20.0]])

    def test_cpu_forward_has_matching_proposal_dimensions(self) -> None:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.zeros((1, 3, 128, 128)))

        self.assertEqual(tuple(outputs["pred_logits"].shape), (1, 1024, 2))
        self.assertEqual(tuple(outputs["pred_points"].shape), (1, 1024, 2))
        self.assertTrue(torch.isfinite(outputs["pred_logits"]).all())
        self.assertTrue(torch.isfinite(outputs["pred_points"]).all())


if __name__ == "__main__":
    unittest.main()
