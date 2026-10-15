import os
import subprocess
import sys
import unittest
from pathlib import Path


class GpuDeterminismTests(unittest.TestCase):
    def test_train_import_sets_the_cublas_workspace_contract(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; "
                    "import Experiments.p2pnet_nereus.train; "
                    "print(os.environ.get('CUBLAS_WORKSPACE_CONFIG'))"
                ),
            ],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), ":4096:8")

    @unittest.skipUnless(
        os.environ.get("P2PNET_RUN_CUDA_TESTS") == "1",
        "set P2PNET_RUN_CUDA_TESTS=1 for the CUDA integration test",
    )
    def test_matching_and_classification_emit_no_determinism_warning(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        environment = os.environ.copy()
        environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
        child = """
from Experiments.p2pnet_nereus.train import seed_everything, weighted_loss
from Experiments.p2pnet_nereus.p2pnet_adapter import build_p2pnet
import torch

seed_everything(42, 0)
_, criterion = build_p2pnet(training=True)
criterion = criterion.cuda()
outputs = {
    "pred_logits": torch.zeros(1, 16, 2, device="cuda", requires_grad=True),
    "pred_points": torch.zeros(1, 16, 2, device="cuda", requires_grad=True),
}
targets = [{
    "point": torch.tensor([[10.0, 20.0]], device="cuda"),
    "labels": torch.ones(1, dtype=torch.int64, device="cuda"),
}]
losses = criterion(outputs, targets)
weighted_loss(losses).backward()
print("finite", all(torch.isfinite(value) for value in losses.values()))
"""
        result = subprocess.run(
            [sys.executable, "-c", child],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("finite True", result.stdout)
        self.assertNotIn("operation is not deterministic", result.stderr)
        self.assertNotIn(
            "does not have a deterministic implementation", result.stderr
        )


if __name__ == "__main__":
    unittest.main()
