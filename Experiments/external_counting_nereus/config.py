"""Immutable default paths for the external counting-model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EXPECTED_TEST_SIZE = 1129
TEXT_QUERY = "fish"
COUNTGD_THRESHOLD = 0.23
CLIPCOUNT_SCALE_FACTOR = 60.0


@dataclass(frozen=True)
class ExternalExperimentPaths:
    project_root: Path
    model_root: Path
    experiment_root: Path
    test_images_manifest: Path
    test_targets: Path
    output_root: Path

    @classmethod
    def defaults(cls) -> "ExternalExperimentPaths":
        project_root = Path(__file__).resolve().parents[2]
        experiment_root = (
            project_root / "Experiments" / "external_counting_nereus"
        )
        shared_data = (
            project_root / "Experiments" / "p2pnet_nereus" / "data"
        )
        return cls(
            project_root=project_root,
            model_root=Path("models"),
            experiment_root=experiment_root,
            test_images_manifest=shared_data / "test_images.jsonl",
            test_targets=shared_data / "test.jsonl",
            output_root=experiment_root / "outputs",
        )
