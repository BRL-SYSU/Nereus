"""Immutable paths and constants for the Nereus P2PNet experiment."""

from dataclasses import dataclass
from pathlib import Path


EXPECTED_SPLIT_SIZES = {"train": 3945, "val": 563, "test": 1129}
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
MODEL_ROOT = Path("models/P2PNet")


@dataclass(frozen=True)
class ExperimentPaths:
    project_root: Path
    split_file: Path
    image_dir: Path
    annotation_dir: Path
    manifest_dir: Path
    model_root: Path

    @classmethod
    def from_project_root(
        cls, project_root: Path | None = None
    ) -> "ExperimentPaths":
        root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        return cls(
            project_root=root,
            split_file=root
            / "Experiments"
            / "canonical_dataset"
            / "splits"
            / "splits.json",
            image_dir=root / "test_dataset" / "images",
            annotation_dir=root / "test_dataset" / "final_annotations",
            manifest_dir=root / "Experiments" / "p2pnet_nereus" / "data",
            model_root=MODEL_ROOT,
        )
