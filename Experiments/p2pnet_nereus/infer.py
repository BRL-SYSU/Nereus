"""Frozen-checkpoint inference without access to test point labels."""

import argparse
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .artifacts import sha256_file
from .config import IMAGE_HEIGHT, IMAGE_WIDTH, MODEL_ROOT, ExperimentPaths
from .dataset import transform_eval
from .grid_metrics import GRID_ORDER, densest_region, grid_counts
from .p2pnet_adapter import build_p2pnet, decode_predictions


class NereusImageDataset(Dataset):
    """Image-only manifest view that deliberately discards annotation fields."""

    def __init__(self, manifest_path: Path, limit: int | None = None) -> None:
        records: list[dict[str, object]] = []
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            source = json.loads(line)
            records.append(
                {
                    "image_id": str(source["image_id"]),
                    "image_path": str(source["image_path"]),
                    "width": int(source["width"]),
                    "height": int(source["height"]),
                }
            )
        self.records = records[:limit] if limit is not None else records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, dict[str, object]]:
        record = self.records[index]
        with Image.open(str(record["image_path"])) as source:
            image, size = transform_eval(source)
        if size != (record["width"], record["height"]):
            raise ValueError(f"image size mismatch for {record['image_id']}")
        return image, {
            "image_id": record["image_id"],
            "original_size": size,
        }


def _collate_images(
    batch: list[tuple[torch.Tensor, dict[str, object]]],
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    images, metadata = zip(*batch)
    return torch.stack(images), list(metadata)


def validate_prediction_record(record: dict[str, Any]) -> None:
    required = {
        "image_id",
        "pred_count",
        "pred_points_xy",
        "pred_grid_counts",
        "pred_densest_grid_region",
        "threshold",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"prediction record missing keys: {sorted(missing)}")
    if not isinstance(record["image_id"], str) or not record["image_id"]:
        raise ValueError("image_id must be a non-empty string")
    count = record["pred_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("pred_count must be a nonnegative integer")
    points = np.asarray(record["pred_points_xy"], dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pred_points_xy must have shape (N, 2)")
    if len(points) != count:
        raise ValueError("pred_count does not match point length")

    width = int(record.get("image_width", IMAGE_WIDTH))
    height = int(record.get("image_height", IMAGE_HEIGHT))
    recomputed_grid = grid_counts(points, width, height)
    supplied_grid = record["pred_grid_counts"]
    if not isinstance(supplied_grid, dict) or set(supplied_grid) != set(
        GRID_ORDER
    ):
        raise ValueError("prediction grid must contain exactly nine cells")
    normalized_grid: dict[str, int] = {}
    for name in GRID_ORDER:
        value = supplied_grid[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("prediction grid values must be nonnegative integers")
        normalized_grid[name] = value
    if normalized_grid != recomputed_grid or sum(normalized_grid.values()) != count:
        raise ValueError("prediction grid is inconsistent with points")
    if record["pred_densest_grid_region"] != densest_region(recomputed_grid):
        raise ValueError("predicted densest grid region is inconsistent")
    threshold = float(record["threshold"])
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")


def load_frozen_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    artifact_sha256 = sha256_file(MODEL_ROOT / "DOWNLOAD_MANIFEST.json")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint["artifact_manifest_sha256"] != artifact_sha256:
        raise ValueError("checkpoint artifact manifest hash mismatch")
    model = build_p2pnet(training=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _atomic_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    count = 0
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        record, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
                count += 1
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return count


def iter_inference(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> Iterator[dict[str, Any]]:
    with torch.inference_mode():
        for images, metadata in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            image_sizes = [
                tuple(int(value) for value in item["original_size"])
                for item in metadata
            ]
            decoded = decode_predictions(outputs, threshold, image_sizes)
            for item, prediction, (width, height) in zip(
                metadata, decoded, image_sizes
            ):
                points = np.asarray(
                    prediction["points_xy"], dtype=np.float32
                ).reshape(-1, 2)
                counts = grid_counts(points, width, height)
                record = {
                    "image_id": str(item["image_id"]),
                    "image_width": width,
                    "image_height": height,
                    "pred_count": int(prediction["count"]),
                    "pred_points_xy": points.tolist(),
                    "pred_grid_counts": counts,
                    "pred_densest_grid_region": densest_region(counts),
                    "threshold": float(threshold),
                }
                validate_prediction_record(record)
                yield record


def run_inference(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> list[dict[str, Any]]:
    return list(
        iter_inference(
            model=model,
            loader=loader,
            device=device,
            threshold=threshold,
        )
    )


def _parse_args() -> argparse.Namespace:
    paths = ExperimentPaths.from_project_root()
    output_dir = paths.project_root / "Experiments" / "p2pnet_nereus" / "outputs"
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument(
        "--checkpoint", type=Path, default=MODEL_ROOT / "checkpoints" / "best.pth"
    )
    parser.add_argument(
        "--calibration", type=Path, default=output_dir / "calibration.json"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    checkpoint_sha256 = sha256_file(args.checkpoint)
    if calibration["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("calibration was produced by a different checkpoint")
    threshold = float(calibration["selected_threshold"])

    paths = ExperimentPaths.from_project_root()
    dataset = NereusImageDataset(
        paths.manifest_dir / "test_images.jsonl", limit=args.limit
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate_images,
        persistent_workers=args.num_workers > 0,
    )
    model, _ = load_frozen_model(args.checkpoint, device)
    default_name = (
        "test_predictions.jsonl"
        if args.limit is None
        else f"test_predictions_smoke{args.limit}.jsonl"
    )
    output = (
        args.output
        if args.output is not None
        else paths.project_root
        / "Experiments"
        / "p2pnet_nereus"
        / "outputs"
        / default_name
    )
    record_count = _atomic_jsonl(
        iter_inference(
            model=model,
            loader=loader,
            device=device,
            threshold=threshold,
        ),
        output,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "records": record_count,
                "sha256": sha256_file(output),
                "threshold": threshold,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
