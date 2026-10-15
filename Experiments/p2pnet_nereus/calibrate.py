"""Validation-only threshold calibration for P2PNet."""

import argparse
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .artifacts import sha256_file
from .config import MODEL_ROOT, ExperimentPaths
from .dataset import NereusPointDataset, collate_points
from .grid_metrics import aggregate_metrics, densest_region, grid_counts
from .infer import load_frozen_model
from .p2pnet_adapter import decode_predictions


THRESHOLDS = tuple(round(step * 0.05, 2) for step in range(1, 20))


def select_threshold(
    candidate_records: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    if not candidate_records:
        raise ValueError("threshold candidates cannot be empty")
    normalized: list[dict[str, float]] = []
    for record in candidate_records:
        candidate = {
            "threshold": float(record["threshold"]),
            "mae": float(record["mae"]),
            "rmse": float(record["rmse"]),
            "exact": float(record["exact"]),
        }
        if not all(math.isfinite(value) for value in candidate.values()):
            raise ValueError("threshold metrics must be finite")
        normalized.append(candidate)
    return min(
        normalized,
        key=lambda record: (
            record["mae"],
            record["rmse"],
            -record["exact"],
            -record["threshold"],
        ),
    )


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def iter_validation_candidates(
    *, model: nn.Module, loader: DataLoader, device: torch.device
) -> Iterator[dict[str, Any]]:
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            image_sizes = [
                tuple(int(value) for value in target["original_size"])
                for target in targets
            ]
            decoded = decode_predictions(
                outputs, threshold=min(THRESHOLDS), image_sizes=image_sizes
            )
            for target, prediction, (width, height) in zip(
                targets, decoded, image_sizes
            ):
                target_points = target["point"].detach().cpu().numpy()
                yield {
                    "image_id": str(target["image_id"]),
                    "image_width": width,
                    "image_height": height,
                    "target_count": int(target["target_count"]),
                    "target_densest_grid_region": densest_region(
                        grid_counts(target_points, width, height)
                    ),
                    "candidate_points_xy": prediction["points_xy"],
                    "candidate_scores": prediction["scores"],
                    "minimum_threshold": min(THRESHOLDS),
                }


def collect_validation_candidates(
    *, model: nn.Module, loader: DataLoader, device: torch.device
) -> list[dict[str, Any]]:
    return list(
        iter_validation_candidates(model=model, loader=loader, device=device)
    )


def evaluate_thresholds(
    validation_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, float]]:
    if not validation_records:
        raise ValueError("validation candidate records cannot be empty")
    candidates: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        targets: list[dict[str, object]] = []
        predictions: list[dict[str, object]] = []
        for record in validation_records:
            points = np.asarray(
                record["candidate_points_xy"], dtype=np.float32
            ).reshape(-1, 2)
            scores = np.asarray(record["candidate_scores"], dtype=np.float32)
            if len(points) != len(scores) or not np.isfinite(scores).all():
                raise ValueError("candidate point-score arrays are inconsistent")
            selected = points[scores >= threshold]
            width = int(record["image_width"])
            height = int(record["image_height"])
            targets.append(
                {
                    "count": int(record["target_count"]),
                    "densest": str(record["target_densest_grid_region"]),
                }
            )
            predictions.append(
                {
                    "count": int(len(selected)),
                    "densest": densest_region(
                        grid_counts(selected, width, height)
                    ),
                }
            )
        metrics = aggregate_metrics(targets, predictions)
        candidates.append({"threshold": threshold, **metrics})
    return candidates


def _stream_candidates_and_metrics(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> tuple[list[dict[str, float]], int]:
    targets_by_threshold: dict[float, list[dict[str, object]]] = {
        threshold: [] for threshold in THRESHOLDS
    }
    predictions_by_threshold: dict[float, list[dict[str, object]]] = {
        threshold: [] for threshold in THRESHOLDS
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    record_count = 0
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            for record in iter_validation_candidates(
                model=model, loader=loader, device=device
            ):
                points = np.asarray(
                    record["candidate_points_xy"], dtype=np.float32
                ).reshape(-1, 2)
                scores = np.asarray(
                    record["candidate_scores"], dtype=np.float32
                )
                if len(points) != len(scores) or not np.isfinite(scores).all():
                    raise ValueError(
                        "candidate point-score arrays are inconsistent"
                    )
                width = int(record["image_width"])
                height = int(record["image_height"])
                target = {
                    "count": int(record["target_count"]),
                    "densest": str(
                        record["target_densest_grid_region"]
                    ),
                }
                for threshold in THRESHOLDS:
                    selected = points[scores >= threshold]
                    targets_by_threshold[threshold].append(target)
                    predictions_by_threshold[threshold].append(
                        {
                            "count": int(len(selected)),
                            "densest": densest_region(
                                grid_counts(selected, width, height)
                            ),
                        }
                    )
                stream.write(
                    json.dumps(
                        record, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
                record_count += 1
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    candidates = [
        {
            "threshold": threshold,
            **aggregate_metrics(
                targets_by_threshold[threshold],
                predictions_by_threshold[threshold],
            ),
        }
        for threshold in THRESHOLDS
    ]
    return candidates, record_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=MODEL_ROOT / "checkpoints" / "best.pth"
    )
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
    paths = ExperimentPaths.from_project_root()
    validation_manifest = paths.manifest_dir / "val.jsonl"
    dataset: NereusPointDataset | Subset = NereusPointDataset(
        validation_manifest, training=False, seed=42
    )
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_points,
        persistent_workers=args.num_workers > 0,
    )
    output_dir = (
        paths.project_root / "Experiments" / "p2pnet_nereus" / "outputs"
    )
    suffix = "" if args.limit is None else f"_smoke{args.limit}"
    candidate_path = output_dir / f"validation_candidates{suffix}.jsonl"
    calibration_path = output_dir / f"calibration{suffix}.json"
    model, checkpoint = load_frozen_model(args.checkpoint, device)
    candidates, record_count = _stream_candidates_and_metrics(
        model=model,
        loader=loader,
        device=device,
        output_path=candidate_path,
    )
    selected = select_threshold(candidates)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    calibration = {
        "selected_threshold": selected["threshold"],
        "selected_metrics": selected,
        "all_candidates": candidates,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_manifest": str(validation_manifest),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "validation_records": record_count,
        "candidate_records": str(candidate_path),
        "candidate_records_sha256": sha256_file(candidate_path),
        "selection_key": ["mae", "rmse", "-exact", "-threshold"],
    }
    _atomic_json(calibration, calibration_path)
    print(
        json.dumps(
            {
                "calibration": str(calibration_path),
                "records": record_count,
                "selected_threshold": selected["threshold"],
                "selected_metrics": selected,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
