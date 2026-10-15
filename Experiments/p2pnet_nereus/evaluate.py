"""Strict Nereus Track A aggregation and Table 3 rendering."""

import json
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import scipy
import torch
import torchvision

from .artifacts import sha256_file
from .config import EXPECTED_SPLIT_SIZES, MODEL_ROOT, ExperimentPaths
from .grid_metrics import (
    aggregate_metrics,
    densest_region,
    grid_counts,
)
from .infer import validate_prediction_record


def _unique_by_id(
    records: Sequence[Mapping[str, Any]], role: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        image_id = str(record["image_id"])
        if image_id in indexed:
            raise ValueError(f"duplicate {role} image ID: {image_id}")
        indexed[image_id] = record
    return indexed


def evaluate_records(
    target_records: Sequence[Mapping[str, Any]],
    prediction_records: Sequence[Mapping[str, Any]],
    *,
    expected_size: int,
) -> dict[str, float]:
    if len(target_records) != expected_size:
        raise ValueError(
            f"expected {expected_size} targets, got {len(target_records)}"
        )
    targets_by_id = _unique_by_id(target_records, "target")
    predictions_by_id = _unique_by_id(prediction_records, "prediction")
    if set(targets_by_id) != set(predictions_by_id):
        missing = sorted(set(targets_by_id) - set(predictions_by_id))
        extra = sorted(set(predictions_by_id) - set(targets_by_id))
        raise ValueError(
            f"prediction image ID set mismatch; missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    if len(prediction_records) != expected_size:
        raise ValueError(
            f"expected {expected_size} predictions, got {len(prediction_records)}"
        )

    metric_targets: list[dict[str, object]] = []
    metric_predictions: list[dict[str, object]] = []
    thresholds: set[float] = set()
    for image_id in sorted(targets_by_id):
        target = targets_by_id[image_id]
        prediction = dict(predictions_by_id[image_id])
        width = int(target["width"])
        height = int(target["height"])
        target_points = np.asarray(
            target["points_xy"], dtype=np.float64
        ).reshape(-1, 2)
        if int(target["count"]) != len(target_points):
            raise ValueError(f"target count mismatch for {image_id}")
        target_grid = grid_counts(target_points, width, height)

        prediction["image_width"] = width
        prediction["image_height"] = height
        validate_prediction_record(prediction)
        thresholds.add(float(prediction["threshold"]))
        metric_targets.append(
            {
                "count": int(target["count"]),
                "densest": densest_region(target_grid),
            }
        )
        metric_predictions.append(
            {
                "count": int(prediction["pred_count"]),
                "densest": str(prediction["pred_densest_grid_region"]),
            }
        )
    if len(thresholds) != 1:
        raise ValueError("all test predictions must use one frozen threshold")
    return aggregate_metrics(metric_targets, metric_predictions)


def render_table3_row(metrics: Mapping[str, float]) -> str:
    return (
        "| P2PNet | Nereus train, ImageNet init | "
        f"{metrics['mae']:.2f} | {metrics['rmse']:.2f} | "
        f"{metrics['exact']:.2f} | {metrics['acc_at_5']:.2f} | "
        f"{metrics['acc_at_10']:.2f} | {metrics['densest_cell']:.2f} | "
        "N/A | N/A | N/A |"
    )


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    paths = ExperimentPaths.from_project_root()
    experiment_root = (
        paths.project_root / "Experiments" / "p2pnet_nereus"
    )
    output_dir = experiment_root / "outputs"
    test_manifest = paths.manifest_dir / "test.jsonl"
    predictions_path = output_dir / "test_predictions.jsonl"
    calibration_path = output_dir / "calibration.json"
    checkpoint_path = MODEL_ROOT / "checkpoints" / "best.pth"
    artifact_manifest_path = MODEL_ROOT / "DOWNLOAD_MANIFEST.json"

    targets = _read_jsonl(test_manifest)
    predictions = _read_jsonl(predictions_path)
    metrics = evaluate_records(
        targets,
        predictions,
        expected_size=EXPECTED_SPLIT_SIZES["test"],
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    prediction_thresholds = {
        float(record["threshold"]) for record in predictions
    }
    if prediction_thresholds != {float(calibration["selected_threshold"])}:
        raise ValueError("test threshold differs from validation calibration")
    if calibration["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ValueError("evaluated checkpoint differs from calibration")

    artifact_manifest = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest_path.is_file()
        else {}
    )
    split_hashes = {
        split: sha256_file(paths.manifest_dir / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    split_hashes["test_images"] = sha256_file(
        paths.manifest_dir / "test_images.jsonl"
    )
    row = render_table3_row(metrics)
    report = {
        "method": "P2PNet",
        "training_data": "Nereus train split only",
        "external_initialization": "ImageNet VGG16-BN only",
        "track_a": metrics,
        "camouflage_f1": "N/A",
        "track_b": "N/A",
        "track_c": "N/A",
        "selected_threshold": float(calibration["selected_threshold"]),
        "selected_epoch": int(calibration["checkpoint_epoch"]),
        "seed": 42,
        "records": EXPECTED_SPLIT_SIZES["test"],
        "table3_row": row,
        "provenance": {
            "upstream_repository": artifact_manifest["upstream"]["repository"],
            "upstream_commit": artifact_manifest["upstream"]["commit"],
            "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "calibration_sha256": sha256_file(calibration_path),
            "prediction_sha256": sha256_file(predictions_path),
            "split_manifest_sha256": split_hashes,
            "gpu_visibility": run_manifest.get("gpu_visibility", "4,5,6,7"),
            "gpu_names": run_manifest.get(
                "gpu_names", ["NVIDIA GeForce RTX 4090"] * 4
            ),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pillow": PIL.__version__,
            },
        },
    }
    metrics_path = output_dir / "metrics.json"
    table_path = output_dir / "table3_track_a.md"
    _atomic_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", metrics_path
    )
    table = (
        "| Method | Training / Init | MAE | RMSE | Exact | Acc@5 | "
        "Acc@10 | Densest Cell | Camouflage F1 | Track B | Track C |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"{row}\n"
    )
    _atomic_text(table, table_path)
    print(
        json.dumps(
            {
                "metrics": metrics,
                "metrics_path": str(metrics_path),
                "table_path": str(table_path),
                "row": row,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
