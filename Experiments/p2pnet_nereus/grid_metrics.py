"""Nereus Track A grid and counting metrics."""

from collections.abc import Mapping, Sequence

import numpy as np


GRID_ORDER = (
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
)


def grid_counts(
    points_xy: np.ndarray, width: int, height: int
) -> dict[str, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    points = np.asarray(points_xy, dtype=np.float64)
    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("non-finite point")
    if len(points) and (
        (points[:, 0] < 0).any()
        or (points[:, 0] >= width).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] >= height).any()
    ):
        raise ValueError("out-of-bounds point")

    counts = {name: 0 for name in GRID_ORDER}
    for x_coordinate, y_coordinate in points:
        column = min(2, int(3 * x_coordinate / width))
        row = min(2, int(3 * y_coordinate / height))
        counts[GRID_ORDER[row * 3 + column]] += 1
    return counts


def densest_region(counts: Mapping[str, int]) -> str:
    if set(counts) != set(GRID_ORDER):
        raise ValueError("grid counts must contain exactly nine regions")
    normalized = {name: int(counts[name]) for name in GRID_ORDER}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("grid counts cannot be negative")
    maximum = max(normalized.values())
    return next(name for name in GRID_ORDER if normalized[name] == maximum)


def aggregate_metrics(
    targets: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    if len(targets) != len(predictions):
        raise ValueError("target and prediction lengths differ")
    if not targets:
        raise ValueError("at least one sample is required")

    target_counts = np.asarray(
        [int(target["count"]) for target in targets], dtype=np.int64
    )
    predicted_counts = np.asarray(
        [int(prediction["count"]) for prediction in predictions],
        dtype=np.int64,
    )
    if (target_counts < 0).any() or (predicted_counts < 0).any():
        raise ValueError("counts cannot be negative")
    absolute_error = np.abs(predicted_counts - target_counts)
    squared_error = np.square(
        predicted_counts.astype(np.float64)
        - target_counts.astype(np.float64)
    )
    densest_matches = np.asarray(
        [
            prediction["densest"] == target["densest"]
            for target, prediction in zip(targets, predictions)
        ],
        dtype=np.float64,
    )
    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "exact": float(100.0 * (absolute_error == 0).mean()),
        "acc_at_5": float(
            100.0 * (absolute_error <= 0.05 * target_counts).mean()
        ),
        "acc_at_10": float(
            100.0 * (absolute_error <= 0.10 * target_counts).mean()
        ),
        "densest_cell": float(100.0 * densest_matches.mean()),
    }
