"""Pure conversion functions for external counting-model predictions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from Experiments.p2pnet_nereus.grid_metrics import (
    GRID_ORDER,
    densest_region,
    grid_counts,
)


def round_nonnegative_count(value: float) -> int:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("count must be finite")
    return int(math.floor(max(0.0, value) + 0.5))


def _density_grid_masses(
    density_map: np.ndarray, scale_factor: float
) -> dict[str, float]:
    density = np.asarray(density_map, dtype=np.float64)
    if density.ndim != 2 or not density.size:
        raise ValueError("density_map must be a non-empty 2D array")
    if not np.isfinite(density).all():
        raise ValueError("density_map must be finite")
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be positive and finite")
    row_edges = np.linspace(0, density.shape[0], 4, dtype=np.int64)
    column_edges = np.linspace(0, density.shape[1], 4, dtype=np.int64)
    masses: dict[str, float] = {}
    for row in range(3):
        for column in range(3):
            cell = density[
                row_edges[row] : row_edges[row + 1],
                column_edges[column] : column_edges[column + 1],
            ]
            masses[GRID_ORDER[row * 3 + column]] = float(
                cell.sum() / scale_factor
            )
    return masses


def countgd_boxes_to_prediction(
    *,
    image_id: str,
    width: int,
    height: int,
    boxes_cxcywh: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    boxes = np.asarray(boxes_cxcywh, dtype=np.float64).reshape(-1, 4)
    normalized_scores = np.asarray(scores).reshape(-1)
    if len(boxes) != len(normalized_scores):
        raise ValueError("box and score lengths differ")
    if not np.isfinite(boxes).all() or not np.isfinite(normalized_scores).all():
        raise ValueError("box predictions must be finite")
    threshold = np.asarray(
        confidence_threshold, dtype=normalized_scores.dtype
    ).item()
    selected = boxes[normalized_scores > threshold]
    points = np.column_stack(
        (selected[:, 0] * int(width), selected[:, 1] * int(height))
    )
    counts = grid_counts(points, int(width), int(height))
    record: dict[str, Any] = {
        "method": "CountGD",
        "setting": "External pretrained Direct-ZS",
        "image_id": str(image_id),
        "image_width": int(width),
        "image_height": int(height),
        "pred_count": int(len(points)),
        "pred_points_xy": points.tolist(),
        "pred_grid_counts": counts,
        "pred_densest_grid_region": densest_region(counts),
        "text_query": "fish",
        "confidence_threshold": float(confidence_threshold),
        "checkpoint_sha256": str(checkpoint_sha256),
    }
    validate_prediction_record(record)
    return record


def clip_density_to_prediction(
    *,
    image_id: str,
    width: int,
    height: int,
    density_map: np.ndarray,
    scale_factor: float,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    masses = _density_grid_masses(density_map, float(scale_factor))
    count_float = float(sum(masses.values()))
    record: dict[str, Any] = {
        "method": "CLIP-Count",
        "setting": "External pretrained Direct-ZS",
        "image_id": str(image_id),
        "image_width": int(width),
        "image_height": int(height),
        "pred_count_float": count_float,
        "pred_count": round_nonnegative_count(count_float),
        "pred_grid_mass": masses,
        "pred_densest_grid_region": max(
            GRID_ORDER, key=lambda name: masses[name]
        ),
        "text_query": "fish",
        "checkpoint_sha256": str(checkpoint_sha256),
    }
    validate_prediction_record(record)
    return record


def validate_prediction_record(record: Mapping[str, Any]) -> None:
    required = {
        "method",
        "setting",
        "image_id",
        "image_width",
        "image_height",
        "pred_count",
        "pred_densest_grid_region",
        "text_query",
        "checkpoint_sha256",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"prediction record missing keys: {sorted(missing)}")
    if not str(record["image_id"]):
        raise ValueError("image_id must be non-empty")
    width = int(record["image_width"])
    height = int(record["image_height"])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    count = record["pred_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("pred_count must be a nonnegative integer")
    if record["pred_densest_grid_region"] not in GRID_ORDER:
        raise ValueError("invalid densest grid region")

    method = str(record["method"])
    if method == "CountGD":
        points = np.asarray(record["pred_points_xy"], dtype=np.float64).reshape(
            -1, 2
        )
        if len(points) != count:
            raise ValueError("CountGD point count mismatch")
        counts = grid_counts(points, width, height)
        supplied = {
            name: int(record["pred_grid_counts"][name]) for name in GRID_ORDER
        }
        if counts != supplied or densest_region(counts) != record[
            "pred_densest_grid_region"
        ]:
            raise ValueError("CountGD grid prediction is inconsistent")
    elif method == "CLIP-Count":
        masses = record["pred_grid_mass"]
        if set(masses) != set(GRID_ORDER):
            raise ValueError("CLIP-Count grid must contain nine cells")
        normalized = {name: float(masses[name]) for name in GRID_ORDER}
        if not all(math.isfinite(value) for value in normalized.values()):
            raise ValueError("CLIP-Count grid mass must be finite")
        count_float = float(record["pred_count_float"])
        if not math.isclose(
            sum(normalized.values()), count_float, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("CLIP-Count total and grid mass differ")
        if round_nonnegative_count(count_float) != count:
            raise ValueError("CLIP-Count integer rounding is inconsistent")
        predicted_densest = max(
            GRID_ORDER, key=lambda name: normalized[name]
        )
        if predicted_densest != record["pred_densest_grid_region"]:
            raise ValueError("CLIP-Count densest grid is inconsistent")
    else:
        raise ValueError(f"unsupported method: {method}")
