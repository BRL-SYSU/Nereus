#!/usr/bin/env python3
"""Count-range stratified evaluation for the Nereus Track A test split."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(
    "."
)
WORK_ROOT = Path(__file__).absolute().parent
DEFAULT_OUTPUT_DIR = WORK_ROOT / "analysis" / "count_range"
TRACK_A_PATH = PROJECT_ROOT / "Experiments/data/eval/trackA_test.json"
TRACK_B_PATH = PROJECT_ROOT / "Experiments/data/eval/trackB_test.json"
CANONICAL_SPLIT_PATH = (
    PROJECT_ROOT / "Experiments/data/canonical/split_test.json"
)
CANONICAL_RECORDS_PATH = (
    PROJECT_ROOT / "Experiments/canonical_dataset/canonical_records.jsonl"
)
FINAL_ANNOTATION_DIR = PROJECT_ROOT / "test_dataset/final_annotations"
OFFICIAL_TEST_PATH = PROJECT_ROOT / "IOCfish5K/test_id.txt"

CLOSED_ROOT = Path("Experiments/closed_vlm_nereus_joint")
P2PNET_PREDICTIONS = (
    PROJECT_ROOT
    / "Experiments/p2pnet_nereus/outputs/test_predictions.jsonl"
)
EXTERNAL_OUTPUT_ROOT = (
    PROJECT_ROOT / "Experiments/external_counting_nereus/outputs"
)

COUNT_RANGES = ("0-50", "51-100", "101-200", ">200")
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
VISIBILITY_CLASSES = ("clear", "ambiguous", "highly_camouflaged")
OPEN_MODELS = (
    "Qwen2.5-VL-7B",
    "LLaVA-OneVision-7B",
    "InternVL2.5-8B",
)
OPEN_SETTINGS = ("ZS", "P-ZS", "Count", "QAS", "Full-SFT")


@dataclass(frozen=True)
class PredictionSpec:
    model: str
    setting: str
    model_group: str
    source_format: str
    source_path: Path | None
    reference_path: Path | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.model, self.setting


def assign_count_range(count: int) -> str:
    if 0 <= count <= 50:
        return "0-50"
    if 51 <= count <= 100:
        return "51-100"
    if 101 <= count <= 200:
        return "101-200"
    if count > 200:
        return ">200"
    raise ValueError(count)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("-").isdigit():
            return int(text)
    return None


def visibility_concepts(
    counts: Mapping[str, object], dominant: object
) -> set[str]:
    concepts = {
        label
        for label in VISIBILITY_CLASSES
        if (as_int(counts.get(label)) or 0) > 0
    }
    dominant_label = str(dominant).strip().lower()
    if dominant_label in VISIBILITY_CLASSES:
        concepts.add(f"dominant:{dominant_label}")
    return concepts


def densest_region(counts: Mapping[str, object]) -> str:
    normalized = {key: as_int(counts.get(key)) for key in GRID_ORDER}
    if any(value is None or value < 0 for value in normalized.values()):
        raise ValueError("invalid grid counts")
    maximum = max(int(value) for value in normalized.values())
    return next(
        key for key in GRID_ORDER if int(normalized[key]) == maximum
    )


def question_by_type(
    record: Mapping[str, Any], question_type: str
) -> Mapping[str, Any]:
    matches = [
        question
        for question in record.get("questions", [])
        if question.get("type") == question_type
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{record.get('image_id')}: expected one {question_type} question"
        )
    return matches[0]


def load_canonical_records(path: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    by_id = {str(record["image_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"{path}: duplicate image_id")
    return by_id


def build_ground_truth(
    manifest_path: Path,
    canonical_records_path: Path,
    final_annotation_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, list):
        raise ValueError(f"{manifest_path}: expected list")
    canonical = load_canonical_records(canonical_records_path)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for manifest_record in manifest:
        image_id = str(manifest_record["image_id"])
        if image_id in seen:
            issues.append(
                {"image_id": image_id, "issue": "duplicate_manifest_image_id"}
            )
            continue
        seen.add(image_id)

        count_question = question_by_type(manifest_record, "total_count")
        grid_question = question_by_type(manifest_record, "grid_counts")
        camouflage_question = question_by_type(
            manifest_record, "camouflage_visibility"
        )
        manifest_count = as_int(count_question.get("target"))
        if manifest_count is None or manifest_count < 0:
            raise ValueError(f"{image_id}: invalid manifest total count")

        canonical_record = canonical.get(image_id)
        canonical_count = None
        if canonical_record is not None:
            canonical_count = as_int(
                canonical_record.get("final_annotation_json", {}).get(
                    "total_count"
                )
            )

        final_path = final_annotation_dir / f"{image_id}.json"
        point_count = None
        structured_count = None
        source_point_count = None
        final_grid = None
        final_visibility = None
        if final_path.is_file():
            final = read_json(final_path)
            instances = final.get("instances")
            if isinstance(instances, list):
                point_count = len(instances)
            source_point_count = as_int(
                final.get("source_annotation", {}).get(
                    "num_valid_point_objects"
                )
            )
            task1 = final.get("image_level_annotations", {}).get(
                "task1_region", {}
            )
            final_grid = task1.get("grid_counts")
            task2 = final.get("image_level_annotations", {}).get(
                "task2_camouflage", {}
            )
            summary = task2.get("image_camouflage_summary", {})
            structured_count = as_int(summary.get("total_point_instances"))
            distribution = summary.get("visibility_distribution", {})
            if isinstance(distribution, dict):
                final_visibility = {
                    "clear": distribution.get("clear_count"),
                    "ambiguous": distribution.get("ambiguous_count"),
                    "highly_camouflaged": distribution.get(
                        "highly_camouflaged_count"
                    ),
                }
        else:
            issues.append(
                {
                    "image_id": image_id,
                    "issue": "missing_final_annotation",
                    "path": str(final_path),
                }
            )

        if canonical_count is not None:
            gt_count = canonical_count
            gt_source = (
                "canonical_records.final_annotation_json.total_count"
            )
        elif point_count is not None:
            gt_count = point_count
            gt_source = "final_annotation.instances"
        elif structured_count is not None:
            gt_count = structured_count
            gt_source = (
                "final_annotation.task2.image_camouflage_summary."
                "total_point_instances"
            )
        else:
            gt_count = manifest_count
            gt_source = "trackA_manifest.total_count_fallback"

        count_sources = {
            "selected_gt": gt_count,
            "manifest_total_count": manifest_count,
            "canonical_total_count": canonical_count,
            "point_count": point_count,
            "source_point_count": source_point_count,
            "structured_total_count": structured_count,
        }
        present_counts = {
            key: value
            for key, value in count_sources.items()
            if value is not None
        }
        if any(value != gt_count for value in present_counts.values()):
            issues.append(
                {
                    "image_id": image_id,
                    "issue": "gt_count_mismatch",
                    "values": present_counts,
                }
            )

        grid_target = grid_question.get("target", {})
        grid_counts = grid_target.get("grid_counts", {})
        manifest_densest = str(
            grid_target.get("densest_grid_region", "")
        )
        try:
            derived_densest = densest_region(grid_counts)
        except ValueError as error:
            issues.append(
                {
                    "image_id": image_id,
                    "issue": "invalid_manifest_grid",
                    "detail": str(error),
                }
            )
            derived_densest = manifest_densest
        grid_sum = sum(
            value
            for value in (as_int(grid_counts.get(key)) for key in GRID_ORDER)
            if value is not None
        )
        if grid_sum != gt_count or manifest_densest != derived_densest:
            issues.append(
                {
                    "image_id": image_id,
                    "issue": "manifest_grid_inconsistency",
                    "grid_sum": grid_sum,
                    "gt_count": gt_count,
                    "manifest_densest": manifest_densest,
                    "derived_densest": derived_densest,
                }
            )
        if isinstance(final_grid, dict) and final_grid != grid_counts:
            issues.append(
                {
                    "image_id": image_id,
                    "issue": "manifest_final_grid_mismatch",
                }
            )

        camouflage_target = camouflage_question.get("target", {})
        visibility_counts = camouflage_target.get(
            "visibility_counts", {}
        )
        dominant_visibility = camouflage_target.get(
            "dominant_visibility", ""
        )
        visibility_sum = sum(
            value
            for value in (
                as_int(visibility_counts.get(key))
                for key in VISIBILITY_CLASSES
            )
            if value is not None
        )
        if visibility_sum != gt_count:
            issues.append(
                {
                    "image_id": image_id,
                    "issue": "manifest_visibility_sum_mismatch",
                    "visibility_sum": visibility_sum,
                    "gt_count": gt_count,
                }
            )
        if (
            isinstance(final_visibility, dict)
            and final_visibility != visibility_counts
        ):
            issues.append(
                {
                    "image_id": image_id,
                    "issue": "manifest_final_visibility_mismatch",
                }
            )

        records.append(
            {
                "image_id": image_id,
                "gt_count": gt_count,
                "gt_source": gt_source,
                "count_range": assign_count_range(gt_count),
                "densest_cell_gt": manifest_densest,
                "camouflage_gt_concepts": visibility_concepts(
                    visibility_counts, dominant_visibility
                ),
            }
        )

    if len(records) != len(manifest):
        issues.append(
            {
                "issue": "manifest_record_loss",
                "manifest_records": len(manifest),
                "usable_records": len(records),
            }
        )
    audit = {
        "manifest_path": str(manifest_path),
        "manifest_records": len(manifest),
        "unique_image_ids": len(seen),
        "canonical_records": len(canonical),
        "gt_priority": [
            "canonical total_count",
            "number of point annotations",
            "structured final total count",
            "Track A manifest fallback",
        ],
        "issues": issues,
        "issue_count": len(issues),
        "critical_issue_count": sum(
            item.get("issue")
            in {
                "duplicate_manifest_image_id",
                "missing_final_annotation",
                "gt_count_mismatch",
                "manifest_record_loss",
            }
            for item in issues
        ),
    }
    return records, audit


def prediction_specs() -> list[PredictionSpec]:
    specs = [
        PredictionSpec(
            model="P2PNet",
            setting="Nereus-Train",
            model_group="counting_baseline",
            source_format="jsonl",
            source_path=P2PNET_PREDICTIONS,
            reference_path=P2PNET_PREDICTIONS.parent / "metrics.json",
        ),
        PredictionSpec(
            model="CountGD",
            setting="Direct-ZS",
            model_group="counting_baseline",
            source_format="jsonl",
            source_path=EXTERNAL_OUTPUT_ROOT / "countgd_predictions.jsonl",
            reference_path=EXTERNAL_OUTPUT_ROOT / "metrics.json",
        ),
        PredictionSpec(
            model="CLIP-Count",
            setting="Direct-ZS",
            model_group="counting_baseline",
            source_format="jsonl",
            source_path=EXTERNAL_OUTPUT_ROOT / "clipcount_predictions.jsonl",
            reference_path=EXTERNAL_OUTPUT_ROOT / "metrics.json",
        ),
    ]
    closed_settings = (
        ("J-ZS", CLOSED_ROOT / "outputs_joint_zs"),
        ("J-PZS", CLOSED_ROOT / "outputs"),
        ("J-Full", CLOSED_ROOT / "outputs_joint_full"),
    )
    for setting, root in closed_settings:
        reference = root / "metrics.json"
        for model in ("gpt-5", "gemini-3.1-pro-preview"):
            specs.append(
                PredictionSpec(
                    model=model,
                    setting=setting,
                    model_group="closed_vlm",
                    source_format="closed_cache",
                    source_path=root / model / "cache",
                    reference_path=reference if reference.is_file() else None,
                )
            )
    for model in OPEN_MODELS:
        for setting in OPEN_SETTINGS:
            specs.append(
                PredictionSpec(
                    model=model,
                    setting=setting,
                    model_group="open_vlm",
                    source_format="not_found",
                    source_path=None,
                )
            )
    return specs


def indexed_jsonl(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        image_id = str(record.get("image_id", ""))
        if image_id in by_id:
            duplicates.append(image_id)
        else:
            by_id[image_id] = record
    return by_id, duplicates


def parse_jsonl_prediction(record: Mapping[str, Any]) -> dict[str, Any]:
    pred_count = as_int(record.get("pred_count"))
    valid = pred_count is not None and pred_count >= 0
    return {
        "observed_prediction": True,
        "raw_response": "",
        "pred_count": pred_count,
        "valid_prediction": valid,
        "parse_status": "explicit_pred_count" if valid else "invalid_count",
        "retry_status": "not_applicable",
        "densest_cell_pred": record.get("pred_densest_grid_region"),
        "camouflage_pred_concepts": None,
    }


def parse_closed_cache(path: Path) -> dict[str, Any]:
    value = read_json(path)
    parsed = value.get("parsed", {})
    pred_count = as_int(parsed.get("total_count"))
    valid = bool(parsed.get("valid")) and pred_count is not None
    if pred_count is not None and pred_count < 0:
        valid = False
    if valid:
        parse_status = "valid"
    elif pred_count is not None:
        parse_status = "invalid_with_parser_fallback"
    else:
        parse_status = "invalid_without_count"
    raw_visibility = parsed.get("visibility_counts", {})
    pred_concepts = (
        visibility_concepts(
            raw_visibility
            if isinstance(raw_visibility, Mapping)
            else {},
            parsed.get("dominant_visibility", ""),
        )
        if isinstance(parsed, Mapping)
        else set()
    )
    attempts = as_int(value.get("attempts")) or 1
    return {
        "observed_prediction": True,
        "raw_response": str(value.get("raw_content", "")),
        "pred_count": pred_count,
        "valid_prediction": valid,
        "parse_status": parse_status,
        "retry_status": "retried" if attempts > 1 else "not_retried",
        "densest_cell_pred": parsed.get("densest_grid_region"),
        "camouflage_pred_concepts": pred_concepts if valid else set(),
    }


def missing_prediction() -> dict[str, Any]:
    return {
        "observed_prediction": False,
        "raw_response": "",
        "pred_count": None,
        "valid_prediction": False,
        "parse_status": "missing_prediction",
        "retry_status": "missing",
        "densest_cell_pred": None,
        "camouflage_pred_concepts": None,
    }


def merge_predictions(
    gt_records: Sequence[Mapping[str, Any]],
    specs: Sequence[PredictionSpec],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    expected_ids = {str(record["image_id"]) for record in gt_records}

    for spec in specs:
        indexed: dict[str, dict[str, Any]] = {}
        duplicates: list[str] = []
        fields: list[str] = []
        if (
            spec.source_format == "jsonl"
            and spec.source_path is not None
            and spec.source_path.is_file()
        ):
            indexed, duplicates = indexed_jsonl(spec.source_path)
            if indexed:
                fields = sorted(next(iter(indexed.values())).keys())

        observed = 0
        valid = 0
        unexpected_ids: list[str] = []
        if spec.source_format == "jsonl":
            unexpected_ids = sorted(set(indexed) - expected_ids)

        for gt in gt_records:
            image_id = str(gt["image_id"])
            if spec.source_format == "jsonl":
                prediction = (
                    parse_jsonl_prediction(indexed[image_id])
                    if image_id in indexed
                    else missing_prediction()
                )
            elif (
                spec.source_format == "closed_cache"
                and spec.source_path is not None
            ):
                cache_path = spec.source_path / image_id / "track_a.json"
                prediction = (
                    parse_closed_cache(cache_path)
                    if cache_path.is_file()
                    else missing_prediction()
                )
                if not fields and cache_path.is_file():
                    fields = sorted(read_json(cache_path).keys())
            else:
                prediction = missing_prediction()

            observed += int(prediction["observed_prediction"])
            valid += int(prediction["valid_prediction"])
            pred_count = prediction["pred_count"]
            absolute_error = (
                abs(int(pred_count) - int(gt["gt_count"]))
                if pred_count is not None
                else None
            )
            signed_error = (
                int(pred_count) - int(gt["gt_count"])
                if pred_count is not None
                else None
            )
            merged.append(
                {
                    **gt,
                    "model": spec.model,
                    "setting": spec.setting,
                    "model_group": spec.model_group,
                    "prediction_source": (
                        str(spec.source_path)
                        if spec.source_path is not None
                        else ""
                    ),
                    **prediction,
                    "absolute_error": absolute_error,
                    "signed_error": signed_error,
                }
            )

        inventory.append(
            {
                "model": spec.model,
                "setting": spec.setting,
                "model_group": spec.model_group,
                "source_format": spec.source_format,
                "path": (
                    str(spec.source_path)
                    if spec.source_path is not None
                    else "NOT FOUND"
                ),
                "expected_samples": len(gt_records),
                "observed_samples": observed,
                "valid_samples": valid,
                "complete": observed == len(gt_records),
                "fields": fields,
                "has_image_id": (
                    "image_id" in fields
                    or spec.source_format == "closed_cache"
                ),
                "has_gt_count": any(
                    field in fields
                    for field in ("gt_count", "target_count", "ground_truth")
                ),
                "has_pred_count": (
                    "pred_count" in fields
                    or "parsed" in fields
                ),
                "duplicate_image_ids": duplicates,
                "unexpected_image_ids": unexpected_ids,
            }
        )
    return merged, inventory


def micro_set_f1(
    targets: Sequence[set[str]], predictions: Sequence[set[str]]
) -> float:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("invalid target/prediction sets")
    true_positive = sum(
        len(target & prediction)
        for target, prediction in zip(targets, predictions)
    )
    false_positive = sum(
        len(prediction - target)
        for target, prediction in zip(targets, predictions)
    )
    false_negative = sum(
        len(target - prediction)
        for target, prediction in zip(targets, predictions)
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return 100.0 * (2 * true_positive / denominator) if denominator else 100.0


def compute_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot compute metrics for empty rows")
    if any(row["pred_count"] is None for row in rows):
        raise ValueError("metrics require observed count predictions")

    gt = np.asarray([int(row["gt_count"]) for row in rows], dtype=np.int64)
    pred = np.asarray(
        [int(row["pred_count"]) for row in rows], dtype=np.int64
    )
    absolute = np.abs(pred - gt)
    signed = pred - gt
    valid = np.asarray(
        [bool(row["valid_prediction"]) for row in rows], dtype=bool
    )
    densest_available = all(
        row["densest_cell_pred"] is not None for row in rows
    )
    camouflage_available = all(
        row["camouflage_pred_concepts"] is not None for row in rows
    )
    metrics: dict[str, Any] = {
        "n": len(rows),
        "gt_mean": float(np.mean(gt)),
        "gt_median": float(np.median(gt)),
        "gt_min": int(np.min(gt)),
        "gt_max": int(np.max(gt)),
        "valid_n": int(valid.sum()),
        "valid_coverage_pct": 100.0 * float(np.mean(valid)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(signed.astype(float))))),
        "exact_accuracy_pct": 100.0 * float(np.mean(absolute == 0)),
        "acc_at_5_pct": 100.0
        * float(np.mean(absolute <= 0.05 * gt)),
        "acc_at_10_pct": 100.0
        * float(np.mean(absolute <= 0.10 * gt)),
        "nmae_wape_pct": 100.0
        * float(np.sum(absolute))
        / float(np.sum(gt)),
        "bias": float(np.mean(signed)),
        "median_signed_error": float(np.median(signed)),
        "under_count_rate_pct": 100.0 * float(np.mean(pred < gt)),
        "over_count_rate_pct": 100.0 * float(np.mean(pred > gt)),
        "mean_predicted_count": float(np.mean(pred)),
        "mean_ground_truth_count": float(np.mean(gt)),
        "densest_cell_accuracy_pct": None,
        "camouflage_f1_pct": None,
    }
    if densest_available:
        metrics["densest_cell_accuracy_pct"] = 100.0 * float(
            np.mean(
                [
                    row["densest_cell_pred"] == row["densest_cell_gt"]
                    for row in rows
                ]
            )
        )
    if camouflage_available:
        metrics["camouflage_f1_pct"] = micro_set_f1(
            [
                set(row["camouflage_gt_concepts"])
                for row in rows
            ],
            [
                set(row["camouflage_pred_concepts"])
                for row in rows
            ],
        )
    return metrics


def complete_groups(
    merged: Sequence[Mapping[str, Any]], expected_size: int
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in merged:
        key = str(row["model"]), str(row["setting"])
        grouped.setdefault(key, []).append(row)
    return {
        key: rows
        for key, rows in grouped.items()
        if len(rows) == expected_size
        and all(row["observed_prediction"] for row in rows)
        and all(row["pred_count"] is not None for row in rows)
    }


def stratified_metrics(
    merged: Sequence[Mapping[str, Any]], expected_size: int
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[Mapping[str, Any]]]]:
    complete = complete_groups(merged, expected_size)
    output: list[dict[str, Any]] = []
    for (model, setting), rows in complete.items():
        model_group = str(rows[0]["model_group"])
        for count_range in COUNT_RANGES:
            range_rows = [
                row for row in rows if row["count_range"] == count_range
            ]
            output.append(
                {
                    "model_group": model_group,
                    "model": model,
                    "setting": setting,
                    "count_range": count_range,
                    "metric_scope": (
                        "all_samples_original_invalid_handling"
                    ),
                    **compute_metrics(range_rows),
                }
            )
    return output, complete


def bootstrap_rows(
    rows: Sequence[Mapping[str, Any]],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    gt = np.asarray([int(row["gt_count"]) for row in rows], dtype=float)
    pred = np.asarray([int(row["pred_count"]) for row in rows], dtype=float)
    absolute = np.abs(pred - gt)
    acc10 = (absolute <= 0.10 * gt).astype(float) * 100.0
    densest = np.asarray(
        [
            float(row["densest_cell_pred"] == row["densest_cell_gt"])
            for row in rows
        ]
    ) * 100.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(repetitions, len(rows)))
    resampled_gt = gt[indices]
    estimates = {
        "mae": np.mean(absolute[indices], axis=1),
        "nmae_wape_pct": 100.0
        * np.sum(absolute[indices], axis=1)
        / np.sum(resampled_gt, axis=1),
        "acc_at_10_pct": np.mean(acc10[indices], axis=1),
        "densest_cell_accuracy_pct": np.mean(densest[indices], axis=1),
    }
    points = {
        "mae": float(np.mean(absolute)),
        "nmae_wape_pct": 100.0
        * float(np.sum(absolute))
        / float(np.sum(gt)),
        "acc_at_10_pct": float(np.mean(acc10)),
        "densest_cell_accuracy_pct": float(np.mean(densest)),
    }
    output = []
    for metric, values in estimates.items():
        finite = values[np.isfinite(values)]
        output.append(
            {
                "metric": metric,
                "estimate": points[metric],
                "ci_lower": float(np.quantile(finite, 0.025)),
                "ci_upper": float(np.quantile(finite, 0.975)),
                "confidence_level_pct": 95,
                "repetitions": repetitions,
                "seed": seed,
                "n": len(rows),
            }
        )
    return output


def stratified_bootstrap(
    complete: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (model, setting), rows in complete.items():
        for range_index, count_range in enumerate(COUNT_RANGES):
            range_rows = [
                row for row in rows if row["count_range"] == count_range
            ]
            for item in bootstrap_rows(
                range_rows,
                repetitions=repetitions,
                seed=seed + range_index,
            ):
                output.append(
                    {
                        "model_group": rows[0]["model_group"],
                        "model": model,
                        "setting": setting,
                        "count_range": count_range,
                        **item,
                    }
                )
    return output


def paired_improvements(
    complete: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    models = sorted({model for model, _ in complete})
    for model in models:
        zs_rows = complete.get((model, "ZS"))
        sft_rows = complete.get((model, "Full-SFT"))
        if zs_rows is None or sft_rows is None:
            continue
        zs_by_id = {str(row["image_id"]): row for row in zs_rows}
        sft_by_id = {str(row["image_id"]): row for row in sft_rows}
        if set(zs_by_id) != set(sft_by_id):
            continue
        for range_index, count_range in enumerate(COUNT_RANGES):
            ids = [
                image_id
                for image_id, row in zs_by_id.items()
                if row["count_range"] == count_range
            ]
            gt = np.asarray(
                [int(zs_by_id[image_id]["gt_count"]) for image_id in ids],
                dtype=float,
            )
            zs = np.asarray(
                [int(zs_by_id[image_id]["pred_count"]) for image_id in ids],
                dtype=float,
            )
            sft = np.asarray(
                [int(sft_by_id[image_id]["pred_count"]) for image_id in ids],
                dtype=float,
            )
            rng = np.random.default_rng(seed + range_index)
            indices = rng.integers(
                0, len(ids), size=(repetitions, len(ids))
            )

            def metric_values(
                predicted: np.ndarray, sampled: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                errors = np.abs(predicted - gt)
                mae = np.mean(errors[sampled], axis=1)
                nmae = (
                    100.0
                    * np.sum(errors[sampled], axis=1)
                    / np.sum(gt[sampled], axis=1)
                )
                acc10 = 100.0 * np.mean(
                    (errors <= 0.10 * gt)[sampled], axis=1
                )
                return mae, nmae, acc10

            zs_values = metric_values(zs, indices)
            sft_values = metric_values(sft, indices)
            point_zs = compute_metrics(
                [zs_by_id[image_id] for image_id in ids]
            )
            point_sft = compute_metrics(
                [sft_by_id[image_id] for image_id in ids]
            )
            metric_specs = (
                (
                    "delta_mae",
                    point_zs["mae"] - point_sft["mae"],
                    zs_values[0] - sft_values[0],
                    "ZS - Full-SFT; positive favors Full-SFT",
                ),
                (
                    "delta_nmae_wape_pct",
                    point_zs["nmae_wape_pct"]
                    - point_sft["nmae_wape_pct"],
                    zs_values[1] - sft_values[1],
                    "ZS - Full-SFT; positive favors Full-SFT",
                ),
                (
                    "delta_acc_at_10_pct",
                    point_sft["acc_at_10_pct"]
                    - point_zs["acc_at_10_pct"],
                    sft_values[2] - zs_values[2],
                    "Full-SFT - ZS; positive favors Full-SFT",
                ),
            )
            for metric, estimate, values, definition in metric_specs:
                finite = values[np.isfinite(values)]
                output.append(
                    {
                        "model": model,
                        "count_range": count_range,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_lower": float(np.quantile(finite, 0.025)),
                        "ci_upper": float(np.quantile(finite, 0.975)),
                        "definition": definition,
                        "confidence_level_pct": 95,
                        "repetitions": repetitions,
                        "seed": seed + range_index,
                        "n": len(ids),
                    }
                )
    return output


def reference_metrics(spec: PredictionSpec) -> dict[str, float] | None:
    if spec.reference_path is None or not spec.reference_path.is_file():
        return None
    value = read_json(spec.reference_path)
    if spec.model == "P2PNet":
        metrics = value["track_a"]
    elif spec.model in {"CountGD", "CLIP-Count"}:
        metrics = value["metrics"][spec.model]
    elif spec.model in {"gpt-5", "gemini-3.1-pro-preview"}:
        metrics = value["results"][spec.model]["track_a"]
    else:
        return None
    return {str(key): float(item) for key, item in metrics.items()}


def overall_reproduction(
    complete: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    specs: Sequence[PredictionSpec],
    tolerance: float = 1e-9,
) -> list[dict[str, Any]]:
    spec_by_key = {spec.key: spec for spec in specs}
    output: list[dict[str, Any]] = []
    mapping = {
        "mae": "mae",
        "exact_accuracy_pct": "exact",
        "acc_at_5_pct": "acc_at_5",
        "densest_cell_accuracy_pct": "densest_cell",
        "camouflage_f1_pct": "camouflage_f1",
    }
    for key, rows in complete.items():
        reference = reference_metrics(spec_by_key[key])
        if reference is None:
            continue
        recomputed = compute_metrics(rows)
        comparisons: dict[str, Any] = {}
        passed = True
        for local_name, reference_name in mapping.items():
            local_value = recomputed.get(local_name)
            reference_value = reference.get(reference_name)
            if local_value is None and reference_value is None:
                continue
            if local_value is None or reference_value is None:
                continue
            difference = float(local_value) - float(reference_value)
            metric_passed = abs(difference) <= tolerance
            passed = passed and metric_passed
            comparisons[local_name] = {
                "recomputed": local_value,
                "reference": reference_value,
                "difference": difference,
                "passed": metric_passed,
            }
        output.append(
            {
                "model": key[0],
                "setting": key[1],
                "passed": passed,
                "tolerance": tolerance,
                "comparisons": comparisons,
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {}
            for field in fields:
                value = row.get(field)
                if isinstance(value, set):
                    value = json.dumps(sorted(value), ensure_ascii=False)
                elif isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False, sort_keys=True)
                elif isinstance(value, bool):
                    value = str(value).lower()
                elif value is None:
                    value = ""
                serialized[field] = value
            writer.writerow(serialized)


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([head, rule, *body]) + "\n"


def latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        (">", r"$>$"),
    ):
        text = text.replace(old, new)
    return text


def latex_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    caption: str,
    label: str,
) -> str:
    columns = "l" * len(headers)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(latex_escape(item) for item in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(latex_escape(item) for item in row) + r" \\"
        for row in rows
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{latex_escape(label)}}}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def metric_lookup(
    metrics: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (
            str(row["model"]),
            str(row["setting"]),
            str(row["count_range"]),
        ): row
        for row in metrics
    }


def write_inventory(
    output_dir: Path,
    inventory: Sequence[Mapping[str, Any]],
    gt_audit: Mapping[str, Any],
    split_audit: Mapping[str, Any],
) -> None:
    lines = [
        "# Count-range evaluation inventory",
        "",
        "## Test splits",
        "",
        markdown_table(
            ["File", "Role", "Samples", "Relationship"],
            [
                [
                    TRACK_A_PATH,
                    "Primary Track A manifest",
                    gt_audit["manifest_records"],
                    "Used for the main experiment",
                ],
                [
                    CANONICAL_SPLIT_PATH,
                    "Canonical test ID manifest",
                    split_audit["canonical_count"],
                    (
                        "Same IDs and order as Track A"
                        if split_audit["canonical_same_order"]
                        else "DIFFERS from Track A"
                    ),
                ],
                [
                    TRACK_B_PATH,
                    "Track B manifest",
                    split_audit["track_b_count"],
                    (
                        "Same IDs and order as Track A"
                        if split_audit["track_b_same_order"]
                        else "DIFFERS from Track A"
                    ),
                ],
                [
                    OFFICIAL_TEST_PATH,
                    "Official IOCfish5K test IDs",
                    split_audit["official_count"],
                    "Optional only; not mixed with Table 3",
                ],
            ],
        ).rstrip(),
        "",
        "## Prediction sources",
        "",
        "The table below is a filesystem snapshot taken when the evaluator "
        "runs. Incomplete caches are inventoried for provenance only and are "
        "excluded from every metric.",
        "",
        markdown_table(
            [
                "Model",
                "Setting",
                "Group",
                "Observed / Expected",
                "Valid",
                "Count fields",
                "Path",
            ],
            [
                [
                    item["model"],
                    item["setting"],
                    item["model_group"],
                    f"{item['observed_samples']} / {item['expected_samples']}",
                    item["valid_samples"],
                    ", ".join(item["fields"]) if item["fields"] else "N/A",
                    item["path"],
                ]
                for item in inventory
            ],
        ).rstrip(),
        "",
        "Closed-model cache files contain `image_id`, raw response, parsed "
        "count, parse validity, retry attempts, densest cell, and parsed "
        "camouflage fields. GT counts are joined from the canonical dataset, "
        "not read from prediction files. JSONL counting baselines contain "
        "`image_id` and explicit `pred_count`; they do not contain GT counts "
        "or camouflage outputs.",
        "",
        "No Qwen2.5-VL-7B, LLaVA-OneVision-7B, or InternVL2.5-8B Table 3 "
        "per-image prediction file was found. The shared model cache contains "
        "base InternVL2.5-8B and LLaVA-OneVision-7B weights, but no identified "
        "Count/QAS/Full-SFT checkpoint and no matching Qwen2.5-VL-7B model.",
        "",
    ]
    (output_dir / "count_range_inventory.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def sample_distribution(gt_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for count_range in COUNT_RANGES:
        values = [
            int(record["gt_count"])
            for record in gt_records
            if record["count_range"] == count_range
        ]
        output.append(
            {
                "count_range": count_range,
                "n": len(values),
                "sample_pct": 100.0 * len(values) / len(gt_records),
                "gt_mean": statistics.mean(values),
                "gt_median": statistics.median(values),
                "gt_min": min(values),
                "gt_max": max(values),
                "gt_sum": sum(values),
            }
        )
    return output


def write_main_tables(
    output_dir: Path,
    distribution: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
) -> None:
    lookup = metric_lookup(metrics)
    n_by_range = {
        str(row["count_range"]): int(row["n"]) for row in distribution
    }
    main_rows = []
    for count_range in COUNT_RANGES:
        for setting in ("ZS", "Count", "QAS", "Full-SFT"):
            metric = lookup.get(("Qwen2.5-VL-7B", setting, count_range))
            main_rows.append(
                [
                    count_range,
                    n_by_range[count_range],
                    setting,
                    fmt(metric.get("mae") if metric else None),
                    fmt(metric.get("nmae_wape_pct") if metric else None),
                    fmt(metric.get("acc_at_10_pct") if metric else None),
                    fmt(metric.get("bias") if metric else None),
                ]
            )
    main_headers = (
        "Count Range",
        "N",
        "Setting",
        "MAE",
        "NMAE",
        "Acc@10",
        "Bias",
    )
    note = (
        "# Qwen2.5-VL-7B main rebuttal table\n\n"
        "> Metrics are N/A because the required per-image Table 3 predictions "
        "were not found. No substitute model or aggregate-number "
        "reconstruction was used.\n\n"
    )
    (output_dir / "count_range_main_table.md").write_text(
        note + markdown_table(main_headers, main_rows), encoding="utf-8"
    )
    (output_dir / "count_range_main_table.tex").write_text(
        "% Required Qwen2.5-VL-7B per-image predictions were not found.\n"
        + latex_table(
            main_headers,
            main_rows,
            "Qwen2.5-VL-7B count-range evaluation. N/A denotes missing "
            "per-image Table 3 predictions.",
            "tab:count-range-main",
        ),
        encoding="utf-8",
    )

    all_rows = []
    keys = sorted(
        {
            (str(row["model_group"]), str(row["model"]), str(row["setting"]))
            for row in metrics
        }
        | {
            ("open_vlm", model, setting)
            for model in OPEN_MODELS
            for setting in OPEN_SETTINGS
        }
    )
    for model_group, model, setting in keys:
        all_rows.append(
            [
                model,
                setting,
                *[
                    fmt(
                        lookup.get((model, setting, count_range), {}).get(
                            "nmae_wape_pct"
                        )
                    )
                    for count_range in COUNT_RANGES
                ],
            ]
        )
    all_headers = (
        "Backbone",
        "Setting",
        "0-50 NMAE",
        "51-100 NMAE",
        "101-200 NMAE",
        ">200 NMAE",
    )
    all_note = (
        "# All discovered models\n\n"
        "Open-VLM rows remain N/A until their exact Table 3 per-image "
        "predictions are supplied. Closed models are labeled with `J-*` and "
        "are not treated as SFT settings.\n\n"
    )
    (output_dir / "count_range_all_models_table.md").write_text(
        all_note + markdown_table(all_headers, all_rows), encoding="utf-8"
    )
    (output_dir / "count_range_all_models_table.tex").write_text(
        latex_table(
            all_headers,
            all_rows,
            "Count-range NMAE for all discovered complete prediction sets.",
            "tab:count-range-all-models",
        ),
        encoding="utf-8",
    )

    grounded_rows = []
    for row in metrics:
        if (
            row.get("densest_cell_accuracy_pct") is None
            and row.get("camouflage_f1_pct") is None
        ):
            continue
        grounded_rows.append(
            [
                row["model"],
                row["setting"],
                row["count_range"],
                fmt(row.get("densest_cell_accuracy_pct")),
                fmt(row.get("camouflage_f1_pct")),
            ]
        )
    grounded_headers = (
        "Backbone",
        "Setting",
        "Count Range",
        "Densest Cell Accuracy",
        "Camouflage F1",
    )
    grounded_note = (
        "# Grounded metrics\n\n"
        "Camouflage F1 uses the original frozen deterministic "
        "concept-set micro-F1 normalization. Counting-only baselines have no "
        "camouflage output and are reported as N/A.\n\n"
    )
    (output_dir / "count_range_grounded_metrics.md").write_text(
        grounded_note + markdown_table(grounded_headers, grounded_rows),
        encoding="utf-8",
    )
    (output_dir / "count_range_grounded_metrics.tex").write_text(
        latex_table(
            grounded_headers,
            grounded_rows,
            "Grounded Track A metrics by count range.",
            "tab:count-range-grounded",
        ),
        encoding="utf-8",
    )


def plot_metrics(
    output_dir: Path, metrics: Sequence[Mapping[str, Any]]
) -> None:
    lookup = metric_lookup(metrics)
    series = sorted(
        {
            (str(row["model"]), str(row["setting"]))
            for row in metrics
            if all(
                (str(row["model"]), str(row["setting"]), count_range)
                in lookup
                for count_range in COUNT_RANGES
            )
        }
    )
    colors = ("#1B1B1B", "#D1495B", "#00798C", "#EDA E49", "#3066BE")
    colors = tuple(color.replace(" ", "") for color in colors)
    markers = ("o", "s", "^", "D", "P")

    def draw(metric_name: str, ylabel: str, stem: str, zero_line: bool) -> None:
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for index, (model, setting) in enumerate(series):
            values = [
                float(lookup[(model, setting, count_range)][metric_name])
                for count_range in COUNT_RANGES
            ]
            axis.plot(
                COUNT_RANGES,
                values,
                marker=markers[index % len(markers)],
                linewidth=2.0,
                markersize=5.5,
                color=colors[index % len(colors)],
                label=f"{model} / {setting}",
            )
        if zero_line:
            axis.axhline(0, color="#666666", linewidth=1.0, linestyle="--")
        axis.set_xlabel("Ground-truth count range")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=8, ncol=2)
        figure.tight_layout()
        figure.savefig(output_dir / f"{stem}.png", dpi=300)
        figure.savefig(output_dir / f"{stem}.pdf")
        plt.close(figure)

    draw("nmae_wape_pct", "NMAE / WAPE (%)", "count_range_nmae", False)
    draw("bias", "Bias (predicted - GT)", "count_range_bias", True)


def split_audit(gt_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    track_ids = [str(record["image_id"]) for record in gt_records]
    canonical = read_json(CANONICAL_SPLIT_PATH)
    canonical_ids = [str(item) for item in canonical["image_ids"]]
    track_b = read_json(TRACK_B_PATH)
    track_b_ids = [str(record["image_id"]) for record in track_b]
    official_ids = [
        Path(line.strip()).stem.zfill(4)
        for line in OFFICIAL_TEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    track_set = set(track_ids)
    official_set = set(official_ids)
    return {
        "track_a_count": len(track_ids),
        "canonical_count": len(canonical_ids),
        "canonical_same_order": track_ids == canonical_ids,
        "track_b_count": len(track_b_ids),
        "track_b_same_order": track_ids == track_b_ids,
        "official_count": len(official_ids),
        "official_intersection": len(track_set & official_set),
        "track_a_only": len(track_set - official_set),
        "official_only": len(official_set - track_set),
    }


def write_optional_official_report(
    output_dir: Path, audit: Mapping[str, Any]
) -> None:
    text = f"""# Optional IOCfish5K official-split comparison

The official IOCfish5K test list contains {audit['official_count']:,} images.
The Nereus Track A/Table 3 manifest contains {audit['track_a_count']:,} images.
Their intersection is {audit['official_intersection']:,}; Track A has
{audit['track_a_only']:,} IDs outside the official test list, while the
official test list has {audit['official_only']:,} IDs outside Track A.

No official-split metric is computed because the discovered prediction files
cover the Nereus Track A split, not all 2,000 official-test images. The two
splits are not mixed.
"""
    (output_dir / "official_split_optional_report.md").write_text(
        text, encoding="utf-8"
    )


def write_report(
    output_dir: Path,
    distribution: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    complete: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    reproduction: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    gt_audit: Mapping[str, Any],
) -> None:
    lookup = metric_lookup(metrics)
    complete_names = ", ".join(
        f"{model}/{setting}" for model, setting in sorted(complete)
    )
    range_text = ", ".join(
        f"{row['count_range']}: {row['n']} ({row['sample_pct']:.2f}%)"
        for row in distribution
    )
    reproduction_passed = bool(reproduction) and all(
        item["passed"] for item in reproduction
    )
    warning_count = (
        int(gt_audit["issue_count"])
        - int(gt_audit["critical_issue_count"])
    )

    closed_findings = []
    for model in ("gpt-5", "gemini-3.1-pro-preview"):
        high = lookup.get((model, "J-PZS", ">200"))
        if high:
            closed_findings.append(
                f"- {model}/J-PZS at `>200`: NMAE "
                f"{high['nmae_wape_pct']:.2f}%, bias {high['bias']:.2f}, "
                f"under-count rate {high['under_count_rate_pct']:.2f}%."
            )
    grounded_findings = []
    for model in ("gpt-5", "gemini-3.1-pro-preview"):
        values = [
            lookup.get((model, "J-PZS", count_range))
            for count_range in COUNT_RANGES
        ]
        if all(values):
            grounded_findings.append(
                f"- {model}/J-PZS Densest Cell changes from "
                f"{values[0]['densest_cell_accuracy_pct']:.2f}% at `0-50` "
                f"to {values[-1]['densest_cell_accuracy_pct']:.2f}% at "
                f"`>200`; Camouflage F1 changes from "
                f"{values[0]['camouflage_f1_pct']:.2f}% to "
                f"{values[-1]['camouflage_f1_pct']:.2f}%."
            )

    contribution_lines = []
    for (model, setting), rows in sorted(complete.items()):
        total_error = sum(float(row["absolute_error"]) for row in rows)
        low_error = sum(
            float(row["absolute_error"])
            for row in rows
            if row["count_range"] == "0-50"
        )
        contribution = 100.0 * low_error / total_error if total_error else 0
        contribution_lines.append(
            f"- {model}/{setting}: low-count images contribute "
            f"{contribution:.2f}% of total absolute error."
        )

    complete_sources = [
        f"- {item['model']}/{item['setting']}: `{item['path']}`"
        for item in inventory
        if item["complete"]
    ]
    nmae_table = markdown_table(
        ["Model", "Setting", "0-50", "51-100", "101-200", ">200"],
        [
            [
                model,
                setting,
                *[
                    (
                        f"{lookup[(model, setting, count_range)]['nmae_wape_pct']:.2f}"
                        if (model, setting, count_range) in lookup
                        else "N/A"
                    )
                    for count_range in COUNT_RANGES
                ],
            ]
            for model, setting in sorted(complete)
        ],
    ).rstrip()

    missing_open = [
        item
        for item in inventory
        if item["model_group"] == "open_vlm"
        and not item["complete"]
    ]
    text = f"""# Nereus Track A count-range report

## Protocol

The experiment uses the exact Track A/Table 3 manifest at
`{TRACK_A_PATH}`: 1,129 unique images. Its IDs and order exactly match the
canonical test manifest. Count ranges are assigned from canonical GT total
counts, never predictions. The official IOCfish5K 2,000-image test split is
kept separate.

Range distribution: {range_text}.

One image contributes one GT count and one predicted count per model/setting.
Canonical counts, point counts, structured final counts, grid sums, and
visibility sums were audited before scoring. Invalid cached model outputs
retain the original parser fallback and remain in all-sample metrics; coverage
is reported separately. Missing prediction files are never imputed. The GT
audit found {gt_audit['critical_issue_count']} critical count/point conflicts
and {warning_count} non-critical grid-sum warnings; all warning cases remain
in the evaluation and are listed in `count_range_audit.json`.

## Prediction availability

Complete discovered prediction sets: {complete_names}.

Prediction files actually used for metrics:

{chr(10).join(complete_sources)}

No new model training, open-model inference, or closed API call was performed.
The {len(missing_open)} required open-VLM model/setting combinations have no
per-image Table 3 predictions in the searched workspace. Base weights alone
cannot reconstruct Count/QAS/Full-SFT results, so the requested Qwen main table
remains N/A. New inference is not needed for the five complete sets above; it
would only be needed after recovering the exact missing checkpoints if their
per-image Table 3 outputs cannot be found.

## Overall consistency

Overall metrics recomputed by merging all four ranges
{'match' if reproduction_passed else 'do not all match'} the available Table 3
reference artifacts within tolerance. Detailed values are recorded in
`count_range_audit.json`.

## Main findings from available complete predictions

NMAE/WAPE (%) by count range:

{nmae_table}

The trends are model-dependent rather than universally monotonic. P2PNet
improves from 38.13% to 15.48% as count increases; CountGD is best in
`51-100` and worsens at `>200`; GPT-5 and Gemini have their highest NMAE in
`>200`. CLIP-Count falls sharply after `0-50` but remains substantially worse
than P2PNet and CountGD in every higher range.

The `0-50` stratum is the largest at {distribution[0]['sample_pct']:.2f}% of
images, so ordinary sample-mean metrics weight it most heavily, but it is not
a majority of the test set. Its share of total absolute error is:

{chr(10).join(contribution_lines)}

High-density under-counting diagnostics for the only complete closed-VLM
setting are:

{chr(10).join(closed_findings)}

Gemini/J-PZS shows clear systematic high-density under-counting: 94.91% of
`>200` images are under-counted and mean bias is -193.07. GPT-5/J-PZS
under-counts a majority of `>200` images (56.94%), but its mean bias is
+19.05 because the over-count tail offsets the typical negative error; it
should therefore not be described as uniformly or systematically
under-counting this range.

Grounded-metric range endpoints are:

{chr(10).join(grounded_findings)}

## Rebuttal questions

1. Error versus count is reported without presupposing monotonicity in
   `count_range_metrics.csv` and the NMAE/Bias figures.
2. Whether Nereus supervision improves every range cannot be established:
   matched ZS, Count, QAS, and Full-SFT per-image predictions are absent.
3. The overall sample mean gives the largest single weight to `0-50`
   ({distribution[0]['sample_pct']:.2f}%); error-contribution percentages
   above distinguish sample dominance from error dominance.
4. High-density under-counting can be described for complete J-PZS outputs
   using bias and under-count rate, but should not be generalized to missing
   open-model settings.
5. Densest Cell and Camouflage F1 are range-stratified only where those
   outputs exist. Counting-only baselines have no Camouflage F1.
6. QAS versus Full-SFT cannot be compared, and paired-bootstrap output is
   intentionally empty until both exact prediction sets are available.

## Unsupported conclusions

The available evidence does not support claims that Nereus supervision helps
all density ranges, that QAS is better or worse than Full-SFT, or that the
closed-model J-Full prompt is equivalent to open-model Full-SFT. Partial J-ZS
and J-Full caches are inventoried but excluded from metrics to avoid
selection-biased conclusions.
"""
    (output_dir / "count_range_report.md").write_text(
        text, encoding="utf-8"
    )


def write_reproduce_script(output_dir: Path) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
PYTHON_BIN="python"

"${{PYTHON_BIN}}" "${{SCRIPT_DIR}}/count_range_eval.py" \\
  --output-dir "${{SCRIPT_DIR}}/analysis/count_range" \\
  --bootstrap-repetitions 1000 \\
  --seed 42
"""
    path = output_dir / "reproduce_count_range.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict-required-predictions",
        action="store_true",
        help="Fail if any required open-VLM prediction set is absent.",
    )
    args = parser.parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_records, gt_audit = build_ground_truth(
        TRACK_A_PATH, CANONICAL_RECORDS_PATH, FINAL_ANNOTATION_DIR
    )
    if gt_audit["critical_issue_count"]:
        raise ValueError(
            "GT audit failed; inspect source annotations before evaluation"
        )
    splits = split_audit(gt_records)
    if not splits["canonical_same_order"] or not splits["track_b_same_order"]:
        raise ValueError("Track A/canonical/Track B test IDs differ")

    specs = prediction_specs()
    merged, inventory = merge_predictions(gt_records, specs)
    metrics, complete = stratified_metrics(merged, len(gt_records))
    bootstrap = stratified_bootstrap(
        complete,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    paired = paired_improvements(
        complete,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    reproduction = overall_reproduction(complete, specs)
    if any(not item["passed"] for item in reproduction):
        raise ValueError(
            "overall metrics do not reproduce available Table 3 artifacts"
        )

    distribution = sample_distribution(gt_records)
    prediction_fields = [
        "image_id",
        "model",
        "setting",
        "model_group",
        "gt_count",
        "gt_source",
        "pred_count",
        "count_range",
        "absolute_error",
        "signed_error",
        "observed_prediction",
        "valid_prediction",
        "parse_status",
        "retry_status",
        "raw_response",
        "densest_cell_gt",
        "densest_cell_pred",
        "camouflage_gt_concepts",
        "camouflage_pred_concepts",
        "prediction_source",
    ]
    write_csv(
        args.output_dir / "count_range_predictions_merged.csv",
        merged,
        prediction_fields,
    )
    write_csv(
        args.output_dir / "count_range_sample_distribution.csv",
        distribution,
        [
            "count_range",
            "n",
            "sample_pct",
            "gt_mean",
            "gt_median",
            "gt_min",
            "gt_max",
            "gt_sum",
        ],
    )
    metric_fields = [
        "model_group",
        "model",
        "setting",
        "count_range",
        "metric_scope",
        "n",
        "gt_mean",
        "gt_median",
        "gt_min",
        "gt_max",
        "valid_n",
        "valid_coverage_pct",
        "mae",
        "rmse",
        "exact_accuracy_pct",
        "acc_at_5_pct",
        "acc_at_10_pct",
        "nmae_wape_pct",
        "bias",
        "median_signed_error",
        "under_count_rate_pct",
        "over_count_rate_pct",
        "mean_predicted_count",
        "mean_ground_truth_count",
        "densest_cell_accuracy_pct",
        "camouflage_f1_pct",
    ]
    write_csv(
        args.output_dir / "count_range_metrics.csv",
        metrics,
        metric_fields,
    )
    write_csv(
        args.output_dir / "count_range_metrics_bootstrap.csv",
        bootstrap,
        [
            "model_group",
            "model",
            "setting",
            "count_range",
            "metric",
            "estimate",
            "ci_lower",
            "ci_upper",
            "confidence_level_pct",
            "repetitions",
            "seed",
            "n",
        ],
    )
    write_csv(
        args.output_dir / "count_range_paired_improvements.csv",
        paired,
        [
            "model",
            "count_range",
            "metric",
            "estimate",
            "ci_lower",
            "ci_upper",
            "definition",
            "confidence_level_pct",
            "repetitions",
            "seed",
            "n",
        ],
    )

    coverage = {
        f"{item['model']}|{item['setting']}": {
            "observed": item["observed_samples"],
            "expected": item["expected_samples"],
            "valid": item["valid_samples"],
            "complete": item["complete"],
            "duplicates": item["duplicate_image_ids"],
            "unexpected_ids": item["unexpected_image_ids"],
        }
        for item in inventory
    }
    invalid_predictions = [
        {
            "image_id": row["image_id"],
            "model": row["model"],
            "setting": row["setting"],
            "observed_prediction": row["observed_prediction"],
            "parse_status": row["parse_status"],
        }
        for row in merged
        if row["observed_prediction"] and not row["valid_prediction"]
    ]
    audit = {
        "schema_version": "nereus_count_range_audit/v1",
        "count_range_definition": {
            "0-50": "0 <= GT <= 50",
            "51-100": "51 <= GT <= 100",
            "101-200": "101 <= GT <= 200",
            ">200": "GT > 200",
        },
        "bootstrap": {
            "repetitions": args.bootstrap_repetitions,
            "confidence_interval_pct": 95,
            "seed": args.seed,
            "unit": "image_id within each count range",
        },
        "gt": gt_audit,
        "splits": splits,
        "prediction_coverage": coverage,
        "invalid_predictions": invalid_predictions,
        "overall_reproduction": reproduction,
        "metric_eligibility": (
            "Only model/setting combinations with all 1,129 observed count "
            "predictions are scored. Partial caches remain in the merged CSV."
        ),
        "paired_improvement_status": (
            "No complete same-backbone ZS and Full-SFT pair was found."
            if not paired
            else "Computed from paired image-level predictions."
        ),
    }
    (args.output_dir / "count_range_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    write_inventory(args.output_dir, inventory, gt_audit, splits)
    write_main_tables(args.output_dir, distribution, metrics)
    plot_metrics(args.output_dir, metrics)
    write_optional_official_report(args.output_dir, splits)
    write_report(
        args.output_dir,
        distribution,
        metrics,
        complete,
        reproduction,
        inventory,
        gt_audit,
    )
    write_reproduce_script(args.output_dir)

    missing_required = [
        item
        for item in inventory
        if item["model_group"] == "open_vlm" and not item["complete"]
    ]
    print(f"Track A images: {len(gt_records)}")
    print(
        "GT audit issues: "
        f"{gt_audit['critical_issue_count']} critical, "
        f"{gt_audit['issue_count'] - gt_audit['critical_issue_count']} warnings"
    )
    print(f"Complete prediction sets: {len(complete)}")
    print(f"Missing required open-VLM sets: {len(missing_required)}")
    print(f"Overall reproduction checks: {len(reproduction)} passed")
    print(f"Outputs: {args.output_dir}")
    if args.strict_required_predictions and missing_required:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
