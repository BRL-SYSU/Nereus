"""Strict shared Track A evaluation for external counting models."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from Experiments.p2pnet_nereus.grid_metrics import (
    aggregate_metrics,
    densest_region,
    grid_counts,
)

from .adapters import validate_prediction_record
from .config import EXPECTED_TEST_SIZE, ExternalExperimentPaths
from .io_utils import sha256_file


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
    if len(prediction_records) != expected_size:
        raise ValueError(
            f"expected {expected_size} predictions, "
            f"got {len(prediction_records)}"
        )
    targets = _unique_by_id(target_records, "target")
    predictions = _unique_by_id(prediction_records, "prediction")
    if set(targets) != set(predictions):
        missing = sorted(set(targets) - set(predictions))
        extra = sorted(set(predictions) - set(targets))
        raise ValueError(
            "prediction image ID set mismatch; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    metric_targets: list[dict[str, object]] = []
    metric_predictions: list[dict[str, object]] = []
    methods: set[str] = set()
    checkpoints: set[str] = set()
    prompts: set[str] = set()
    for image_id in sorted(targets):
        target = targets[image_id]
        prediction = predictions[image_id]
        validate_prediction_record(prediction)
        width = int(target["width"])
        height = int(target["height"])
        if (
            int(prediction["image_width"]) != width
            or int(prediction["image_height"]) != height
        ):
            raise ValueError(f"image size mismatch for {image_id}")
        points = np.asarray(target["points_xy"], dtype=np.float64).reshape(
            -1, 2
        )
        if len(points) != int(target["count"]):
            raise ValueError(f"target count mismatch for {image_id}")
        target_densest = densest_region(grid_counts(points, width, height))
        metric_targets.append(
            {"count": int(target["count"]), "densest": target_densest}
        )
        metric_predictions.append(
            {
                "count": int(prediction["pred_count"]),
                "densest": str(prediction["pred_densest_grid_region"]),
            }
        )
        methods.add(str(prediction["method"]))
        checkpoints.add(str(prediction["checkpoint_sha256"]))
        prompts.add(str(prediction["text_query"]))

    if len(methods) != 1:
        raise ValueError("prediction file contains multiple methods")
    if len(checkpoints) != 1:
        raise ValueError("prediction file contains multiple checkpoints")
    if prompts != {"fish"}:
        raise ValueError("all predictions must use the frozen fish prompt")
    return aggregate_metrics(metric_targets, metric_predictions)


def render_table3_row(
    method: str, setting: str, metrics: Mapping[str, float]
) -> str:
    return (
        f"| {method} | {setting} | "
        f"{metrics['mae']:.2f} | {metrics['rmse']:.2f} | "
        f"{metrics['exact']:.2f} | {metrics['acc_at_5']:.2f} | "
        f"{metrics['acc_at_10']:.2f} | "
        f"{metrics['densest_cell']:.2f} | N/A | N/A | N/A |"
    )


def render_table3(rows: Sequence[str]) -> str:
    if not rows:
        raise ValueError("at least one Table 3 row is required")
    return (
        "| Backbone | Setting | MAE | RMSE | Exact | Acc@5 | Acc@10 | "
        "Densest Cell | Camouflage F1 | Track B | Track C |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def _atomic_text(text: str, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    defaults = ExternalExperimentPaths.defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=defaults.test_targets)
    parser.add_argument(
        "--countgd-predictions",
        type=Path,
        default=defaults.output_root / "countgd_predictions.jsonl",
    )
    parser.add_argument(
        "--clipcount-predictions",
        type=Path,
        default=defaults.output_root / "clipcount_predictions.jsonl",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=defaults.output_root / "metrics.json",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=defaults.output_root / "table3_track_a.md",
    )
    args = parser.parse_args()

    targets = _read_jsonl(args.targets)
    methods = (
        ("CountGD", args.countgd_predictions),
        ("CLIP-Count", args.clipcount_predictions),
    )
    metrics_by_method: dict[str, dict[str, float]] = {}
    artifact_report: dict[str, dict[str, object]] = {}
    rows: list[str] = []
    setting = "FSC-147 pretrained Direct-ZS, text=fish"
    for method, prediction_path in methods:
        predictions = _read_jsonl(prediction_path)
        if {str(record.get("method")) for record in predictions} != {
            method
        }:
            raise ValueError(f"method mismatch in {prediction_path}")
        metrics = evaluate_records(
            targets,
            predictions,
            expected_size=EXPECTED_TEST_SIZE,
        )
        metrics_by_method[method] = metrics
        rows.append(render_table3_row(method, setting, metrics))
        artifact_report[method] = {
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "checkpoint_sha256": str(predictions[0]["checkpoint_sha256"]),
            "records": len(predictions),
            "text_query": str(predictions[0]["text_query"]),
        }

    table = render_table3(rows)
    report = {
        "protocol": {
            "name": "Nereus Table 3 Track A external counting",
            "setting": setting,
            "test_records": EXPECTED_TEST_SIZE,
            "nereus_training": False,
            "nereus_validation_calibration": False,
            "countgd_confidence_threshold": 0.23,
            "clipcount_scale_factor": 60.0,
            "integer_count_rule": "floor(max(0, score) + 0.5)",
            "unsupported": {
                "camouflage_f1": "N/A",
                "track_b": "N/A",
                "track_c": "N/A",
            },
        },
        "metrics": metrics_by_method,
        "artifacts": artifact_report,
        "provenance": {
            "CountGD": {
                "repository": "https://github.com/niki-amini-naieni/CountGD",
                "repository_commit": "b6f362b3f5cd20db4a171faa410dfed8f2f466d8",
                "checkpoint_source": (
                    "https://huggingface.co/spaces/nikigoli/countgd/"
                    "resolve/main/checkpoint_best_regular.pth"
                ),
                "checkpoint_source_status": (
                    "first-author-maintained Hugging Face Space; local "
                    "SHA-256 matches the Space LFS object ID"
                ),
            },
            "CLIP-Count": {
                "repository": "https://github.com/songrise/CLIP-Count",
                "repository_commit": "ba57fd0c9cf45e414734b22f7b141da3f0142f19",
                "official_checkpoint_file_id": (
                    "Google Drive 17Dj0tjd29lPGOGYEF5IrE8aPClXUjTrR"
                ),
                "checkpoint_source": (
                    "https://huggingface.co/ozzafar/clip-count/"
                    "resolve/main/pytorch_model.bin"
                ),
                "checkpoint_source_status": (
                    "community mirror; weights-only safe load; all 841 "
                    "parameters strictly match the official default "
                    "architecture with missing=0 and unexpected=0; "
                    "all 302 frozen CLIP tensors exactly match OpenAI "
                    "ViT-B/16; "
                    "bit identity with the inaccessible Google Drive "
                    "file is not independently proven"
                ),
                "openai_clip_vit_b16_sha256": (
                    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
                ),
            },
        },
        "overlap_audit": {
            "pretraining_dataset": "FSC-147",
            "evaluation_dataset": "Nereus/IOCFish5K test split",
            "image_level_duplicate_check": "not completed",
            "reason": (
                "FSC-147 source images were not bundled with either "
                "checkpoint and the dataset metadata endpoint was "
                "unreachable during this run"
            ),
            "known_semantic_overlap": (
                "fish appears in the FSC-147/CountGD training label list"
            ),
            "interpretation": (
                "Nereus-domain direct zero-shot transfer, not "
                "category-unseen zero-shot"
            ),
        },
        "target_manifest": {
            "path": str(args.targets),
            "sha256": sha256_file(args.targets),
        },
        "table3_markdown": table,
    }
    _atomic_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        args.metrics_output,
    )
    _atomic_text(table, args.table_output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
