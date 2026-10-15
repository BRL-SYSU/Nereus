"""Evaluate closed-VLM predictions with shared Nereus Track A metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from Experiments.p2pnet_nereus.grid_metrics import (
    aggregate_metrics,
    densest_region,
    grid_counts,
)


EXPECTED_TEST_SIZE = 1129
DEFAULT_TARGETS = Path(
    "./"
    "Experiments/p2pnet_nereus/data/test.jsonl"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def evaluate(
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    expected_size: int = EXPECTED_TEST_SIZE,
) -> tuple[dict[str, float], dict[str, int]]:
    if len(targets) != expected_size or len(predictions) != expected_size:
        raise ValueError(
            f"expected {expected_size} targets/predictions, got "
            f"{len(targets)}/{len(predictions)}"
        )
    target_by_id = {str(item["image_id"]): item for item in targets}
    prediction_by_id = {str(item["image_id"]): item for item in predictions}
    if len(target_by_id) != len(targets) or len(prediction_by_id) != len(
        predictions
    ):
        raise ValueError("duplicate image ID")
    if set(target_by_id) != set(prediction_by_id):
        raise ValueError("prediction image IDs do not match targets")

    metric_targets: list[dict[str, object]] = []
    metric_predictions: list[dict[str, object]] = []
    count_parse_failures = 0
    grid_parse_failures = 0
    grid_inconsistencies = 0
    for image_id in sorted(target_by_id):
        target = target_by_id[image_id]
        prediction = prediction_by_id[image_id]
        points = np.asarray(target["points_xy"], dtype=np.float64).reshape(
            -1, 2
        )
        target_densest = densest_region(
            grid_counts(
                points,
                int(target["width"]),
                int(target["height"]),
            )
        )
        metric_targets.append(
            {"count": int(target["count"]), "densest": target_densest}
        )
        metric_predictions.append(
            {
                "count": int(prediction["pred_count"]),
                "densest": str(prediction["pred_densest_grid_region"]),
            }
        )
        count_parse_failures += int(
            not prediction.get("count_parse_valid", False)
        )
        grid_parse_failures += int(
            not prediction.get("grid_parse_valid", False)
        )
        grid_inconsistencies += int(
            prediction.get("grid_consistent") is False
        )
    diagnostics = {
        "count_parse_failures": count_parse_failures,
        "grid_parse_failures": grid_parse_failures,
        "grid_inconsistencies": grid_inconsistencies,
    }
    return aggregate_metrics(metric_targets, metric_predictions), diagnostics


def table_row(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| {name} | P-ZS | {metrics['mae']:.2f} | "
        f"{metrics['rmse']:.2f} | {metrics['exact']:.2f} | "
        f"{metrics['acc_at_5']:.2f} | {metrics['acc_at_10']:.2f} | "
        f"{metrics['densest_cell']:.2f} | N/A | N/A | N/A |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeat for each model",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    targets = _read_jsonl(args.targets)
    results: dict[str, Any] = {}
    rows: list[str] = []
    for specification in args.prediction:
        if "=" not in specification:
            parser.error("--prediction must be NAME=PATH")
        name, raw_path = specification.split("=", 1)
        predictions = _read_jsonl(Path(raw_path))
        metrics, diagnostics = evaluate(targets, predictions)
        results[name] = {"metrics": metrics, "diagnostics": diagnostics}
        rows.append(table_row(name, metrics))

    table = (
        "| Backbone | Setting | MAE | RMSE | Exact | Acc@5 | Acc@10 | "
        "Densest Cell | Camouflage F1 | Track B | Track C |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "table3_track_a.md").write_text(
        table,
        encoding="utf-8",
    )
    print(table)


if __name__ == "__main__":
    main()

