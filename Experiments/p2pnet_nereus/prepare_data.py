"""Build leakage-audited P2PNet manifests from Nereus annotations."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import EXPECTED_SPLIT_SIZES, ExperimentPaths


def load_points(annotation_path: Path) -> np.ndarray:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    raw_points = [instance["point_xy"] for instance in annotation["instances"]]
    if not raw_points:
        return np.empty((0, 2), dtype=np.float32)
    points = np.asarray(raw_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"invalid point shape in {annotation_path}: {points.shape}")
    width = int(annotation["image"]["width"])
    height = int(annotation["image"]["height"])
    if (
        not np.isfinite(points).all()
        or (points[:, 0] < 0).any()
        or (points[:, 0] > width).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] > height).any()
    ):
        raise ValueError(f"invalid source point in {annotation_path}")
    points[points[:, 0] == width, 0] = width - 1
    points[points[:, 1] == height, 1] = height - 1
    return points


def build_split_records(
    paths: ExperimentPaths,
) -> dict[str, list[dict[str, Any]]]:
    split_ids = json.loads(paths.split_file.read_text(encoding="utf-8"))
    if set(split_ids) != set(EXPECTED_SPLIT_SIZES):
        raise ValueError(f"unexpected split names: {sorted(split_ids)}")

    records: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "val", "test"):
        split_records: list[dict[str, Any]] = []
        for image_id in split_ids[split_name]:
            annotation_path = paths.annotation_dir / f"{image_id}.json"
            image_path = paths.image_dir / f"{image_id}.jpg"
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            points = load_points(annotation_path)
            image = annotation["image"]
            total_count = int(
                annotation["source_annotation"]["num_valid_point_objects"]
            )
            split_records.append(
                {
                    "image_id": str(image_id),
                    "image_path": str(image_path.resolve()),
                    "annotation_path": str(annotation_path.resolve()),
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "count": total_count,
                    "points_xy": points.tolist(),
                }
            )
        records[split_name] = split_records
    return records


def audit_records(
    records: dict[str, list[dict[str, Any]]],
    expected_sizes: dict[str, int] | None = EXPECTED_SPLIT_SIZES,
) -> None:
    if set(records) != {"train", "val", "test"}:
        raise ValueError(f"unexpected record splits: {sorted(records)}")
    if expected_sizes is not None:
        actual_sizes = {name: len(rows) for name, rows in records.items()}
        if actual_sizes != expected_sizes:
            raise ValueError(
                f"split size mismatch: expected {expected_sizes}, got {actual_sizes}"
            )

    ids_by_split: dict[str, set[str]] = {}
    for split_name, split_records in records.items():
        image_ids = [str(record["image_id"]) for record in split_records]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"duplicate image ID in {split_name}")
        ids_by_split[split_name] = set(image_ids)

        for record in split_records:
            image_path = Path(record["image_path"])
            annotation_path = Path(record["annotation_path"])
            if not image_path.is_file():
                raise ValueError(f"missing RGB image: {image_path}")
            if not annotation_path.is_file():
                raise ValueError(f"missing annotation: {annotation_path}")

            width = int(record["width"])
            height = int(record["height"])
            points = np.asarray(record["points_xy"], dtype=np.float64)
            if points.size == 0:
                points = np.empty((0, 2), dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError(
                    f"invalid point shape for {record['image_id']}: {points.shape}"
                )
            if len(points) != int(record["count"]):
                raise ValueError(
                    f"count mismatch for {record['image_id']}: "
                    f"{record['count']} != {len(points)}"
                )
            if not np.isfinite(points).all():
                raise ValueError(f"non-finite point for {record['image_id']}")
            if len(points) and (
                (points[:, 0] < 0).any()
                or (points[:, 0] >= width).any()
                or (points[:, 1] < 0).any()
                or (points[:, 1] >= height).any()
            ):
                raise ValueError(f"out-of-bounds point for {record['image_id']}")

    names = ("train", "val", "test")
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = ids_by_split[left_name] & ids_by_split[right_name]
            if overlap:
                raise ValueError(
                    f"split overlap between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:3]}"
                )


def write_manifests(
    records: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in records.items():
        destination = output_dir / f"{split_name}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for record in split_records:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        os.replace(temporary, destination)

    image_only_destination = output_dir / "test_images.jsonl"
    image_only_temporary = image_only_destination.with_suffix(".jsonl.tmp")
    with image_only_temporary.open("w", encoding="utf-8") as stream:
        for record in records["test"]:
            image_record = {
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "width": record["width"],
                "height": record["height"],
            }
            stream.write(
                json.dumps(image_record, separators=(",", ":")) + "\n"
            )
    os.replace(image_only_temporary, image_only_destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    paths = ExperimentPaths.from_project_root(args.project_root)
    records = build_split_records(paths)
    audit_records(records)
    write_manifests(records, paths.manifest_dir)
    print(
        json.dumps(
            {name: len(rows) for name, rows in records.items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
