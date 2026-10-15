#!/usr/bin/env python3
"""Build canonical AquaCount-Instruct records keyed by image_id.

Each output record has this logical shape:

image_id
├── RGB underwater image
├── final annotation JSON
├── QA / instruction conversations
└── density heatmap / density overlay / density array

The script writes path-level manifests by default, and can optionally embed QA
payloads for downstream SFT dataset generation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
IMAGE_ID_PATTERN = re.compile(r"^\d{4}$")


def parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Summarize AquaCount-Instruct assets into canonical per-image records.",
    )
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_repo_root / "Experiments" / "canonical_dataset",
        help="Directory for manifest outputs.",
    )
    parser.add_argument(
        "--include-qa",
        action="store_true",
        help="Embed QA conversations/pairs in each JSONL record. Default stores paths and counts only.",
    )
    parser.add_argument(
        "--include-final-annotation",
        action="store_true",
        help="Embed the full final annotation JSON. Default stores path and compact metadata only.",
    )
    parser.add_argument(
        "--make-splits",
        action="store_true",
        help="Also create image-level train/val/test id files from complete records.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--complete-requires",
        nargs="+",
        default=["rgb", "final_annotation", "density", "task1_qa", "task2_qa", "task3_qa"],
        choices=["rgb", "final_annotation", "density", "task1_qa", "task2_qa", "task3_qa"],
        help="Assets required for complete_records.jsonl.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Existing Task1/Task2 files are pretty-printed single JSON objects with a
    # .jsonl suffix. If that fails, fall back to true line-delimited JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_image(images_dir: Path, image_id: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def count_qa_items(path: Path | None, kind: str) -> int:
    if path is None or not path.exists():
        return 0

    payload = read_jsonl_payload(path)
    if kind in {"task1", "task2"}:
        if isinstance(payload, dict):
            return len(payload.get("conversations", [])) // 2
        if isinstance(payload, list):
            return len(payload)
        return 0

    if kind == "task3":
        return len(payload) if isinstance(payload, list) else 0

    return 0


def build_density_assets(density_root: Path, image_id: str) -> dict[str, Any]:
    density_dir = density_root / image_id
    assets = {
        "dir": str(density_dir),
        "density_heatmap": str(density_dir / f"{image_id}_density_heatmap.png"),
        "density_overlay": str(density_dir / f"{image_id}_density_overlay.png"),
        "density_png": str(density_dir / f"{image_id}_density.png"),
        "density_array": str(density_dir / f"{image_id}_density.npy"),
    }
    exists = {key: Path(value).exists() for key, value in assets.items()}
    return {
        "paths": assets,
        "exists": exists,
        "complete": all(exists.values()),
    }


def build_qa_assets(repo_root: Path, image_id: str, include_qa: bool) -> dict[str, Any]:
    task1_path = repo_root / "scripts" / "task1_region" / "QA" / "QA_dataset" / f"{image_id}.jsonl"
    task2_path = repo_root / "scripts" / "task2_Camouflage" / "QAdataset" / f"{image_id}.jsonl"
    task3_dir = repo_root / "scripts" / "task3_density_depth" / "QA" / "VQA Dataset" / image_id
    task3_path = task3_dir / "qa_pairs.json"
    task3_context_path = task3_dir / "generation_context.json"
    task3_stitched_path = task3_dir / f"{image_id}_stitched.png"

    qa_assets: dict[str, Any] = {
        "task1_region": {
            "path": str(task1_path),
            "exists": task1_path.exists(),
            "num_qa_pairs": count_qa_items(task1_path if task1_path.exists() else None, "task1"),
        },
        "task2_camouflage": {
            "path": str(task2_path),
            "exists": task2_path.exists(),
            "num_qa_pairs": count_qa_items(task2_path if task2_path.exists() else None, "task2"),
        },
        "task3_density_depth": {
            "qa_pairs_path": str(task3_path),
            "generation_context_path": str(task3_context_path),
            "stitched_image_path": str(task3_stitched_path),
            "qa_pairs_exists": task3_path.exists(),
            "generation_context_exists": task3_context_path.exists(),
            "stitched_image_exists": task3_stitched_path.exists(),
            "num_qa_pairs": count_qa_items(task3_path if task3_path.exists() else None, "task3"),
        },
    }

    if include_qa:
        if task1_path.exists():
            qa_assets["task1_region"]["payload"] = read_jsonl_payload(task1_path)
        if task2_path.exists():
            qa_assets["task2_camouflage"]["payload"] = read_jsonl_payload(task2_path)
        if task3_path.exists():
            qa_assets["task3_density_depth"]["qa_pairs"] = read_jsonl_payload(task3_path)
        if task3_context_path.exists():
            qa_assets["task3_density_depth"]["generation_context"] = read_json(task3_context_path)

    return qa_assets


def compact_final_annotation(final_payload: dict[str, Any], final_path: Path) -> dict[str, Any]:
    source_annotation = final_payload.get("source_annotation", {})
    image = final_payload.get("image", {})
    task3 = final_payload.get("task3_density_depth", {})
    density = task3.get("density", {}) if isinstance(task3, dict) else {}
    hotspots = task3.get("hotspots", {}) if isinstance(task3, dict) else {}
    scene = task3.get("scene_attribution", {}) if isinstance(task3, dict) else {}
    instances = final_payload.get("instances", [])
    visibility_counts: Counter[str] = Counter()
    grid_counts: Counter[str] = Counter()
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        camouflage = instance.get("task2_camouflage") or {}
        visibility = camouflage.get("visibility")
        if visibility:
            visibility_counts[str(visibility)] += 1
        region = instance.get("task1_region") or {}
        grid_region = region.get("grid_region")
        if grid_region:
            grid_counts[str(grid_region)] += 1

    return {
        "path": str(final_path),
        "exists": final_path.exists(),
        "schema_version": final_payload.get("schema_version"),
        "split": final_payload.get("split"),
        "image_file_name": image.get("file_name"),
        "width": image.get("width"),
        "height": image.get("height"),
        "total_count": source_annotation.get("num_valid_point_objects"),
        "num_raw_objects": source_annotation.get("num_objects"),
        "num_instances": len(instances),
        "visibility_counts": dict(sorted(visibility_counts.items())),
        "grid_counts": dict(sorted(grid_counts.items())),
        "has_task1_region": bool(final_payload.get("image_level_annotations", {}).get("task1_region")),
        "has_task2_camouflage": bool(final_payload.get("image_level_annotations", {}).get("task2_camouflage")),
        "has_task3_density_depth": bool(task3),
        "task3_density_paths": density if isinstance(density, dict) else {},
        "task3_num_hotspots": hotspots.get("num_hotspots") if isinstance(hotspots, dict) else None,
        "task3_num_scene_hotspots": scene.get("num_hotspots") if isinstance(scene, dict) else None,
    }


def has_required_asset(record: dict[str, Any], asset: str) -> bool:
    if asset == "rgb":
        return bool(record["rgb_underwater_image"]["exists"])
    if asset == "final_annotation":
        return bool(record["final_annotation_json"]["exists"])
    if asset == "density":
        return bool(record["density_assets"]["complete"])
    if asset == "task1_qa":
        return bool(record["qa_instruction_conversations"]["task1_region"]["exists"])
    if asset == "task2_qa":
        return bool(record["qa_instruction_conversations"]["task2_camouflage"]["exists"])
    if asset == "task3_qa":
        return bool(record["qa_instruction_conversations"]["task3_density_depth"]["qa_pairs_exists"])
    raise ValueError(f"Unknown asset requirement: {asset}")


def collect_missing_assets(record: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not record["rgb_underwater_image"]["exists"]:
        missing.append("rgb")
    if not record["final_annotation_json"]["exists"]:
        missing.append("final_annotation")
    if not record["density_assets"]["complete"]:
        for key, exists in record["density_assets"]["exists"].items():
            if not exists:
                missing.append(f"density.{key}")
    qa = record["qa_instruction_conversations"]
    if not qa["task1_region"]["exists"]:
        missing.append("qa.task1_region")
    if not qa["task2_camouflage"]["exists"]:
        missing.append("qa.task2_camouflage")
    if not qa["task3_density_depth"]["qa_pairs_exists"]:
        missing.append("qa.task3_density_depth.qa_pairs")
    if not qa["task3_density_depth"]["generation_context_exists"]:
        missing.append("qa.task3_density_depth.generation_context")
    if not qa["task3_density_depth"]["stitched_image_exists"]:
        missing.append("qa.task3_density_depth.stitched_image")
    return missing


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo_root = args.repo_root.resolve()
    dataset_root = repo_root / "test_dataset"
    images_dir = dataset_root / "images"
    final_dir = dataset_root / "final_annotations"
    density_root = repo_root / "scripts" / "task3_density_depth" / "density" / "Density maps"

    records: list[dict[str, Any]] = []
    for final_path in sorted(final_dir.glob("*.json")):
        if not IMAGE_ID_PATTERN.fullmatch(final_path.stem):
            continue
        final_payload = read_json(final_path)
        image_id = str(final_payload.get("image_id") or final_path.stem)
        if not IMAGE_ID_PATTERN.fullmatch(image_id):
            continue
        rgb_path = find_image(images_dir, image_id)
        density_assets = build_density_assets(density_root, image_id)
        qa_assets = build_qa_assets(repo_root, image_id, include_qa=args.include_qa)

        final_annotation: dict[str, Any]
        if args.include_final_annotation:
            final_annotation = {
                "path": str(final_path),
                "exists": True,
                "payload": final_payload,
            }
        else:
            final_annotation = compact_final_annotation(final_payload, final_path)

        record = {
            "image_id": image_id,
            "rgb_underwater_image": {
                "path": str(rgb_path) if rgb_path is not None else str(images_dir / f"{image_id}.jpg"),
                "exists": rgb_path is not None and rgb_path.exists(),
            },
            "final_annotation_json": final_annotation,
            "qa_instruction_conversations": qa_assets,
            "density_assets": density_assets,
        }
        record["missing_assets"] = collect_missing_assets(record)
        record["complete_for_training"] = all(
            has_required_asset(record, asset) for asset in args.complete_requires
        )
        records.append(record)

    return records


def make_splits(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[str]]:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    ids = [record["image_id"] for record in records if record["complete_for_training"]]
    rng = random.Random(args.seed)
    rng.shuffle(ids)

    n_total = len(ids)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }


def write_splits(output_dir: Path, splits: dict[str, list[str]]) -> None:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in splits.items():
        (split_dir / f"{split_name}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    write_json(split_dir / "splits.json", splits)


def build_summary(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    missing_counter: Counter[str] = Counter()
    for record in records:
        missing_counter.update(record["missing_assets"])

    return {
        "repo_root": str(args.repo_root.resolve()),
        "num_records_from_final_annotations": len(records),
        "num_complete_for_training": sum(1 for record in records if record["complete_for_training"]),
        "complete_requires": args.complete_requires,
        "asset_coverage": {
            "rgb": sum(1 for record in records if has_required_asset(record, "rgb")),
            "final_annotation": sum(1 for record in records if has_required_asset(record, "final_annotation")),
            "density_complete": sum(1 for record in records if has_required_asset(record, "density")),
            "task1_qa": sum(1 for record in records if has_required_asset(record, "task1_qa")),
            "task2_qa": sum(1 for record in records if has_required_asset(record, "task2_qa")),
            "task3_qa": sum(1 for record in records if has_required_asset(record, "task3_qa")),
        },
        "missing_asset_counts": dict(sorted(missing_counter.items())),
        "outputs": {
            "all_records": str(args.output_dir / "canonical_records.jsonl"),
            "complete_records": str(args.output_dir / "complete_records.jsonl"),
            "missing_records": str(args.output_dir / "missing_assets.jsonl"),
            "summary": str(args.output_dir / "summary.json"),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(args)
    complete_records = [record for record in records if record["complete_for_training"]]
    missing_records = [record for record in records if record["missing_assets"]]

    write_jsonl(args.output_dir / "canonical_records.jsonl", records)
    write_jsonl(args.output_dir / "complete_records.jsonl", complete_records)
    write_jsonl(args.output_dir / "missing_assets.jsonl", missing_records)
    write_json(args.output_dir / "summary.json", build_summary(records, args))

    if args.make_splits:
        write_splits(args.output_dir, make_splits(records, args))

    print(f"Wrote {len(records)} canonical records to {args.output_dir / 'canonical_records.jsonl'}")
    print(f"Wrote {len(complete_records)} complete records to {args.output_dir / 'complete_records.jsonl'}")
    print(f"Wrote summary to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
