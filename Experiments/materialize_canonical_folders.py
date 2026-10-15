#!/usr/bin/env python3
"""Materialize canonical AquaCount records as per-image folders.

This layout is convenient for inspection and dataset release:

canonical_dataset_folders/
├── 0000/
│   ├── image/0000.jpg
│   ├── final_annotation/0000.json
│   ├── qa/task1_region.jsonl
│   ├── qa/task2_camouflage.jsonl
│   ├── qa/task3_density_depth_qa_pairs.json
│   ├── qa/task3_density_depth_generation_context.json
│   ├── qa/0000_stitched.png
│   ├── density/0000_density_heatmap.png
│   ├── density/0000_density_overlay.png
│   ├── density/0000_density.png
│   ├── density/0000_density.npy
│   └── record.json
└── manifest.jsonl

By default files are symlinked instead of copied to avoid duplicating the image
and density data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    canonical_dir = repo_root / "Experiments" / "canonical_dataset"
    parser = argparse.ArgumentParser(description="Create image_id folders from canonical_records.jsonl.")
    parser.add_argument(
        "--records-jsonl",
        type=Path,
        default=canonical_dir / "complete_records.jsonl",
        help="Input canonical JSONL. Defaults to complete_records.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "Experiments" / "canonical_dataset_folders",
        help="Output folder dataset root.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy", "hardlink"),
        default="symlink",
        help="How to place source files into each image_id folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing destination files/links.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def place_file(src: Path, dst: Path, link_mode: str, overwrite: bool) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return True
        dst.unlink()

    if link_mode == "symlink":
        os.symlink(src, dst)
    elif link_mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def suffix_path(path: Path, fallback_suffix: str) -> str:
    return path.suffix if path.suffix else fallback_suffix


def materialize_record(record: dict[str, Any], output_dir: Path, link_mode: str, overwrite: bool) -> dict[str, Any]:
    image_id = record["image_id"]
    image_dir = output_dir / image_id

    rgb_src = Path(record["rgb_underwater_image"]["path"])
    final_src = Path(record["final_annotation_json"]["path"])
    qa = record["qa_instruction_conversations"]
    density = record["density_assets"]["paths"]

    local_paths = {
        "image": str(Path(image_id) / "image" / f"{image_id}{suffix_path(rgb_src, '.jpg')}"),
        "final_annotation": str(Path(image_id) / "final_annotation" / f"{image_id}.json"),
        "qa": {
            "task1_region": str(Path(image_id) / "qa" / "task1_region.jsonl"),
            "task2_camouflage": str(Path(image_id) / "qa" / "task2_camouflage.jsonl"),
            "task3_density_depth_qa_pairs": str(Path(image_id) / "qa" / "task3_density_depth_qa_pairs.json"),
            "task3_density_depth_generation_context": str(
                Path(image_id) / "qa" / "task3_density_depth_generation_context.json"
            ),
            "task3_stitched_image": str(Path(image_id) / "qa" / f"{image_id}_stitched.png"),
        },
        "density": {
            "heatmap": str(Path(image_id) / "density" / f"{image_id}_density_heatmap.png"),
            "overlay": str(Path(image_id) / "density" / f"{image_id}_density_overlay.png"),
            "png": str(Path(image_id) / "density" / f"{image_id}_density.png"),
            "array": str(Path(image_id) / "density" / f"{image_id}_density.npy"),
        },
        "record": str(Path(image_id) / "record.json"),
    }

    placed = {
        "image": place_file(rgb_src, output_dir / local_paths["image"], link_mode, overwrite),
        "final_annotation": place_file(final_src, output_dir / local_paths["final_annotation"], link_mode, overwrite),
        "task1_qa": place_file(
            Path(qa["task1_region"]["path"]),
            output_dir / local_paths["qa"]["task1_region"],
            link_mode,
            overwrite,
        ),
        "task2_qa": place_file(
            Path(qa["task2_camouflage"]["path"]),
            output_dir / local_paths["qa"]["task2_camouflage"],
            link_mode,
            overwrite,
        ),
        "task3_qa": place_file(
            Path(qa["task3_density_depth"]["qa_pairs_path"]),
            output_dir / local_paths["qa"]["task3_density_depth_qa_pairs"],
            link_mode,
            overwrite,
        ),
        "task3_generation_context": place_file(
            Path(qa["task3_density_depth"]["generation_context_path"]),
            output_dir / local_paths["qa"]["task3_density_depth_generation_context"],
            link_mode,
            overwrite,
        ),
        "task3_stitched_image": place_file(
            Path(qa["task3_density_depth"]["stitched_image_path"]),
            output_dir / local_paths["qa"]["task3_stitched_image"],
            link_mode,
            overwrite,
        ),
        "density_heatmap": place_file(
            Path(density["density_heatmap"]),
            output_dir / local_paths["density"]["heatmap"],
            link_mode,
            overwrite,
        ),
        "density_overlay": place_file(
            Path(density["density_overlay"]),
            output_dir / local_paths["density"]["overlay"],
            link_mode,
            overwrite,
        ),
        "density_png": place_file(
            Path(density["density_png"]),
            output_dir / local_paths["density"]["png"],
            link_mode,
            overwrite,
        ),
        "density_array": place_file(
            Path(density["density_array"]),
            output_dir / local_paths["density"]["array"],
            link_mode,
            overwrite,
        ),
    }

    folder_record = {
        "image_id": image_id,
        "dataset_folder": str(image_dir),
        "link_mode": link_mode,
        "local_paths": local_paths,
        "source_record": record,
        "placed": placed,
    }
    write_json(output_dir / local_paths["record"], folder_record)
    return {
        "image_id": image_id,
        "dataset_folder": str(image_dir),
        "local_paths": local_paths,
        "placed": placed,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.records_jsonl, limit=args.limit)
    manifest_records = [
        materialize_record(record, args.output_dir, args.link_mode, args.overwrite)
        for record in records
    ]

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_json(
        args.output_dir / "summary.json",
        {
            "records_jsonl": str(args.records_jsonl),
            "output_dir": str(args.output_dir),
            "num_records": len(manifest_records),
            "link_mode": args.link_mode,
            "manifest": str(manifest_path),
        },
    )
    print(f"Materialized {len(manifest_records)} folders under {args.output_dir}")
    print(f"Wrote folder manifest to {manifest_path}")


if __name__ == "__main__":
    main()
