#!/usr/bin/env python3
"""Convert canonical AquaCount records into experiment-ready datasets.

The output layout matches the baseline experiment plan:

data/
  canonical/
  sft/
  eval/

Track A files use only the RGB image as model input. Track B density files use
RGB plus density evidence and keep final annotations as targets/metadata only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


GRID_ORDER = [
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]

VISIBILITY_ORDER = ["clear", "ambiguous", "highly_camouflaged"]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    experiments_dir = repo_root / "Experiments"
    parser = argparse.ArgumentParser(
        description="Build canonical, SFT, and eval JSON files from AquaCount canonical records.",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        default=experiments_dir / "canonical_dataset",
        help="Directory containing canonical_records.jsonl and splits/splits.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiments_dir / "data",
        help="Output directory that will contain canonical/, sft/, and eval/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit per split. Omit for the full dataset.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_json_compact(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def ensure_image_token(text: str, num_images: int = 1) -> str:
    prefix = "\n".join("<image>" for _ in range(num_images))
    stripped = text.strip()
    if stripped.startswith("<image>"):
        return stripped
    return f"{prefix}\n{stripped}"


def normalize_grid_counts(grid_counts: dict[str, Any] | None) -> dict[str, int]:
    grid_counts = grid_counts or {}
    return {label: int(grid_counts.get(label, 0) or 0) for label in GRID_ORDER}


def normalize_visibility_counts(visibility_counts: dict[str, Any] | None) -> dict[str, int]:
    visibility_counts = visibility_counts or {}
    return {label: int(visibility_counts.get(label, 0) or 0) for label in VISIBILITY_ORDER}


def top_key(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    max_count = max(counts.values())
    winners = [key for key, value in counts.items() if value == max_count]
    for key in GRID_ORDER + VISIBILITY_ORDER:
        if key in winners:
            return key
    return winners[0] if winners else None


def compact_hotspot(item: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "hotspot_id",
        "point_xy",
        "peak_value",
        "count_estimate",
        "region_area",
        "region_bbox_xywh",
    ]
    return {key: item.get(key) for key in keys if key in item}


def compact_scene_hotspot(item: dict[str, Any]) -> dict[str, Any]:
    semantic = item.get("semantic_result") or {}
    depth = item.get("depth_summary") or {}
    return {
        "hotspot_id": item.get("hotspot_id"),
        "point_xy": item.get("point_xy"),
        "count_estimate": item.get("count_estimate"),
        "semantic_label": semantic.get("semantic_label"),
        "semantic_confidence": semantic.get("confidence"),
        "relative_depth_label": depth.get("relative_depth_label"),
        "coarse_position": depth.get("coarse_position"),
        "depth_edge_description": depth.get("depth_edge_description"),
        "causal_chain": item.get("causal_chain"),
    }


def compact_density_supervision(final_ann: dict[str, Any]) -> dict[str, Any]:
    task3 = final_ann.get("task3_density_depth") or {}
    context = task3.get("generation_context") or {}
    feature_summary = context.get("density_feature_summary") or {}
    hotspots = task3.get("hotspots") or {}
    scene = task3.get("scene_attribution") or {}
    return {
        "density_feature_summary": {
            "total_density_integral": feature_summary.get("total_density_integral"),
            "hotspot_count": feature_summary.get("hotspot_count"),
            "high_response_pixel_ratio": feature_summary.get("high_response_pixel_ratio"),
            "medium_response_pixel_ratio": feature_summary.get("medium_response_pixel_ratio"),
            "low_response_pixel_ratio": feature_summary.get("low_response_pixel_ratio"),
            "highest_density_quadrant": feature_summary.get("highest_density_quadrant"),
            "lowest_density_quadrant": feature_summary.get("lowest_density_quadrant"),
            "spatial_pattern": feature_summary.get("spatial_pattern"),
            "density_concentration": feature_summary.get("density_concentration"),
            "top_hotspots": [compact_hotspot(item) for item in feature_summary.get("top_hotspots", [])],
        },
        "primary_hotspot": context.get("primary_hotspot"),
        "low_response_region": context.get("low_response_region"),
        "hotspots": [compact_hotspot(item) for item in hotspots.get("items", [])],
        "scene_hotspots": [compact_scene_hotspot(item) for item in scene.get("items", [])],
    }


def build_canonical_record(record: dict[str, Any], final_ann: dict[str, Any], split: str) -> dict[str, Any]:
    image = final_ann.get("image") or {}
    source = final_ann.get("source_annotation") or {}
    image_level = final_ann.get("image_level_annotations") or {}
    task1 = image_level.get("task1_region") or {}
    task2 = image_level.get("task2_camouflage") or {}
    task3 = final_ann.get("task3_density_depth") or {}
    total_count = int(source.get("num_valid_point_objects") or len(final_ann.get("instances", [])))
    grid_counts = normalize_grid_counts(task1.get("grid_counts") or record.get("final_annotation_json", {}).get("grid_counts"))
    visibility = task2.get("image_camouflage_summary", {}).get("visibility_distribution", {})
    visibility_counts = normalize_visibility_counts(
        {
            "clear": visibility.get("clear_count"),
            "ambiguous": visibility.get("ambiguous_count"),
            "highly_camouflaged": visibility.get("highly_camouflaged_count"),
        }
        if visibility
        else record.get("final_annotation_json", {}).get("visibility_counts")
    )
    density_paths = (record.get("density_assets") or {}).get("paths") or {}
    qa = record.get("qa_instruction_conversations") or {}

    return {
        "image_id": record["image_id"],
        "split": split,
        "rgb_image_path": (record.get("rgb_underwater_image") or {}).get("path") or image.get("abs_path"),
        "final_annotation_path": (record.get("final_annotation_json") or {}).get("path"),
        "density_paths": density_paths,
        "qa_paths": {
            "task1_region": (qa.get("task1_region") or {}).get("path"),
            "task2_camouflage": (qa.get("task2_camouflage") or {}).get("path"),
            "task3_density_depth": (qa.get("task3_density_depth") or {}).get("qa_pairs_path"),
        },
        "image": {
            "file_name": image.get("file_name"),
            "width": image.get("width"),
            "height": image.get("height"),
            "channels": image.get("channels"),
        },
        "total_count": total_count,
        "grid_counts": grid_counts,
        "densest_grid_region": top_key(grid_counts),
        "visibility_counts": visibility_counts,
        "dominant_visibility": top_key(visibility_counts),
        "scene_counts": task1.get("scene_counts", {}),
        "grid_comparison": task1.get("grid_comparison", {}),
        "camouflage_grid_analysis": task2.get("grid_camouflage_analysis", {}),
        "camouflage_image_explanation": task2.get("image_level_explanation"),
        "density_summary": compact_density_supervision(final_ann)["density_feature_summary"],
        "task3_density_paths": task3.get("density", {}),
    }


def sft_example(
    sample_id: str,
    image_id: str,
    task: str,
    track: str,
    images: list[str],
    user: str,
    assistant: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "id": sample_id,
        "image_id": image_id,
        "task": task,
        "track": track,
        "image": images[0] if images else None,
        "images": images,
        "conversations": [
            {"from": "user", "value": user},
            {"from": "assistant", "value": assistant},
        ],
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def qa_pairs_from_conversation_payload(payload: Any) -> list[tuple[str, str]]:
    if isinstance(payload, dict):
        conversations = payload.get("conversations", [])
    elif isinstance(payload, list):
        conversations = payload
    else:
        return []

    pairs: list[tuple[str, str]] = []
    idx = 0
    while idx + 1 < len(conversations):
        user = conversations[idx]
        assistant = conversations[idx + 1]
        if user.get("from") == "user" and assistant.get("from") == "assistant":
            pairs.append((str(user.get("value", "")), str(assistant.get("value", ""))))
            idx += 2
        else:
            idx += 1
    return pairs


def load_track_a_qa_samples(record: dict[str, Any], canonical: dict[str, Any]) -> list[dict[str, Any]]:
    image_id = record["image_id"]
    rgb = canonical["rgb_image_path"]
    qa = record.get("qa_instruction_conversations") or {}
    samples: list[dict[str, Any]] = []
    for task_name in ["task1_region", "task2_camouflage"]:
        path_value = (qa.get(task_name) or {}).get("path")
        if not path_value:
            continue
        path = Path(path_value)
        if not path.exists():
            continue
        for qa_idx, (question, answer) in enumerate(qa_pairs_from_conversation_payload(read_json_or_jsonl(path)), start=1):
            samples.append(
                sft_example(
                    sample_id=f"{image_id}_{task_name}_{qa_idx:02d}",
                    image_id=image_id,
                    task=task_name,
                    track="A",
                    images=[rgb],
                    user=ensure_image_token(question, 1),
                    assistant=answer.strip(),
                    metadata={"source": "existing_qa"},
                )
            )
    return samples


def load_density_qa_samples(record: dict[str, Any], canonical: dict[str, Any]) -> list[dict[str, Any]]:
    image_id = record["image_id"]
    rgb = canonical["rgb_image_path"]
    density = canonical["density_paths"].get("density_overlay") or canonical["density_paths"].get("density_heatmap")
    qa_path = (record.get("qa_instruction_conversations", {}).get("task3_density_depth") or {}).get("qa_pairs_path")
    if not qa_path or not density:
        return []
    path = Path(qa_path)
    if not path.exists():
        return []
    pairs = read_json_or_jsonl(path)
    samples: list[dict[str, Any]] = []
    for idx, item in enumerate(pairs if isinstance(pairs, list) else [], start=1):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        samples.append(
            sft_example(
                sample_id=f"{image_id}_task3_density_qa_{idx:02d}",
                image_id=image_id,
                task="task3_density_depth_qa",
                track="B",
                images=[rgb, density],
                user=ensure_image_token(question, 2),
                assistant=answer,
                metadata={
                    "source": "existing_density_qa",
                    "question_type": item.get("question_type"),
                    "density_image": density,
                },
            )
        )
    return samples


def structured_final_samples(canonical: dict[str, Any], final_ann: dict[str, Any]) -> list[dict[str, Any]]:
    image_id = canonical["image_id"]
    rgb = canonical["rgb_image_path"]
    grid_counts = canonical["grid_counts"]
    visibility_counts = canonical["visibility_counts"]
    structured_target = {
        "image_id": image_id,
        "total_count": canonical["total_count"],
        "grid_counts": grid_counts,
        "densest_grid_region": canonical["densest_grid_region"],
        "visibility_counts": visibility_counts,
        "dominant_visibility": canonical["dominant_visibility"],
        "scene_counts": canonical["scene_counts"],
    }
    samples = [
        sft_example(
            sample_id=f"{image_id}_structured_count",
            image_id=image_id,
            task="structured_total_count",
            track="A",
            images=[rgb],
            user="<image>\nHow many fish are visible in this underwater image? Answer with only the total count.",
            assistant=str(canonical["total_count"]),
            metadata={"source": "final_annotation"},
        ),
        sft_example(
            sample_id=f"{image_id}_structured_grid",
            image_id=image_id,
            task="structured_grid_counts",
            track="A",
            images=[rgb],
            user="<image>\nEstimate the fish count in each 3x3 grid region and identify the densest region. Return JSON.",
            assistant=json.dumps(
                {"grid_counts": grid_counts, "densest_grid_region": canonical["densest_grid_region"]},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            metadata={"source": "final_annotation"},
        ),
        sft_example(
            sample_id=f"{image_id}_structured_camouflage",
            image_id=image_id,
            task="structured_camouflage",
            track="A",
            images=[rgb],
            user="<image>\nSummarize fish visibility and camouflage in the image. Return JSON.",
            assistant=json.dumps(
                {
                    "visibility_counts": visibility_counts,
                    "dominant_visibility": canonical["dominant_visibility"],
                    "image_level_explanation": canonical["camouflage_image_explanation"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            metadata={"source": "final_annotation"},
        ),
        sft_example(
            sample_id=f"{image_id}_structured_full",
            image_id=image_id,
            task="structured_final_summary",
            track="A",
            images=[rgb],
            user="<image>\nProvide the final structured annotation summary for this image. Return JSON.",
            assistant=json.dumps(structured_target, ensure_ascii=False, separators=(",", ":")),
            metadata={"source": "final_annotation"},
        ),
    ]
    return samples


def density_structured_samples(canonical: dict[str, Any], final_ann: dict[str, Any]) -> list[dict[str, Any]]:
    image_id = canonical["image_id"]
    rgb = canonical["rgb_image_path"]
    density = canonical["density_paths"].get("density_overlay") or canonical["density_paths"].get("density_heatmap")
    if not density:
        return []
    supervision = compact_density_supervision(final_ann)
    return [
        sft_example(
            sample_id=f"{image_id}_density_structured_hotspots",
            image_id=image_id,
            task="density_structured_hotspots",
            track="B",
            images=[rgb, density],
            user="<image>\n<image>\nUsing the RGB image and density evidence, summarize the density hotspots. Return JSON.",
            assistant=json.dumps(
                {
                    "density_feature_summary": supervision["density_feature_summary"],
                    "primary_hotspot": supervision["primary_hotspot"],
                    "low_response_region": supervision["low_response_region"],
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            metadata={"source": "final_annotation_density", "density_image": density},
        ),
        sft_example(
            sample_id=f"{image_id}_density_structured_ecology",
            image_id=image_id,
            task="density_structured_ecology",
            track="B",
            images=[rgb, density],
            user="<image>\n<image>\nExplain the ecological or scene context of the density hotspots. Return JSON.",
            assistant=json.dumps(
                {"scene_hotspots": supervision["scene_hotspots"]},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            metadata={"source": "final_annotation_density", "density_image": density},
        ),
    ]


def track_a_eval_record(canonical: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": canonical["image_id"],
        "track": "A",
        "input": {
            "rgb_image": canonical["rgb_image_path"],
            "allowed_visual_inputs": ["rgb_image"],
            "forbidden_test_time_inputs": ["final_annotation", "density_heatmap", "density_overlay", "density_array"],
        },
        "questions": [
            {
                "id": f"{canonical['image_id']}_count",
                "type": "total_count",
                "prompt": "<image>\nHow many fish are visible in this underwater image? Answer with only the total count.",
                "target": canonical["total_count"],
            },
            {
                "id": f"{canonical['image_id']}_grid",
                "type": "grid_counts",
                "prompt": "<image>\nEstimate the number of fish in each 3x3 grid region and identify the densest region. Return JSON.",
                "target": {
                    "grid_counts": canonical["grid_counts"],
                    "densest_grid_region": canonical["densest_grid_region"],
                },
            },
            {
                "id": f"{canonical['image_id']}_camouflage",
                "type": "camouflage_visibility",
                "prompt": "<image>\nSummarize fish visibility and camouflage. Return JSON.",
                "target": {
                    "visibility_counts": canonical["visibility_counts"],
                    "dominant_visibility": canonical["dominant_visibility"],
                },
            },
        ],
        "metrics": ["count_mae", "count_rmse", "count_acc_at_5_percent", "densest_cell_accuracy", "camouflage_f1"],
    }


def track_b_eval_record(canonical: dict[str, Any], final_ann: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    density = canonical["density_paths"].get("density_overlay") or canonical["density_paths"].get("density_heatmap")
    qa_path = (record.get("qa_instruction_conversations", {}).get("task3_density_depth") or {}).get("qa_pairs_path")
    qa_pairs = read_json_or_jsonl(Path(qa_path)) if qa_path and Path(qa_path).exists() else []
    return {
        "image_id": canonical["image_id"],
        "track": "B",
        "input": {
            "rgb_image": canonical["rgb_image_path"],
            "density_image": density,
            "density_heatmap": canonical["density_paths"].get("density_heatmap"),
            "density_overlay": canonical["density_paths"].get("density_overlay"),
            "allowed_visual_inputs": ["rgb_image", "density_image"],
            "forbidden_test_time_inputs": ["final_annotation"],
        },
        "questions": [
            {
                "id": f"{canonical['image_id']}_density_hotspots",
                "type": "density_hotspots",
                "prompt": "<image>\n<image>\nUsing the RGB image and density evidence, identify the main density hotspots and explain their spatial pattern.",
                "target": compact_density_supervision(final_ann),
            }
        ]
        + [
            {
                "id": f"{canonical['image_id']}_density_qa_{idx:02d}",
                "type": item.get("question_type", "density_qa"),
                "prompt": ensure_image_token(str(item.get("question", "")), 2),
                "target": item.get("answer"),
            }
            for idx, item in enumerate(qa_pairs if isinstance(qa_pairs, list) else [], start=1)
        ],
        "metrics": ["hotspot_localization_accuracy", "density_reasoning_f1", "ecological_reasoning_quality"],
    }


def validate_splits(splits: dict[str, list[str]]) -> None:
    all_ids: list[str] = []
    for split_name in ["train", "val", "test"]:
        ids = splits.get(split_name)
        if not isinstance(ids, list):
            raise ValueError(f"Missing split list: {split_name}")
        all_ids.extend(ids)
    duplicate_count = len(all_ids) - len(set(all_ids))
    if duplicate_count:
        raise ValueError(f"Image-level split leakage: found {duplicate_count} duplicate ids across splits")


def main() -> None:
    args = parse_args()
    canonical_dir = args.canonical_dir.resolve()
    output_dir = args.output_dir.resolve()

    records_path = canonical_dir / "canonical_records.jsonl"
    splits_path = canonical_dir / "splits" / "splits.json"
    if not records_path.exists():
        raise FileNotFoundError(records_path)
    if not splits_path.exists():
        raise FileNotFoundError(splits_path)

    raw_records = read_jsonl(records_path)
    records_by_id = {record["image_id"]: record for record in raw_records}
    splits = read_json(splits_path)
    validate_splits(splits)
    if args.limit is not None:
        splits = {key: value[: args.limit] for key, value in splits.items()}

    split_by_id = {
        image_id: split_name
        for split_name, image_ids in splits.items()
        for image_id in image_ids
    }

    canonical_records: list[dict[str, Any]] = []
    final_by_id: dict[str, dict[str, Any]] = {}
    for image_id in sorted(split_by_id):
        record = records_by_id.get(image_id)
        if record is None:
            raise KeyError(f"Split id not found in canonical records: {image_id}")
        final_path = Path(record["final_annotation_json"]["path"])
        final_ann = read_json(final_path)
        final_by_id[image_id] = final_ann
        canonical_records.append(build_canonical_record(record, final_ann, split_by_id[image_id]))

    canonical_by_id = {record["image_id"]: record for record in canonical_records}

    count_only_train: list[dict[str, Any]] = []
    qa_only_train: list[dict[str, Any]] = []
    structured_final_train: list[dict[str, Any]] = []
    density_qa_train: list[dict[str, Any]] = []
    density_structured_train: list[dict[str, Any]] = []
    track_a_test: list[dict[str, Any]] = []
    track_b_test: list[dict[str, Any]] = []

    for image_id in splits["train"]:
        canonical = canonical_by_id[image_id]
        final_ann = final_by_id[image_id]
        record = records_by_id[image_id]
        count_only_train.append(
            sft_example(
                sample_id=f"{image_id}_count_only",
                image_id=image_id,
                task="count_only",
                track="A",
                images=[canonical["rgb_image_path"]],
                user="<image>\nHow many fish are visible in this underwater image? Answer with only the total count.",
                assistant=str(canonical["total_count"]),
                metadata={"source": "final_annotation", "target_type": "total_count"},
            )
        )
        qa_only_train.extend(load_track_a_qa_samples(record, canonical))
        structured_final_train.extend(structured_final_samples(canonical, final_ann))
        density_qa_train.extend(load_density_qa_samples(record, canonical))
        density_structured_train.extend(density_structured_samples(canonical, final_ann))

    for image_id in splits["test"]:
        canonical = canonical_by_id[image_id]
        final_ann = final_by_id[image_id]
        record = records_by_id[image_id]
        track_a_test.append(track_a_eval_record(canonical))
        track_b_test.append(track_b_eval_record(canonical, final_ann, record))

    qa_structured_train = qa_only_train + structured_final_train
    full_aquacount_train = qa_structured_train + density_qa_train + density_structured_train

    write_jsonl(output_dir / "canonical" / "records.jsonl", canonical_records)
    for split_name in ["train", "val", "test"]:
        write_json(
            output_dir / "canonical" / f"split_{split_name}.json",
            {
                "split": split_name,
                "num_records": len(splits[split_name]),
                "image_ids": splits[split_name],
            },
        )

    write_json_compact(output_dir / "sft" / "count_only_train.json", count_only_train)
    write_json_compact(output_dir / "sft" / "qa_only_train.json", qa_only_train)
    write_json_compact(output_dir / "sft" / "structured_final_train.json", structured_final_train)
    write_json_compact(output_dir / "sft" / "qa_structured_train.json", qa_structured_train)
    write_json_compact(output_dir / "sft" / "density_qa_train.json", density_qa_train)
    write_json_compact(output_dir / "sft" / "density_structured_train.json", density_structured_train)
    write_json_compact(output_dir / "sft" / "full_aquacount_train.json", full_aquacount_train)
    write_json_compact(output_dir / "eval" / "trackA_test.json", track_a_test)
    write_json_compact(output_dir / "eval" / "trackB_test.json", track_b_test)

    summary = {
        "output_dir": str(output_dir),
        "splits": {key: len(value) for key, value in splits.items()},
        "canonical_records": len(canonical_records),
        "sft_counts": {
            "count_only_train": len(count_only_train),
            "qa_only_train": len(qa_only_train),
            "structured_final_train": len(structured_final_train),
            "qa_structured_train": len(qa_structured_train),
            "density_qa_train": len(density_qa_train),
            "density_structured_train": len(density_structured_train),
            "full_aquacount_train": len(full_aquacount_train),
        },
        "eval_counts": {
            "trackA_test": len(track_a_test),
            "trackB_test": len(track_b_test),
        },
        "track_policy": {
            "trackA": "RGB only; final annotations and density assets are targets/metadata, not model inputs.",
            "trackB": "RGB plus density evidence; final annotations are targets/metadata, not model inputs.",
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
