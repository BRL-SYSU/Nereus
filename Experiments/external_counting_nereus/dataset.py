"""Label-free test manifest access for external-model inference."""

from __future__ import annotations

import json
from pathlib import Path


ALLOWED_IMAGE_FIELDS = {"image_id", "image_path", "width", "height"}
PROHIBITED_LABEL_FIELDS = {
    "count",
    "points_xy",
    "grid_counts",
    "densest_grid_region",
    "camouflage",
    "camouflage_labels",
}


def read_image_manifest(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        source = json.loads(line)
        prohibited = PROHIBITED_LABEL_FIELDS & set(source)
        if prohibited:
            raise ValueError(
                "label-bearing field in inference manifest at line "
                f"{line_number}: {sorted(prohibited)}"
            )
        missing = ALLOWED_IMAGE_FIELDS - set(source)
        if missing:
            raise ValueError(
                f"inference manifest missing fields: {sorted(missing)}"
            )
        record = {
            "image_id": str(source["image_id"]),
            "image_path": str(source["image_path"]),
            "width": int(source["width"]),
            "height": int(source["height"]),
        }
        if not record["image_id"] or int(record["width"]) <= 0 or int(
            record["height"]
        ) <= 0:
            raise ValueError("invalid image manifest record")
        records.append(record)
    image_ids = [str(record["image_id"]) for record in records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("duplicate image ID in inference manifest")
    return records
