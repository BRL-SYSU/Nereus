"""Report resumable API inference progress without reading any labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Experiments.external_counting_nereus.dataset import read_image_manifest

from .run import DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, _safe_model_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    records = read_image_manifest(args.manifest)
    cache_root = args.output_root / _safe_model_name(args.model) / "cache"
    count_done = 0
    grid_done = 0
    count_valid = 0
    grid_valid = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for record in records:
        image_root = cache_root / str(record["image_id"])
        for task in ("count", "grid"):
            path = image_root / f"{task}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            usage = payload.get("usage", {})
            if isinstance(usage, dict):
                prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens += int(
                    usage.get("completion_tokens", 0) or 0
                )
                total_tokens += int(usage.get("total_tokens", 0) or 0)
            if task == "count":
                count_done += 1
                count_valid += int(
                    bool(payload.get("parsed", {}).get("valid"))
                )
            else:
                grid_done += 1
                grid_valid += int(
                    bool(payload.get("parsed", {}).get("valid"))
                )
    total_tasks = 2 * len(records)
    done_tasks = count_done + grid_done
    print(
        json.dumps(
            {
                "model": args.model,
                "images": len(records),
                "count_done": count_done,
                "grid_done": grid_done,
                "requests_done": done_tasks,
                "requests_total": total_tasks,
                "progress_percent": round(
                    100.0 * done_tasks / total_tasks,
                    2,
                ),
                "count_parse_valid": count_valid,
                "grid_parse_valid": grid_valid,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
