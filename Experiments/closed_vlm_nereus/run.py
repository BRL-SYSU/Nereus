"""Run resumable APIYi inference for Nereus Table 3 Track A."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from Experiments.external_counting_nereus.dataset import read_image_manifest
from Experiments.external_counting_nereus.io_utils import sha256_file

from .apiyi_client import APIYiClient
from .protocol import (
    PROMPTS,
    image_to_data_uri,
    parse_count,
    parse_grid,
)


DEFAULT_MANIFEST = Path(
    "./Experiments/"
    "p2pnet_nereus/data/test_images.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "./Experiments/"
    "closed_vlm_nereus/outputs"
)
EXPECTED_TEST_SIZE = 1129


def _safe_model_name(model: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in model
    )
    return safe.strip("._") or "model"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _cache_valid(
    record: dict[str, Any],
    *,
    model: str,
    image_id: str,
    image_sha256: str,
    task: str,
) -> bool:
    return (
        record.get("schema_version") == 1
        and record.get("requested_model") == model
        and record.get("image_id") == image_id
        and record.get("image_sha256") == image_sha256
        and record.get("task") == task
        and record.get("prompt_sha256") == _prompt_sha256(PROMPTS[task])
        and isinstance(record.get("parsed"), dict)
    )


def _read_valid_cache(
    path: Path,
    *,
    model: str,
    image_id: str,
    image_sha256: str,
    task: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    return (
        record
        if _cache_valid(
            record,
            model=model,
            image_id=image_id,
            image_sha256=image_sha256,
            task=task,
        )
        else None
    )


def _call_task(
    *,
    client: APIYiClient,
    model: str,
    image_id: str,
    image_sha256: str,
    image_data_uri: str,
    task: str,
    cache_path: Path,
) -> dict[str, Any]:
    cached = _read_valid_cache(
        cache_path,
        model=model,
        image_id=image_id,
        image_sha256=image_sha256,
        task=task,
    )
    if cached is not None:
        return cached
    response = client.complete_image(
        model=model,
        prompt=PROMPTS[task],
        image_data_uri=image_data_uri,
        task=task,
    )
    parsed = (
        parse_count(response.content)
        if task == "count"
        else parse_grid(response.content)
    )
    record = {
        "schema_version": 1,
        "requested_model": model,
        "response_model": response.response_model,
        "response_id": response.response_id,
        "system_fingerprint": response.system_fingerprint,
        "image_id": image_id,
        "image_sha256": image_sha256,
        "task": task,
        "prompt": PROMPTS[task],
        "prompt_sha256": _prompt_sha256(PROMPTS[task]),
        "raw_content": response.content,
        "parsed": parsed,
        "usage": response.usage,
        "latency_seconds": response.latency_seconds,
        "request_attempts": response.request_attempts,
        "created_unix": time.time(),
    }
    _atomic_json(cache_path, record)
    return record


def _run_one(
    *,
    client: APIYiClient,
    model: str,
    output_dir: Path,
    manifest_record: dict[str, object],
) -> dict[str, Any]:
    image_id = str(manifest_record["image_id"])
    image_path = Path(str(manifest_record["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image_sha256 = sha256_file(image_path)
    image_data_uri = image_to_data_uri(image_path)
    task_records: dict[str, dict[str, Any]] = {}
    for task in ("count", "grid"):
        task_records[task] = _call_task(
            client=client,
            model=model,
            image_id=image_id,
            image_sha256=image_sha256,
            image_data_uri=image_data_uri,
            task=task,
            cache_path=output_dir / "cache" / image_id / f"{task}.json",
        )
    return {
        "image_id": image_id,
        "count_valid": bool(task_records["count"]["parsed"]["valid"]),
        "grid_valid": bool(task_records["grid"]["parsed"]["valid"]),
    }


def _prediction_from_cache(
    *,
    model: str,
    output_dir: Path,
    manifest_record: dict[str, object],
) -> dict[str, Any]:
    image_id = str(manifest_record["image_id"])
    image_path = Path(str(manifest_record["image_path"]))
    image_sha256 = sha256_file(image_path)
    records: dict[str, dict[str, Any]] = {}
    for task in ("count", "grid"):
        path = output_dir / "cache" / image_id / f"{task}.json"
        record = _read_valid_cache(
            path,
            model=model,
            image_id=image_id,
            image_sha256=image_sha256,
            task=task,
        )
        if record is None:
            raise ValueError(f"missing or stale cache: {path}")
        records[task] = record
    count_parsed = records["count"]["parsed"]
    grid_parsed = records["grid"]["parsed"]
    return {
        "schema_version": 1,
        "method": model,
        "setting": "P-ZS (frozen Nereus Track A prompts)",
        "image_id": image_id,
        "image_width": int(manifest_record["width"]),
        "image_height": int(manifest_record["height"]),
        "image_sha256": image_sha256,
        "pred_count": int(count_parsed["count"]),
        "pred_densest_grid_region": str(grid_parsed["densest"]),
        "count_parse_valid": bool(count_parsed["valid"]),
        "grid_parse_valid": bool(grid_parsed["valid"]),
        "pred_grid_counts": grid_parsed.get("grid_counts", {}),
        "grid_consistent": grid_parsed.get(
            "explicit_derived_consistent"
        ),
        "count_response_id": str(records["count"]["response_id"]),
        "grid_response_id": str(records["grid"]["response_id"]),
        "count_response_model": str(records["count"]["response_model"]),
        "grid_response_model": str(records["grid"]["response_model"]),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_api_key(
    *, environment_name: str, key_file: Path | None
) -> str:
    if key_file is not None:
        mode = key_file.stat().st_mode & 0o777
        if mode & 0o077:
            raise SystemExit(
                f"API key file permissions must be 600 or stricter, got "
                f"{mode:03o}: {key_file}"
            )
        key = key_file.read_text(encoding="utf-8").strip()
        if key.startswith("APIYI_API_KEY="):
            key = key.split("=", 1)[1].strip()
    else:
        key = os.environ.get(environment_name, "").strip()
    if not key:
        source = str(key_file) if key_file else environment_name
        raise SystemExit(f"missing API key in {source}")
    if "\n" in key or "\r" in key:
        raise SystemExit("API key must be one line")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--base-url", default="https://api.apiyi.com/v1"
    )
    parser.add_argument("--api-key-env", default="APIYI_API_KEY")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help=(
            "600-permission file containing only the key (or "
            "APIYI_API_KEY=...). The key is never written to outputs."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_records = read_image_manifest(args.manifest)
    if len(all_records) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"expected {EXPECTED_TEST_SIZE} manifest rows, "
            f"got {len(all_records)}"
        )
    selected = all_records[
        args.offset : (
            args.offset + args.limit if args.limit is not None else None
        )
    ]
    if not selected:
        raise ValueError("selected manifest slice is empty")
    output_dir = args.output_root / _safe_model_name(args.model)
    if args.dry_run:
        summary = {
            "model": args.model,
            "manifest": str(args.manifest),
            "manifest_records": len(all_records),
            "selected_records": len(selected),
            "requests_if_uncached": 2 * len(selected),
            "output_dir": str(output_dir),
            "prompts": PROMPTS,
            "labels_exposed_to_inference": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    api_key = _read_api_key(
        environment_name=args.api_key_env,
        key_file=args.api_key_file,
    )
    client = APIYiClient(
        api_key=api_key,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    failures: list[dict[str, str]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {
            executor.submit(
                _run_one,
                client=client,
                model=args.model,
                output_dir=output_dir,
                manifest_record=record,
            ): str(record["image_id"])
            for record in selected
        }
        for future in as_completed(future_to_id):
            image_id = future_to_id[future]
            try:
                result = future.result()
            except Exception as error:  # keep other samples resumable
                failures.append(
                    {
                        "image_id": image_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            else:
                completed += 1
                print(
                    f"[{completed + len(failures)}/{len(selected)}] "
                    f"{image_id} count_parse={result['count_valid']} "
                    f"grid_parse={result['grid_valid']}",
                    flush=True,
                )
    _atomic_json(
        output_dir / "last_run.json",
        {
            "model": args.model,
            "selected_records": len(selected),
            "completed_records": completed,
            "failures": failures,
            "labels_exposed_to_inference": False,
        },
    )
    if failures:
        raise SystemExit(
            f"{len(failures)} sample(s) failed; rerun the same command "
            "to resume from cache"
        )
    predictions = [
        _prediction_from_cache(
            model=args.model,
            output_dir=output_dir,
            manifest_record=record,
        )
        for record in selected
    ]
    prediction_path = output_dir / "predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    print(f"wrote {len(predictions)} predictions to {prediction_path}")


if __name__ == "__main__":
    main()
