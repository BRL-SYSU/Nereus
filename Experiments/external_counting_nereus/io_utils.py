"""Atomic and hashable experiment output helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_jsonl(
    records: Iterable[Mapping[str, Any]], output: Path
) -> dict[str, object]:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    record_count = 0
    try:
        with temporary_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(
                        dict(record), sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
                record_count += 1
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "output": str(output),
        "records": record_count,
        "sha256": sha256_file(output),
    }
