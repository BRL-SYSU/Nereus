"""Atomic per-image cache used to resume long external-model inference."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_SAFE_IMAGE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class PredictionCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, image_id: str) -> Path:
        normalized = str(image_id)
        if not _SAFE_IMAGE_ID.fullmatch(normalized):
            raise ValueError(f"unsafe image ID: {normalized!r}")
        return self.directory / f"{normalized}.json"

    def write(self, record: Mapping[str, Any]) -> Path:
        destination = self._path(str(record["image_id"]))
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=self.directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    dict(record),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
            os.replace(temporary_name, destination)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        return destination

    def read(
        self,
        image_id: str,
        *,
        method: str,
        checkpoint_sha256: str,
        text_query: str,
    ) -> dict[str, Any] | None:
        path = self._path(image_id)
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "image_id": str(image_id),
            "method": str(method),
            "checkpoint_sha256": str(checkpoint_sha256),
            "text_query": str(text_query),
        }
        if any(record.get(name) != value for name, value in expected.items()):
            return None
        return record
