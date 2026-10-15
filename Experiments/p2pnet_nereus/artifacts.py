"""Artifact provenance checks for the Nereus P2PNet experiment."""

import hashlib
import json
from pathlib import Path
from typing import Any


PINNED_COMMIT = "5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file(item: dict[str, Any], role: str) -> None:
    artifact = Path(item["path"])
    if not artifact.is_file():
        raise ValueError(f"missing {role} artifact: {artifact}")
    if artifact.stat().st_size != int(item["size_bytes"]):
        raise ValueError(f"{role} size mismatch: {artifact}")
    if sha256_file(artifact) != item["sha256"]:
        raise ValueError(f"{role} SHA-256 mismatch: {artifact}")


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))

    if manifest["upstream"]["commit"] != PINNED_COMMIT:
        raise ValueError("unexpected P2PNet commit")
    if manifest["backbone"]["scope"] != "imagenet_only":
        raise ValueError("non-ImageNet initialization is forbidden")
    if manifest["counting_task_checkpoint_downloaded"]:
        raise ValueError("counting-task checkpoints are forbidden")

    _validate_file(manifest["upstream"]["archive"], "upstream")
    _validate_file(manifest["backbone"], "backbone")
    patch = manifest.get("compatibility_patch")
    if patch is not None:
        _validate_file(patch, "compatibility patch")
    return manifest
