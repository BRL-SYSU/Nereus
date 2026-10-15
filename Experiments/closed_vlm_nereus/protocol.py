"""Frozen prompts, response parsing, and image encoding for Track A."""

from __future__ import annotations

import ast
import base64
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from Experiments.p2pnet_nereus.grid_metrics import GRID_ORDER, densest_region


COUNT_PROMPT = (
    "How many fish are visible in this underwater image? "
    "Answer with only the total count."
)
GRID_PROMPT = (
    "Estimate the number of fish in each 3x3 grid region and identify "
    "the densest region. Return JSON."
)
PROMPTS = {"count": COUNT_PROMPT, "grid": GRID_PROMPT}

_GRID_ALIASES = {
    "top left": "top-left",
    "upper left": "top-left",
    "top middle": "top-center",
    "top centre": "top-center",
    "upper center": "top-center",
    "upper centre": "top-center",
    "upper middle": "top-center",
    "top right": "top-right",
    "upper right": "top-right",
    "middle left": "center-left",
    "centre left": "center-left",
    "center left": "center-left",
    "middle": "center",
    "centre": "center",
    "middle center": "center",
    "middle centre": "center",
    "middle right": "center-right",
    "centre right": "center-right",
    "center right": "center-right",
    "bottom left": "bottom-left",
    "lower left": "bottom-left",
    "bottom middle": "bottom-center",
    "bottom centre": "bottom-center",
    "lower center": "bottom-center",
    "lower centre": "bottom-center",
    "lower middle": "bottom-center",
    "bottom right": "bottom-right",
    "lower right": "bottom-right",
}


def _normalized_key(value: object) -> str:
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    if text in _GRID_ALIASES:
        return _GRID_ALIASES[text]
    hyphenated = text.replace(" ", "-")
    return hyphenated if hyphenated in GRID_ORDER else ""


def image_to_data_uri(path: Path) -> str:
    """Encode an original image without resizing or recompression."""
    data = Path(path).read_bytes()
    if data[:3] == b"\xff\xd8\xff":
        mime = "image/jpeg"
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif data[:6] in (b"GIF87a", b"GIF89a"):
        mime = "image/gif"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError(f"unsupported image signature: {path}")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_count(text: str) -> dict[str, Any]:
    """Normalize a count-only model response without another model call."""
    stripped = str(text).strip()
    candidates: list[object] = []
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, int) and not isinstance(decoded, bool):
        candidates.append(decoded)
    elif isinstance(decoded, Mapping):
        for key in ("total_count", "count", "fish_count", "total"):
            if key in decoded:
                candidates.append(decoded[key])
    if not candidates:
        exact = re.fullmatch(r"\s*(\d+)(?:\.0+)?\s*[.!]?\s*", stripped)
        if exact:
            candidates.append(exact.group(1))
    if not candidates:
        labelled = re.search(
            r"(?i)(?:total(?:\s+count)?|fish(?:\s+count)?|count)"
            r"\s*(?:is|:|=)\s*(\d+)",
            stripped,
        )
        if labelled:
            candidates.append(labelled.group(1))
    if not candidates:
        return {"valid": False, "count": 0, "parse_method": "invalid"}
    try:
        value = int(candidates[0])
    except (TypeError, ValueError):
        return {"valid": False, "count": 0, "parse_method": "invalid"}
    if value < 0:
        return {"valid": False, "count": 0, "parse_method": "invalid"}
    return {"valid": True, "count": value, "parse_method": "normalized"}


def _json_candidates(text: str) -> list[Mapping[str, Any]]:
    stripped = str(text).strip()
    candidates: list[str] = [stripped]
    candidates.extend(
        match.group(1)
        for match in re.finditer(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])
    decoded: list[Mapping[str, Any]] = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(value, Mapping):
            decoded.append(value)
    return decoded


def _grid_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, int] = {}
    for key, count in value.items():
        region = _normalized_key(key)
        if not region:
            continue
        try:
            numeric = int(count)
        except (TypeError, ValueError):
            continue
        if numeric >= 0:
            normalized[region] = numeric
    return normalized


def parse_grid(text: str) -> dict[str, Any]:
    """Parse grid JSON and preserve whether densest was explicit or derived."""
    for decoded in _json_candidates(text):
        grid: dict[str, int] = {}
        for key in ("grid_counts", "counts", "regions", "grid"):
            grid = _grid_mapping(decoded.get(key))
            if grid:
                break
        if not grid:
            grid = _grid_mapping(decoded)

        explicit = ""
        for key in (
            "densest_grid_region",
            "densest_region",
            "densest_cell",
            "densest",
        ):
            explicit = _normalized_key(decoded.get(key, ""))
            if explicit:
                break

        complete = set(grid) == set(GRID_ORDER)
        derived = densest_region(grid) if complete else ""
        selected = explicit or derived
        if selected:
            return {
                "valid": True,
                "densest": selected,
                "grid_counts": (
                    {name: grid[name] for name in GRID_ORDER}
                    if complete
                    else grid
                ),
                "parse_method": "explicit" if explicit else "derived",
                "explicit_derived_consistent": (
                    None if not explicit or not derived else explicit == derived
                ),
            }

    region_match = re.search(
        r"(?i)densest(?:\s+(?:grid\s+)?(?:region|cell|area))?"
        r"\s*(?:is|:|=)\s*[\"']?([a-z_-]+(?:\s+[a-z_-]+)?)",
        str(text),
    )
    if region_match:
        region = _normalized_key(region_match.group(1))
        if region:
            return {
                "valid": True,
                "densest": region,
                "grid_counts": {},
                "parse_method": "labelled_text",
                "explicit_derived_consistent": None,
            }
    return {
        "valid": False,
        "densest": "__invalid__",
        "grid_counts": {},
        "parse_method": "invalid",
        "explicit_derived_consistent": None,
    }
