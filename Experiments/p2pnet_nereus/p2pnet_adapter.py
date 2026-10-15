"""Strict adapter around the pinned official P2PNet implementation."""

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .artifacts import validate_manifest
from .config import MODEL_ROOT


@lru_cache(maxsize=1)
def _validated_manifest() -> dict[str, Any]:
    return validate_manifest(MODEL_ROOT / "DOWNLOAD_MANIFEST.json")


def _upstream_builder() -> Any:
    manifest = _validated_manifest()
    backbone_path = Path(manifest["backbone"]["path"])
    os.environ["P2PNET_VGG16_BN_WEIGHTS"] = str(backbone_path)
    upstream_path = Path(manifest["upstream"]["extracted_path"]).resolve()
    upstream_text = str(upstream_path)
    if upstream_text not in sys.path:
        sys.path.insert(0, upstream_text)

    existing_models = sys.modules.get("models")
    if existing_models is not None:
        module_file = Path(getattr(existing_models, "__file__", "")).resolve()
        if upstream_path not in module_file.parents:
            raise RuntimeError(
                f"Python module name collision for 'models': {module_file}"
            )
    models_module = importlib.import_module("models")
    return models_module.build_model


def build_p2pnet(training: bool) -> nn.Module | tuple[nn.Module, nn.Module]:
    args = SimpleNamespace(
        backbone="vgg16_bn",
        row=2,
        line=2,
        set_cost_class=1.0,
        set_cost_point=0.05,
        point_loss_coef=0.0002,
        eos_coef=0.5,
    )
    return _upstream_builder()(args, training=training)


def decode_predictions(
    outputs: dict[str, torch.Tensor],
    threshold: float,
    image_sizes: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    logits = outputs["pred_logits"]
    points = outputs["pred_points"]
    if logits.ndim != 3 or points.ndim != 3:
        raise ValueError("P2PNet outputs must be rank-three tensors")
    if logits.shape[:2] != points.shape[:2] or logits.shape[-1] != 2:
        raise ValueError("P2PNet output shapes are inconsistent")
    if points.shape[-1] != 2 or len(image_sizes) != logits.shape[0]:
        raise ValueError("point shape or image-size batch is inconsistent")
    if not torch.isfinite(logits).all() or not torch.isfinite(points).all():
        raise ValueError("P2PNet produced a non-finite output")

    foreground_scores = torch.softmax(logits, dim=-1)[..., 1]
    decoded: list[dict[str, Any]] = []
    for batch_index, (width, height) in enumerate(image_sizes):
        sample_points = points[batch_index]
        sample_scores = foreground_scores[batch_index]
        keep = (
            (sample_scores >= threshold)
            & (sample_points[:, 0] >= 0)
            & (sample_points[:, 0] < width)
            & (sample_points[:, 1] >= 0)
            & (sample_points[:, 1] < height)
        )
        kept_points = sample_points[keep].detach().cpu().to(torch.float32)
        kept_scores = sample_scores[keep].detach().cpu().to(torch.float32)
        point_list = kept_points.tolist()
        decoded.append(
            {
                "count": int(len(point_list)),
                "points_xy": point_list,
                "scores": kept_scores.tolist(),
            }
        )
    return decoded
