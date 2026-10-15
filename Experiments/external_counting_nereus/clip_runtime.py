"""Small, dependency-light reproductions of CLIP-Count inference utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def clip_sliding_window(
    image: torch.Tensor,
    window_size: tuple[int, int] = (384, 384),
    stride: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the official CLIP-Count horizontal window extraction exactly."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.ndim != 3:
        raise ValueError("image must have shape [3, 384, width]")
    image_numpy = image.permute(1, 2, 0).detach().cpu().numpy()
    padding = int(stride) - image_numpy.shape[1] % int(stride)
    image_numpy = np.pad(
        image_numpy, ((0, 0), (0, padding), (0, 0)), "constant"
    )
    height, width, _ = image_numpy.shape
    if height != int(window_size[0]):
        raise ValueError("CLIP-Count requires inference height 384")
    if width < int(window_size[1]):
        raise ValueError("resized image width must be at least 384")

    patches: list[np.ndarray] = []
    intervals: list[list[int]] = []
    for start in range(
        0, width - int(window_size[1]) + 1, int(stride)
    ):
        end = start + int(window_size[1])
        patches.append(image_numpy[:, start:end, :])
        intervals.append([start, end])
    if not patches:
        raise ValueError("no CLIP-Count inference windows were produced")
    return (
        np.asarray(patches).transpose(0, 3, 1, 2),
        np.asarray(intervals, dtype=np.int64),
    )


def clip_window_composite(
    patches: torch.Tensor,
    window_size: tuple[int, int] = (384, 384),
    stride: int = 128,
) -> torch.Tensor:
    """Match the official sigmoid-blended CLIP-Count window composition."""
    if not isinstance(patches, torch.Tensor) or patches.ndim != 4:
        raise ValueError("patches must have shape [N, C, 384, 384]")
    if patches.shape[0] < 1:
        raise ValueError("at least one patch is required")
    patch_width = int(window_size[1])
    blend_width = patch_width - int(stride)
    if blend_width <= 0:
        raise ValueError("stride must be smaller than window width")

    image = patches[0].clone()
    for patch in patches[1:]:
        previous = image[:, :, -blend_width:]
        following = patch[:, :, :blend_width]
        factor = torch.sigmoid(
            torch.tensor(
                np.linspace(-3, 3, blend_width),
                device=image.device,
                dtype=image.dtype,
            )
        )
        image[:, :, -blend_width:] = (
            (1.0 - factor) * previous + factor * following
        )
        image = torch.cat((image, patch[:, :, blend_width:]), dim=-1)
    return image


def extract_clipcount_state_dict(
    checkpoint: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract the inner CLIPCount weights from a Lightning checkpoint."""
    nested_state = checkpoint.get("state_dict")
    if isinstance(nested_state, Mapping):
        raw_state = nested_state
    elif checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        raw_state = checkpoint
    else:
        raise ValueError("CLIP-Count checkpoint has no state_dict")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        normalized_name = str(name)
        if normalized_name.startswith("model."):
            normalized_name = normalized_name.removeprefix("model.")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"non-tensor checkpoint value: {name}")
        state[normalized_name] = value
    if not state:
        raise ValueError("CLIP-Count checkpoint state_dict is empty")
    return state
