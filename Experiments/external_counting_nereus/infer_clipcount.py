"""Run label-isolated CLIP-Count inference on Nereus Track A test images."""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as vision_functional
from PIL import Image

from .adapters import clip_density_to_prediction, validate_prediction_record
from .clip_runtime import (
    clip_sliding_window,
    clip_window_composite,
    extract_clipcount_state_dict,
)
from .config import (
    CLIPCOUNT_SCALE_FACTOR,
    EXPECTED_TEST_SIZE,
    TEXT_QUERY,
    ExternalExperimentPaths,
)
from .dataset import read_image_manifest
from .io_utils import atomic_write_jsonl, sha256_file
from .prediction_cache import PredictionCache


_MODEL_DEFAULTS: dict[str, Any] = {
    "decoder_depth": 4,
    "decoder_head": 8,
    "use_coop": True,
    "use_vpt": True,
    "coop_width": 2,
    "vpt_width": 20,
    "vpt_depth": 10,
    "backbone": "b16",
    "use_fim": False,
    "use_mixed_fim": True,
    "unfreeze_vit": False,
}


def _checkpoint_hparams(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    hyperparameters = checkpoint.get("hyper_parameters", {})
    if not isinstance(hyperparameters, Mapping):
        return dict(_MODEL_DEFAULTS)
    raw_args = hyperparameters.get("args", hyperparameters)
    if isinstance(raw_args, argparse.Namespace):
        raw_args = vars(raw_args)
    if not isinstance(raw_args, Mapping):
        return dict(_MODEL_DEFAULTS)
    return {
        name: raw_args.get(name, default)
        for name, default in _MODEL_DEFAULTS.items()
    }


def _build_model(
    *,
    repo: Path,
    checkpoint_path: Path,
    clip_base_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    sys.path.insert(0, str(repo))
    import clip.clip as clip_implementation

    expected_clip_sha256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
    if sha256_file(clip_base_path) != expected_clip_sha256:
        raise ValueError("OpenAI ViT-B/16 checkpoint hash mismatch")

    def use_local_clip_checkpoint(url: str, root: str) -> str:
        del root
        if expected_clip_sha256 not in url:
            raise ValueError(f"unexpected OpenAI CLIP URL: {url}")
        return str(clip_base_path)

    clip_implementation._download = use_local_clip_checkpoint
    from models.clip_count import CLIPCount

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    hparams = _checkpoint_hparams(checkpoint)
    model = CLIPCount(
        fim_depth=int(hparams["decoder_depth"]),
        fim_num_heads=int(hparams["decoder_head"]),
        use_coop=bool(hparams["use_coop"]),
        use_vpt=bool(hparams["use_vpt"]),
        coop_width=int(hparams["coop_width"]),
        vpt_width=int(hparams["vpt_width"]),
        vpt_depth=int(hparams["vpt_depth"]),
        backbone=str(hparams["backbone"]),
        use_fim=bool(hparams["use_fim"]),
        use_mixed_fim=bool(hparams["use_mixed_fim"]),
        unfreeze_vit=bool(hparams["unfreeze_vit"]),
    )
    state = extract_clipcount_state_dict(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    non_clip_missing = [
        name
        for name in incompatible.missing_keys
        if not (
            name.startswith("clip.")
            or name.startswith("img_encoder.vit.")
            or name.startswith("text_encoder.clip.")
        )
    ]
    if non_clip_missing:
        raise ValueError(
            "CLIP-Count checkpoint is missing trained parameters: "
            f"{non_clip_missing[:10]}"
        )
    if incompatible.unexpected_keys:
        raise ValueError(
            "CLIP-Count checkpoint has unexpected parameters: "
            f"{incompatible.unexpected_keys[:10]}"
        )
    print(
        "CLIP-Count checkpoint loaded: "
        f"allowed_frozen_clip_missing={len(incompatible.missing_keys)}",
        flush=True,
    )
    return model.to(device).eval()


def _prepare_image(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = vision_functional.resize(tensor, 384, antialias=False)
    return torch.clamp(tensor.float() / 255.0, 0.0, 1.0)


def _progress(index: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    seconds_per_image = elapsed / index
    eta_seconds = seconds_per_image * (total - index)
    print(
        f"CLIP-Count {index}/{total} ({100.0 * index / total:.2f}%) "
        f"elapsed={elapsed / 60:.1f}m eta={eta_seconds / 60:.1f}m",
        flush=True,
    )


def main() -> None:
    defaults = ExternalExperimentPaths.defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=defaults.test_images_manifest
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=defaults.model_root / "CLIP-Count" / "repo",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=defaults.model_root
        / "CLIP-Count"
        / "checkpoints"
        / "clipcount_pretrained.ckpt",
    )
    parser.add_argument(
        "--clip-base",
        type=Path,
        default=defaults.model_root
        / "CLIP-Count"
        / "clip_cache"
        / "ViT-B-16.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults.output_root / "clipcount_predictions.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=defaults.output_root / "clipcount_cache",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    records = read_image_manifest(args.manifest)
    if len(records) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"expected {EXPECTED_TEST_SIZE} test images, got {len(records)}"
        )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]

    checkpoint_sha256 = sha256_file(args.checkpoint)
    device = torch.device(args.device)
    model = _build_model(
        repo=args.repo.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        clip_base_path=args.clip_base.resolve(),
        device=device,
    )
    cache = PredictionCache(args.cache_dir)
    predictions: list[dict[str, object]] = []
    started = time.monotonic()
    for index, image_record in enumerate(records, start=1):
        image_id = str(image_record["image_id"])
        cached = cache.read(
            image_id,
            method="CLIP-Count",
            checkpoint_sha256=checkpoint_sha256,
            text_query=TEXT_QUERY,
        )
        if cached is not None:
            validate_prediction_record(cached)
            prediction = cached
        else:
            image_path = Path(str(image_record["image_path"]))
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            width, height = image.size
            if width != int(image_record["width"]) or height != int(
                image_record["height"]
            ):
                raise ValueError(f"image dimension mismatch: {image_id}")
            resized = _prepare_image(image)
            resized_width = int(resized.shape[-1])
            patches_numpy, _ = clip_sliding_window(resized, stride=128)
            patches = torch.from_numpy(patches_numpy).float().to(device)
            prompts = [TEXT_QUERY] * len(patches)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16
            ):
                density_patches = model(patches, prompts).unsqueeze(1)
                density = clip_window_composite(
                    density_patches, stride=128
                ).squeeze(0)
                density = density[:, :resized_width]
            prediction = clip_density_to_prediction(
                image_id=image_id,
                width=width,
                height=height,
                density_map=density.detach().float().cpu().numpy(),
                scale_factor=CLIPCOUNT_SCALE_FACTOR,
                checkpoint_sha256=checkpoint_sha256,
            )
            cache.write(prediction)
        predictions.append(prediction)
        if index == 1 or index % 10 == 0 or index == len(records):
            _progress(index, len(records), started)

    result = atomic_write_jsonl(predictions, args.output)
    print(result, flush=True)


if __name__ == "__main__":
    main()
