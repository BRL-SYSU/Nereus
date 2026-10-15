"""Run label-isolated CountGD inference on the Nereus Track A test images."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .adapters import countgd_boxes_to_prediction, validate_prediction_record
from .config import (
    COUNTGD_THRESHOLD,
    EXPECTED_TEST_SIZE,
    TEXT_QUERY,
    ExternalExperimentPaths,
)
from .dataset import read_image_manifest
from .io_utils import atomic_write_jsonl, sha256_file
from .prediction_cache import PredictionCache


def _build_model_and_transform(
    *,
    repo: Path,
    config_path: Path,
    checkpoint_path: Path,
    bert_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, object]:
    sys.path.insert(0, str(repo))
    from datasets_inference import transforms as transforms
    from models.registry import MODULE_BUILD_FUNCS
    from util.slconfig import SLConfig

    config = SLConfig.fromfile(str(config_path))
    config.merge_from_dict({"text_encoder_type": str(bert_path)})
    args = argparse.Namespace(device=str(device))
    for name, value in config._cfg_dict.to_dict().items():
        setattr(args, name, value)

    if args.modelname not in MODULE_BUILD_FUNCS._module_dict:
        raise ValueError(f"unregistered CountGD model: {args.modelname}")
    model, _, _ = MODULE_BUILD_FUNCS.get(args.modelname)(args)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise ValueError("CountGD checkpoint has no model state")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    print(
        "CountGD checkpoint loaded: "
        f"missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}",
        flush=True,
    )
    model.eval()

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    transform = transforms.Compose(
        [transforms.RandomResize([800], max_size=1333), normalize]
    )
    return model, transform


def _progress(index: int, total: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    seconds_per_image = elapsed / index
    eta_seconds = seconds_per_image * (total - index)
    print(
        f"CountGD {index}/{total} ({100.0 * index / total:.2f}%) "
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
        "--repo", type=Path, default=defaults.model_root / "CountGD" / "repo"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=defaults.model_root
        / "CountGD"
        / "repo"
        / "config"
        / "cfg_fsc147_vit_b.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=defaults.model_root
        / "CountGD"
        / "checkpoints"
        / "checkpoint_fsc147_best.pth",
    )
    parser.add_argument(
        "--bert",
        type=Path,
        default=defaults.model_root
        / "CountGD"
        / "checkpoints"
        / "bert-base-uncased",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=defaults.output_root / "countgd_predictions.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=defaults.output_root / "countgd_cache",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=COUNTGD_THRESHOLD)
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
    model, transform = _build_model_and_transform(
        repo=args.repo.resolve(),
        config_path=args.config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        bert_path=args.bert.resolve(),
        device=device,
    )
    cache = PredictionCache(args.cache_dir)
    predictions: list[dict[str, object]] = []
    started = time.monotonic()
    for index, image_record in enumerate(records, start=1):
        image_id = str(image_record["image_id"])
        cached = cache.read(
            image_id,
            method="CountGD",
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
            transformed, target = transform(
                image, {"exemplars": torch.tensor([])}
            )
            transformed = transformed.to(device)
            exemplars = target["exemplars"].to(device)
            with torch.inference_mode():
                output = model(
                    transformed.unsqueeze(0),
                    [exemplars],
                    [torch.tensor([0], device=device)],
                    captions=[f"{TEXT_QUERY} ."],
                )
                logits = output["pred_logits"][0].sigmoid()
                boxes = output["pred_boxes"][0]
                scores = logits.max(dim=-1).values
            prediction = countgd_boxes_to_prediction(
                image_id=image_id,
                width=width,
                height=height,
                boxes_cxcywh=boxes.detach().cpu().numpy(),
                scores=scores.detach().cpu().numpy(),
                confidence_threshold=args.threshold,
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
