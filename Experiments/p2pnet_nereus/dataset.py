"""Point-safe PyTorch data loading for Nereus."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as vision_functional


CROP_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransformParams:
    scale: float
    flip: bool
    crop_x: int
    crop_y: int


def sample_transform_params(
    *, width: int, height: int, seed: int, epoch: int, image_id: str
) -> TransformParams:
    digest = hashlib.sha256(
        f"{seed}:{epoch}:{image_id}".encode("utf-8")
    ).digest()
    generator = random.Random(int.from_bytes(digest[:8], byteorder="big"))
    scale = generator.uniform(0.8, 1.2)
    scaled_width = round(width * scale)
    scaled_height = round(height * scale)
    if scaled_width < CROP_SIZE or scaled_height < CROP_SIZE:
        raise ValueError(
            f"scaled image is smaller than {CROP_SIZE}: "
            f"{scaled_width}x{scaled_height}"
        )
    return TransformParams(
        scale=scale,
        flip=generator.random() < 0.5,
        crop_x=generator.randint(0, scaled_width - CROP_SIZE),
        crop_y=generator.randint(0, scaled_height - CROP_SIZE),
    )


def _normalized_tensor(image: Image.Image) -> torch.Tensor:
    tensor = vision_functional.to_tensor(image)
    return vision_functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)


def transform_train(
    image: Image.Image, points_xy: np.ndarray, params: TransformParams
) -> tuple[torch.Tensor, torch.Tensor]:
    image = image.convert("RGB")
    original_width, original_height = image.size
    scaled_width = round(original_width * params.scale)
    scaled_height = round(original_height * params.scale)
    if scaled_width < CROP_SIZE or scaled_height < CROP_SIZE:
        raise ValueError("scaled training image is smaller than the crop")
    if not (
        0 <= params.crop_x <= scaled_width - CROP_SIZE
        and 0 <= params.crop_y <= scaled_height - CROP_SIZE
    ):
        raise ValueError("crop lies outside the scaled image")

    image = image.resize(
        (scaled_width, scaled_height), resample=Image.Resampling.BILINEAR
    )
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2).copy()
    if len(points):
        points[:, 0] *= scaled_width / original_width
        points[:, 1] *= scaled_height / original_height

    if params.flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if len(points):
            points[:, 0] = scaled_width - 1 - points[:, 0]

    image = image.crop(
        (
            params.crop_x,
            params.crop_y,
            params.crop_x + CROP_SIZE,
            params.crop_y + CROP_SIZE,
        )
    )
    if len(points):
        points[:, 0] -= params.crop_x
        points[:, 1] -= params.crop_y
        in_crop = (
            (points[:, 0] >= 0)
            & (points[:, 0] < CROP_SIZE)
            & (points[:, 1] >= 0)
            & (points[:, 1] < CROP_SIZE)
        )
        points = points[in_crop]
    points_tensor = torch.from_numpy(points.reshape(-1, 2)).to(torch.float32)
    return _normalized_tensor(image), points_tensor


def transform_eval(image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
    image = image.convert("RGB")
    width, height = image.size
    tensor = _normalized_tensor(image)
    padded_width = ((width + 15) // 16) * 16
    padded_height = ((height + 15) // 16) * 16
    tensor = torch_functional.pad(
        tensor, (0, padded_width - width, 0, padded_height - height)
    )
    return tensor, (width, height)


class NereusPointDataset(Dataset):
    def __init__(
        self, manifest_path: Path, *, training: bool, seed: int = 42
    ) -> None:
        self.records = [
            json.loads(line)
            for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.training = training
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        points = np.asarray(record["points_xy"], dtype=np.float32).reshape(-1, 2)
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")

        if self.training:
            params = sample_transform_params(
                width=image.width,
                height=image.height,
                seed=self.seed,
                epoch=self.epoch,
                image_id=record["image_id"],
            )
            image_tensor, point_tensor = transform_train(image, points, params)
            original_size = (image.width, image.height)
        else:
            image_tensor, original_size = transform_eval(image)
            point_tensor = torch.from_numpy(points).to(torch.float32)

        target: dict[str, Any] = {
            "point": point_tensor,
            "labels": torch.ones(len(point_tensor), dtype=torch.int64),
            "image_id": record["image_id"],
            "original_size": original_size,
            "target_count": int(record["count"]),
        }
        return image_tensor, target


def collate_points(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
