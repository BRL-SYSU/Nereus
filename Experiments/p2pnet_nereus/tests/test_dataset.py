import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DistributedSampler, TensorDataset

from Experiments.p2pnet_nereus.dataset import (
    NereusPointDataset,
    TransformParams,
    collate_points,
    sample_transform_params,
    transform_eval,
    transform_train,
)
from Experiments.p2pnet_nereus.train import build_training_loader


class DatasetTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = Image.new("RGB", (1920, 1080), color=(127, 127, 127))

    def test_scale_flip_crop_preserves_xy_order(self) -> None:
        params = TransformParams(
            scale=1.0, flip=True, crop_x=100, crop_y=50
        )

        tensor, points = transform_train(
            self.image,
            np.array([[1719.0, 100.0]], dtype=np.float32),
            params,
        )

        self.assertEqual(tuple(tensor.shape), (3, 512, 512))
        torch.testing.assert_close(
            points, torch.tensor([[100.0, 50.0]], dtype=torch.float32)
        )

    def test_crop_discards_points_outside_the_crop(self) -> None:
        _, points = transform_train(
            self.image,
            np.array([[10.0, 10.0], [200.0, 100.0]], dtype=np.float32),
            TransformParams(scale=1.0, flip=False, crop_x=100, crop_y=50),
        )

        torch.testing.assert_close(
            points, torch.tensor([[100.0, 50.0]], dtype=torch.float32)
        )

    def test_empty_points_keep_two_columns(self) -> None:
        _, points = transform_train(
            self.image,
            np.empty((0, 2), dtype=np.float32),
            TransformParams(scale=1.0, flip=False, crop_x=0, crop_y=0),
        )

        self.assertEqual(tuple(points.shape), (0, 2))
        self.assertEqual(points.dtype, torch.float32)

    def test_eval_pads_bottom_to_multiple_of_sixteen(self) -> None:
        tensor, original_size = transform_eval(self.image)

        self.assertEqual(original_size, (1920, 1080))
        self.assertEqual(tuple(tensor.shape), (3, 1088, 1920))
        torch.testing.assert_close(
            tensor[:, 1080:, :],
            torch.zeros((3, 8, 1920), dtype=torch.float32),
        )

    def test_sampling_is_stable_for_seed_epoch_and_image(self) -> None:
        first = sample_transform_params(
            width=1920, height=1080, seed=42, epoch=7, image_id="0003"
        )
        second = sample_transform_params(
            width=1920, height=1080, seed=42, epoch=7, image_id="0003"
        )

        self.assertEqual(first, second)
        self.assertGreaterEqual(first.scale, 0.8)
        self.assertLessEqual(first.scale, 1.2)

    def test_training_loader_restarts_workers_for_epoch_updates(self) -> None:
        dataset = TensorDataset(torch.arange(4))
        sampler = DistributedSampler(
            dataset,
            num_replicas=1,
            rank=0,
            shuffle=False,
        )

        loader = build_training_loader(
            dataset=dataset,
            sampler=sampler,
            batch_size=2,
            num_workers=1,
            seed=42,
            rank=0,
        )

        self.assertFalse(loader.persistent_workers)

    def test_eval_dataset_and_collate_preserve_target_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (32, 18), color=(10, 20, 30)).save(image_path)
            manifest = root / "val.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "image_id": "sample",
                        "image_path": str(image_path),
                        "annotation_path": str(root / "sample.json"),
                        "width": 32,
                        "height": 18,
                        "count": 1,
                        "points_xy": [[10.0, 11.0]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = NereusPointDataset(manifest, training=False, seed=42)
            image, target = dataset[0]
            images, targets = collate_points([(image, target), (image, target)])

        self.assertEqual(tuple(image.shape), (3, 32, 32))
        self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
        self.assertEqual(target["image_id"], "sample")
        self.assertEqual(target["original_size"], (32, 18))
        self.assertEqual(target["target_count"], 1)
        torch.testing.assert_close(
            target["point"], torch.tensor([[10.0, 11.0]])
        )
        torch.testing.assert_close(targets[1]["labels"], torch.ones(1, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
