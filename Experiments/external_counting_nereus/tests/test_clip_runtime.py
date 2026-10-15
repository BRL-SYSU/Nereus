from __future__ import annotations

import unittest

import numpy as np
import torch

from Experiments.external_counting_nereus.clip_runtime import (
    clip_sliding_window,
    clip_window_composite,
    extract_clipcount_state_dict,
)


class ClipSlidingWindowTests(unittest.TestCase):
    def test_official_padding_and_stride_are_reproduced(self) -> None:
        image = torch.arange(
            3 * 384 * 500, dtype=torch.float32
        ).reshape(1, 3, 384, 500)

        patches, intervals = clip_sliding_window(image, stride=128)

        self.assertEqual(tuple(patches.shape), (2, 3, 384, 384))
        np.testing.assert_array_equal(
            intervals, np.asarray([[0, 384], [128, 512]])
        )
        np.testing.assert_array_equal(
            patches[0, :, :, 128:384],
            patches[1, :, :, 0:256],
        )
        self.assertTrue(np.all(patches[1, :, :, 372:384] == 0.0))

    def test_soft_composite_preserves_shape_and_constant_boundaries(self) -> None:
        patches = torch.stack(
            (
                torch.ones((1, 384, 384), dtype=torch.float32),
                torch.full((1, 384, 384), 3.0, dtype=torch.float32),
            )
        )

        image = clip_window_composite(patches, stride=128)

        self.assertEqual(tuple(image.shape), (1, 384, 512))
        self.assertTrue(torch.all(image[:, :, :128] == 1.0))
        self.assertTrue(torch.all(image[:, :, 384:] == 3.0))
        self.assertGreater(float(image[0, 0, 128]), 1.0)
        self.assertLess(float(image[0, 0, 383]), 3.0)


class ClipCheckpointTests(unittest.TestCase):
    def test_lightning_model_prefix_is_removed_without_clip_parameters(self) -> None:
        checkpoint = {
            "state_dict": {
                "model.density_decoder.output_layer.weight": torch.ones(1),
                "model.patch_feat_proj.bias": torch.zeros(1),
            }
        }

        state = extract_clipcount_state_dict(checkpoint)

        self.assertEqual(
            set(state),
            {
                "density_decoder.output_layer.weight",
                "patch_feat_proj.bias",
            },
        )

    def test_raw_model_state_dict_is_supported(self) -> None:
        checkpoint = {
            "density_decoder.output_layer.weight": torch.ones(1),
            "patch_feat_proj.bias": torch.zeros(1),
        }

        state = extract_clipcount_state_dict(checkpoint)

        self.assertEqual(state, checkpoint)


if __name__ == "__main__":
    unittest.main()
