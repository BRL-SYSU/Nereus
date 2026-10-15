from __future__ import annotations

import unittest

from Experiments.external_counting_nereus.config import ExternalExperimentPaths


class ExternalExperimentPathTests(unittest.TestCase):
    def test_defaults_use_shared_nereus_test_split_and_requested_model_root(
        self,
    ) -> None:
        paths = ExternalExperimentPaths.defaults()

        self.assertEqual(paths.test_images_manifest.name, "test_images.jsonl")
        self.assertEqual(paths.test_targets.name, "test.jsonl")
        self.assertEqual(paths.model_root.name, "models")
        self.assertIn("p2pnet_nereus/data", str(paths.test_images_manifest))
        self.assertIn(
            "models", str(paths.model_root)
        )


if __name__ == "__main__":
    unittest.main()
