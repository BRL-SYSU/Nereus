import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from Experiments.p2pnet_nereus.artifacts import (
    PINNED_COMMIT,
    sha256_file,
    validate_manifest,
)


MODEL_ROOT = Path("models/P2PNet")
PATCH_PATH = Path(__file__).resolve().parents[1] / "patches" / "p2pnet-local.patch"


class ArtifactTests(unittest.TestCase):
    def test_compatibility_patch_is_valid_unified_diff(self) -> None:
        result = subprocess.run(
            ["git", "apply", "--numstat", str(PATCH_PATH)],
            cwd=MODEL_ROOT / "upstream",
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_download_manifest_pins_only_imagenet_initialization(self) -> None:
        manifest = validate_manifest(MODEL_ROOT / "DOWNLOAD_MANIFEST.json")

        self.assertEqual(
            manifest["upstream"]["commit"],
            "5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455",
        )
        self.assertEqual(manifest["backbone"]["scope"], "imagenet_only")
        self.assertFalse(manifest["counting_task_checkpoint_downloaded"])

    def test_vgg_hash_matches_the_official_filename_prefix(self) -> None:
        digest = sha256_file(
            MODEL_ROOT / "pretrained" / "vgg16_bn-6c64b313.pth"
        )

        self.assertTrue(digest.startswith("6c64b313"))

    def test_manifest_rejects_a_counting_task_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "source.tar.gz"
            backbone = root / "vgg16_bn-6c64b313.pth"
            archive.write_bytes(b"source")
            backbone.write_bytes(b"backbone")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "upstream": {
                            "commit": PINNED_COMMIT,
                            "archive": {
                                "path": str(archive),
                                "sha256": sha256_file(archive),
                                "size_bytes": archive.stat().st_size,
                            },
                        },
                        "backbone": {
                            "path": str(backbone),
                            "sha256": sha256_file(backbone),
                            "size_bytes": backbone.stat().st_size,
                            "scope": "imagenet_only",
                        },
                        "counting_task_checkpoint_downloaded": True,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "counting-task checkpoints are forbidden"
            ):
                validate_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
