#!/usr/bin/env bash
set -euo pipefail

model_root="models/P2PNet"
source_dir="${model_root}/source"
pretrained_dir="${model_root}/pretrained"
upstream_dir="${model_root}/upstream"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
patch_path="${script_dir}/patches/p2pnet-local.patch"
python_bin="python"
commit="5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455"
archive_url="https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/archive/${commit}.tar.gz"
backbone_url="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
archive_path="${source_dir}/CrowdCounting-P2PNet-${commit}.tar.gz"
backbone_path="${pretrained_dir}/vgg16_bn-6c64b313.pth"
archive_size="95014249"
archive_sha256="d4c8207d4bdf25ac51169bf10c7b4cb943149cde7dc347dd1db2f6a4b05cfc97"
backbone_size="553507836"
backbone_sha256="6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a"

verify_file() {
    local path="$1"
    local expected_size="$2"
    local expected_sha256="$3"
    local actual_size
    local actual_sha256
    actual_size="$(stat -c '%s' "${path}")"
    actual_sha256="$(sha256sum "${path}")"
    actual_sha256="${actual_sha256%% *}"
    if [[ "${actual_size}" != "${expected_size}" || \
          "${actual_sha256}" != "${expected_sha256}" ]]; then
        echo "Artifact integrity failure: ${path}" >&2
        exit 1
    fi
}

mkdir -p "${source_dir}" "${pretrained_dir}" "${model_root}/checkpoints" "${model_root}/logs"

if [[ ! -f "${archive_path}" ]]; then
    /usr/bin/wget -c -O "${archive_path}.part" "${archive_url}"
    verify_file "${archive_path}.part" "${archive_size}" "${archive_sha256}"
    mv "${archive_path}.part" "${archive_path}"
fi
verify_file "${archive_path}" "${archive_size}" "${archive_sha256}"
tar -tzf "${archive_path}" >/dev/null

if [[ ! -d "${upstream_dir}" ]]; then
    extract_dir="$(mktemp -d "${model_root}/.extract.XXXXXX")"
    tar -xzf "${archive_path}" -C "${extract_dir}"
    extracted_root="${extract_dir}/CrowdCounting-P2PNet-${commit}"
    if [[ ! -d "${extracted_root}" ]]; then
        echo "Expected archive root is missing: ${extracted_root}" >&2
        exit 1
    fi
    mv "${extracted_root}" "${upstream_dir}"
    rmdir "${extract_dir}"
fi

if git -C "${upstream_dir}" apply --check "${patch_path}" 2>/dev/null; then
    git -C "${upstream_dir}" apply "${patch_path}"
elif ! git -C "${upstream_dir}" apply --reverse --check "${patch_path}" 2>/dev/null; then
    echo "P2PNet source is neither pristine nor patched as expected" >&2
    exit 1
fi

if [[ ! -f "${backbone_path}" ]]; then
    /usr/bin/wget -c -O "${backbone_path}.part" "${backbone_url}"
    verify_file "${backbone_path}.part" "${backbone_size}" "${backbone_sha256}"
    mv "${backbone_path}.part" "${backbone_path}"
fi
verify_file "${backbone_path}" "${backbone_size}" "${backbone_sha256}"

"${python_bin}" - \
    "${model_root}" "${archive_path}" "${archive_url}" \
    "${backbone_path}" "${backbone_url}" "${patch_path}" "${commit}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


model_root, archive, archive_url, backbone, backbone_url, patch, commit = sys.argv[1:]
model_root = Path(model_root)
archive = Path(archive)
backbone = Path(backbone)
patch = Path(patch)
manifest = {
    "upstream": {
        "repository": "https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet",
        "commit": commit,
        "archive": {
            "url": archive_url,
            "path": str(archive),
            "size_bytes": archive.stat().st_size,
            "sha256": digest(archive),
        },
        "extracted_path": str(model_root / "upstream"),
    },
    "backbone": {
        "url": backbone_url,
        "path": str(backbone),
        "size_bytes": backbone.stat().st_size,
        "sha256": digest(backbone),
        "scope": "imagenet_only",
    },
    "compatibility_patch": {
        "path": str(patch),
        "size_bytes": patch.stat().st_size,
        "sha256": digest(patch),
        "changes": [
            "local_vgg_path",
            "input_device_anchor_points",
            "robust_torchvision_version_check",
            "point_loss_weight_key",
            "deterministic_cross_entropy_shape",
        ],
    },
    "counting_task_checkpoint_downloaded": False,
}
target = model_root / "DOWNLOAD_MANIFEST.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
