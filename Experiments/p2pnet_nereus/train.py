"""Deterministic distributed training for the Nereus P2PNet baseline."""

import argparse
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from .artifacts import sha256_file
from .config import MODEL_ROOT, ExperimentPaths
from .dataset import NereusPointDataset, collate_points
from .grid_metrics import aggregate_metrics, densest_region, grid_counts
from .p2pnet_adapter import build_p2pnet, decode_predictions


POINT_LOSS_COEFFICIENT = 0.0002
HEAD_LR = 1e-4
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 0.1
FIXED_THRESHOLD = 0.5


def validate_training_topology(
    world_size: int, batch_size_per_gpu: int
) -> int:
    if world_size not in (2, 4):
        raise ValueError("DDP world size must be two or four")
    global_batch_size = world_size * batch_size_per_gpu
    if global_batch_size != 8:
        raise ValueError("global batch size 8 is required")
    return global_batch_size


def seed_everything(seed: int, rank: int) -> None:
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def weighted_loss(
    loss_dict: Mapping[str, torch.Tensor | float],
) -> torch.Tensor | float:
    if "loss_ce" not in loss_dict or "loss_point" not in loss_dict:
        raise KeyError("criterion must emit loss_ce and loss_point")
    return (
        loss_dict["loss_ce"]
        + POINT_LOSS_COEFFICIENT * loss_dict["loss_point"]
    )


def checkpoint_rank(
    metrics: Mapping[str, float | int],
) -> tuple[float, float, float, int]:
    values = (
        float(metrics["mae"]),
        float(metrics["rmse"]),
        -float(metrics["exact"]),
        int(metrics["epoch"]),
    )
    if not all(math.isfinite(value) for value in values[:3]):
        raise ValueError("checkpoint metrics must be finite")
    return values


def make_checkpoint(
    *,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    epoch: int,
    config: Mapping[str, Any],
    validation: Mapping[str, Any],
    artifact_manifest_sha256: str,
) -> dict[str, object]:
    if (
        len(artifact_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in artifact_manifest_sha256
        )
    ):
        raise ValueError("artifact manifest SHA-256 must be lowercase hex")
    return {
        "model": dict(model_state),
        "optimizer": dict(optimizer_state),
        "epoch": int(epoch),
        "config": dict(config),
        "validation": dict(validation),
        "artifact_manifest_sha256": artifact_manifest_sha256,
    }


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _move_training_targets(
    targets: list[dict[str, Any]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "point": target["point"].to(device, non_blocking=True),
            "labels": target["labels"].to(device, non_blocking=True),
        }
        for target in targets
    ]


def train_one_epoch(
    *,
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    smoke_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    criterion.train()
    totals = {"loss": 0.0, "loss_ce": 0.0, "loss_point": 0.0}
    steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        device_targets = _move_training_targets(targets, device)
        outputs = model(images)
        loss_dict = criterion(outputs, device_targets)
        loss = weighted_loss(loss_dict)
        if not isinstance(loss, torch.Tensor) or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss: {loss}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["loss_ce"] += float(loss_dict["loss_ce"].detach())
        totals["loss_point"] += float(loss_dict["loss_point"].detach())
        steps += 1
        if smoke_batches is not None and steps >= smoke_batches:
            break

    if steps == 0:
        raise RuntimeError("training loader produced no batches")
    stats_tensor = torch.tensor(
        [totals["loss"], totals["loss_ce"], totals["loss_point"], steps],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_initialized():
        dist.all_reduce(stats_tensor)
    total_steps = float(stats_tensor[3].item())
    return {
        "loss": float(stats_tensor[0].item() / total_steps),
        "loss_ce": float(stats_tensor[1].item() / total_steps),
        "loss_point": float(stats_tensor[2].item() / total_steps),
    }


@torch.inference_mode()
def evaluate_at_half(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    targets_for_metrics: list[dict[str, object]] = []
    predictions_for_metrics: list[dict[str, object]] = []
    try:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            image_sizes = [
                tuple(int(value) for value in target["original_size"])
                for target in targets
            ]
            decoded = decode_predictions(
                outputs, threshold=FIXED_THRESHOLD, image_sizes=image_sizes
            )
            for target, prediction, (width, height) in zip(
                targets, decoded, image_sizes
            ):
                target_points = target["point"].detach().cpu().numpy()
                predicted_points = np.asarray(
                    prediction["points_xy"], dtype=np.float32
                ).reshape(-1, 2)
                targets_for_metrics.append(
                    {
                        "count": int(target["target_count"]),
                        "densest": densest_region(
                            grid_counts(target_points, width, height)
                        ),
                    }
                )
                predictions_for_metrics.append(
                    {
                        "count": int(prediction["count"]),
                        "densest": densest_region(
                            grid_counts(predicted_points, width, height)
                        ),
                    }
                )
    finally:
        if was_training:
            model.train()
    return aggregate_metrics(targets_for_metrics, predictions_for_metrics)


def _worker_init(seed: int, rank: int):
    def initialize(worker_id: int) -> None:
        worker_seed = seed + 1000 * rank + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return initialize


def build_training_loader(
    *,
    dataset: Dataset,
    sampler: Sampler,
    batch_size: int,
    num_workers: int,
    seed: int,
    rank: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_points,
        worker_init_fn=_worker_init(seed, rank),
        # Parent-side dataset.set_epoch() is visible to newly forked workers.
        persistent_workers=False,
    )


def ddp_options() -> dict[str, bool]:
    # The pinned head declares conv3/conv4 modules that its forward omits.
    return {"find_unused_parameters": True}


def _training_config(args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    paths = ExperimentPaths.from_project_root()
    manifests = {
        split: str(paths.manifest_dir / f"{split}.jsonl")
        for split in ("train", "val")
    }
    return {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "global_batch_size": args.batch_size_per_gpu * world_size,
        "world_size": world_size,
        "crop_size": 512,
        "head_lr": HEAD_LR,
        "backbone_lr": BACKBONE_LR,
        "weight_decay": WEIGHT_DECAY,
        "point_loss_coefficient": POINT_LOSS_COEFFICIENT,
        "max_grad_norm": MAX_GRAD_NORM,
        "eval_every": args.eval_every,
        "checkpoint_threshold": FIXED_THRESHOLD,
        "smoke_batches": args.smoke_batches,
        "manifests": manifests,
        "manifest_sha256": {
            split: sha256_file(Path(path)) for split, path in manifests.items()
        },
        "initialization": "ImageNet VGG16-BN only",
        "upstream_commit": (
            "5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455"
        ),
    }


def _optimizer(model: nn.Module) -> torch.optim.AdamW:
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in backbone_ids
    ]
    head = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not backbone or not head:
        raise RuntimeError("optimizer parameter partition is empty")
    return torch.optim.AdamW(
        [
            {"params": head, "lr": HEAD_LR},
            {"params": backbone, "lr": BACKBONE_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


def _load_resume(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_config: Mapping[str, Any],
    expected_artifact_sha256: str,
) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["artifact_manifest_sha256"] != expected_artifact_sha256:
        raise ValueError("resume artifact manifest hash mismatch")
    if checkpoint["config"] != dict(expected_config):
        raise ValueError("resume training config mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["epoch"]) + 1, dict(checkpoint["validation"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke-batches", type=int)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    validate_training_topology(world_size, args.batch_size_per_gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval interval must be positive")
    if args.smoke_batches is not None and args.smoke_batches <= 0:
        raise ValueError("smoke batch count must be positive")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
    seed_everything(args.seed, rank)
    started = time.time()
    try:
        paths = ExperimentPaths.from_project_root()
        training_config = _training_config(args, world_size)
        artifact_manifest_path = MODEL_ROOT / "DOWNLOAD_MANIFEST.json"
        artifact_sha256 = sha256_file(artifact_manifest_path)
        checkpoint_dir = MODEL_ROOT / "checkpoints"
        if args.smoke_batches is not None:
            checkpoint_dir = checkpoint_dir / "smoke"
        output_dir = paths.project_root / "Experiments" / "p2pnet_nereus" / "outputs"

        train_dataset = NereusPointDataset(
            paths.manifest_dir / "train.jsonl", training=True, seed=args.seed
        )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        train_loader = build_training_loader(
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            seed=args.seed,
            rank=rank,
        )

        model, criterion = build_p2pnet(training=True)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        criterion = criterion.to(device)
        optimizer = _optimizer(model)
        start_epoch = 1
        best_validation: dict[str, Any] = {}
        if args.resume is not None:
            start_epoch, best_validation = _load_resume(
                path=args.resume,
                model=model,
                optimizer=optimizer,
                expected_config=training_config,
                expected_artifact_sha256=artifact_sha256,
            )
        distributed_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            **ddp_options(),
        )

        validation_loader: DataLoader | None = None
        if rank == 0 and args.smoke_batches is None:
            validation_dataset = NereusPointDataset(
                paths.manifest_dir / "val.jsonl", training=False, seed=args.seed
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_points,
                persistent_workers=args.num_workers > 0,
            )

        if rank == 0:
            gpu_names = [
                torch.cuda.get_device_name(index) for index in range(world_size)
            ]
            _atomic_json_dump(
                {
                    "status": "running",
                    "started_unix": started,
                    "config": training_config,
                    "artifact_manifest": str(artifact_manifest_path),
                    "artifact_manifest_sha256": artifact_sha256,
                    "checkpoint_dir": str(checkpoint_dir),
                    "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu_names": gpu_names,
                },
                output_dir / "run_manifest.json",
            )

        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_dataset.set_epoch(epoch)
            epoch_stats = train_one_epoch(
                model=distributed_model,
                criterion=criterion,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                smoke_batches=args.smoke_batches,
            )
            should_evaluate = (
                args.smoke_batches is None and epoch % args.eval_every == 0
            )
            if rank == 0 and should_evaluate:
                assert validation_loader is not None
                validation = evaluate_at_half(model, validation_loader, device)
                validation["epoch"] = epoch
                if (
                    not best_validation
                    or checkpoint_rank(validation)
                    < checkpoint_rank(best_validation)
                ):
                    best_validation = validation
                    best_checkpoint = make_checkpoint(
                        model_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        epoch=epoch,
                        config=training_config,
                        validation=best_validation,
                        artifact_manifest_sha256=artifact_sha256,
                    )
                    _atomic_torch_save(
                        best_checkpoint, checkpoint_dir / "best.pth"
                    )

            if rank == 0:
                latest_checkpoint = make_checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    config=training_config,
                    validation=best_validation,
                    artifact_manifest_sha256=artifact_sha256,
                )
                _atomic_torch_save(
                    latest_checkpoint, checkpoint_dir / "latest.pth"
                )
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train": epoch_stats,
                            "best_validation": best_validation,
                            "elapsed_seconds": time.time() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            dist.barrier()

        if rank == 0:
            _atomic_json_dump(
                {
                    "status": "complete",
                    "started_unix": started,
                    "completed_unix": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "config": training_config,
                    "artifact_manifest": str(artifact_manifest_path),
                    "artifact_manifest_sha256": artifact_sha256,
                    "checkpoint_dir": str(checkpoint_dir),
                    "best_validation": best_validation,
                    "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "gpu_names": gpu_names,
                },
                output_dir / "run_manifest.json",
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
