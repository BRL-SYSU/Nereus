# P2PNet on Nereus Track A

This package reproduces a counting-specific P2PNet baseline on the fixed
Nereus train/validation/test split. It reports only the requested Track A
metrics: MAE, RMSE, Exact, Acc@5, Acc@10, and Densest Cell. Camouflage F1,
Track B, and Track C are rendered as `N/A`.

## Leakage boundary

- External initialization is limited to the official ImageNet VGG16-BN
  checkpoint recorded in
  `models/P2PNet/DOWNLOAD_MANIFEST.json`.
- No P2PNet crowd-counting checkpoint, IOCFormer checkpoint, or test-derived
  model state is loaded.
- Training uses `train.jsonl`; checkpoint selection and threshold calibration
  use `val.jsonl`.
- Test inference reads the physically separate `test_images.jsonl`, whose only
  fields are image ID, path, width, and height. Training does not open or hash
  either test manifest. Test points are consumed only later by `evaluate.py`,
  after predictions and the threshold are frozen.

## Fixed configuration

- Upstream commit:
  `5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455`
- Split sizes: train 3,945; validation 563; test 1,129
- Seed 42; 300 epochs; GPUs 4,5,6,7
- Global batch 8 (2 per GPU); deterministic 512 × 512 crops
- AdamW: head LR `1e-4`, backbone LR `1e-5`, weight decay `1e-4`
- Loss: `loss_ce + 0.0002 * loss_point`; gradient clipping `0.1`
- Checkpoint selection at threshold 0.50 every five epochs by
  `(MAE, RMSE, -Exact, epoch)`
- Final validation-only threshold grid: 0.05 through 0.95 in steps of 0.05,
  ranked by `(MAE, RMSE, -Exact, -threshold)`
- Densest-cell ties use row-major order.

The local compatibility patch is auditable at
`patches/p2pnet-local.patch`. Besides local weight/device compatibility and
modern torchvision version parsing, it corrects the pinned upstream
`loss_points` key typo to the actually emitted `loss_point`; otherwise point
regression would be silently unweighted.

## Reproduce

Run commands from the AquaCount-instruct project root:

```bash
bash Experiments/p2pnet_nereus/download_assets.sh

python \
  -m Experiments.p2pnet_nereus.prepare_data

python \
  -m unittest discover -s Experiments/p2pnet_nereus/tests -v
```

Two-GPU one-batch smoke on physical GPUs 0 and 3:

```bash
CUDA_VISIBLE_DEVICES=0,3 \
torchrun \
  --standalone --nproc_per_node=2 \
  -m Experiments.p2pnet_nereus.train \
  --smoke-batches 1 --epochs 1 --num-workers 0 \
  --batch-size-per-gpu 4
```

Full training:

```bash
CUDA_VISIBLE_DEVICES=0,3 \
torchrun \
  --standalone --nproc_per_node=2 \
  -m Experiments.p2pnet_nereus.train \
  --epochs 300 --batch-size-per-gpu 4 --eval-every 5 --seed 42
```

Resume only with the same immutable configuration and artifact hash:

```bash
CUDA_VISIBLE_DEVICES=0,3 \
torchrun \
  --standalone --nproc_per_node=2 \
  -m Experiments.p2pnet_nereus.train \
  --epochs 300 --batch-size-per-gpu 4 --eval-every 5 --seed 42 \
  --resume models/P2PNet/checkpoints/latest.pth
```

Freeze the threshold, predict test images on physical GPU 0, and render the
Table 3 row:

```bash
CUDA_VISIBLE_DEVICES=0 \
python \
  -m Experiments.p2pnet_nereus.calibrate

CUDA_VISIBLE_DEVICES=0 \
python \
  -m Experiments.p2pnet_nereus.infer --split test

python \
  -m Experiments.p2pnet_nereus.evaluate
```

Final artifacts are written to `outputs/`; the model checkpoints are under
`models/P2PNet/checkpoints/`.

## License

The pinned upstream P2PNet source retains its original license, which permits
academic research use and is not a general commercial-use grant.
