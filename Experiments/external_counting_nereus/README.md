# Nereus Track A external counting evaluation

This module evaluates pretrained counting-specific models directly on the
1,129-image Nereus test split. It does not train on Nereus and does not use
the Nereus validation labels for threshold selection.

## Frozen protocol

- Setting: `FSC-147 pretrained Direct-ZS, text=fish`
- CountGD: official resize (short side 800, max side 1333), confidence
  threshold `0.23`, predicted box centers used for the 3x3 grid
- CLIP-Count: height 384, window 384, stride 128, sigmoid window blending,
  density scale factor 60
- Integer density count:
  `floor(max(0, predicted_density_sum / 60) + 0.5)`
- Count metrics: MAE, RMSE, Exact, relative Acc@5, relative Acc@10
- Spatial metric: 3x3 Densest Cell accuracy
- Camouflage F1, Track B, and Track C: `N/A`

Inference only reads
`Experiments/p2pnet_nereus/data/test_images.jsonl`. Evaluation reads the
label-bearing `test.jsonl` only after both prediction files are complete.

## Commands

Run from the repository root. Install each model's upstream dependencies and
stage source/checkpoints below `models/CountGD` and `models/CLIP-Count`, or pass
the explicit `--repo`, `--checkpoint`, and model-specific path options shown
by each inference module's `--help`. Source repositories and weights are not
included in this release. Only load checkpoints from trusted sources.

```bash
CUDA_VISIBLE_DEVICES=0 python -m \
  Experiments.external_counting_nereus.infer_countgd

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=3 python -m \
  Experiments.external_counting_nereus.infer_clipcount

python -m Experiments.external_counting_nereus.evaluate
```

The `env -u LD_LIBRARY_PATH` prefix is required on this host because its
global path exposes cuDNN 8.5 before the cuDNN 8.9.2 bundled with PyTorch.

## Result

| Backbone | Setting | MAE | RMSE | Exact | Acc@5 | Acc@10 | Densest Cell | Camouflage F1 | Track B | Track C |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountGD | FSC-147 pretrained Direct-ZS, text=fish | 32.33 | 80.16 | 6.64 | 14.35 | 25.69 | 67.67 | N/A | N/A | N/A |
| CLIP-Count | FSC-147 pretrained Direct-ZS, text=fish | 81.20 | 149.04 | 1.06 | 3.90 | 7.79 | 33.22 | N/A | N/A | N/A |

The CountGD checkpoint came from the first author's Hugging Face Space and
matches its LFS SHA-256. Google Drive was unreachable from this host, so the
CLIP-Count checkpoint came from a community Hugging Face mirror. It was
loaded in weights-only mode and all 841 parameters strictly matched the
official architecture (`missing=0`, `unexpected=0`). All 302 frozen CLIP
tensors exactly matched OpenAI's official ViT-B/16; bit identity of the
remaining trained tensors with the official Google Drive file could not be
independently proven.

FSC-147 and Nereus/IOCFish5K are distinct named datasets, but an image-level
duplicate audit could not be completed because FSC-147 source images were not
available during the run. The category `fish` is present in FSC-147/CountGD,
so these rows represent Nereus-domain direct zero-shot transfer, not
category-unseen zero-shot.
