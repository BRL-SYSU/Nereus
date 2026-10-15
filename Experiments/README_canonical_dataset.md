# Canonical Dataset Manifest

Use `build_canonical_dataset.py` to organize AquaCount-Instruct into one record per `image_id`:

```text
image_id
├── RGB underwater image
├── final annotation JSON
├── QA / instruction conversations
└── density heatmap / density overlay / density array
```

## Run

```bash
cd .
python3 Experiments/build_canonical_dataset.py --make-splits
```

Default output directory:

```text
./Experiments/canonical_dataset
```

## Outputs

```text
canonical_records.jsonl   all records found from test_dataset/final_annotations
complete_records.jsonl    records with RGB + final annotation + density + Task1/2/3 QA
missing_assets.jsonl      records with at least one missing asset
summary.json              coverage statistics
splits/train_ids.txt      image-level split from complete_records
splits/val_ids.txt
splits/test_ids.txt
splits/splits.json
```

## Optional Full Payloads

The default manifest stores paths and compact metadata only. To embed QA payloads:

```bash
python3 Experiments/build_canonical_dataset.py \
  --output-dir Experiments/canonical_dataset_with_qa \
  --include-qa
```

To also embed full final annotation JSON payloads:

```bash
python3 Experiments/build_canonical_dataset.py \
  --output-dir Experiments/canonical_dataset_full_payload \
  --include-qa \
  --include-final-annotation
```

Use full payload mode only when needed, because the JSONL files become much larger.

## Folder Layout Version

For a clearer human-readable dataset layout, materialize one folder per `image_id`:

```bash
python3 Experiments/materialize_canonical_folders.py \
  --output-dir Experiments/canonical_dataset_folders \
  --overwrite
```

Default input is:

```text
Experiments/canonical_dataset/complete_records.jsonl
```

The script uses symlinks by default, so it does not duplicate the original image
and density files.

Each sample folder looks like:

```text
Experiments/canonical_dataset_folders/0000/
├── image/0000.jpg
├── final_annotation/0000.json
├── qa/task1_region.jsonl
├── qa/task2_camouflage.jsonl
├── qa/task3_density_depth_qa_pairs.json
├── qa/task3_density_depth_generation_context.json
├── qa/0000_stitched.png
├── density/0000_density_heatmap.png
├── density/0000_density_overlay.png
├── density/0000_density.png
├── density/0000_density.npy
└── record.json
```

The top-level `manifest.jsonl` lists all materialized folders and local paths.
Use the folder tree for inspection/release, and use `manifest.jsonl` as the
stable index for generating model-specific SFT JSONL files.
