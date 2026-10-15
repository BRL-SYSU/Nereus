# Nereus count-range evaluation

This directory contains the deterministic post-processing evaluation for the
NeurIPS 2026 rebuttal. It uses the exact 1,129-image Nereus Track A test
manifest and never bins by predicted count.

Run:

```bash
python \
  Experiments/rebuttal/count_range/count_range_eval.py
```

Run from the repository root. Outputs are written to
`Experiments/rebuttal/count_range/analysis/count_range/`. The evaluator inventories all
required open-VLM settings, preserves missing and invalid predictions in the
merged CSV, and scores only complete 1,129-image prediction sets. Prediction
files and intermediate canonical/QA artifacts are not included in this public
release. Inspect the input-path constants and `--help` before staging inputs.
Joint closed-VLM results are expected under `Experiments/closed_vlm_nereus_joint`;
they are not interchangeable with the separate-question baseline outputs.

Use `--strict-required-predictions` in CI when all required open-model
prediction files have been staged; it exits with status 2 while any required
set remains missing.
