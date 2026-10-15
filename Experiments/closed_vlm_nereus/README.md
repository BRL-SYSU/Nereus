# Closed VLM Nereus Track A

This module evaluates `gemini-3.1-pro-preview` and `gpt-5` on the frozen
Nereus Table 3 Track A test split.

- Test images: 1,129
- Inputs exposed during inference: RGB image plus frozen question only
- Questions per image: total count and 3x3 grid/densest cell
- Requests per model: 2,258
- Setting: prompted zero-shot (`P-ZS`), no Nereus training or calibration
- Reported: MAE, RMSE, Exact, Acc@5, Acc@10, Densest Cell
- N/A: Camouflage F1, Track B, Track C

Original images are sent as Base64 data URIs without resizing or
recompression. Every task response is cached atomically, so interrupted runs
resume without repeating completed requests.

Dry-run:

```bash
cd .
python -m \
  Experiments.closed_vlm_nereus.run \
  --model gemini-3.1-pro-preview --limit 3 --dry-run
```

Progress:

```bash
python -m \
  Experiments.closed_vlm_nereus.status --model gpt-5
```

For live requests, place the API key in a local file that contains either the
raw token or one line in the form `APIYI_API_KEY=...`. Restrict it to the
current user and pass only the file path:

```bash
chmod 600 .secrets/apiyi_api_key
python -m \
  Experiments.closed_vlm_nereus.run \
  --model gemini-3.1-pro-preview --limit 3 --workers 1 \
  --api-key-file .secrets/apiyi_api_key
```

The token is read into memory only. It is never written to response caches,
prediction JSONL files, progress output, or reports.
