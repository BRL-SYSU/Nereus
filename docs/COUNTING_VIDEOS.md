# Counting Annotation Films

Three silent, downloadable annotation visualizations for the existing Nereus player. No site components, styles, dependency files, input images, or datasets are changed by this pipeline.

## Delivered Media

All clips are 1280 x 720, 30 fps, 14 seconds, H.264 with `yuv420p` pixels, no audio stream, and fast-start MP4 metadata. Posters are 1280 x 720 WebP images extracted from the encoded films. Captions are English WebVTT.

| ID | Sample | MP4 bytes | MiB | Story |
| --- | --- | ---: | ---: | --- |
| `region` | `0000` | 1,213,657 | 1.16 | 90 points; densest top-right cell 31; open water 82 versus reef rock 5 |
| `camouflage` | `0001` | 1,569,863 | 1.50 | 134 clear, 130 ambiguous, 6 highly camouflaged; true-coordinate detail crop |
| `density` | `0001`, `0003` | See manifest | See manifest | Two localized hotspots: visual evidence, environment/depth tags, then ecological interpretation |

Each ID has exactly these files under `website/public/assets/videos/`:

- `{id}.mp4`
- `{id}-poster.webp`
- `{id}.vtt`

`manifest.json` is an array ordered `region`, `camouflage`, `density`. Paths are relative public asset paths, without a leading slash, so the parent site can apply its deployment base path.

## Manifest Contract

| Fields | Meaning |
| --- | --- |
| `id`, `title`, `description` | Stable player identifier and accessible description |
| `src`, `poster`, `captions` | `assets/videos/{id}.mp4`, `assets/videos/{id}-poster.webp`, `assets/videos/{id}.vtt` |
| `duration`, `width`, `height`, `fps`, `mimeType`, `bytes` | 14, 1280, 720, 30, `video/mp4`, actual encoded byte size |
| `sampleIds`, `annotationFacts`, `annotationDerived` | Curated sample provenance, derived numeric facts, `true` |
| `captionsLanguage` | `en` |
| `license`, `source`, `credit`, `annotationSource` | Dataset attribution and original annotation reference |
| `sourceData`, `sourceDataSha256` | Repository-relative curated sample JSON and its SHA-256 |
| `sourceAssets` | Array of `{src, sha256}` for original source photographs and heatmap |

Exact `annotationFacts` objects:

```json
{
  "region": {"total":90,"densestCell":"top-right","densestCount":31,"openWater":82,"reefRock":5,"difference":77},
  "camouflage": {"total":270,"clear":134,"ambiguous":130,"highlyCamouflaged":6},
  "density": {"totalPoints":270,"hotspots":10,"openWaterHotspots":9,"coralReefHotspots":1}
}
```

## Annotation Fidelity

`scripts/video/story.mjs` reads the curated records in `website/src/data/samples.json`. Totals, grid cells, visibility categories, region membership, and hotspot counts are computed from individual point or hotspot records. The same frame state feeds point reveals and counters; counters do not run independently of the visible annotation sequence.

The renderer uses source-derived photographs and heatmaps for 0000, 0001 and 0003. Camera transforms, annotation rings, and heatmap crossfades are overlays on the supplied assets. There is no generated imagery, new model inference, image retouching, or newly inferred ecological cause. Scaling and final video compression necessarily resample source pixels. Original dataset files remain untouched.

The camouflage detail settles on four of the six highly camouflaged annotations, at original coordinates. Region comparisons use annotation membership, not a new geometric reassignment. Other semantic labels are omitted from the 82-versus-5 comparison. Density hotspot numbers are not fish counts. The habitat descriptions are supplied annotation labels and visual context, not evidence of habitat preference, shelter benefits, or ecological causes.

The current density film replaces the earlier 0001/hotspot10 example because its
crop did not clearly show coral. It compares 0001/hotspot1 at (1327,215) and
0003/hotspot1 at (886,446), seven seconds each. Each scene localizes the density
response, zooms into RGB evidence, reveals environment and relative-depth tags,
then holds an annotation interpretation. Source `evidence`, `ecological_hint`,
`depth_summary` and `causal_chain` are preserved in `density-evidence.json`.
Open swimming space may support schooling; coral structure may offer shelter
and food near an aggregation. These are source annotation hypotheses, not proof
of causes, feeding or habitat preference. Relative depth is not metric depth.
Dashed outlines are source context patches, not fish boxes; the almost-full-frame
SAM box in 0003 is deliberately not portrayed as precise instance localization.
Display crops are presentation choices. The legacy density `annotationFacts`
counts describe 0001 only, explicitly scoped in the manifest. `reasoningCases`
records both scenes and source annotation hashes. Only density is rerendered;
region and camouflage media remain unchanged.

Credit: Guolei Sun et al., IOCfish5K (CVPR 2023), with Nereus annotations; CC BY-NC 4.0, as recorded in the curated sample metadata. Preserve this attribution and the noncommercial license in downstream use.

## Reproduce

Run commands from the repository root. Existing `website/node_modules` supplies Playwright, Sharp, and the Manrope font. No global packages or site dependency files are modified. Use Node compatible with the website's declared engine.

Set `FFMPEG` and `FFPROBE` to installed executable paths if they are not on `PATH`. Alternatively place local binaries at `scripts/video/.local/ffmpeg` and `scripts/video/.local/ffprobe`. FFmpeg must support PNG decoding plus `libx264` or `libopenh264` encoding. Chromium defaults to `website/.cache/ms-playwright`; override `PLAYWRIGHT_BROWSERS_PATH` for another existing installation.

```bash
node scripts/export-density-reasoning.mjs "$COUNTING_ROOT" "$EXPERIMENTS_ROOT"
node scripts/video/render.mjs --preview
node scripts/video/render.mjs
node scripts/video/verify.mjs
node --test website/tests/videos.test.mjs
npm --prefix website test
```

`--preview` renders eight lossless preview frames per clip without changing public media. `--only=region`, `--only=camouflage`, or `--only=density` restricts rendering. Encoding uses lossless PNG frame transport, then H.264. Posters are extracted from the encoded MP4, not from a separate mockup. A loopback-only asset server serves a fixed allowlist and is closed when rendering finishes. No content is uploaded.

After any rerender, rerun strict verification before committing artifacts. `scripts/video/.gitignore` excludes `.cache/` and `.local/`; media, captions, posters, the manifest, and `scripts/video/media-verification.json` are intended deliverables.

## Verification

The test suite was first run with missing story/media assertions and failed, then the implementation was added. The final tests check numeric facts, annotation identity and all 420 frame states per film, exact media paths, size limits, source integrity, captions, and video evidence.

`node scripts/video/verify.mjs` is strict: both FFmpeg and FFprobe are required. It reruns codec/duration/dimension/frame-count checks; parses fast-start MP4 atoms; decodes every frame; checks source hashes; extracts eight actual video frames per clip; checks nonblank image regions; and tests decoding, seeking, and advancing playback in Chromium at 1280 x 720 and 390 x 844 viewports. It records fresh evidence bound to the MP4/poster/caption SHA-256 hashes.

Regular `npm test` does not require FFprobe on CI. When available it probes directly; otherwise it validates the hash-bound recorded probe results. Changed media cannot silently reuse stale evidence. Strict verification is a separate local release step, not an optional fallback.

Inspection outputs, all in the ignored `scripts/video/.cache/` directory:

- `region-contact.webp`, `camouflage-contact.webp`, `density-contact.webp`: decoded frames at 0.5, 2, 4, 6, 8, 10, 11.5, and 13.5 seconds, left-to-right then top-to-bottom.
- `{id}-decoded-{time}.png`: full-size extracted frames.
- `{id}-browser-1280.png`, `{id}-browser-390.png`: actual browser video screenshots.
- `layout-report.json`: renderer exceptions and out-of-bounds text checks.

Contact sheets and browser frames support visual review. Fine source notes and
reasoning text benefit from fullscreen, captions, or the player's transcript on
narrow screens. Density now emphasizes the evidence chain rather than large
hotspot-count numerals.
