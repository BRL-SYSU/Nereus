# Website Asset Manifest

Updated: 2026-10-14. Only four explicitly selected counting samples were
exported. No recursive dataset import. All source directories are immutable.
Source-root aliases keep private server locations out of the repository and HTML.
The exact local root mappings are in ignored `LOCAL_SOURCE_PATHS.md`.

## Roots

- `COUNTING`: owner-supplied grounded_counting project.
- `EXPERIMENTS`: AquaCount-instruct/Experiments project.
- `RELEASE`: Nereus_dataset release snapshot.
- `MANUSCRIPT`: supplied anonymous Nereus.pdf, internal reference only.

## Curated Records

| ID | Scientific purpose | RGB source | Annotation | Density source | Export |
| --- | --- | --- | --- | --- | --- |
| 0000 | Reef/open-water distribution, 90 points, 4 hotspots | COUNTING/images/0000.jpg | COUNTING/annotations/final_annotations/0000.json | EXPERIMENTS/canonical_dataset_folders/0000/density/0000_density_heatmap.png | website/public/assets/counting/0000-* |
| 0001 | Dense school, 270 points, 10 hotspots, QA inconsistency | COUNTING/images/0001.jpg | COUNTING/annotations/final_annotations/0001.json | EXPERIMENTS/canonical_dataset_folders/0001/density/0001_density_heatmap.png | website/public/assets/counting/0001-* |
| 0003 | Hotspot 1: fish near branching coral; annotation-based environmental interpretation | COUNTING/images/0003.jpg | COUNTING/annotations/final_annotations/0003.json | EXPERIMENTS/canonical_dataset_folders/0003/density/0003_density_heatmap.png | website/public/assets/counting/0003-* |
| 5044 | Occlusion, 20 points, 1 hotspot | COUNTING/images/5044.jpg | COUNTING/annotations/final_annotations/5044.json | EXPERIMENTS/canonical_dataset_folders/5044/density/5044_density_heatmap.png | website/public/assets/counting/5044-* |

Each RGB source and density source is 1920 x 1080. RGB exports: 640/1280/1920
pixels wide, WebP quality82; density export:1280 wide, WebP quality90. No color
enhancement, retouching, added organisms or altered counts. EXIF/private metadata
not retained. Explorer geometry remains in original 1920x1080 coordinates and
scales with an identically sized SVG viewport. Hero may responsively crop its
decorative display, but scientific explorer images are never cropped. Website
favicon is a 64x64 center crop of the exported 0000 RGB, PNG.

All four source images/original points: IOCfish5K, Guolei Sun et al., CVPR2023,
CC BY-NC4.0. Nereus enrichments: release default CC BY-NC4.0, subject to upstream
rights. Noncommercial academic homepage reuse follows upstream terms with credit,
license link and transformation disclosure. Publication remains owner-controlled.
The counting exports have recorded rights. On 2026-10-14 the owner confirmed
public-display permission for the three attribute samples and current animation.
Their original ND licenses and attribution remain unchanged (see below).

## Annotations and QA

- `samples.json` exports ONLY whitelisted fields: sample ID/dimensions, points,
  region memberships/counts/boxes, grid, visibility, hotspot locations/SAM boxes
  and semantic evidence, selected question/answer text, credit and source alias.
- No raw model responses, original filenames containing names, absolute paths,
  source split metadata, generation logs or private locations are exported.
- QA source: `COUNTING/qa/task1_region/{id}.jsonl` (JSON object despite extension).
  0000 pairs1/3/4; 0001 pair4; 5044 pairs1/4 are shown verbatim.
- Known bad 0001 pair3 excerpt appears only in the annotation-failure disclosure.
- Scene boxes are labeled boxes, not exact masks. Semantic counts can overlap.
- Hotspot SAM bounding boxes are stored source geometry, not new segmentation.
- Relevant paper: Section4.2 and Figures8-10. Samples0000/5044 are curated release
  examples, not asserted to be published figure panels or benchmark predictions.

## Attribute and Diagram Assets

`website/src/data/attributes.json` retains the earlier Figure6 text excerpts and
Figure7 field names for reference. The active explorer now uses
`website/src/data/attribute-samples.json`: three real test records and full-frame
WebP images with original target boxes. The public/assets/attributes manifest
records source URLs, revision, downloaded hashes and ND image licenses.
Five display categories organize source fields; no missing values are inferred.
The owner confirmed public-display permission for the existing overlays/zoom
on 2026-10-14; this does not grant additional downstream adaptation rights.
See ATTRIBUTE_PREVIEW.md for transformations and limitations.
The original HTML/CSS pipeline and result graphics retain manuscript values.
At the authors' request, Figures 2 and 4 are additionally rendered directly from
the supplied manuscript and displayed immediately after Abstract, in that order.
`assets/paper/manifest.json` records source PDF hash, pages and crop rectangles.
The source PDF is not distributed. Full figures and original captions are
preserved; only page margins and adjacent text are removed. WebP variants are
1440px and approximately 2400px wide, with an accessible enlarged-image dialog.

## Dependencies and Rights

Fonts: Manrope and DM Serif Display, bundled through Fontsource, SIL OFL1.1.
Icons: @lucide/astro, ISC. Copies of notices accompany public assets.
No academic-project-astro-template implementation was copied.

## Local Review Artifacts (Ignored)

`.local/contact-sheet.html` compares original RGB and density exports.
Browser screenshots and QA reports remain in `.local/` or ignored test outputs;
they are not production assets. `scripts/export-assets.mjs` regenerates only the
three selected examples after receiving source roots as command arguments.
# Counting Film Addendum (2026-10-14)

`website/public/assets/videos/` contains three 14-second annotation visualizations
derived exclusively from the existing curated IOCfish5K/Nereus assets. Region
counting uses0000; camouflage counting uses0001; density-hotspot reasoning now
compares hotspot1 of0001 and0003.
MP4s, WebP posters and English VTT tracks preserve source attribution and
CC BY-NC4.0 terms. They add camera motion, counters and annotation overlays, not
model predictions or new biological observations. Source hashes, numeric facts,
rendering instructions and media verification are in COUNTING_VIDEOS.md and the
video manifest. Figure exports are limited to the two author-requested diagrams.

Density evidence excerpts in `assets/videos/density-evidence.json` preserve
semantic evidence, ecological hints, relative-depth labels and source causal
chains for the two selected hotspots. Display lines are English summaries;
interpretations are hypotheses, not established ecological causes. No raw model
responses or absolute server paths are included. The dataset itself is unchanged.
