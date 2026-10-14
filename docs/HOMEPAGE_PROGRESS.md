# Nereus Homepage Delivery Report

Updated: 2026-10-14 (Asia/Shanghai). Initial delivery: 2026-10-13.
Status: **Implemented and locally verified; owner review required before publication.**
Branch: `feat/research-homepage`. No commits, push, PR or remote deployment made.
No original dataset, annotations, experiments or manuscript was edited.

## Completed

- Centered six-author header and affiliations confirmed by the owner: Jingtao
  Deng and Zihao Jin contributed equally; Dan Zeng is corresponding author.
  Paper/arXiv remain pending. Dataset, code and provisional BibTeX links work;
  citation copying uses the confirmed author order, without an invented arXiv ID.
- Expanded Abstract immediately after the author/resource header, before dataset
  overview and statistics. The original anonymous PDF remains excluded.
- Three 14-second, 1280x720 H.264 annotation films: region-aware counting,
  camouflage-state counting and density-hotspot reasoning. Real sample0000/0001
  images, original point coordinates and provided density evidence; no model
  inference or invented imagery. Posters, English captions, transcripts,
  replay/fullscreen/download controls, keyboard task tabs, reduced-motion and
  offscreen/background pause handling are included.
- Original Astro7/MDX/TypeScript/Tailwind static homepage; self-hosted fonts,
  mobile navigation, visible Nereus identity, actual underwater-image hero.
- Exact title, owner-confirmed venue, verbatim manuscript abstract, motivation,
  two task families, sourced manuscript-version statistics and source distribution.
- Five-category Attribute Explorer with three real Figure6 text excerpts,
  organism switching, keyboard-operable category tabs and honest absent fields.
- Grounded Counting Explorer with0000/0001/5044, original RGB, ground-truth
  points, semantic boxes/grid, visibility, precomputed density, hotspot SAM boxes,
  recorded local scene evidence, genuine QA and downloadable curated annotations.
- Coordinate/count validation, image-loading and retry states, stale-request
  protection, full-width images with invariant16:9 overlay alignment.
- Two original pipeline diagrams and explicit mixed LMM/human quality process.
- Complete displayed overall Table2 rows; all18 TrackA and11 TrackB rows;
  four transfer rows and five selected supervision/degradation ablation rows.
  Scientific protocols, metric directions, caveats and sources are visible.
- Curated qualitative annotation cases, real0001 QA grounding failure,
  limitations, actual dataset/code links and honest pending resource states.
- IOCfish CC BY-NC4.0 attribution, source notices/licenses, font/icon notices.
- Source manifest, full scientific audit, maintenance guide and review checklist.
- Manual-only GitHub Pages workflow configured for `/Nereus/`. It requires main,
  publication_approved=true and the github-pages deployment environment.

## Partial by Design

| Requested capability | Delivered | Reason for fallback |
| --- | --- | --- |
| Attribute RGB + target box + structured records | Interactive published text excerpts/taxonomy | Local component contains only README/LICENSE_INFO; image rights and aligned payload absent |
| Semantic masks | Same-sample recorded semantic bounding boxes + spatial grid | No verified matching semantic pixel masks exported; boxes are never mislabeled masks |
| Published figure panels | Original scientific diagrams and genuine counting assets | Anonymous PDF not authorized; mixed-source figure image permissions not verified |
| Prediction versus ground truth qualitative comparison | Honest annotation examples and annotation failure | No verified published model-prediction pair selected; no inference run |
| Public paper/arXiv/final citation/contact | Confirmed authors and affiliations, provisional manuscript BibTeX, pending paper/arXiv buttons | Owner explicitly deferred the public PDF; arXiv and final publication metadata are not yet supplied |

These gaps are not hidden behind fabricated examples. The site is usable locally
and represents both task families, but research release approval remains separate.

## Selected Source Assets

Original images: `COUNTING/images/{0000,0001,5044}.jpg`.
Annotations: `COUNTING/annotations/final_annotations/{0000,0001,5044}.json`.
Density: `EXPERIMENTS/canonical_dataset_folders/{id}/density/{id}_density_heatmap.png`.
QA: `COUNTING/qa/task1_region/{id}.jsonl`.
Counts90/270/20; hotspots4/10/1. Source dimensions1920x1080.
Full paths are in ignored LOCAL_SOURCE_PATHS.md for local review. Public ASSET_MANIFEST
records provenance, transformations, licenses and exact exported asset patterns.

No generated underwater imagery, fabricated ground truth, changed measurements,
GPU work or new model inference was used. Only three selected source records
were processed for export; the entire dataset was not enumerated or hashed.

## Scientific Results Shown

Headlines:88,555 images,2.68M QA,136 attribute dimensions,2,690 species, with
manuscript references. Secondary/contested totals remain explicitly versioned.
Table2 Nereus overall accuracy: LLaVA48.91%, InternVL48.10%, Qwen41.55%.
TrackA InternVL QAS MAE25.99 vs Full30.48. TrackB is supplied GT-derived density
interpretation, not autonomous RGB-only prediction. Source tables are transcribed,
not rerun, and no statistical-significance or universal-superiority claim is made.

## Verification Evidence

- `npm test`: **11/11 passed**, including author order/contributions, all displayed benchmark row fixtures,
  point-derived grid membership, box/hotspot bounds, image dimensions, count and
  visibility totals, public/bundled annotation equality and source references.
  Video tests additionally validate all420 frame states per clip, annotation
  identity, hotspot wording, source hashes and hash-bound media verification.
- `npm run check`: **0 errors, 0 warnings, 0 hints** across27 files.
- `npm run build`: **passed**, one static homepage under the configured base.
- `npm run verify:build`: **passed**;43 build files,5,889,356 bytes (~5.62MiB),
  35 internal resource references,65 IDs/anchors; no restricted artifacts/private paths.
- Playwright: **28/28 passed** across1440x1000 and390x844. Covers resource loading,
  six modes, three samples, point counts, layer alignment, attribute keyboard tabs,
  model filter, clipboard, mobile navigation, slow loading and failure/retry.
  New checks cover author order, pending paper/arXiv, expanded abstract position,
  BibTeX copying, all three videos playing/seeking/replaying, keyboard film tabs,
  automatic playback, single-video playback, load-failure recovery, reduced
  motion, and quiet cancellation of intentionally paused loading videos.
- Axe WCAG2A/AA and2.1AA checks: no detected violations in tested initial states.
  This is automated coverage, not a complete accessibility certification.
- Visual checks at320,390,768,1440,1920px widths: no document horizontal overflow
  or browser script errors. Screenshots and pixel-variation checks confirm nonblank
  rendering. All three films also pass decoded-pixel and changing-frame checks
  in desktop and mobile browsers. Reviewed header, author block, counting films,
  attribute, point, region and density views.
- Independent review checked manuscript table/excerpt agreement and highlighted
  loading and test-coverage issues; both were addressed and retested. The new
  player review found intentional-pause warnings; a failing real-browser test
  reproduced the issue before its fix. The mobile film-tab contrast failure was
  also fixed and the full suite rerun. Hotspot10 is described as a source
  annotation label rather than independently asserting visible coral texture.

Local evidence: `.local/visual-review.json`, `.local/*-first.png`, `*-full.png`,
`*-counting.png`, `*-attributes.png`, `*-regions.png`, `*-visibility.png`,
`*-density.png`, `*-hotspots.png`, and contact-sheet.html. These stay ignored.
New film screenshots are `*-film-{region,camouflage,density}.png`. Decoded video
contact sheets and strict media evidence are documented in COUNTING_VIDEOS.md.
Element screenshots hide the fixed navigation only during screenshot capture;
full-page and first-viewport screenshots retain the real navigation.

## Repository Layout

```text
Nereus/
  .github/workflows/deploy-pages.yml
  website/
    public/assets/{counting,licenses,videos}/
    src/{components,content,data,layouts,lib,pages,styles}/
    tests/{content.test.mjs,benchmarks.test.mjs,publication.test.mjs,videos.test.mjs,browser/}
    astro.config.mjs
    package.json
    package-lock.json
  scripts/{export-assets,verify-build,capture-review}.mjs
  scripts/video/{story,renderer,render,runtime,verify}.mjs
  docs/{SCIENTIFIC_CONTENT_AUDIT,ASSET_MANIFEST,HOMEPAGE_PLAN,
        HOMEPAGE_PROGRESS,HOMEPAGE_MAINTENANCE,RELEASE_CHECKLIST}.md
  README.md
  NOTICE
  .gitignore
```

## Preview and Publication

Production preview remains running on server port4321 at `/Nereus/`.
Local-machine URL: `http://localhost:4321/Nereus/`.
Use the research server's reachable address or SSH port forwarding from a laptop.
To restart: from website/, `npm run build` then
`npm run preview -- --host :: --port 4321 --background`.

Production has NOT been deployed. After owner review, approved commit/push/merge,
enable GitHub Pages via Actions and manually run the prepared workflow on main.
Follow HOMEPAGE_MAINTENANCE.md and RELEASE_CHECKLIST.md. GitHub availability of the
empty repository could not be independently fetched by the web tool; its URL
matches the local origin and explicit owner instruction. Hugging Face dataset
card was reachable, and its licenses/split namespaces were inspected.

## Outstanding Owner Decisions

Authorized public paper/arXiv/final citation/contact; point/object aggregate conflicts;
Table3/11 baseline differences; new-attribute versus release license scope;
permission-cleared attribute examples; model prediction examples; website code
license; final content approval and deliberate publication. These do not prevent
local use of the delivered homepage.
