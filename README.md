# Nereus

Research homepage for **Nereus: A Large-Scale Underwater Dataset for Fine-Grained
Attribute Understanding and Grounded Counting Perception**.

NeurIPS2026 Datasets & Benchmarks (venue supplied by project owner).
Intended website: https://brl-sysu.github.io/Nereus/
Dataset: https://huggingface.co/datasets/Nereusdata/Nereus

## Local Development

Requires Node22.12+ (Node24 used for verification).

```bash
cd website
npm ci
npm run dev
```

Open the printed local URL followed by `/Nereus/`.

```bash
npm test
npm run check
npm run build
npm run verify:build
npm run preview -- --host :: --port 4321 --background
```

No backend, API key, GPU, training run or original dataset is required to run the
website. Curated assets and annotations are included. Production publication is
manual and requires owner review; see docs/RELEASE_CHECKLIST.md.

## Repository

```text
.github/workflows/    Manual GitHub Pages deployment
website/
  public/assets/     Curated preview exports and per-source attribution
  src/components/    Scientific sections and explorers
  src/content/       Manuscript abstract (MDX)
  src/data/          Verified benchmark values and curated records
  src/lib/           Native TypeScript interactions
  src/pages/         Static homepage
  src/styles/        Responsive editorial design
  tests/             Data and browser checks
docs/                Scientific audit, provenance, progress and maintenance
scripts/             Bounded asset export and build-safety checks
```

Future evaluation code can be added independently without changing website/.
The anonymous source manuscript is intentionally excluded. Some research release
materials remain pending. On 2026-10-14 the project owner confirmed permission
to publicly show the three attribute examples and their interactive presentation.
Original image licenses and attribution remain unchanged; this confirmation is
not a blanket permission for downstream adaptations.
See docs/ATTRIBUTE_PREVIEW.md for the latest implementation and
permission boundaries, and docs/HOMEPAGE_PROGRESS.md for earlier milestones.

## Publication and Task Films

The author header follows the owner-confirmed author order, with equal
contribution for Jingtao Deng and Zihao Jin and corresponding authorship for
Dan Zeng. Author affiliations and resource URLs live in
`website/src/data/publication.json`. Paper and arXiv remain pending; the anonymous
PDF is not included. BibTeX is explicitly a provisional manuscript citation.

Three short counting films illustrate region counts, camouflage-state counts,
and density/habitat evidence using genuine source images and annotations.
They are annotated-data demonstrations, not model predictions. Rendering and
media verification are documented in `docs/COUNTING_VIDEOS.md`; existing videos
are served directly and do not require video tools to build the website.

## Rights

IOCfish5K images/points: CC BY-NC4.0, Guolei Sun et al. (CVPR2023). Nereus additions
retain their source terms. Website code licensing is pending owner selection;
no blanket software license is asserted. See NOTICE and docs/ASSET_MANIFEST.md.
