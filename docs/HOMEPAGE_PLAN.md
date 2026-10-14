# Nereus Research Homepage Implementation Plan

**Goal:** Accurate, polished static research homepage at
`https://brl-sysu.github.io/Nereus/`, locally verified, not published.

**Architecture:** Astro, MDX abstract, structured JSON science data, TypeScript
interactions, Tailwind and editorial CSS. Only curated licensed assets.

**Spec:** Complete user homepage brief supplied 2026-10-13. The user explicitly
authorizes autonomous planning, implementation and local verification.

## Decisions

- Compared plain HTML, React SPA and Astro: select Astro for static output and
  small client-side interactions. Original design; no template code copied.
- White editorial sections, deep ink, teal/coral accents, real IOCfish imagery.
- Source datasets immutable, explicit small sample list, no inference/training.
- Empty repo verified, remote BRL-SYSU/Nereus; branch feat/research-homepage.
- No commits, push or publication before owner review. GitHub Pages manual only.
- Sites local workflow applies; no external Site registration or publication.
- Never publish anonymous manuscript, unknown authors or unlicensed imagery.
- Statistics retain manuscript version and citations, conflicts documented.

## Review Focus

1. Image coordinates preserve source dimensions and align after scaling.
2. Missing data produces honest fallbacks, never invented labels or masks.
3. All resources work under /Nereus/ on GitHub Pages.
4. Mobile and keyboard workflows remain usable without layout overflow.
5. Build excludes manuscript, credentials and private server paths.

## Tasks

- [x] 0. Verify repository, source boundaries, remote; create development branch.
- [x] 1. Paper and provenance audit; scientific/asset documentation.
- [x] 2. Data tests first (red), explicit-sample exporter and content (green).
- [x] 3. Astro sections and TypeScript explorer; six modes and sample switching.
- [x] 4. Benchmark tables, protocols, ablations and faithful interpretation.
- [x] 5. Typecheck, tests, build, production preview, desktop/mobile browser QA.
- [x] 6. Manual deployment workflow, safety audit, maintenance and progress docs.

## Rulings

- Local attribute component contains documentation only. Use paper taxonomy,
  not fabricated sample images/boxes. Record missing payload/permissions.
- Semantic boxes are not masks. Only display verified geometry with correct labels.
- Source aliases go in public docs; server mappings stay in ignored local docs.
- Completed implementation uses declared fallbacks, not a claim that blocked
  research materials are available. See HOMEPAGE_PROGRESS.md for partial scope.
- Independent review found an asynchronous image/annotation swap risk. Fixed
  with decoded-image readiness, stale request guards, hidden loading evidence
  and retry; slow-load and failed-load regression tests now pass.
