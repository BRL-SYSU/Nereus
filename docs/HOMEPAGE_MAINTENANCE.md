# Homepage Maintenance

## Content Ownership

- `website/src/data/publication.json`: owner-confirmed author order, affiliations,
  equal-contribution and corresponding-author roles, and resource URLs.
  The order is Jingtao Deng, Zihao Jin, Yuehe Chen, Feixiang He, Shuiwang Li,
  Dan Zeng. The first two authors contributed equally; Dan Zeng corresponds.
  Paper/arXiv URLs intentionally remain null. Add only an approved public PDF or
  URL; do not copy the anonymous manuscript into public assets. Update the
  pending-link assertions in publication tests when these resources are released.
- `website/src/lib/citation.ts`: provisional manuscript BibTeX from confirmed
  authors. Replace the provisional note/type with verified publication metadata
  when available; do not invent a DOI or arXiv identifier.
- `website/src/data/science.json`: manuscript-version statistics and benchmark
  tables. Maintain sources and change tests alongside independently verified data.
- `website/src/content/Abstract.mdx`: verbatim supplied abstract. Change only when
  authorized camera-ready copy is available; retain version discrepancy notes.
- `website/src/data/attributes.json`: Figure6 text excerpts and Figure7 field names.
- `website/src/data/attribute-samples.json`: active real test examples; regenerate
  with `scripts/export-attribute-samples.mjs`. See ATTRIBUTE_PREVIEW.md before
  public use; image licenses contain ND restrictions.
  These are NOT full release records. Empty categories are intentional.
- `website/src/data/samples.json` and public counterpart: generated whitelist of
  three explicit counting samples. Never manually relabel points or correct QA.
- Components separate scientific copy, visualization, and resource states.
- `src/lib/counting.ts` coordinates image decoding and annotation swaps. Keep
  loading and stale-request protections when adding samples.

## Local Preview

From repository root:

```bash
cd website
npm ci
npm run dev -- --port 4321
```

Browse `http://localhost:4321/Nereus/`. For a remote research server, use its
reachable host with the same port or SSH port forwarding. Do not assume your
laptop's localhost points to the server.

Production preview:

```bash
npm run build
npm run preview -- --host :: --port 4321 --background
```

Astro7 may retain the preview as a background service. Inspect with
`npx astro preview status`; stop the owned preview with `npx astro preview stop`.
Do not kill unrelated processes. Select a free port if4321 is already occupied.

## Verification

```bash
npm test
npm run check
npm run build
npm run verify:build
PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright npx playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=.cache/ms-playwright npm run test:browser
```

Start the production preview before browser tests. They use port4321 and test
desktop1440x1000/mobile390x844. Review screenshots at additional narrow320,
tablet768 and wide1920 widths. Test keyboard tabs, menu, data filters, copying,
loading failure/retry and fast sample changes. The automated axe check is not a
complete manual accessibility certification.

The counting film player is `website/src/components/CountingFilms.astro`.
It serves three pre-rendered MP4s with captions, transcripts, replay, download,
and native playback controls. Only one film may play at a time; leaving the
section or hiding the browser tab pauses playback. Reduced-motion visitors
start with a still poster and can explicitly play any film. See
`docs/COUNTING_VIDEOS.md` for reproducible rendering and source attribution.

`verify-build` parses output HTML, resolves local resources/anchors under /Nereus/,
checks unique IDs and image alt attributes, rejects private serverpaths/secrets,
restricted research artifacts, individual files>=5MiB and total output>=12MiB.
It scans only the small generated website, never source research directories.

## Regenerating Selected Assets

Supply source roots from ignored local documentation:

```bash
node scripts/export-assets.mjs "$COUNTING_ROOT" "$EXPERIMENTS_ROOT"
```

The exporter reads exactly0000,0001,5044 and writes only the repository's web
assets/data/contact sheet. Verify permission and attribution before adding IDs.
Dimensions, counts, point bounds, grid and visibility totals must agree. Keep
original full images in explorer; no enhancement or altered measurements.
Retain author watermarks. The existing maps are precomputed ground-truth-derived
evidence, not model predictions. If upstream files are unavailable, committed
exports suffice for website build and testing.

## Publishing, Only After Approval

1. Complete docs/RELEASE_CHECKLIST.md and review staged files. No commits or
   remote pushes were made during initial implementation.
2. Create approved commits on the development branch; push and open a PR.
3. Merge reviewed changes to main. In repository Settings > Pages, choose
   GitHub Actions as source. Configure github-pages environment reviewers if needed.
4. In Actions run "Deploy research homepage" on main, explicitly setting
   publication_approved=true. No push trigger deploys automatically.
5. Inspect the completed deployment URL and validate assets/navigation under
   `https://brl-sysu.github.io/Nereus/`.

Build job has read-only contents access; deploy job alone has pages:write and
id-token:write. Source dataset roots are not used in CI. Reference:
https://docs.astro.build/en/guides/deploy/github/ and
https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages

## License and Resource Changes

Author names and affiliations were confirmed by the owner on 2026-10-13.
Do not replace pending paper/arXiv/contact details with inferred metadata.
No blanket license is assigned to all Nereus images. Preserve IOCfish CC BY-NC4.0,
font OFL and icon ISC notices. Software licensing is an owner decision. Attribute
photographs require per-item credit and permission manifest before adding assets.

The suggested external Astro template was only a reference; implementation is
original. Keep future evaluation code separate from website dependencies.
