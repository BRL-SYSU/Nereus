# Interactive Attribute Preview

Updated 2026-10-14. The project owner confirmed public-display permission for
the three examples and their current animation, and authorized full-page release.
This records the owner's confirmation, not an independent legal determination.
Dataset contents and original source licenses are unchanged.

## Experience

- Full-height opening image with natural scrolling, a modest 65px maximum
  parallax offset and grouped author reveals. Reduced motion and no-JavaScript
  browsing preserve readable content. No wheel interception or scroll locking.
- Three real test records: Scorpaenidae, cuttlefish and Narella, with 19, 13 and
  17 existing attributes respectively. No new biological labels are generated.
- A 14-second browser animation: source image (0-2s), whole-object bbox (2-4s),
  target zoom (4-6s), progressive attribute evidence (6-13s), full-frame end state.
- Sample selection, five perspectives, highlights, all attributes, pause/replay,
  seek, target zoom, raw annotation inspection and JSON download.
- Offscreen/background playback stops. Reduced-motion mode has no automatic
  playback or automatic zoom; intentional manual zoom changes without animation.
- Image decode must finish before annotations appear. Failed images can retry;
  stale asynchronous responses cannot replace the currently selected sample.

## Provenance and Limits

Samples are from the public Nereus Hugging Face test release. The exporter checks
image dimensions and bbox x/y/width/height against the FathomNet source record.
The manifest pins the dataset revision and records downloaded file hashes.
The upstream binary hashes differ from the Nereus files; this is disclosed,
not presented as a byte-identical upstream copy. These checks do not constitute
independent biological validation. Upstream bbox review states are UNVERIFIED.

All three upstream image records specify CC-BY-NC-ND-4.0. Source bbox records
specify CC-BY-NC-4.0; this does not assign a new license to Nereus attributes.
The image files remain full-frame (resize/format conversion only). Zoom and
boxes are separate interactive browser layers. No composited film, segmentation,
part boxes or cropped target image is exported. Local preview does not establish
permission for public deployment. The owner subsequently confirmed permission
for this page on 2026-10-14. Retain original contributor credits and licenses;
that confirmation is not a blanket license for downstream adaptations.

Source records and original URLs are linked per image in
website/public/assets/attributes/manifest.json. Do not infer attribution from
email addresses or blanket-license the FathomNet collection.

## Maintenance

Run `node scripts/export-attribute-samples.mjs` from the repository root to
re-export curated samples. It uses the ignored annotation cache under `.local`;
raw images stay there. Do not commit that cache. Source field values remain
verbatim, including list-like strings. Display labels only replace underscores
with spaces. Hand-curated highlight selections and category assignment are
presentation choices, not new source annotations.

Run the full existing test/build/check pipeline plus
`tests/browser/immersive.spec.ts`. Review desktop and mobile screenshots.
Paper and arXiv remain unavailable until the owner supplies approved URLs.
The three existing counting videos and author ordering are unchanged.
