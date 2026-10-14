import { readFile, writeFile, mkdir, access } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { createHash } from "node:crypto";
const require = createRequire(
  new URL("../website/package.json", import.meta.url),
);
const sharp = require("sharp");
const root = new URL("../", import.meta.url).pathname;
const cache = `${root}.local`;
const out = `${root}website/public/assets/attributes`;
await mkdir(out, { recursive: true });
const records = JSON.parse(
  await readFile(`${cache}/attribute-test-annotations.json`, "utf8"),
);
const repo = "https://huggingface.co/datasets/Nereusdata/Nereus";
function get(url) {
  return execFileSync(
    "/usr/bin/curl",
    ["-fsSL", "--connect-timeout", "10", "--max-time", "90", url],
    { maxBuffer: 60 * 1024 * 1024 },
  );
}
const revision = JSON.parse(
  get("https://huggingface.co/api/datasets/Nereusdata/Nereus"),
).sha;
const pinnedRecords = JSON.parse(
  get(
    `${repo}/resolve/${revision}/object_attribute_understanding/test_annotations.json`,
  ),
);
const selected = [
  [
    "fish",
    "2c852905-9fb3-46bb-927f-4d90195afb0f",
    "Fish",
    "FathomNet / FathomVerse (see original source record)",
  ],
  [
    "cuttlefish",
    "13460870-cb45-4597-a743-6c0c0ba333ec",
    "Cephalopod",
    "FathomNet / FathomVerse (see original source record)",
  ],
  [
    "coral",
    "00c103b2-6d12-4929-8867-a1614cada73d",
    "Coral",
    "NOAA Ocean Exploration / Benthic Animal Guide",
  ],
];
const samples = [];
for (const [id, uuid, label, credit] of selected) {
  const record = records.find((r) => r.image === `${uuid}.jpg`);
  if (!record || record.objects.length !== 1)
    throw new Error(`Ambiguous sample ${id}`);
  const pinnedRecord = pinnedRecords.find((r) => r.image === record.image);
  if (JSON.stringify(record) !== JSON.stringify(pinnedRecord))
    throw new Error(`Cached annotation differs from pinned release: ${id}`);
  const source = `https://database.fathomnet.org/api/images/${uuid}`;
  const provenance = JSON.parse(get(source));
  const object = record.objects[0];
  const b = provenance.boundingBoxes.find((b) => b.concept === object.concept);
  if (
    !b ||
    JSON.stringify([b.x, b.y, b.width, b.height]) !==
      JSON.stringify(object.bbox)
  )
    throw new Error(`BBox mismatch ${id}`);
  const file = `${cache}/attribute-source-${id}.jpg`;
  try {
    await access(file);
  } catch {
    await writeFile(
      file,
      get(
        `${repo}/resolve/${revision}/object_attribute_understanding/test/${record.image}`,
      ),
    );
  }
  const raw = await readFile(file);
  const metadata = await sharp(raw).metadata();
  if (metadata.width !== record.width || metadata.height !== record.height)
    throw new Error(`Dimensions mismatch ${id}`);
  await sharp(raw)
    .resize({ width: 1600, withoutEnlargement: true })
    .webp({ quality: 88 })
    .toFile(`${out}/${id}.webp`);
  const filename = `${id}.json`;
  await writeFile(`${out}/${filename}`, JSON.stringify(record, null, 2) + "\n");
  samples.push({
    id,
    label,
    credit,
    image: `assets/attributes/${id}.webp`,
    annotation: `assets/attributes/${filename}`,
    record,
    provenance: {
      source,
      imageUrl: provenance.url,
      imageLicense: provenance.imageLicense,
      bboxLicense: b.annotationLicense,
      reviewState: b.reviewState,
      dataset: repo,
      revision,
      sha256: createHash("sha256").update(raw).digest("hex"),
      sourceImageHashMatches:
        createHash("sha256").update(raw).digest("hex") === provenance.sha256,
      transformations:
        "Full-frame resize and WebP encoding. Interactive annotation and zoom are separate browser layers; no baked overlays or cropped image export.",
      publicationStatus:
        "Local research preview; review ND image terms before public deployment.",
    },
  });
}
await writeFile(
  `${root}website/src/data/attribute-samples.json`,
  JSON.stringify(samples, null, 2) + "\n",
);
await writeFile(
  `${out}/manifest.json`,
  JSON.stringify(
    samples.map(({ image, annotation, credit, provenance }) => ({
      image,
      annotation,
      credit,
      ...provenance,
    })),
    null,
    2,
  ) + "\n",
);
console.log(
  JSON.stringify(
    samples.map((s) => ({
      id: s.id,
      dimensions: [s.record.width, s.record.height],
      attributes: Object.keys(s.record.objects[0].attributes).length,
      license: s.provenance.imageLicense,
      hashMatch: s.provenance.sourceImageHashMatches,
    })),
    null,
    2,
  ),
);
