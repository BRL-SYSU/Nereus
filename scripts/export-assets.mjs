import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
const require = createRequire(
  new URL("../website/package.json", import.meta.url),
);
const sharp = require("sharp");
const [countingRoot, experimentsRoot] = process.argv.slice(2);
if (!countingRoot || !experimentsRoot)
  throw new Error(
    "Usage: node scripts/export-assets.mjs COUNTING_ROOT EXPERIMENTS_ROOT",
  );
const out = new URL("../website/public/assets/counting/", import.meta.url);
await mkdir(out, { recursive: true });
await mkdir(new URL("../website/src/data/", import.meta.url), {
  recursive: true,
});
await mkdir(new URL("../.local/", import.meta.url), { recursive: true });
const selection = [
  { id: "0000", name: "Reef and open water", qaIndexes: [0, 2, 3] },
  { id: "0001", name: "A dense school", qaIndexes: [3] },
  { id: "5044", name: "Partial visibility", qaIndexes: [0, 3] },
];
const samples = [];
for (const choice of selection) {
  const { id } = choice;
  const d = JSON.parse(
    await readFile(
      join(countingRoot, "annotations/final_annotations", id + ".json"),
      "utf8",
    ),
  );
  assert.equal(d.image_id, id);
  const rawImage = join(countingRoot, "images", id + ".jpg");
  const density = join(
    experimentsRoot,
    "canonical_dataset_folders",
    id,
    "density",
    id + "_density_heatmap.png",
  );
  const metadata = await sharp(rawImage).metadata();
  const densityMetadata = await sharp(density).metadata();
  assert.equal(metadata.width, d.image.width);
  assert.equal(metadata.height, d.image.height);
  assert.equal(densityMetadata.width, d.image.width);
  assert.equal(densityMetadata.height, d.image.height);
  const points = d.instances.map((p) => ({
    id: p.id,
    x: p.point_xy[0],
    y: p.point_xy[1],
    visibility: p.task2_camouflage.visibility,
    regions: p.task1_region.regions,
  }));
  const grid = d.image_level_annotations.task1_region.grid_counts;
  assert.equal(
    Object.values(grid).reduce((a, b) => a + b, 0),
    points.length,
  );
  for (const p of points)
    assert.ok(
      p.x >= 0 && p.y >= 0 && p.x < d.image.width && p.y < d.image.height,
    );
  const visibility = Object.fromEntries(
    ["clear", "ambiguous", "highly_camouflaged"].map((v) => [
      v,
      points.filter((p) => p.visibility === v).length,
    ]),
  );
  const conversations = JSON.parse(
    await readFile(
      join(countingRoot, "qa/task1_region", id + ".jsonl"),
      "utf8",
    ),
  ).conversations;
  const qa = choice.qaIndexes.map((index) => ({
    question: conversations[index * 2].value.replace("<image>\n", ""),
    answer: conversations[index * 2 + 1].value,
    source: `qa/task1_region/${id}.jsonl, pair ${index + 1}`,
  }));
  for (const size of [640, 1280, 1920])
    await sharp(rawImage)
      .resize({ width: size, withoutEnlargement: true })
      .webp({ quality: 82 })
      .toFile(fileURLToPath(new URL(`${id}-${size}.webp`, out)));
  await sharp(density)
    .resize({ width: 1280 })
    .webp({ quality: 90 })
    .toFile(fileURLToPath(new URL(`${id}-density.webp`, out)));
  const hotspots = d.task3_density_depth.hotspots.items.map((h) => {
    const attribution = d.task3_density_depth.scene_attribution.items.find(
      (a) => a.hotspot_id === h.hotspot_id,
    );
    return {
      id: h.hotspot_id,
      x: h.point_xy[0],
      y: h.point_xy[1],
      box: h.sam_bbox_xyxy,
      label: attribution?.semantic_result?.semantic_label || "Not reported",
      evidence:
        attribution?.semantic_result?.evidence ||
        "No semantic evidence recorded.",
    };
  });
  samples.push({
    id,
    name: choice.name,
    width: d.image.width,
    height: d.image.height,
    count: points.length,
    image: `assets/counting/${id}-1280.webp`,
    heatmap: `assets/counting/${id}-density.webp`,
    points,
    grid,
    visibility,
    boxes: d.image_level_annotations.task1_region.scene_bboxes,
    regionCounts: d.image_level_annotations.task1_region.scene_counts,
    visibilityExplanation:
      d.image_level_annotations.task2_camouflage.image_level_explanation,
    hotspots,
    qa,
    license: "CC BY-NC 4.0",
    source: "IOCfish5K + Nereus annotations",
    credit: "Guolei Sun et al., IOCfish5K (CVPR 2023)",
    annotationSource: `annotations/final_annotations/${id}.json`,
  });
}
await sharp(fileURLToPath(new URL("0000-640.webp", out)))
  .resize(64, 64, { fit: "cover" })
  .png()
  .toFile(
    fileURLToPath(
      new URL("../website/public/assets/favicon.png", import.meta.url),
    ),
  );
await writeFile(
  new URL("../website/src/data/samples.json", import.meta.url),
  JSON.stringify(samples, null, 2) + "\n",
);
await writeFile(
  new URL("../website/public/assets/counting/samples.json", import.meta.url),
  JSON.stringify(samples) + "\n",
);
const sheet =
  '<!doctype html><html lang="en"><meta charset="utf-8"><title>Nereus selected assets</title><style>body{font:16px system-ui;background:#fff;color:#183d41;margin:30px}main{display:grid;grid-template-columns:repeat(2,1fr);gap:24px}img{width:100%}figure{margin:0}p{max-width:70ch}</style><h1>Selected IOCfish5K samples</h1><p>Original RGB / precomputed density. No retouching or changed measurements. CC BY-NC 4.0; Guolei Sun et al.</p><main>' +
  samples
    .map(
      (s) =>
        `<figure><img src="../website/public/${s.image}"><figcaption>${s.id}: ${s.name}, ${s.count} points</figcaption></figure><figure><img src="../website/public/${s.heatmap}"><figcaption>${s.id}: precomputed density response</figcaption></figure>`,
    )
    .join("") +
  "</main></html>";
await writeFile(
  new URL("../.local/contact-sheet.html", import.meta.url),
  sheet,
);
console.log(
  JSON.stringify(
    {
      exported: samples.map((s) => ({
        id: s.id,
        points: s.count,
        hotspots: s.hotspots.length,
      })),
      sourceWrites: 0,
    },
    null,
    2,
  ),
);
