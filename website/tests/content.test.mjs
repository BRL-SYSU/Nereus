import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
const data = new URL("../src/data/", import.meta.url);
test("verified content exists separately from presentation", () => {
  assert.ok(
    existsSync(new URL("science.json", data)),
    "scientific dataset not implemented",
  );
});
test("manuscript statistics and benchmarks retain source references", () => {
  const path = new URL("science.json", data);
  assert.ok(existsSync(path), "scientific dataset not implemented");
  const s = JSON.parse(readFileSync(path));
  assert.equal(s.statistics.find((x) => x.id === "images").value, 88555);
  assert.equal(s.statistics.find((x) => x.id === "dimensions").value, 136);
  for (const item of s.statistics) assert.ok(item.source);
  assert.equal(
    s.attributeResults.find((x) => x.model === "LLaVA-OneVision-7B").nereus,
    48.91,
  );
  assert.equal(
    s.countingResults.find(
      (x) => x.model === "InternVL2.5-8B" && x.setting === "QAS",
    ).mae,
    25.99,
  );
  assert.match(s.protocols.trackB, /ground.truth/i);
});
test("curated samples preserve coordinates, counts and attribution", () => {
  const path = new URL("samples.json", data);
  assert.ok(existsSync(path), "curated samples not exported");
  const samples = JSON.parse(readFileSync(path));
  assert.ok(samples.length >= 3);
  for (const s of samples) {
    assert.equal(s.count, s.points.length);
    assert.equal(
      Object.values(s.grid).reduce((a, b) => a + b, 0),
      s.count,
    );
    assert.equal(
      Object.values(s.visibility).reduce((a, b) => a + b, 0),
      s.count,
    );
    assert.ok(s.qa.length > 0);
    assert.equal(s.license, "CC BY-NC 4.0");
    for (const p of s.points) {
      assert.ok(p.x >= 0 && p.x < s.width);
      assert.ok(p.y >= 0 && p.y < s.height);
    }
    for (const name of [s.image, s.heatmap])
      assert.ok(existsSync(new URL("../public/" + name, import.meta.url)));
    assert.doesNotMatch(
      JSON.stringify(s),
      /\/mnt\/|\/data\/Jinzh|\/home\/|raw_response/,
    );
  }
});
test("spatial memberships, image dimensions and download stay aligned", async () => {
  const samples = JSON.parse(readFileSync(new URL("samples.json", data)));
  assert.deepEqual(
    samples,
    JSON.parse(
      readFileSync(
        new URL("../public/assets/counting/samples.json", import.meta.url),
      ),
    ),
  );
  const { default: sharp } = await import("sharp");
  const labels = [
    "top-left",
    "top-center",
    "top-right",
    "center-left",
    "center",
    "center-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
  ];
  for (const s of samples) {
    const grid = Object.fromEntries(labels.map((k) => [k, 0]));
    for (const p of s.points)
      grid[
        labels[
          Math.min(2, Math.floor(p.y / (s.height / 3))) * 3 +
            Math.min(2, Math.floor(p.x / (s.width / 3)))
        ]
      ]++;
    assert.deepEqual(s.grid, grid);
    for (const path of [s.image, s.heatmap]) {
      const image = await sharp(
        new URL("../public/" + path, import.meta.url).pathname,
      ).metadata();
      assert.equal(image.width, 1280);
      assert.equal(image.height, 720);
    }
    for (const h of s.hotspots) {
      assert.ok(h.x >= 0 && h.x < s.width && h.y >= 0 && h.y < s.height);
      assert.ok(h.id > 0);
    }
    for (const b of [
      ...Object.values(s.boxes).flat(),
      ...s.hotspots.map((h) => h.box).filter(Boolean),
    ]) {
      assert.ok(
        b[0] >= 0 &&
          b[1] >= 0 &&
          b[2] <= s.width &&
          b[3] <= s.height &&
          b[2] >= b[0] &&
          b[3] >= b[1],
      );
    }
  }
});
