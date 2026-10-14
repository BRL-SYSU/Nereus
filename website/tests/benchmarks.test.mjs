import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
const s = JSON.parse(
  readFileSync(new URL("../src/data/science.json", import.meta.url)),
);
// Independently transcribed manuscript row fixtures, in printed table order.
test("every displayed Table 3A numeric row matches the manuscript", () => {
  const expected = [
    ["Gemini 2.0 Flash", "ZS", 52.8, 9.08, 11.12, 28.79, 18.94],
    ["Gemini 2.0 Flash", "P-ZS", 50.5, 8.86, 13.29, 15.5, 17.09],
    ["Gemini 2.0 Flash", "Full", 55.5, 9.04, 12.68, 7.62, 12.73],
    ["Qwen2.5-VL-7B", "ZS", 98.33, 5.51, 5.51, 7.62, 9.58],
    ["Qwen2.5-VL-7B", "P-ZS", 94.34, 7.26, 7.44, 4.61, 2.38],
    ["Qwen2.5-VL-7B", "Count", 38.57, 9.92, 16.83, 6.11, 7.41],
    ["Qwen2.5-VL-7B", "QAS", 31.98, 10.01, 18.51, 49.16, 52.85],
    ["Qwen2.5-VL-7B", "Full", 36.19, 10.36, 16.56, 47.39, 54.06],
    ["LLaVA-OneVision-7B", "ZS", 113.61, 4.63, 4.63, 0, 10.26],
    ["LLaVA-OneVision-7B", "P-ZS", 104.38, 6.24, 6.24, 0.18, 13.33],
    ["LLaVA-OneVision-7B", "Count", 79.47, 7.26, 8.77, 0.09, 16.9],
    ["LLaVA-OneVision-7B", "QAS", 52.81, 8.06, 12.84, 32.68, 28.91],
    ["LLaVA-OneVision-7B", "Full", 52.27, 8.06, 12.22, 36.67, 35.57],
    ["InternVL2.5-8B", "ZS", 87.38, 2.22, 2.22, 22.85, 21.26],
    ["InternVL2.5-8B", "P-ZS", 167.55, 0, 1.16, 20.73, 18.98],
    ["InternVL2.5-8B", "Count", 31.72, 10.01, 19.04, 20.81, 14.72],
    ["InternVL2.5-8B", "QAS", 25.99, 10.45, 20.46, 52.7, 50.02],
    ["InternVL2.5-8B", "Full", 30.48, 9.74, 18.95, 51.9, 52.59],
  ];
  assert.deepEqual(
    s.countingResults.map((r) => [
      r.model,
      r.setting,
      r.mae,
      r.exact,
      r.acc5,
      r.cell,
      r.camo,
    ]),
    expected,
  );
});
test("all Table 3B, Table 9 and ablation values match manuscript fixtures", () => {
  assert.deepEqual(
    s.densityResults.map((r) => [
      r.model,
      r.setting,
      r.hotspot,
      r.density,
      r.grounding,
    ]),
    [
      ["Gemini 2.0 Flash", "ZS", 44.88, 18.35, 16.48],
      ["Gemini 2.0 Flash", "Full", 41.91, 18.22, 16.15],
      ["Qwen2.5-VL-7B", "ZS", 41.75, 18.13, 16.04],
      ["Qwen2.5-VL-7B", "D-QA", 37.39, 45.26, 37.19],
      ["Qwen2.5-VL-7B", "Full", 35.37, 45.24, 37.14],
      ["LLaVA-OneVision-7B", "ZS", 32.86, 18.98, 17.34],
      ["LLaVA-OneVision-7B", "D-QA", 1.42, 28.27, 23.9],
      ["LLaVA-OneVision-7B", "Full", 47.48, 30.32, 25.22],
      ["InternVL2.5-8B", "ZS", 42.13, 19.8, 18.37],
      ["InternVL2.5-8B", "D-QA", 37.59, 45.26, 36.35],
      ["InternVL2.5-8B", "Full", 38.7, 43.24, 35.0],
    ],
  );
  assert.deepEqual(
    s.transferResults.map((r) => [
      r.setting,
      r.mae,
      r.rmse,
      r.exact,
      r.acc5,
      r.acc10,
      r.cell,
      r.camo,
    ]),
    [
      ["Nautilus counting SFT", 39.96, 158.5, 9.65, 16.3, 24.53, 6.11, 8.48],
      ["Nereus Count", 38.57, 89.59, 9.92, 16.83, 26.66, 6.11, 7.41],
      ["Nereus QAS", 31.98, 78.25, 10.01, 18.51, 29.85, 49.16, 52.85],
      ["Nereus Full", 36.19, 84.74, 10.36, 16.56, 25.33, 47.39, 54.06],
    ],
  );
  assert.deepEqual(
    s.ablations.map((r) => [r.accuracy, r.similarity]),
    [
      [39.08, 55.2],
      [16.78, 61.6],
      [41.55, 73.1],
      [40.51, 74.6],
      [37.36, 71.8],
    ],
  );
  assert.deepEqual(
    s.attributeResults.map((r) => [
      r.model,
      r.zs,
      r.naut,
      r.marine,
      r.nereus,
      r.similarity,
    ]),
    [
      ["LLaVA-OneVision-7B", 10.48, 12.06, 13.56, 48.91, 74.1],
      ["InternVL2.5-8B", 11.12, 16.08, 9.97, 48.1, 76.2],
      ["Qwen2.5-VL-7B", 13.22, 17.09, 14.64, 41.55, 73.1],
    ],
  );
});
