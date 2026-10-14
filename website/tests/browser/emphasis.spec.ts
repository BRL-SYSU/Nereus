import { test, expect } from "@playwright/test";
test("venue and co-first authors have clear visual emphasis", async ({
  page,
}) => {
  await page.goto("./");
  const size = await page
    .locator(".hero-venue")
    .evaluate((e) => parseFloat(getComputedStyle(e).fontSize));
  expect(size).toBeGreaterThanOrEqual(
    page.viewportSize()!.width < 761 ? 11 : 15,
  );
  const emphasis = await page
    .locator(".author.equal-contributor")
    .first()
    .evaluate((e) => {
      const name = e.querySelector(".author-name")!;
      const other = document.querySelector(
        ".author:not(.equal-contributor) .author-name",
      )!;
      return {
        weight: +getComputedStyle(name).fontWeight,
        size: parseFloat(getComputedStyle(name).fontSize),
        other: parseFloat(getComputedStyle(other).fontSize),
        stroke: parseFloat(getComputedStyle(name).webkitTextStrokeWidth),
      };
    });
  expect(emphasis.weight).toBeGreaterThanOrEqual(800);
  expect(emphasis.size).toBeGreaterThan(emphasis.other);
  expect(emphasis.stroke).toBeGreaterThan(0);
});
