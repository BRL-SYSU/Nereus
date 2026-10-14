import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
test("page, assets and navigation work under the GitHub Pages subpath", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const response = await page.goto("./");
  expect(response?.status()).toBe(200);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Nereus.");
  for (const id of [
    "overview",
    "attributes",
    "counting",
    "construction",
    "benchmarks",
    "qualitative",
    "resources",
  ])
    await expect(page.locator("#" + id)).toBeAttached();
  await page.locator("#resources").scrollIntoViewIfNeeded();
  await page
    .locator("img")
    .evaluateAll((imgs) =>
      imgs.forEach((i) => ((i as HTMLImageElement).loading = "eager")),
    );
  await expect
    .poll(() =>
      page
        .locator("img")
        .evaluateAll((imgs) =>
          imgs.every(
            (i) =>
              (i as HTMLImageElement).complete &&
              (i as HTMLImageElement).naturalWidth > 0,
          ),
        ),
    )
    .toBe(true);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});
test("counting modes and samples retain correct evidence", async ({ page }) => {
  await page.goto("./#counting");
  await page
    .locator("#evidence-image")
    .evaluate(async (i) => await (i as HTMLImageElement).decode());
  await expect(page.locator("#evidence-image")).toBeVisible();
  await expect(page.locator("#sample-count")).toHaveText("90");
  for (const mode of [
    "rgb",
    "points",
    "regions",
    "visibility",
    "density",
    "hotspots",
  ]) {
    await page.locator(`[data-mode="${mode}"]`).click();
    await expect(page.locator(`[data-mode="${mode}"]`)).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await expect(page.locator("#mode-caption")).not.toBeEmpty();
  }
  await page.locator('[data-sample="0001"]').click();
  await expect(page.locator("#sample-count")).toHaveText("270");
  await page.locator('[data-mode="points"]').click();
  await expect(page.locator("#evidence-layer circle")).toHaveCount(270);
  await page.locator('[data-mode="regions"]').click();
  await expect(page.locator("#sample-grid")).toContainText("87");
  await page.locator('[data-sample="5044"]').click();
  await expect(page.locator("#sample-count")).toHaveText("20");
  const aligned = await page.locator("#evidence-image").evaluate((i) => {
    const image = i.getBoundingClientRect(),
      layer = document
        .querySelector("#evidence-layer")!
        .getBoundingClientRect();
    return (
      Math.abs(image.width - layer.width) < 1 &&
      Math.abs(image.height - layer.height) < 1 &&
      Math.abs(image.width / image.height - 16 / 9) < 0.01
    );
  });
  expect(aligned).toBe(true);
});
test("attribute tabs support keyboard and honest missing-category states", async ({
  page,
}) => {
  await page.goto("./#attributes");
  await page.locator("#attr-tab-0").focus();
  await page.keyboard.press("ArrowDown");
  await expect(page.locator("#attr-tab-1")).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await page.locator('[data-organism="1"]').click();
  await expect(page.locator("#attribute-values")).toContainText("tapered");
  await page.locator("#attr-tab-3").click();
  await expect(page.locator("#attribute-values")).toContainText("not shown");
});
test("benchmark view changes and model filter expose source values", async ({
  page,
}) => {
  await page.goto("./#benchmarks");
  await page.locator('[data-benchmark="trackA"]').click();
  await page.locator("#model-filter").selectOption("InternVL2.5-8B");
  await expect(page.locator("#benchmark-trackA tbody")).toContainText("25.99");
  await expect(page.locator("#benchmark-trackA tbody tr:visible")).toHaveCount(
    5,
  );
  await page.locator('[data-benchmark="trackB"]').click();
  await expect(page.locator("#benchmark-trackB")).toContainText(
    "ground-truth-derived",
  );
});
test("visible page meets automated accessibility checks", async ({ page }) => {
  await page.goto("./");
  const result = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(result.violations).toEqual([]);
});
test("slow sample loads never display new points over a previous image", async ({
  page,
}) => {
  await page.route("**/5044-1280.webp", async (route) => {
    await new Promise((r) => setTimeout(r, 900));
    await route.continue();
  });
  await page.goto("./#counting");
  await expect(page.locator(".evidence-stage")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await page.locator('[data-sample="5044"]').click();
  await expect(page.locator(".evidence-stage")).toHaveAttribute(
    "aria-busy",
    "true",
  );
  await expect(page.locator("#evidence-layer")).not.toBeVisible();
  await expect(page.locator(".evidence-stage")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await expect(page.locator("#sample-count")).toHaveText("20");
});
test("image failure can be retried without showing mismatched evidence", async ({
  page,
}) => {
  let fail = true;
  await page.route("**/5044-1280.webp", async (route) => {
    if (fail) await route.abort();
    else await route.continue();
  });
  await page.goto("./#counting");
  await expect(page.locator(".evidence-stage")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await page.locator('[data-sample="5044"]').click();
  await expect(page.locator("#retry-evidence")).toBeVisible();
  await expect(page.locator("#evidence-layer")).not.toBeVisible();
  fail = false;
  await page.locator("#retry-evidence").click();
  await expect(page.locator("#sample-count")).toHaveText("20");
  await expect(page.locator("#evidence-layer")).toBeVisible();
});
test("navigation and citation clipboard remain usable", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("./");
  if (await page.locator(".menu-toggle").isVisible()) {
    await page.locator(".menu-toggle").click();
    await expect(page.locator("#main-nav")).toBeVisible();
    await page.locator('#main-nav a[href="#attributes"]').click();
    await expect(page.locator(".menu-toggle")).toHaveAttribute(
      "aria-expanded",
      "false",
    );
  }
  await page.locator("#copy-bibtex").click();
  await expect(page.locator("#copy-status")).toHaveText("BibTeX copied");
  expect(await page.evaluate(() => navigator.clipboard.readText())).toMatch(
    /title = \{Nereus: A Large-Scale/,
  );
});
