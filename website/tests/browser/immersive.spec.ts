import { test, expect } from "@playwright/test";

test("hero fills the opening viewport and author groups reveal on natural scroll", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  await page.goto("./");
  const hero = page.locator(".hero");
  const box = await hero.boundingBox();
  expect(box!.height).toBeGreaterThan(page.viewportSize()!.height * 0.8);
  await expect(page.locator(".hero-scroll")).toBeInViewport();
  const first = page.locator(".publication-header .authors");
  await expect(first).toHaveCSS("opacity", "0");
  await page.mouse.wheel(0, page.viewportSize()!.height * 0.75);
  await expect(first).toHaveCSS("opacity", "1");
  expect(
    await hero.evaluate((e) => e.style.getPropertyValue("--hero-shift")),
  ).not.toBe("0px");
  await page.locator(".publication-links").scrollIntoViewIfNeeded();
  await expect(page.locator(".publication-links")).toHaveCSS("opacity", "1");
});

test("reduced motion retains readable authors and a static hero", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("./");
  await expect(page.locator(".publication-header .authors")).toHaveCSS(
    "opacity",
    "1",
  );
  await expect(page.locator(".hero-image")).toHaveCSS("transform", "none");
});

test("real attribute samples keep image, box and annotations aligned", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("./#attributes");
  await expect(page.locator("#attribute-story")).toBeVisible();
  for (const [index, name] of [
    "Scorpaenidae",
    "cuttlefish",
    "Narella",
  ].entries()) {
    await page.locator(`[data-specimen="${index}"]`).click();
    await expect(page.locator("#attribute-story")).toHaveAttribute(
      "aria-busy",
      "false",
    );
    await expect(page.locator("#specimen-name")).toHaveText(name);
    await page.getByRole("button", { name: "Show all attributes" }).click();
    await expect(page.locator("#attribute-values > div")).toHaveCount([19, 13, 17][index]);
    await expect(page.locator("#specimen-box")).toBeVisible();
    await expect(page.locator("#attribute-values > div")).not.toHaveCount(0);
    await page.getByText("Full source annotation", { exact: true }).click();
    await expect(page.locator("#specimen-json")).toContainText(name);
    await page.getByText("Full source annotation", { exact: true }).click();
    const aligned = await page.locator("#specimen-box").evaluate((box) => {
      const r = box.getBoundingClientRect(),
        i = document.querySelector("#specimen-image")!.getBoundingClientRect();
      return (
        r.left >= i.left - 2 &&
        r.top >= i.top - 2 &&
        r.right <= i.right + 2 &&
        r.bottom <= i.bottom + 2
      );
    });
    expect(aligned).toBe(true);
  }
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test('attribute autoplay zooms in view and stops outside the section', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto('./#attributes');
  await expect.poll(() => page.locator('#attribute-progress').inputValue()).not.toBe('0');
  await page.locator('#attribute-progress').fill('5');
  await expect(page.locator('#attribute-zoom')).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(() => page.locator('#specimen-plane').evaluate(e => new DOMMatrix(getComputedStyle(e).transform).a)).toBeGreaterThan(1.1);
  await page.getByRole('button', { name: 'Play attribute sequence', exact: true }).click();
  await page.locator('#abstract').scrollIntoViewIfNeeded();
  await expect(page.locator('#attribute-story')).toHaveAttribute('data-playing', 'false');
});

test('authors and source annotation remain readable without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4321/Nereus/');
  await expect(page.locator('.authors')).toHaveCSS('opacity', '1');
  await expect(page.locator('#specimen-name')).toHaveText('Scorpaenidae');
  await page.getByText('Full source annotation', { exact: true }).click();
  await expect(page.locator('#specimen-json')).toContainText('Scorpaenidae');
  await context.close();
});

test("attribute playback advances, pauses and replays", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("./#attributes");
  await expect(page.locator("#attribute-story")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await page.getByRole("button", { name: "Replay attribute sequence" }).click();
  await expect
    .poll(() => page.locator("#attribute-progress").inputValue())
    .not.toBe("0");
  await page.getByRole("button", { name: "Pause attribute sequence" }).click();
  const t = await page.locator("#attribute-progress").inputValue();
  await page.waitForTimeout(250);
  expect(await page.locator("#attribute-progress").inputValue()).toBe(t);
  await page.locator("#attribute-progress").fill("10");
  await expect(
    page.locator("#attribute-values > div:not([hidden])"),
  ).not.toHaveCount(0);
});

test("failed attribute image hides evidence and supports retry", async ({
  page,
}) => {
  let fail = true;
  await page.route("**/assets/attributes/cuttlefish.webp", async (route) => {
    if (fail) await route.abort();
    else await route.continue();
  });
  await page.goto("./#attributes");
  await page.locator('[data-specimen="1"]').click();
  await expect(page.locator("#specimen-retry")).toBeVisible();
  await expect(page.locator("#specimen-box")).toBeHidden();
  fail = false;
  await page.locator("#specimen-retry").click();
  await expect(page.locator("#attribute-story")).toHaveAttribute(
    "aria-busy",
    "false",
  );
  await expect(page.locator("#specimen-image")).toHaveJSProperty(
    "naturalWidth",
    1600,
  );
});
