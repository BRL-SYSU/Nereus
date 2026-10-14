import { test, expect } from "@playwright/test";

test("automatic playback starts in view and deliberate loading cancellation is quiet", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/assets/videos/region.mp4", async (route) => {
    await held;
    await route.continue().catch(() => {});
  });
  try {
    await page.goto("./");
    await page.locator("#counting-films").scrollIntoViewIfNeeded();
    await expect(page.locator("#film-region")).toHaveJSProperty(
      "paused",
      false,
    );
    await page.locator("#abstract").scrollIntoViewIfNeeded();
    await expect(page.locator("#film-region")).toHaveJSProperty("paused", true);
    await expect(page.locator("#film-panel-region .film-status")).toBeHidden();
  } finally {
    release();
  }
});

test("automatic films start, switch without overlap, and recover from a failed video load", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "no-preference" });
  let fail = true;
  await page.route("**/assets/videos/camouflage.mp4", async (route) => {
    if (fail) await route.abort();
    else await route.continue();
  });
  await page.goto("./#counting-films");
  await expect
    .poll(() =>
      page
        .locator("#film-region")
        .evaluate((v: HTMLVideoElement) => v.currentTime),
    )
    .toBeGreaterThan(0);
  await page.locator("#film-tab-camouflage").click();
  await expect(page.locator("#film-region")).toHaveJSProperty("paused", true);
  await expect(
    page.locator("#film-panel-camouflage .film-status"),
  ).toBeVisible();
  fail = false;
  await page.getByRole("button", { name: "Replay film" }).click();
  await expect
    .poll(() =>
      page
        .locator("#film-camouflage")
        .evaluate((v: HTMLVideoElement) => v.currentTime),
    )
    .toBeGreaterThan(0);
  await expect(
    page.locator("#film-panel-camouflage .film-status"),
  ).toBeHidden();
  await page.locator("#film-tab-density").click();
  await expect
    .poll(() =>
      page
        .locator("#film-density")
        .evaluate((v: HTMLVideoElement) => v.currentTime),
    )
    .toBeGreaterThan(0);
  await expect(page.locator("#film-camouflage")).toHaveJSProperty(
    "paused",
    true,
  );
});

test("author header is complete and abstract is expanded before the overview", async ({
  page,
}) => {
  await page.goto("./");
  await expect(page.locator(".author-name")).toHaveText([
    "Jingtao Deng",
    "Zihao Jin",
    "Yuehe Chen",
    "Feixiang He",
    "Shuiwang Li",
    "Dan Zeng",
  ]);
  await expect(page.locator(".author.equal-contributor")).toHaveCount(2);
  await expect(page.locator(".author.corresponding-author")).toContainText(
    "Dan Zeng",
  );
  await expect(
    page.getByRole("button", { name: "Paper coming soon" }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "arXiv coming soon" }),
  ).toBeDisabled();
  await expect(page.locator("a[href$='.pdf']")).toHaveCount(0);
  await expect(page.locator("#abstract .prose")).toBeVisible();
  expect(
    await page.evaluate(() =>
      Boolean(
        document
          .querySelector("#abstract")!
          .compareDocumentPosition(document.querySelector("#overview")!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
      ),
    ),
  ).toBe(true);
});

test("BibTeX entry can be reached and copied with confirmed author order", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("./");
  await page
    .locator(".publication-links")
    .getByRole("link", { name: "BibTeX" })
    .click();
  await expect(page.locator("#bibtex")).toBeInViewport();
  await page.getByRole("button", { name: "Copy BibTeX" }).click();
  await expect(page.locator("#copy-status")).toHaveText("BibTeX copied");
  const citation = await page.evaluate(() => navigator.clipboard.readText());
  expect(citation).toMatch(
    /author = \{Deng, Jingtao and Jin, Zihao and Chen, Yuehe and He, Feixiang and Li, Shuiwang and Zeng, Dan\}/,
  );
  expect(citation).not.toMatch(/eprint|arxiv\.org|Anonymous/);
});

test("three films play, seek, switch and replay without simultaneous playback", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("./#counting-films");
  await expect(page.locator(".counting-film")).toHaveCount(3);
  for (const id of ["region", "camouflage", "density"]) {
    await page.locator(`#film-tab-${id}`).click();
    const video = page.locator(`#film-${id}`);
    await expect(video).toBeVisible();
    await expect(video).toHaveJSProperty("paused", true);
    await video.evaluate(async (v: HTMLVideoElement) => {
      v.muted = true;
      await v.play();
    });
    await expect
      .poll(() => video.evaluate((v: HTMLVideoElement) => v.currentTime))
      .toBeGreaterThan(0);
    const duration = await video.evaluate((v: HTMLVideoElement) => v.duration);
    expect(duration).toBeGreaterThanOrEqual(10);
    expect(duration).toBeLessThanOrEqual(15.1);
    await video.evaluate((v: HTMLVideoElement) => {
      v.pause();
      v.currentTime = 6;
    });
    await page.getByRole("button", { name: "Replay film" }).click();
    await expect
      .poll(() => video.evaluate((v: HTMLVideoElement) => v.currentTime))
      .toBeLessThan(3);
    expect(
      await page
        .locator(".counting-film")
        .evaluateAll(
          (vs) => vs.filter((v) => !(v as HTMLVideoElement).paused).length,
        ),
    ).toBe(1);
  }
  await page.locator("#film-tab-density").focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#film-tab-region")).toBeFocused();
  await expect(page.locator("#film-tab-region")).toHaveAttribute(
    "aria-selected",
    "true",
  );
});

test("films respect reduced motion and pause when the section leaves view", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("./#counting-films");
  const video = page.locator("#film-region");
  await expect(video).toBeVisible();
  await expect(video).toHaveJSProperty("paused", true);
  await video.evaluate(async (v: HTMLVideoElement) => {
    v.muted = true;
    await v.play();
  });
  await expect
    .poll(() => video.evaluate((v: HTMLVideoElement) => v.currentTime))
    .toBeGreaterThan(0);
  await page.locator("#abstract").scrollIntoViewIfNeeded();
  await expect(video).toHaveJSProperty("paused", true);
});
