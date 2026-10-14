import { test, expect } from "@playwright/test";
import sharp from "sharp";

test("original figures follow abstract and can be enlarged accessibly", async ({
  page,
}) => {
  await page.goto("./");
  const section = page.locator("#paper-figures");
  await expect(section).toHaveCount(1);
  expect(
    await page.locator("#abstract").evaluate((e) => e.nextElementSibling?.id),
  ).toBe("paper-figures");
  await expect(section.locator("figure h2")).toHaveText([
    "Motivation",
    "Pipeline",
  ]);
  for (const id of ["motivation", "pipeline"]) {
    const figure = page.locator(`#paper-${id}`);
    await figure.scrollIntoViewIfNeeded();
    await expect(figure.locator("img")).toBeVisible();
    await figure.locator("img").evaluate((e: HTMLImageElement) => e.decode());
    const pixels = await sharp(await figure.locator("img").screenshot()).stats();
    expect(pixels.channels.some(channel => channel.stdev > 30)).toBe(true);
    const iconBox = await figure.locator('.paper-expand').boundingBox();
    const imageBox = await figure.locator('img').boundingBox();
    expect(iconBox!.y + iconBox!.height).toBeLessThanOrEqual(imageBox!.y);
    expect(
      await figure
        .locator("img")
        .evaluate(async (e: HTMLImageElement) => {
          // srcset naturalWidth is density-corrected; check source pixels directly.
          const source = new Image();
          source.src = e.currentSrc;
          await source.decode();
          return source.naturalWidth;
        }),
    ).toBeGreaterThan(1000);
    await figure.locator("button").click();
    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole("heading")).toHaveText(
      id === "motivation" ? "Motivation" : "Pipeline",
    );
    await page.keyboard.press("Escape");
    await expect(dialog).not.toBeVisible();
    await expect(figure.locator("button")).toBeFocused();
  }
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
  ).toBe(false);
  await expect(page.locator('a[href$=".pdf"]')).toHaveCount(0);
});
