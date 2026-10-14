import { createRequire } from "node:module";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
const require = createRequire(
  new URL("../website/package.json", import.meta.url),
);
const { chromium } = require("@playwright/test");
const sharp = require("sharp");
const out = new URL("../.local/", import.meta.url);
await mkdir(out, { recursive: true });
const url = process.env.PREVIEW_URL || "http://127.0.0.1:4321/Nereus/";
const browser = await chromium.launch({ headless: true });
const results = [];
for (const [name, width, height] of [
  ["desktop", 1440, 1000],
  ["mobile", 390, 844],
  ["narrow", 320, 740],
  ["tablet", 768, 1024],
  ["wide", 1920, 1080],
]) {
  const context = await browser.newContext({
      viewport: { width, height },
      reducedMotion: "reduce",
    }),
    page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(url);
  await page.evaluate(() => document.fonts.ready);
  await page
    .locator("img")
    .evaluateAll((imgs) => imgs.forEach((i) => (i.loading = "eager")));
  await page.waitForFunction(() =>
    Array.from(document.images).every((i) => i.complete && i.naturalWidth > 0),
  );
  await page.waitForFunction(
    () =>
      document.querySelector(".evidence-stage").getAttribute("aria-busy") ===
      "false",
  );
  await page.screenshot({
    path: fileURLToPath(new URL(`${name}-first.png`, out)),
  });
  if (["desktop", "mobile"].includes(name)) {
    await page.screenshot({
      path: fileURLToPath(new URL(`${name}-full.png`, out)),
      fullPage: true,
    });
    for (const id of ["region", "camouflage", "density"]) {
      await page.locator(`#film-tab-${id}`).click();
      const video = page.locator(`#film-${id}`);
      const frames = [];
      for (const time of [2, 10]) {
        const pixels = await video.evaluate(async (element, target) => {
          element.muted = true;
          await element.play();
          element.pause();
          await new Promise((resolve, reject) => {
            element.addEventListener("seeked", resolve, { once: true });
            element.addEventListener("error", reject, { once: true });
            element.currentTime = target;
          });
          const canvas = document.createElement("canvas");
          canvas.width = 160;
          canvas.height = 90;
          const ctx = canvas.getContext("2d");
          ctx.drawImage(element, 0, 0, 160, 90);
          const rgba = ctx.getImageData(0, 0, 160, 90).data;
          const values = [];
          for (let i = 0; i < rgba.length; i += 4)
            values.push((rgba[i] + rgba[i + 1] + rgba[i + 2]) / 3);
          const mean = values.reduce((a, b) => a + b, 0) / values.length;
          const variance =
            values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
            values.length;
          return {
            duration: element.duration,
            width: element.videoWidth,
            height: element.videoHeight,
            stdev: Math.sqrt(variance),
            signature: values
              .filter((_, i) => i % 89 === 0)
              .map(Math.round)
              .join(","),
          };
        }, time);
        frames.push(pixels);
      }
      await page.locator(`#film-panel-${id}`).screenshot({
        path: fileURLToPath(new URL(`${name}-film-${id}.png`, out)),
        style: ".site-header,.skip-link{visibility:hidden!important}",
      });
      const nonblank = frames.every(
        (f) => f.stdev > 10 && f.width === 1280 && f.height === 720,
      );
      const moving = frames[0].signature !== frames[1].signature;
      results.push({
        name: `${name}-film-${id}`,
        nonblank,
        moving,
        duration: frames[0].duration,
      });
    }
    await page.locator("#counting-explorer").screenshot({
      path: fileURLToPath(new URL(`${name}-counting.png`, out)),
      style: ".site-header,.skip-link{visibility:hidden!important}",
    });
    await page.locator("#attribute-story").screenshot({
      path: fileURLToPath(new URL(`${name}-attributes.png`, out)),
      style: ".site-header,.skip-link{visibility:hidden!important}",
    });
    for (const [index, id] of ["fish", "cuttlefish", "coral"].entries()) {
      await page.locator(`[data-specimen="${index}"]`).click();
      await page.waitForFunction(
        () =>
          document
            .querySelector("#attribute-story")
            .getAttribute("aria-busy") === "false",
      );
      await page.locator("#attr-tab-all").click();
      const image = page.locator("#specimen-image");
      const pixels = await image.evaluate((i) => {
        const c = document.createElement("canvas");
        c.width = 80;
        c.height = 60;
        const ctx = c.getContext("2d");
        ctx.drawImage(i, 0, 0, 80, 60);
        const data = ctx.getImageData(0, 0, 80, 60).data;
        let min = 255,
          max = 0;
        for (let n = 0; n < data.length; n += 4) {
          min = Math.min(min, data[n]);
          max = Math.max(max, data[n]);
        }
        return { nonblank: max - min > 40, loaded: i.naturalWidth > 0 };
      });
      await page
        .locator("#attribute-story")
        .screenshot({
          path: fileURLToPath(new URL(`${name}-attribute-${id}.png`, out)),
          style: ".site-header,.skip-link{visibility:hidden!important}",
        });
      results.push({ name: `${name}-attribute-${id}`, ...pixels });
    }
    for (const mode of ["regions", "visibility", "density", "hotspots"]) {
      await page.locator(`[data-mode="${mode}"]`).click();
      await page.waitForFunction(
        () =>
          document
            .querySelector(".evidence-stage")
            .getAttribute("aria-busy") === "false",
      );
      await page.locator("#counting-explorer").screenshot({
        path: fileURLToPath(new URL(`${name}-${mode}.png`, out)),
        style: ".site-header,.skip-link{visibility:hidden!important}",
      });
    }
  }
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > innerWidth,
  );
  const pixel = await sharp(
    fileURLToPath(new URL(`${name}-first.png`, out)),
  ).stats();
  results.push({
    name,
    width,
    height,
    overflow,
    browserErrors: errors,
    pixelStdDev: pixel.channels.map((c) => Number(c.stdev.toFixed(2))),
  });
  await context.close();
}
await browser.close();
await writeFile(
  new URL("visual-review.json", out),
  JSON.stringify(results, null, 2) + "\n",
);
console.log(JSON.stringify(results, null, 2));
if (
  results.some(
    (r) =>
      r.overflow ||
      r.browserErrors?.length ||
      r.nonblank === false ||
      r.moving === false,
  )
)
  process.exitCode = 1;
