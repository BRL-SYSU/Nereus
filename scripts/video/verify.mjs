import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { root, outputDir, cacheDir, findBinary, probe, requireWebsite } from './runtime.mjs';

const hash = data => createHash('sha256').update(data).digest('hex');
const sharp = requireWebsite('sharp');
const manifest = JSON.parse(await readFile(join(outputDir, 'manifest.json')));
assert.deepEqual(manifest.map(v => v.id), ['region', 'camouflage', 'density']);
const ffmpeg = findBinary('ffmpeg');
findBinary('ffprobe'); // Strict verification never falls back to previously recorded evidence.
await mkdir(cacheDir, { recursive: true });
const evidence = { schemaVersion: 1, videos: [] };
const times = [0.5, 2, 4, 6, 8, 10, 11.5, 13.5];
for (const video of manifest) {
  const path = join(root, 'website/public', video.src);
  const bytes = await readFile(path);
  const metadata = probe(path);
  const stream = metadata.streams[0];
  assert.equal(metadata.streams.length, 1);
  assert.equal(stream.codec_name, 'h264');
  assert.equal(stream.pix_fmt, 'yuv420p');
  assert.equal(stream.width, 1280);
  assert.equal(stream.height, 720);
  assert.equal(stream.nb_frames, '420');
  assert.equal(stream.r_frame_rate, '30/1');
  assert.ok(Math.abs(Number(metadata.format.duration) - 14) < 0.05);
  assert.ok(bytes.length < 5 * 1048576);
  assert.equal(bytes.length, video.bytes);
  const atoms = [];
  for (let offset = 0; offset + 8 <= bytes.length;) {
    let size = bytes.readUInt32BE(offset);
    const type = bytes.toString('ascii', offset + 4, offset + 8);
    if (size === 1) size = Number(bytes.readBigUInt64BE(offset + 8));
    assert.ok(size >= 8 && offset + size <= bytes.length, `valid MP4 atom ${type}`);
    atoms.push(type); offset += size;
  }
  assert.ok(atoms.includes('moov') && atoms.includes('mdat'));
  assert.ok(atoms.indexOf('moov') < atoms.indexOf('mdat'));
  execFileSync(ffmpeg, ['-v', 'error', '-xerror', '-i', path, '-f', 'null', '-'], { stdio: 'pipe' });
  const snapshots = [];
  const frames = [];
  for (const [index, time] of times.entries()) {
    const frame = execFileSync(ffmpeg, ['-v', 'error', '-ss', String(time), '-i', path, '-frames:v', '1', '-f', 'image2pipe', '-c:v', 'png', '-threads', '1', 'pipe:1'], { maxBuffer: 8 * 1048576 });
    await writeFile(join(cacheDir, `${video.id}-decoded-${time}.png`), frame);
    const stats = await sharp(frame).extract({ left: 32, top: 140, width: 864, height: 486 }).stats();
    assert.ok(stats.channels.some(c => c.stdev > 20), `${video.id} frame ${time} has nonblank source imagery`);
    snapshots.push({ time, sha256: hash(frame), imageChannelStdev: stats.channels.map(c => Number(c.stdev.toFixed(2))) });
    frames.push({ input: await sharp(frame).resize(640, 360).png().toBuffer(), left: index % 2 * 640, top: Math.floor(index / 2) * 360 });
  }
  assert.equal(new Set(snapshots.map(s => s.sha256)).size, times.length, 'sampled animation frames differ');
  await sharp({ create: { width: 1280, height: 1440, channels: 3, background: '#f4f8f7' } }).composite(frames).webp({ quality: 88 }).toFile(join(cacheDir, `${video.id}-contact.webp`));
  const poster = await sharp(join(root, 'website/public', video.poster)).metadata();
  assert.equal(poster.width, 1280); assert.equal(poster.height, 720);
  assert.equal(video.sourceDataSha256, hash(await readFile(join(root, video.sourceData))));
  for (const asset of video.sourceAssets) assert.equal(asset.sha256, hash(await readFile(join(root, 'website/public', asset.src))));
  delete metadata.format.filename;
  evidence.videos.push({ id: video.id, sha256: hash(bytes), probe: metadata, atoms, fullDecode: true,
    posterSha256: hash(await readFile(join(root, 'website/public', video.poster))),
    captionsSha256: hash(await readFile(join(root, 'website/public', video.captions))), snapshots });
  console.log(`${video.id}: 14s / 420 frames / H.264 / silent / ${(bytes.length / 1048576).toFixed(2)} MiB / full decode OK`);
}

// Verify the exported files in a real browser, independent of the parent site's UI.
const files = new Map(await Promise.all(manifest.map(async v => [`/${v.id}.mp4`, await readFile(join(root, 'website/public', v.src))])));
const server = createServer((req, res) => {
  if (req.url === '/') {
    res.setHeader('Content-Type', 'text/html');
    res.end('<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0;background:#f4f8f7}video{display:block;width:100%;height:auto}</style><video muted playsinline preload="auto"></video>');
    return;
  }
  const data = files.get(req.url);
  if (!data) { res.writeHead(404); res.end(); return; }
  const range = req.headers.range?.match(/^bytes=(\d+)-(\d*)$/);
  const start = range ? Number(range[1]) : 0;
  const end = range && range[2] ? Math.min(Number(range[2]), data.length - 1) : data.length - 1;
  res.writeHead(range ? 206 : 200, { 'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes',
    'Content-Length': end - start + 1, ...(range ? { 'Content-Range': `bytes ${start}-${end}/${data.length}` } : {}) });
  res.end(data.subarray(start, end + 1));
});
server.listen(0, '127.0.0.1'); await once(server, 'listening');
process.env.PLAYWRIGHT_BROWSERS_PATH ||= join(root, 'website/.cache/ms-playwright');
let browser;
try {
  browser = await requireWebsite('playwright').chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage();
  for (const viewport of [{ width: 1280, height: 720 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    for (const video of manifest) {
      const result = await page.evaluate(async id => {
        const el = document.querySelector('video');
        const loaded = new Promise((resolve, reject) => { el.onloadeddata = resolve; el.onerror = () => reject(new Error('Video decode failed')); });
        el.src = `/${id}.mp4`; await loaded;
        const seeked = new Promise(resolve => { el.onseeked = resolve; });
        el.currentTime = 11.5; await seeked;
        const box = el.getBoundingClientRect();
        return { width: el.videoWidth, height: el.videoHeight, duration: el.duration, renderedWidth: box.width, renderedHeight: box.height };
      }, video.id);
      assert.equal(result.width, 1280); assert.equal(result.height, 720);
      assert.equal(result.duration, 14); assert.equal(result.renderedWidth, viewport.width);
      const screenshot = await page.locator('video').screenshot({ path: join(cacheDir, `${video.id}-browser-${viewport.width}.png`) });
      const stats = await sharp(screenshot).stats();
      assert.ok(stats.channels.some(c => c.stdev > 35), 'browser video is not blank');
      await page.evaluate(async () => { const el = document.querySelector('video'); await el.play(); });
      await page.waitForFunction(() => document.querySelector('video').currentTime > 11.75);
      await page.evaluate(() => document.querySelector('video').pause());
      const record = evidence.videos.find(v => v.id === video.id);
      (record.browserChecks ||= []).push({ viewport, ...result, playbackAdvanced: true });
    }
  }
} finally { await browser?.close(); server.close(); }
await writeFile(join(root, 'scripts/video/media-verification.json'), JSON.stringify(evidence, null, 2) + '\n');
console.log('Strict media evidence saved; desktop/mobile decode, seeking, and playback passed.');
