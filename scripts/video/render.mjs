import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, rename } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { spawn, execFileSync } from 'node:child_process';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { deriveStories, captionCues, toVtt, DURATION, FPS, reasoningCases } from './story.mjs';
import { root, outputDir, cacheDir, requireWebsite, findBinary, probe } from './runtime.mjs';

const samplesBytes = await readFile(join(root, 'website/src/data/samples.json'));
const samples = JSON.parse(samplesBytes);
const stories = deriveStories(samples);
const selected = process.argv.find(arg => arg.startsWith('--only='))?.split('=')[1];
if (selected && !stories.some(s => s.id === selected)) throw new Error('Unknown --only video ID');
const preview = process.argv.includes('--preview');
await mkdir(cacheDir, { recursive: true });
await mkdir(outputDir, { recursive: true });
process.env.PLAYWRIGHT_BROWSERS_PATH ||= join(root, 'website/.cache/ms-playwright');
const { chromium } = requireWebsite('playwright');
const sharp = requireWebsite('sharp');
const ffmpeg = findBinary('ffmpeg');
const encoders = execFileSync(ffmpeg, ['-hide_banner', '-encoders'], { encoding: 'utf8' });
const encoder = encoders.includes('libx264') ? 'libx264' : 'libopenh264';
if (!encoders.includes(encoder)) throw new Error('FFmpeg needs libx264 or libopenh264');
const hash = data => createHash('sha256').update(data).digest('hex');
const assets = new Map([
  ['/story.mjs', join(root, 'scripts/video/story.mjs')],
  ['/renderer.mjs', join(root, 'scripts/video/renderer.mjs')],
  ['/website/src/data/density-reasoning.json', join(root, 'website/src/data/density-reasoning.json')],
  ['/font.woff2', join(root, 'website/node_modules/@fontsource-variable/manrope/files/manrope-latin-wght-normal.woff2')],
]);
for (const sample of samples.filter(s => ['0000', '0001'].includes(s.id))) {
  for (const asset of [`assets/counting/${sample.id}-1920.webp`, sample.heatmap]) assets.set(`/${asset}`, join(root, 'website/public', asset));
}
for (const c of reasoningCases) for (const asset of [c.image,c.heatmap]) assets.set(`/${asset}`,join(root,'website/public',asset));
const server = createServer(async (req, res) => {
  if (req.url === '/') {
    res.setHeader('Content-Type', 'text/html');
    res.end('<!doctype html><html lang="en"><meta charset="utf-8"><title>Nereus video renderer</title><style>html,body{margin:0;background:#f4f8f7}canvas{display:block;width:100%;height:auto}</style><canvas id="film" width="1280" height="720"></canvas></html>');
    return;
  }
  const path = assets.get(req.url);
  if (!path) { res.writeHead(404); res.end(); return; }
  try {
    res.setHeader('Content-Type', path.endsWith('.mjs') ? 'text/javascript' : path.endsWith('.json') ? 'application/json' : path.endsWith('.woff2') ? 'font/woff2' : 'image/webp');
    res.end(await readFile(path));
  } catch { res.writeHead(500); res.end(); }
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');
let browser;
try {
  browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.evaluate(async ({ samples, stories }) => {
    const { createRenderer } = await import('/renderer.mjs');
    window.renderer = await createRenderer(document.querySelector('canvas'), samples, stories);
  }, { samples, stories });
  const layoutIssues = [];
  for (const story of stories.filter(s => !selected || s.id === selected)) {
    const frames = preview ? [0, 2, 4, 6, 8, 10, 11.5, 13.5].map(t => Math.round(t * FPS)) : Array.from({ length: DURATION * FPS }, (_, i) => i);
    const partial = join(cacheDir, `${story.id}.partial.mp4`);
    let child;
    let completion;
    let log = '';
    if (!preview) {
      const codecArgs = encoder === 'libx264' ? ['-preset', 'slow', '-crf', '23'] : ['-profile:v', 'high', '-b:v', '1250k'];
      child = spawn(ffmpeg, ['-y', '-hide_banner', '-loglevel', 'warning', '-f', 'image2pipe', '-framerate', String(FPS), '-vcodec', 'png', '-i', 'pipe:0', '-an', '-c:v', encoder, ...codecArgs, '-maxrate', '1400k', '-bufsize', '2800k', '-pix_fmt', 'yuv420p', '-g', '60', '-threads', '4', '-movflags', '+faststart', partial], { stdio: ['pipe', 'ignore', 'pipe'] });
      child.stderr.on('data', data => { log += data; });
      child.stdin.on('error', () => {});
      completion = new Promise((resolve, reject) => {
        child.on('error', reject);
        child.on('close', code => code === 0 ? resolve() : reject(new Error(`FFmpeg exited ${code}: ${log}`)));
      });
      completion.catch(() => {});
    }
    try {
      for (const frame of frames) {
        const t = frame / FPS;
        const result = await page.evaluate(({ id, t }) => {
          const { textBoxes } = window.renderer.render(id, t);
          return { data: document.querySelector('canvas').toDataURL('image/png').split(',')[1],
            issues: textBoxes.filter(b => b.x < 0 || b.y < 0 || b.x + b.width > 1280 || b.y + b.height > 714) };
        }, { id: story.id, t });
        layoutIssues.push(...result.issues.map(issue => ({ id: story.id, t, ...issue })));
        const bytes = Buffer.from(result.data, 'base64');
        if (preview) await writeFile(join(cacheDir, `${story.id}-${String(frame).padStart(3, '0')}.png`), bytes);
        else {
          if (child.exitCode !== null) throw new Error(`Encoder stopped: ${log}`);
          if (!child.stdin.write(bytes)) await once(child.stdin, 'drain');
        }
        if (frame === 120 || frame === 240 || frame === 360) console.log(`${story.id}: frame ${frame}/${DURATION * FPS}`);
      }
      if (!preview) {
        child.stdin.end();
        await completion;
        const metadata = probe(partial);
        if (Math.abs(Number(metadata.format.duration) - DURATION) > 0.05 || metadata.streams[0].codec_name !== 'h264') throw new Error('Unexpected video metadata');
        if (Number(metadata.format.size) >= 5 * 1024 * 1024) throw new Error('Video exceeds 5 MiB');
        await rename(partial, join(outputDir, `${story.id}.mp4`));
        const posterTime = story.id === 'region' ? 4 : story.id === 'camouflage' ? 13.5 : 11.5;
        const posterPNG = execFileSync(ffmpeg, ['-v', 'error', '-ss', String(posterTime), '-i', join(outputDir, `${story.id}.mp4`), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-threads', '1', 'pipe:1'], { maxBuffer: 8 * 1024 * 1024 });
        await sharp(posterPNG).webp({ quality: 90 }).toFile(join(outputDir, `${story.id}-poster.webp`));
        await writeFile(join(outputDir, `${story.id}.vtt`), toVtt(captionCues(story.id, story.annotationFacts, samples.find(s => s.id === story.sampleIds[0]))));
        console.log(`${story.id}: ${metadata.format.duration}s H.264, ${(Number(metadata.format.size) / 1048576).toFixed(2)} MiB`);
      }
    } finally {
      if (child && child.exitCode === null) { child.stdin.destroy(); child.kill('SIGTERM'); await completion.catch(() => {}); }
    }
  }
  await writeFile(join(cacheDir, 'layout-report.json'), JSON.stringify({ errors, layoutIssues }, null, 2));
  if (errors.length || layoutIssues.length) throw new Error(`Renderer errors: ${errors.length}; out-of-bounds labels: ${layoutIssues.length}`);
  if (!preview && stories.every(s => existsSync(join(outputDir, `${s.id}.mp4`)))) {
    const manifest = [];
    for (const story of stories) {
      const sample = samples.find(s => s.id === story.sampleIds[0]);
      const sourceAssets = story.id === 'density' ? [...reasoningCases.flatMap(c=>[c.image,c.heatmap]),'assets/videos/density-evidence.json'] : [`assets/counting/${sample.id}-1920.webp`];
      manifest.push({ ...story, bytes: Number(probe(join(outputDir, `${story.id}.mp4`)).format.size),
        sourceData: 'website/src/data/samples.json', sourceDataSha256: hash(samplesBytes),
        sourceAssets: await Promise.all(sourceAssets.map(async src => ({ src, sha256: hash(await readFile(join(root, 'website/public', src))) }))) });
    }
    await writeFile(join(outputDir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
    console.log('Published manifest.json for all three clips.');
  }
} finally {
  await browser?.close();
  server.close();
}
