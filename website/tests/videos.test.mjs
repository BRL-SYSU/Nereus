import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, statSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const root = new URL('../../', import.meta.url);
const samples = JSON.parse(readFileSync(new URL('website/src/data/samples.json', root)));
const storyURL = new URL('scripts/video/story.mjs', root);
const manifestURL = new URL('website/public/assets/videos/manifest.json', root);
const loadStory = async () => {
  assert.ok(existsSync(storyURL), 'reproducible annotation story module exists');
  return import(storyURL.href);
};

test('video story facts are computed from curated annotation records', async () => {
  const { deriveStories } = await loadStory();
  const stories = deriveStories(samples);
  assert.deepEqual(stories.map(s => s.id), ['region', 'camouflage', 'density']);
  assert.deepEqual(stories[0].annotationFacts, {
    total: 90, densestCell: 'top-right', densestCount: 31,
    openWater: 82, reefRock: 5, difference: 77,
  });
  assert.deepEqual(stories[1].annotationFacts, {
    total: 270, clear: 134, ambiguous: 130, highlyCamouflaged: 6,
  });
  assert.deepEqual(stories[2].annotationFacts, {
    totalPoints: 270, hotspots: 10, openWaterHotspots: 9, coralReefHotspots: 1,
  });
  // Changing a real record must change the story, not leave a hardcoded counter.
  const edited = structuredClone(samples);
  edited[0].points.find(p => p.regions.includes('reef_rock')).regions = ['open_water'];
  const changed = deriveStories(edited)[0].annotationFacts;
  assert.equal(changed.reefRock, 4);
  assert.equal(changed.openWater, 83);
});

test('sequential reveals and counters share the same true point records', async () => {
  const { deriveStories, frameState } = await loadStory();
  for (const story of deriveStories(samples)) {
    const sample = samples.find(s => s.id === story.sampleIds[0]);
    const sourceIds = new Set(sample.points.map(p => p.id));
    let changed = false;
    let previous = -1;
    for (let frame = 0; frame < 420; frame++) {
      const state = frameState(story.id, frame / 30, sample);
      assert.equal(state.totalVisible, state.points.length);
      assert.equal(new Set(state.points.map(p => p.id)).size, state.points.length);
      for (const point of state.points) {
        assert.ok(sourceIds.has(point.id));
        assert.deepEqual(point, sample.points.find(p => p.id === point.id));
      }
      for (const category of ['clear', 'ambiguous', 'highly_camouflaged']) {
        assert.equal(state.visibility[category], state.points.filter(p => p.visibility === category).length);
      }
      assert.equal(state.hotspotCount, state.hotspots.length);
      state.hotspots.forEach(h => assert.deepEqual(h, sample.hotspots.find(p => p.id === h.id)));
      changed ||= previous >= 0 && previous !== state.totalVisible;
      previous = state.totalVisible;
    }
    assert.ok(changed, `${story.id} reveals points over time`);
  }
  const sample = samples.find(s => s.id === '0001');
  assert.equal(frameState('camouflage', 13.9, sample).visibility.highly_camouflaged, 6);
  assert.equal(frameState('density', 13.9, sample).hotspotCount, 10);
});

test('density crop and text report the source annotation without an unsupported visual claim', async () => {
  const { hotspotDetail, captionCues, deriveStories } = await loadStory();
  assert.equal(typeof hotspotDetail, 'function');
  const sample = samples.find(s => s.id === '0001');
  const detail = hotspotDetail(sample, 10);
  assert.deepEqual(detail.hotspot, sample.hotspots.find(h => h.id === 10));
  assert.deepEqual(detail.crop, { x: 718, y: 398.25, w: 280, h: 157.5 });
  assert.deepEqual(detail.hotspot.box, [831, 471, 882, 497]);
  assert.equal(detail.label, 'Hotspot 10. Source annotation: coral reef.');
  const density = deriveStories(samples).find(s => s.id === 'density');
  const cue = captionCues('density', density.annotationFacts, sample).find(([start]) => start === 11);
  assert.match(cue[2], /shelter and food/);
  const changed = structuredClone(sample);
  changed.hotspots.find(h => h.id === 10).label = 'uncertain';
  assert.equal(hotspotDetail(changed, 10).label, 'Hotspot 10. Source annotation: uncertain.');
  assert.doesNotMatch(captionCues('density', density.annotationFacts, sample).map(c => c[2]).join(' '), /coral-textured/);
});

test('exactly three final clips have accessible assets and verified H.264 media', async () => {
  assert.ok(existsSync(manifestURL), 'final video manifest exists');
  const manifest = JSON.parse(readFileSync(manifestURL));
  assert.deepEqual(manifest.map(m => m.id), ['region', 'camouflage', 'density']);
  const { deriveStories } = await loadStory();
  const { findBinary } = await import(new URL('scripts/video/runtime.mjs', root));
  let ffprobe;
  try { ffprobe = findBinary('ffprobe'); } catch { /* CI verifies hash-bound probe evidence without system FFmpeg. */ }
  const evidence = JSON.parse(readFileSync(new URL('scripts/video/media-verification.json', root)));
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  assert.deepEqual(readdirSync(new URL('website/public/assets/videos/', root)).filter(name => name.endsWith('.mp4')).sort(), ['camouflage.mp4', 'density.mp4', 'region.mp4']);
  for (const [index, item] of manifest.entries()) {
    assert.equal(item.mimeType, 'video/mp4');
    assert.deepEqual(item.annotationFacts, deriveStories(samples)[index].annotationFacts);
    assert.equal(item.width, 1280);
    assert.equal(item.height, 720);
    assert.equal(item.duration, 14);
    assert.ok(item.title.length > 5);
    assert.ok(item.description.length > 30);
    assert.deepEqual(item.sampleIds, index === 2 ? ['0001','0003'] : [index === 0 ? '0000' : '0001']);
    for (const key of ['src', 'poster', 'captions']) {
      assert.match(item[key], /^assets\/videos\/[a-z-]+\.(mp4|webp|vtt)$/);
      assert.ok(existsSync(new URL(`website/public/${item[key]}`, root)));
    }
    const path = fileURLToPath(new URL(`website/public/${item.src}`, root));
    assert.ok(statSync(path).size < 5 * 1024 * 1024, `${item.id} is under 5 MiB`);
    const bytes = readFileSync(path);
    const recorded = evidence.videos.find(video => video.id === item.id);
    assert.ok(recorded, 'strict local media verification was recorded');
    assert.equal(recorded.sha256, hash(bytes));
    assert.equal(recorded.fullDecode, true);
    assert.deepEqual(recorded.browserChecks.map(check => check.viewport.width), [1280, 390]);
    assert.ok(recorded.browserChecks.every(check => check.playbackAdvanced));
    assert.equal(new Set(recorded.snapshots.map(frame => frame.sha256)).size, 8);
    assert.equal(recorded.posterSha256, hash(readFileSync(new URL(`website/public/${item.poster}`, root))));
    assert.equal(recorded.captionsSha256, hash(readFileSync(new URL(`website/public/${item.captions}`, root))));
    assert.equal(item.sourceDataSha256, hash(readFileSync(new URL(item.sourceData, root))));
    for (const asset of item.sourceAssets) assert.equal(asset.sha256, hash(readFileSync(new URL(`website/public/${asset.src}`, root))));
    const probe = ffprobe ? JSON.parse(execFileSync(ffprobe, ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', path])) : recorded.probe;
    assert.equal(probe.streams.length, 1, 'silent video has no audio stream');
    assert.equal(probe.streams[0].codec_name, 'h264');
    assert.equal(probe.streams[0].pix_fmt, 'yuv420p');
    assert.equal(probe.streams[0].width, item.width);
    assert.equal(probe.streams[0].height, item.height);
    assert.equal(probe.streams[0].nb_frames, '420');
    assert.ok(Math.abs(Number(probe.format.duration) - item.duration) < 0.05);
    assert.ok(bytes.indexOf('moov') < bytes.indexOf('mdat'), 'fast-start metadata precedes video payload');
    const vtt = readFileSync(new URL(`website/public/${item.captions}`, root), 'utf8');
    assert.ok(vtt.startsWith('WEBVTT\n'));
    assert.match(vtt, /00:00:14\.000/);
    assert.ok(vtt.split('-->').length >= 4);
    if (item.id === 'density') {
      assert.match(vtt, /Sample 0003, hotspot 1/);
      assert.match(vtt, /Annotation interpretation: Reef structure may offer shelter and food/);
      assert.doesNotMatch(vtt, /coral-textured/);
    }
  }
});
