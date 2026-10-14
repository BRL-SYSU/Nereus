import reasoning from '../../website/src/data/density-reasoning.json' with { type: 'json' };
export const reasoningCases = reasoning.cases;
export const DURATION = 14;
export const FPS = 30;
export const GRID = ['top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'];
export const clamp = value => Math.max(0, Math.min(1, value));
export const ease = value => { const t = clamp(value); return t * t * (3 - 2 * t); };
export const progress = (time, start, end) => clamp((time - start) / (end - start));
export const gridCell = (point, sample) => GRID[Math.min(2, Math.floor(point.y / sample.height * 3)) * 3 + Math.min(2, Math.floor(point.x / sample.width * 3))];
const countBy = (items, keys, predicate) => Object.fromEntries(keys.map(key => [key, items.filter(item => predicate(item, key)).length]));
const visibilityCounts = points => countBy(points, ['clear', 'ambiguous', 'highly_camouflaged'], (p, key) => p.visibility === key);

export function deriveStories(samples) {
  const region = samples.find(s => s.id === '0000');
  const school = samples.find(s => s.id === '0001');
  const grid = countBy(region.points, GRID, (p, key) => gridCell(p, region) === key);
  const densestCell = GRID.reduce((best, key) => grid[key] > grid[best] ? key : best);
  const openWater = region.points.filter(p => p.regions.includes('open_water')).length;
  const reefRock = region.points.filter(p => p.regions.includes('reef_rock')).length;
  const visibility = visibilityCounts(school.points);
  const stories = [
    { id: 'region', title: 'Region-aware counting', sampleIds: ['0000'],
      description: 'True point annotations reveal the densest grid cell, then compare open-water and reef-rock region labels.',
      annotationFacts: { total: region.points.length, densestCell, densestCount: grid[densestCell], openWater, reefRock, difference: openWater - reefRock } },
    { id: 'camouflage', title: 'Camouflage-aware counting', sampleIds: ['0001'],
      description: 'Sequential visibility-category annotations lead to a source-image detail of highly camouflaged points. Categories are annotations, not model predictions.',
      annotationFacts: { total: school.points.length, clear: visibility.clear, ambiguous: visibility.ambiguous, highlyCamouflaged: visibility.highly_camouflaged } },
    { id: 'density', title: 'Density-hotspot reasoning', sampleIds: reasoningCases.map(c => c.sampleId),
      description: 'Two localized hotspots connect visible environment and relative-depth annotations to tentative explanations of schooling and aggregation near coral. These interpretations do not establish ecological causes.',
      annotationFactsScope: 'Counts describe sample 0001 only; the film compares hotspot 1 in samples 0001 and 0003.',
      reasoningCases: reasoningCases.map(c => ({sampleId:c.sampleId, hotspotId:c.hotspotId, point:c.point, annotationSource:c.annotationSource, annotationSha256:c.annotationSha256})),
      annotationFacts: { totalPoints: school.points.length, hotspots: school.hotspots.length,
        openWaterHotspots: school.hotspots.filter(h => h.label === 'open water').length,
        coralReefHotspots: school.hotspots.filter(h => h.label === 'coral reef').length } },
  ];
  return stories.map(story => ({ ...story, src: `assets/videos/${story.id}.mp4`,
    poster: `assets/videos/${story.id}-poster.webp`, captions: `assets/videos/${story.id}.vtt`,
    duration: DURATION, width: 1280, height: 720, fps: FPS, mimeType: 'video/mp4',
    captionsLanguage: 'en', license: 'CC BY-NC 4.0', source: 'IOCfish5K + Nereus annotations',
    credit: 'Guolei Sun et al., IOCfish5K (CVPR 2023)', annotationDerived: true,
    annotationSource: samples.find(s => s.id === story.sampleIds[0]).annotationSource,
  }));
}

export function frameState(id, time, sample) {
  const sorted = [...sample.points].sort((a, b) => a.x - b.x || a.y - b.y);
  const reveal = (items, start, end) => items.slice(0, Math.floor(items.length * progress(time, start, end)));
  let points;
  let phase;
  if (id === 'region') {
    phase = time < 4.2 ? 'grid' : time < 7.8 ? 'focus' : time < 10.6 ? 'water' : 'regions';
    points = phase === 'grid' ? reveal(sorted, 0.5, 3.7) : phase === 'focus' ? sorted :
      [...reveal(sorted.filter(p => p.regions.includes('open_water')), 7.8, 10.2),
        ...reveal(sorted.filter(p => !p.regions.includes('open_water')), 10.6, 11.9)];
  } else if (id === 'camouflage') {
    phase = time < 3.5 ? 'clear' : time < 6.7 ? 'ambiguous' : time < 8.7 ? 'camouflaged' : time < 12 ? 'detail' : 'summary';
    points = [...reveal(sorted.filter(p => p.visibility === 'clear'), 0.5, 3.3),
      ...reveal(sorted.filter(p => p.visibility === 'ambiguous'), 3.5, 6.5),
      ...reveal(sorted.filter(p => p.visibility === 'highly_camouflaged'), 6.7, 8.3)];
  } else if (id === 'density') {
    phase = time < 3.2 ? 'points' : time < 5 ? 'heatmap' : time < 8.2 ? 'hotspots' : time < 10.7 ? 'water-detail' : time < 12.8 ? 'reef-detail' : 'summary';
    points = reveal(sorted, 0.5, 2.9);
  } else throw new Error(`Unknown video: ${id}`);
  const hotspots = id === 'density' ? reveal(sample.hotspots, 5, 7.8) : [];
  return { phase, points, totalVisible: points.length, visibility: visibilityCounts(points),
    grid: countBy(points, GRID, (p, key) => gridCell(p, sample) === key),
    regions: countBy(points, ['open_water', 'reef_rock'], (p, key) => p.regions.includes(key)),
    hotspots, hotspotCount: hotspots.length };
}

export function hotspotDetail(sample, id) {
  const hotspot = sample.hotspots.find(h => h.id === id);
  if (!hotspot) throw new Error(`Missing hotspot ${id}`);
  return { hotspot, crop: { x: hotspot.x - 140, y: hotspot.y - 78.75, w: 280, h: 157.5 },
    label: `Hotspot ${hotspot.id}. Source annotation: ${hotspot.label}.` };
}

export function densityReasoningAt(time) {
  const index = time < 7 ? 0 : 1;
  const localTime = Math.max(0, Math.min(7, time - index * 7));
  return {case: reasoningCases[index], index, localTime,
    phase: localTime < 2 ? 'LOCALIZE' : localTime < 4 ? 'GROUND' : 'INTERPRET',
    showEvidence: localTime >= 1.5, showContext: localTime >= 2.5, showInterpretation: localTime >= 4};
}

export function captionCues(id, f, sample) {
  if (id === 'region') return [
    [0, 4.2, `Region-aware counting. Sample 0000. ${f.total} annotated fish appear point by point across a three-by-three grid.`],
    [4.2, 7.8, `Which cell is densest? Top-right: ${f.densestCount} point annotations. The camera moves into that cell.`],
    [7.8, 10.6, `Semantic region labels: ${f.openWater} fish are annotated as open water.`],
    [10.6, 14, `${f.reefRock} are annotated as reef rock. Open water has ${f.difference} more. These are annotation labels, not predictions; other region labels are not shown in this comparison.`],
  ];
  if (id === 'camouflage') return [
    [0, 3.5, `Camouflage-aware counting. Sample 0001. ${f.clear} clear-visibility annotations appear in teal.`],
    [3.5, 6.7, `${f.ambiguous} ambiguous-visibility annotations appear in gold.`],
    [6.7, 8.7, `${f.highlyCamouflaged} highly camouflaged annotations appear in coral. These are source labels, not model predictions.`],
    [8.7, 12, 'Source-image detail: four of the six highly camouflaged point annotations lie in this crop. Coral rings mark their original coordinates.'],
    [12, 14, `${f.clear} clear, ${f.ambiguous} ambiguous, and ${f.highlyCamouflaged} highly camouflaged. ${f.total} point annotations in total.`],
  ];
  return reasoningCases.flatMap((c, index) => {
    const start = index * 7;
    return [
      [start, start + 2, `Sample ${c.sampleId}, hotspot ${c.hotspotId}: localize the existing density response at (${c.point.join(', ')}).`],
      [start + 2, start + 4, `Visual evidence: ${c.evidenceLines.join(' ')} Source environment: ${c.semanticResult.semantic_label}. Relative-depth annotation: ${c.depthLabel.toLowerCase()}; not a metric measurement.`],
      [start + 4, start + 7, `Annotation interpretation: ${c.interpretationLines.join(' ')} This is a hypothesis, not a verified ecological cause.`],
    ];
  });
}

export function toVtt(cues) {
  const stamp = seconds => `00:00:${seconds.toFixed(3).padStart(6, '0')}`;
  return 'WEBVTT\n\n' + cues.map(([start, end, text], i) => `${i + 1}\n${stamp(start)} --> ${stamp(end)}\n${text}\n`).join('\n');
}
