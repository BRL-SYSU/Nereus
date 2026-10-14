import test from 'node:test';
import assert from 'node:assert/strict';
import * as story from '../../scripts/video/story.mjs';
import { readFileSync } from 'node:fs';

test('density film follows localized evidence into environmental interpretation', () => {
  assert.equal(typeof story.densityReasoningAt, 'function');
  const water = story.densityReasoningAt(6);
  const reef = story.densityReasoningAt(13);
  assert.equal(water.case.sampleId, '0001');
  assert.equal(reef.case.sampleId, '0003');
  assert.deepEqual(water.case.point, [1327, 215]);
  assert.deepEqual(reef.case.point, [886, 446]);
  assert.equal(water.case.semanticResult.semantic_label, 'open water');
  assert.equal(reef.case.semanticResult.semantic_label, 'coral reef');
  assert.match(water.case.semanticResult.ecological_hint, /schooling/);
  assert.match(reef.case.semanticResult.ecological_hint, /shelter and food/);
  assert.equal(water.showInterpretation, true);
  assert.equal(reef.showInterpretation, true);
  assert.equal(story.densityReasoningAt(1).showInterpretation, false);
  assert.equal(story.densityReasoningAt(7.5).showInterpretation, false);
  assert.equal(water.case.depthLabel, 'Relatively deeper');
  assert.equal(reef.case.depthLabel, 'Relatively shallower');
});

test('density captions include source-backed reasoning and its uncertainty', () => {
  const samples = JSON.parse(readFileSync(new URL('../src/data/samples.json', import.meta.url)));
  const sample = samples.find(s => s.id === '0001');
  const text = story.captionCues('density', story.deriveStories(samples)[2].annotationFacts, sample).map(c => c[2]).join(' ');
  assert.match(text, /schooling/i);
  assert.match(text, /shelter/i);
  assert.match(text, /interpretation/i);
  assert.match(text, /not.*caus/i);
  assert.doesNotMatch(text, /Hotspot 10/);
});
