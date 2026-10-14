import {readFile, writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
const require = createRequire(new URL('../website/package.json', import.meta.url));
const sharp = require('sharp');
const [counting, experiments] = process.argv.slice(2);
if (!counting || !experiments) throw new Error('Pass COUNTING_ROOT and EXPERIMENTS_ROOT');
const choices = [
  {sampleId:'0001', hotspotId:1, title:'Schooling in open water', crop:{x:960,y:80,w:800,h:450},
    evidenceLines:['Blue water around fish;', 'no visible bottom structure.'], interpretationLines:['Open swimming space', 'may support schooling', 'or temporary aggregation.']},
  {sampleId:'0003', hotspotId:1, title:'Aggregation near coral', crop:{x:550,y:290,w:800,h:450},
    evidenceLines:['Branching coral structures;', 'fish swimming nearby.'], interpretationLines:['Reef structure may offer', 'shelter and food, relating', 'to nearby aggregation.']},
];
const cases = [];
for (const choice of choices) {
  const bytes = await readFile(join(counting,'annotations/final_annotations',`${choice.sampleId}.json`));
  const record = JSON.parse(bytes);
  const h = record.task3_density_depth.hotspots.items.find(h=>h.hotspot_id===choice.hotspotId);
  const a = record.task3_density_depth.scene_attribution.items.find(h=>h.hotspot_id===choice.hotspotId);
  assert.deepEqual(h.point_xy,a.point_xy);
  const rawImage = join(counting,'images',`${choice.sampleId}.jpg`);
  const rawHeatmap = join(experiments,'canonical_dataset_folders',choice.sampleId,'density',`${choice.sampleId}_density_heatmap.png`);
  const meta = await sharp(rawImage).metadata();
  assert.equal(meta.width,record.image.width); assert.equal(meta.height,record.image.height);
  const image=`assets/counting/${choice.sampleId}-1920.webp`, heatmap=`assets/counting/${choice.sampleId}-density.webp`;
  if (choice.sampleId !== '0001') {
    await sharp(rawImage).resize({width:1920,withoutEnlargement:true}).webp({quality:82}).toFile(fileURLToPath(new URL(`../website/public/${image}`,import.meta.url)));
    const hm = await sharp(rawHeatmap).metadata(); assert.equal(hm.width,meta.width); assert.equal(hm.height,meta.height);
    await sharp(rawHeatmap).resize({width:1280}).webp({quality:90}).toFile(fileURLToPath(new URL(`../website/public/${heatmap}`,import.meta.url)));
  }
  const depthLabel = {'相对更深':'Relatively deeper','相对更浅':'Relatively shallower','相近':'Similar relative depth'}[a.depth_summary.relative_depth_label];
  assert.ok(depthLabel,'Translate a known relative-depth label, never infer meters');
  cases.push({...choice, image, heatmap, width:meta.width, height:meta.height, point:h.point_xy,
    densityRegion:h.region_bbox_xywh, selectedPatchType:a.selected_patch_type,
    contextPatch:{x:h.patch_origin_xy[0],y:h.patch_origin_xy[1],w:h.patch_size,h:h.patch_size},
    semanticResult:{semantic_label:a.semantic_result.semantic_label,evidence:a.semantic_result.evidence,ecological_hint:a.semantic_result.ecological_hint},
    depthSummary:a.depth_summary, depthLabel, causalChain:a.causal_chain,
    annotationSource:`annotations/final_annotations/${choice.sampleId}.json`,
    annotationSha256:createHash('sha256').update(bytes).digest('hex')});
}
const output={schemaVersion:1,license:'CC BY-NC 4.0',credit:'Guolei Sun et al., IOCfish5K; Nereus annotations',
  disclosure:'English display lines summarize original evidence, ecological_hint and causal_chain. Interpretations are hypotheses, not verified ecological causes. Depth labels are annotation-derived and relative, not metric measurements. Camera crops are presentation choices, not new annotations.',cases};
for (const path of ['../website/src/data/density-reasoning.json','../website/public/assets/videos/density-evidence.json']) await writeFile(new URL(path,import.meta.url),JSON.stringify(output,null,2)+'\n');
console.log(cases.map(c=>({sample:c.sampleId,hotspot:c.hotspotId,point:c.point,environment:c.semanticResult.semantic_label,depth:c.depthLabel})));
