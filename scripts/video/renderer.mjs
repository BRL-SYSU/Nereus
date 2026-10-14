import { frameState, ease, progress, gridCell, GRID, hotspotDetail, reasoningCases, densityReasoningAt } from './story.mjs';

const C = { paper: '#f4f8f7', ink: '#173b3b', muted: '#526e6c', teal: '#087f7c', mint: '#5ff5ce', coral: '#ed775f', gold: '#f2c453', line: '#d4e2df', white: '#ffffff' };
const V = { x: 32, y: 140, w: 864, h: 486 };
const colorFor = visibility => ({ clear: C.mint, ambiguous: C.gold, highly_camouflaged: C.coral })[visibility];

export async function createRenderer(canvas, samples, stories) {
  const ctx = canvas.getContext('2d', { alpha: false });
  const font = new FontFace('Manrope', 'url(/font.woff2)', { weight: '200 800' });
  document.fonts.add(await font.load());
  await document.fonts.ready;
  const images = {};
  const load = async src => { const image = new Image(); image.src = src; await image.decode(); return image; };
  for (const sample of samples.filter(s => ['0000', '0001'].includes(s.id))) {
    images[sample.id] = await load(`/assets/counting/${sample.id}-1920.webp`);
    images[`${sample.id}-heatmap`] = await load(`/${sample.heatmap}`);
  }
  let textBoxes = [];
  for (const c of reasoningCases) {
    images[c.sampleId] ||= await load(`/${c.image}`);
    images[`${c.sampleId}-heatmap`] ||= await load(`/${c.heatmap}`);
  }
  const text = (value, x, y, size = 26, color = C.ink, weight = 600, align = 'left') => {
    ctx.font = `${weight} ${size}px Manrope`;
    ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = 'alphabetic';
    ctx.fillText(String(value), x, y);
    const width = ctx.measureText(String(value)).width;
    textBoxes.push({ text: String(value), x: align === 'right' ? x - width : align === 'center' ? x - width / 2 : x, y: y - size, width, height: size });
  };
  const line = (x1, y1, x2, y2, color = C.line, width = 1) => {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
  };
  const circle = (x, y, r, color, width = 2.5) => {
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke();
  };
  const dot = (x, y, r, color) => { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); };
  const rect = (x, y, w, h, color) => { ctx.fillStyle = color; ctx.fillRect(x, y, w, h); };
  const metric = (label, value, y, color, total, active = true) => {
    ctx.globalAlpha = active ? 1 : 0.46;
    dot(938, y - 8, 6, color);
    text(label, 955, y, 24, C.ink);
    text(value, 1246, y + 57, 56, color, 800, 'right');
    rect(930, y + 75, 316, 5, C.line);
    rect(930, y + 75, 316 * value / total, 5, color);
    ctx.globalAlpha = 1;
  };
  const camera = (id, t, sample) => {
    let amount = 0; let target = [0, 0, sample.width, sample.height];
    if (id === 'region') {
      amount = ease(progress(t, 4.2, 5.4)) * (1 - ease(progress(t, 7, 7.8)));
      target = [1190, 0, 730, 410.625];
    } else if (id === 'camouflage') {
      amount = ease(progress(t, 8.7, 9.8)) * (1 - ease(progress(t, 11.7, 12.7)));
      target = [552, 620, 660, 371.25];
    } else {
      amount = ease(progress(t, 4.8, 7.5)) * (1 - ease(progress(t, 12.3, 13.5)));
      target = [320, 40, 1600, 900];
    }
    return { x: target[0] * amount, y: target[1] * amount,
      w: sample.width + (target[2] - sample.width) * amount,
      h: sample.height + (target[3] - sample.height) * amount };
  };
  const drawPhoto = (image, sample, crop, alpha = 1, viewport = V) => {
    ctx.globalAlpha = alpha;
    const scaleX = image.width / sample.width, scaleY = image.height / sample.height;
    ctx.drawImage(image, crop.x * scaleX, crop.y * scaleY, crop.w * scaleX, crop.h * scaleY, viewport.x, viewport.y, viewport.w, viewport.h);
    ctx.globalAlpha = 1;
  };
  const header = (story, state, t) => {
    rect(0, 0, 1280, 720, C.paper);
    rect(32, 30, 7, 24, C.teal);
    text('NEREUS', 51, 50, 22, C.teal, 800);
    text('ANNOTATION STUDIES', 186, 50, 18, C.muted, 600);
    text(`SAMPLE ${story.sampleIds[0]}`, 1248, 50, 18, C.muted, 700, 'right');
    const titles = { region: 'Where are the fish?', camouflage: 'When visibility changes', density: 'From density to context' };
    text(titles[story.id], 32, 107, 44, C.ink, 800);
    const phases = { grid: '01 / LOCATE', focus: '02 / FOCUS', water: '03 / COMPARE', regions: '03 / COMPARE', clear: '01 / CLEAR', ambiguous: '02 / AMBIGUOUS', camouflaged: '03 / CAMOUFLAGED', detail: '04 / LOOK CLOSER', summary: 'ANNOTATION SUMMARY', points: '01 / POINTS', heatmap: '02 / DENSITY', hotspots: '03 / HOTSPOTS', 'water-detail': '04 / CONTEXT', 'reef-detail': '04 / CONTEXT' };
    text(phases[state.phase], 1248, 105, 18, C.teal, 800, 'right');
    line(912, 140, 912, 626);
    rect(0, 714, 1280, 6, C.line); rect(0, 714, 1280 * t / 14, 6, C.teal);
  };
  const footer = (main, sub) => {
    text(main, 32, 662, 26, C.ink, 700);
    text(sub, 32, 694, 17, C.muted, 500);
  };
  const sourceTag = (label, color = C.teal) => {
    const width = label.length * 11 + 30;
    rect(V.x + 14, V.y + V.h - 44, width, 30, C.paper);
    text(label, V.x + 28, V.y + V.h - 23, 17, color, 800);
  };

  function renderDensity(t) {
    const state = densityReasoningAt(t);
    const c = state.case, u = state.localTime;
    const accent = state.index ? C.coral : C.gold;
    const zoom = ease(progress(u, 1.5, 3.3));
    const crop = {x:c.crop.x * zoom, y:c.crop.y * zoom,
      w:c.width + (c.crop.w-c.width)*zoom, h:c.height + (c.crop.h-c.height)*zoom};
    const project = ([x,y]) => ({x:V.x+(x-crop.x)/crop.w*V.w,y:V.y+(y-crop.y)/crop.h*V.h});
    rect(0,0,1280,720,C.paper);
    rect(32,30,7,24,C.teal);
    text('NEREUS',51,50,22,C.teal,800);
    text('GROUNDED REASONING',186,50,18,C.muted);
    text(`SAMPLE ${c.sampleId} / HOTSPOT ${c.hotspotId}`,1248,50,18,C.muted,700,'right');
    text('From density to grounded reasoning',32,107,40,C.ink,800);
    text(`${state.index+1} / 2  ${state.phase}`,1248,105,18,C.teal,800,'right');
    ctx.save(); ctx.beginPath(); ctx.rect(V.x,V.y,V.w,V.h); ctx.clip();
    drawPhoto(images[c.sampleId],c,crop);
    drawPhoto(images[`${c.sampleId}-heatmap`],c,crop,
      0.8*ease(progress(u,0,0.6))*(1-ease(progress(u,1.6,2.7))));
    const p=project(c.point);
    circle(p.x,p.y,13+3*Math.sin(u*2.5),accent,3);
    circle(p.x,p.y,4,C.white,2);
    const box=c.contextPatch, a=project([box.x,box.y]);
    ctx.globalAlpha=ease(progress(u,2.2,3.2));
    ctx.setLineDash([8,6]); ctx.strokeStyle=accent; ctx.lineWidth=2;
    ctx.strokeRect(a.x,a.y,box.w/crop.w*V.w,box.h/crop.h*V.h);
    ctx.setLineDash([]); ctx.globalAlpha=1;
    ctx.restore();
    sourceTag(u < 2.7 ? 'SOURCE DENSITY + HOTSPOT' : 'RGB EVIDENCE / SOURCE PATCH');
    line(912,140,912,626);
    const step=(label,lines,y,start) => {
      const alpha=ease(progress(u,start,start+0.5));
      ctx.globalAlpha=alpha;
      const offset=(1-alpha)*9;
      text(label,932,y+offset,16,C.teal,800);
      lines.forEach((value,i)=>text(value,932,y+37+i*30+offset,22,C.ink,650));
      ctx.globalAlpha=1;
    };
    text('LOCALIZED HOTSPOT',932,168,17,C.muted,800);
    text(`(${c.point.join(', ')})`,932,205,26,C.ink,800);
    line(932,228,1248,228);
    step('01  VISUAL EVIDENCE',c.evidenceLines,259,1.5);
    step('02  ENVIRONMENT + DEPTH',[c.semanticResult.semantic_label,c.depthLabel+' *'],381,2.5);
    step('03  ANNOTATION INTERPRETATION',c.interpretationLines,503,4);
    footer(c.title,'Interpretations are hypotheses, not verified causes. * Source relative-depth tag, not metric depth.');
    rect(0,714,1280,6,C.line); rect(0,714,1280*t/14,6,C.teal);
    return {state,textBoxes,crop};
  }

  function render(id, t) {
    textBoxes = [];
    if (id === 'density') return renderDensity(t);
    const story = stories.find(s => s.id === id);
    const sample = samples.find(s => s.id === story.sampleIds[0]);
    const state = frameState(id, t, sample);
    const f = story.annotationFacts;
    const crop = camera(id, t, sample);
    const project = p => ({ x: V.x + (p.x - crop.x) / crop.w * V.w, y: V.y + (p.y - crop.y) / crop.h * V.h });
    header(story, state, t);
    ctx.save(); ctx.beginPath(); ctx.rect(V.x, V.y, V.w, V.h); ctx.clip();
    drawPhoto(images[sample.id], sample, crop);
    if (id === 'density') drawPhoto(images[`${sample.id}-heatmap`], sample, crop, ease(progress(t, 3.2, 4.6)));

    if (id === 'region' && t < 7.8) {
      ctx.globalAlpha = ease(progress(t, 0.3, 1));
      for (let i = 1; i < 3; i++) {
        const a = project({ x: sample.width * i / 3, y: 0 });
        const b = project({ x: 0, y: sample.height * i / 3 });
        line(a.x, V.y, a.x, V.y + V.h, '#ffffffa0', 1.5);
        line(V.x, b.y, V.x + V.w, b.y, '#ffffffa0', 1.5);
      }
      const a = project({ x: sample.width * 2 / 3, y: 0 });
      const b = project({ x: sample.width, y: sample.height / 3 });
      if (t >= 3.7) {
        ctx.strokeStyle = C.gold; ctx.lineWidth = 4;
        ctx.strokeRect(a.x + 2, a.y + 2, b.x - a.x - 4, b.y - a.y - 4);
      }
      ctx.globalAlpha = 1;
    }
    const pointAlpha = id === 'density' ? 1 - ease(progress(t, 3.2, 4.3)) : 1;
    for (const [index, point] of state.points.entries()) {
      const p = project(point);
      if (p.x < V.x - 15 || p.x > V.x + V.w + 15 || p.y < V.y - 15 || p.y > V.y + V.h + 15) continue;
      let color = C.mint; let alpha = pointAlpha; let radius = 4;
      if (id === 'region') {
        if (state.phase === 'focus' && gridCell(point, sample) !== 'top-right') alpha = 0.18;
        if (t >= 7.8) color = point.regions.includes('open_water') ? C.mint : point.regions.includes('reef_rock') ? C.coral : C.gold;
        radius = state.phase === 'focus' ? 6 : 4.5;
      }
      if (id === 'camouflage') {
        color = colorFor(point.visibility);
        if (state.phase === 'detail' && point.visibility !== 'highly_camouflaged') alpha = 0.12;
        if (point.visibility === 'highly_camouflaged') radius = state.phase === 'detail' ? 12 : 7;
      }
      ctx.globalAlpha = alpha;
      // Coordinates stay attached to source pixels during every camera move.
      circle(p.x, p.y, radius, color, 2);
      dot(p.x, p.y, 1.5, C.white);
      if (index >= state.points.length - 3 && t < 3.7) circle(p.x, p.y, radius + 4, color, 1);
    }
    ctx.globalAlpha = 1;
    if (id === 'region' && state.phase === 'grid') {
      GRID.forEach((key, i) => {
        const p = project({ x: (i % 3) * sample.width / 3, y: Math.floor(i / 3) * sample.height / 3 });
        rect(p.x + 12, p.y + 12, 52, 39, key === 'top-right' && t >= 3.7 ? C.gold : C.paper);
        text(state.grid[key], p.x + 38, p.y + 41, 27, C.ink, 800, 'center');
      });
    }
    if (id === 'density') {
      state.hotspots.forEach(h => {
        const p = project(h);
        const color = h.label === 'coral reef' ? C.coral : C.gold;
        circle(p.x, p.y, h.id === 1 || h.id === 10 ? 12 : 6, color, 2.5);
        if (h.id === 1 || h.id === 10) {
          line(p.x, p.y - 12, p.x, p.y - 36, color, 2);
          rect(p.x - 19, p.y - 69, 38, 32, color);
          text(h.id, p.x, p.y - 45, 21, C.ink, 800, 'center');
        }
      });
    }
    ctx.restore();

    if (id === 'region') {
      if (t < 7.8) {
        text(t < 4.2 ? 'POINT ANNOTATIONS' : 'DENSEST GRID CELL', 932, 170, 18, C.muted, 800);
        text(t < 4.2 ? state.totalVisible : f.densestCount, 928, 277, 104, t < 4.2 ? C.teal : C.ink, 800);
        text(t < 4.2 ? 'fish located' : 'top-right', 932, 318, 30, C.ink, 700);
        line(930, 342, 1246, 342);
        const cell = 82, gap = 15;
        GRID.forEach((key, i) => {
          const x = 943 + i % 3 * (cell + gap), y = 377 + Math.floor(i / 3) * 70;
          rect(x, y, cell, 55, key === 'top-right' ? C.gold : '#e3edeb');
          text(state.grid[key], x + cell / 2, y + 39, 30, C.ink, 800, 'center');
        });
        text('3 x 3 spatial counts', 932, 610, 21, C.muted);
        footer(t < 4.2 ? 'Every marker is a source point annotation.' : 'Which cell is densest? Top-right, with 31 fish.', 'IOCfish5K + Nereus annotations / Sample 0000 / Annotation-derived, not a prediction');
      } else {
        text('SEMANTIC REGION LABELS', 932, 170, 18, C.muted, 800);
        metric('Open water', state.regions.open_water, 224, C.teal, f.total);
        metric('Reef rock', state.regions.reef_rock, 377, C.coral, f.total, t >= 10.6);
        if (t >= 11.9) {
          text(`+${f.difference}`, 932, 572, 58, C.ink, 800);
          text('more in open water', 932, 608, 24, C.muted);
        }
        footer('82 open-water labels. 5 reef-rock labels.', 'Selected region labels; other categories are not included in this comparison.');
      }
      sourceTag(t >= 7.8 ? 'SOURCE IMAGE + REGION LABELS' : 'SOURCE IMAGE + TRUE POINTS');
    }
    if (id === 'camouflage') {
      text('VISIBILITY ANNOTATIONS', 932, 170, 18, C.muted, 800);
      metric('Clear', state.visibility.clear, 216, C.teal, f.total);
      metric('Ambiguous', state.visibility.ambiguous, 356, '#a77910', f.total, t >= 3.5);
      metric('Highly camouflaged', state.visibility.highly_camouflaged, 496, '#c75440', f.total, t >= 6.7);
      text(`${state.totalVisible} / ${f.total} points`, 932, 619, 21, C.muted, 700);
      sourceTag(state.phase === 'detail' ? 'SOURCE DETAIL / 4 OF 6' : 'SOURCE IMAGE + VISIBILITY');
      footer(state.phase === 'detail' ? 'A closer look: four of the six highly camouflaged points.' : state.phase === 'summary' ? '134 clear + 130 ambiguous + 6 highly camouflaged = 270.' : 'Same scene. Three annotated visibility categories.', 'IOCfish5K + Nereus annotations / Sample 0001 / Source labels, not model predictions');
    }
    if (id === 'density') {
      text(t < 5 ? 'ANNOTATED FISH' : 'ANNOTATED HOTSPOTS', 932, 170, 18, C.muted, 800);
      text(t < 5 ? state.totalVisible : state.hotspotCount, 928, 267, 94, C.teal, 800);
      text(t < 5 ? 'point locations' : 'habitat labels', 932, 306, 27, C.ink, 700);
      if (t >= 5) {
        line(930, 324, 1246, 324);
        dot(939, 352, 6, C.gold); text('Open water', 958, 360, 25);
        text(state.hotspots.filter(h => h.label === 'open water').length, 1246, 361, 32, C.ink, 800, 'right');
        dot(939, 394, 6, C.coral); text('Coral reef', 958, 402, 25);
        text(state.hotspots.filter(h => h.label === 'coral reef').length, 1246, 403, 32, C.ink, 800, 'right');
      } else {
        text('Existing density map', 932, 378, 24, C.muted);
        text('No new prediction', 932, 412, 24, C.muted);
      }
      if (t >= 8.2) {
        const { hotspot, crop: context } = hotspotDetail(sample, t < 10.7 ? 1 : 10);
        drawPhoto(images[sample.id], sample, context, 1, { x: 930, y: 446, w: 316, h: 177.75 });
        const color = hotspot.id === 1 ? C.gold : C.coral;
        const [x1, y1, x2, y2] = hotspot.box;
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.strokeRect(930 + (x1 - context.x) / context.w * 316, 446 + (y1 - context.y) / context.h * 177.75, (x2 - x1) / context.w * 316, (y2 - y1) / context.h * 177.75);
        rect(940, 454, 133, 27, C.paper);
        text(`SOURCE / ${hotspot.id}`, 950, 473, 16, C.ink, 800);
      }
      sourceTag(t < 3.7 ? 'SOURCE IMAGE + TRUE POINTS' : 'EXISTING DENSITY HEATMAP');
      footer(t < 5 ? 'From point locations to the existing density heatmap.' : t < 8.2 ? '10 hotspots: 9 open water, 1 coral reef.' : t < 12.8 ? hotspotDetail(sample, t < 10.7 ? 1 : 10).label : 'Visual habitat context, not an ecological cause.', 'IOCfish5K + Nereus annotations / Sample 0001 / Hotspot labels are not fish counts');
    }
    return { state, textBoxes, crop };
  }
  return { render };
}
