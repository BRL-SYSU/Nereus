import samples from "../data/samples.json";
import { asset } from "./paths";
let selected = 0,
  mode = "points";
let renderVersion = 0;
const $ = <T extends Element = HTMLElement>(selector: string) =>
  document.querySelector<T>(selector)!;
const colors: Record<string, string> = {
  clear: "#78e4c9",
  ambiguous: "#ffdb76",
  highly_camouflaged: "#ff907e",
};
const regionColors = ["#fcab8f", "#ffdb76", "#c5b7ff", "#78e4c9", "#dfeaf0"];
const labels: Record<string, string> = {
  clear: "Clear",
  ambiguous: "Ambiguous",
  highly_camouflaged: "Highly camouflaged",
};
function text(tag: string, value: string) {
  const el = document.createElement(tag);
  el.textContent = value;
  return el;
}
function legend(items: { label: string; color: string }[]) {
  const el = $("#mode-legend");
  el.replaceChildren();
  items.forEach((item) => {
    const group = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.background = item.color;
    group.append(swatch, document.createTextNode(item.label));
    el.append(group);
  });
}
function detail(title: string, body: string) {
  $("#mode-detail").replaceChildren(text("h3", title), text("p", body));
}
function renderQA() {
  const s = samples[selected],
    qa = s.qa[Number($<HTMLSelectElement>("#qa-select").value) || 0];
  $("#qa-question").textContent = qa.question;
  $("#qa-answer").textContent = qa.answer;
}
async function render() {
  const s = samples[selected];
  const version = ++renderVersion,
    stage = $<HTMLElement>(".evidence-stage");
  stage.classList.add("is-loading");
  stage.setAttribute("aria-busy", "true");
  stage.dataset.loading = "Loading evidence…";
  $("#retry-evidence").setAttribute("hidden", "");
  try {
    const rgb = new Image(),
      density = new Image();
    rgb.src = asset(s.image);
    density.src = asset(s.heatmap);
    await Promise.all([rgb.decode(), density.decode()]);
  } catch {
    if (version !== renderVersion) return;
    stage.dataset.loading = "Evidence could not be loaded.";
    stage.setAttribute("aria-busy", "false");
    $("#retry-evidence").removeAttribute("hidden");
    return;
  }
  if (version !== renderVersion) return;
  document
    .querySelectorAll<HTMLElement>("[data-sample]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.sample === s.id)),
    );
  document
    .querySelectorAll<HTMLElement>("[data-mode]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.mode === mode)),
    );
  const img = $<HTMLImageElement>("#evidence-image");
  img.src = asset(s.image);
  img.alt = `${s.name}, IOCfish5K sample ${s.id}`;
  const heatmap = $<HTMLImageElement>("#density-image");
  heatmap.src = asset(s.heatmap);
  heatmap.hidden = mode !== "density";
  $("#sample-stamp").textContent = `IOCfish5K / ${s.id}`;
  $("#sample-count").textContent = String(s.count);
  const grid = $("#sample-grid");
  grid.replaceChildren();
  const max = Math.max(...Object.values(s.grid));
  Object.entries(s.grid).forEach(([k, v]) => {
    const cell = document.createElement("div");
    cell.classList.toggle("is-densest", v === max);
    cell.setAttribute("aria-label", `${k}: ${v}`);
    cell.append(
      text("span", String(v)),
      text("small", k.replace("center-", "mid-")),
    );
    grid.append(cell);
  });
  const svg = $<SVGElement>("#evidence-layer");
  let shapes = "";
  svg.setAttribute(
    "aria-label",
    `${mode} evidence for ${s.count} fish in sample ${s.id}`,
  );
  if (mode === "points" || mode === "visibility") {
    shapes = s.points
      .map(
        (p) =>
          `<circle cx="${p.x}" cy="${p.y}" r="${mode === "visibility" ? 8 : 6}" fill="${mode === "visibility" ? colors[p.visibility] : "#ffdb76"}" stroke="#173d43" stroke-width="2"/>`,
      )
      .join("");
    if (mode === "visibility") {
      legend(
        Object.entries(s.visibility).map(([k, v]) => ({
          label: `${labels[k]} · ${v}`,
          color: colors[k],
        })),
      );
      $("#mode-caption").textContent =
        "Dataset visibility labels, propagated to individual counting points.";
      detail("Camouflage and visibility", s.visibilityExplanation);
    } else {
      legend([{ label: "Ground-truth point", color: "#ffdb76" }]);
      $("#mode-caption").textContent =
        "Original ground-truth points; one point per annotated fish.";
      const densest = Object.entries(s.grid)
        .filter(([, v]) => v === max)
        .map(([k]) => k)
        .join(" and ");
      detail(
        "Beyond the total",
        `The ${densest} cell${densest.includes(" and ") ? "s each contain" : " contains"} ${max} of ${s.count} points. Spatial supervision preserves where the count comes from.`,
      );
    }
  } else if (mode === "regions") {
    for (let i = 1; i < 3; i++)
      shapes += `<path d="M ${(i * s.width) / 3} 0 V ${s.height} M 0 ${(i * s.height) / 3} H ${s.width}" stroke="#ffffff" stroke-opacity="0.7" stroke-width="3" stroke-dasharray="12 9"/>`;
    const entries = Object.entries(s.boxes).filter(([, v]) => v.length);
    entries.forEach(([, boxes], i) =>
      boxes.forEach(
        (b) =>
          (shapes += `<rect x="${b[0]}" y="${b[1]}" width="${b[2] - b[0]}" height="${b[3] - b[1]}" fill="${regionColors[i]}" fill-opacity="0.12" stroke="${regionColors[i]}" stroke-width="4"/>`),
      ),
    );
    legend(
      entries.map(([k], i) => ({
        label: k.replaceAll("_", " "),
        color: regionColors[i],
      })),
    );
    $("#mode-caption").textContent =
      "Recorded semantic-region boxes and equal-area grid, not pixel masks.";
    detail(
      "Region-aware counting",
      Object.entries(s.regionCounts)
        .filter(([, v]) => v > 0)
        .map(([k, v]) => `${k.replaceAll("_", " ")}: ${v}`)
        .join("; ") + ". Region memberships may overlap and are not additive.",
    );
  } else if (mode === "density") {
    legend([
      { label: "Low response", color: "#1832c5" },
      { label: "High response", color: "#d93228" },
    ]);
    $("#mode-caption").textContent =
      "Precomputed density heatmap derived from ground-truth points.";
    detail(
      "Supplied density evidence",
      "The colors encode the stored density response, not a model prediction or a calibrated fish count at each pixel. Track B supplies this evidence to the model alongside RGB.",
    );
  } else if (mode === "hotspots") {
    s.hotspots.forEach((h) => {
      if (h.box) {
        const b = h.box;
        shapes += `<rect x="${b[0]}" y="${b[1]}" width="${b[2] - b[0]}" height="${b[3] - b[1]}" fill="#ffdb76" fill-opacity="0.1" stroke="#ffdb76" stroke-width="3"/>`;
      }
      shapes += `<circle cx="${h.x}" cy="${h.y}" r="17" fill="#ffdb76" stroke="#173d43" stroke-width="2"/><text x="${h.x}" y="${h.y + 6}" text-anchor="middle" fill="#173d43" font-size="19" font-family="sans-serif" font-weight="700">${h.id}</text>`;
    });
    legend([{ label: "Recorded hotspot + SAM box", color: "#ffdb76" }]);
    $("#mode-caption").textContent =
      `${s.hotspots.length} recorded density hotspots with local SAM bounding boxes.`;
    const el = $("#mode-detail");
    el.replaceChildren(text("h3", "Local scene evidence"));
    const label = text("label", "Hotspot");
    label.setAttribute("for", "hotspot-select");
    const select = document.createElement("select");
    select.id = "hotspot-select";
    s.hotspots.forEach((h) => {
      const o = document.createElement("option");
      o.value = String(h.id);
      o.textContent = `${h.id} · ${h.label}`;
      select.append(o);
    });
    const p = text("p", s.hotspots[0]?.evidence || "No recorded hotspots.");
    select.addEventListener("change", () => {
      p.textContent =
        s.hotspots.find((h) => h.id === Number(select.value))?.evidence || "";
    });
    el.append(label, select, p);
  } else {
    legend([]);
    $("#mode-caption").textContent =
      "Original RGB image, uncropped; no annotation layer.";
    detail(
      "A challenging visual scene",
      "Distance, background texture and occlusion affect visibility. The other views expose the annotation evidence attached to this exact image.",
    );
  }
  svg.innerHTML = shapes;
  renderQA();
  stage.classList.remove("is-loading");
  stage.setAttribute("aria-busy", "false");
}
document.querySelectorAll<HTMLElement>("[data-mode]").forEach((b) =>
  b.addEventListener("click", () => {
    mode = b.dataset.mode!;
    render();
  }),
);
document.querySelectorAll<HTMLElement>("[data-sample]").forEach((b) =>
  b.addEventListener("click", () => {
    selected = samples.findIndex((s) => s.id === b.dataset.sample);
    const select = $<HTMLSelectElement>("#qa-select");
    select.replaceChildren(
      ...samples[selected].qa.map((_, i) => {
        const o = document.createElement("option");
        o.value = String(i);
        o.textContent = `Question ${i + 1}`;
        return o;
      }),
    );
    render();
    renderQA();
  }),
);
$("#qa-select").addEventListener("change", renderQA);
$("#retry-evidence").addEventListener("click", () => void render());
render();
