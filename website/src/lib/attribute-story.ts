import samples from "../data/attribute-samples.json";
import { asset } from "./paths";

const highlights: Record<string, string[]> = {
  fish: [
    "species",
    "body_shape_lateral",
    "color",
    "texture",
    "camouflage",
    "body_damage",
  ],
  cuttlefish: [
    "cephalopod_group",
    "mantle_shape",
    "dominant_color",
    "skin_pattern",
    "camouflage_visibility",
    "posture",
  ],
  coral: [
    "coral_group",
    "growth_form",
    "color",
    "surface_texture",
    "bleaching_state",
    "tissue_condition",
  ],
};
function category(key: string) {
  if (["species", "diagnosis", "cephalopod_group", "coral_group"].includes(key))
    return 0;
  if (
    [
      "color",
      "dominant_color",
      "texture",
      "skin_pattern",
      "skin_relief",
      "colony_relief",
      "surface_texture",
    ].includes(key)
  )
    return 2;
  if (
    [
      "body_damage",
      "bleaching_state",
      "tissue_condition",
      "algal_overgrowth",
      "sediment_cover",
      "coral_preciousness_grade",
    ].includes(key)
  )
    return 3;
  if (
    [
      "camouflage",
      "camouflage_visibility",
      "posture",
      "interaction",
      "occlusion",
      "neighboring_contact",
      "corallite_visibility",
      "sucker_visibility_type",
    ].includes(key)
  )
    return 4;
  return 1;
}

export function initAttributeStory() {
  const root = document.querySelector<HTMLElement>("#attribute-story");
  if (!root) return;
  const get = <T extends HTMLElement = HTMLElement>(id: string) =>
    root.querySelector<T>(`#${id}`)!;
  const image = get<HTMLImageElement>("specimen-image");
  const plane = get("specimen-plane"),
    box = get("specimen-box");
  const progress = get<HTMLInputElement>("attribute-progress");
  const play = get<HTMLButtonElement>("attribute-play");
  const zoom = get<HTMLButtonElement>("attribute-zoom");
  const values = get("attribute-values");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const tabs = Array.from(
    root.querySelectorAll<HTMLButtonElement>("[data-perspective]"),
  );
  let selected = 0,
    perspective = -1,
    time = reduced.matches ? 14 : 0;
  let playing = false,
    visible = false,
    ready = false,
    started = false,
    zoomed = false;
  let request = 0,
    frame = 0,
    previous = 0;
  root.dataset.enhanced = "true";

  function renderFields() {
    const s = samples[selected];
    const fields = s.record.objects[0].attributes as Record<string, string>;
    const entries =
      perspective === -1
        ? highlights[s.id].filter((k) => k in fields).map((k) => [k, fields[k]])
        : Object.entries(fields).filter(
            ([k]) => perspective === -2 || category(k) === perspective,
          );
    values.replaceChildren(
      ...entries.map(([k, v]) => {
        const row = document.createElement("div"),
          dt = document.createElement("dt"),
          dd = document.createElement("dd");
        dt.textContent = k.replaceAll("_", " ");
        dd.textContent = v;
        row.append(dt, dd);
        return row;
      }),
    );
    if (!entries.length) {
      const p = document.createElement("p");
      p.textContent = "This category is not shown in this source annotation.";
      values.append(p);
    }
    tabs.forEach((tab) => {
      const active = Number(tab.dataset.perspective) === perspective;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active)
        get("attribute-panel").setAttribute("aria-labelledby", tab.id);
    });
  }
  function paint() {
    progress.value = String(Math.round(time * 10) / 10);
    get("attribute-time").textContent =
      `${String(Math.floor(time)).padStart(2, "0")} / 14`;
    box.hidden = !ready || time < 2;
    get("specimen-phase").textContent =
      time < 2
        ? "01 / Source image"
        : time < 4
          ? "02 / Annotated target"
          : time < 6
            ? "03 / Target detail"
            : "04 / Attribute evidence";
    const useZoom =
      ready && (zoomed || (!reduced.matches && time >= 4 && time < 13));
    const s = samples[selected].record;
    const [x, y, w, h] = s.objects[0].bbox;
    const scale = useZoom
      ? Math.max(1, Math.min(1.6, 0.92 / Math.max(w / s.width, h / s.height)))
      : 1;
    const tx = useZoom
      ? Math.max(
          1 - scale,
          Math.min(0, 0.5 - ((x + w / 2) / s.width) * scale),
        ) * 100
      : 0;
    const ty = useZoom
      ? Math.max(
          1 - scale,
          Math.min(0, 0.5 - ((y + h / 2) / s.height) * scale),
        ) * 100
      : 0;
    plane.style.transform = `translate(${tx}%, ${ty}%) scale(${scale})`;
    zoom.setAttribute("aria-pressed", String(useZoom));
    Array.from(values.children).forEach((row, i) => {
      const shown = ready && (time >= 13 || time >= 6 + i * 0.75);
      (row as HTMLElement).style.opacity = shown ? "1" : "0";
      (row as HTMLElement).style.transform = shown ? "none" : "translateY(8px)";
      row.setAttribute("aria-hidden", String(!shown));
    });
  }
  function stop() {
    playing = false;
    cancelAnimationFrame(frame);
    frame = 0;
    play.setAttribute("aria-label", "Play attribute sequence");
    root!.dataset.playing = "false";
  }
  function tick(now: number) {
    if (!playing) return;
    time = Math.min(14, time + Math.min((now - previous) / 1000, 0.1));
    previous = now;
    paint();
    if (time >= 14) stop();
    else frame = requestAnimationFrame(tick);
  }
  function start() {
    if (!ready) return;
    if (time >= 14) time = 0;
    stop();
    playing = true;
    started = true;
    previous = performance.now();
    play.setAttribute("aria-label", "Pause attribute sequence");
    root!.dataset.playing = "true";
    frame = requestAnimationFrame(tick);
  }
  async function load(index: number) {
    const token = ++request;
    stop();
    ready = false;
    selected = index;
    zoomed = false;
    time = reduced.matches || perspective !== -1 ? 14 : 0;
    root!.setAttribute("aria-busy", "true");
    image.style.visibility = "hidden";
    box.hidden = true;
    get("specimen-loading").hidden = false;
    get("specimen-retry").hidden = true;
    get("specimen-loading").textContent = "Loading specimen...";
    const sample = samples[index],
      record = sample.record,
      object = record.objects[0];
    root!
      .querySelectorAll("[data-specimen]")
      .forEach((b, i) => b.setAttribute("aria-pressed", String(i === index)));
    get("specimen-name").textContent = object.concept;
    get("specimen-group").textContent =
      `${object.group} · ${Object.keys(object.attributes).length} annotated attributes`;
    get("specimen-filename").textContent = record.image;
    get("specimen-json").textContent = JSON.stringify(record, null, 2);
    get<HTMLAnchorElement>("specimen-download").href = asset(sample.annotation);
    get<HTMLAnchorElement>("specimen-source").href = sample.provenance.source;
    get("specimen-credit").textContent = sample.credit;
    renderFields();
    paint();
    try {
      const pending = new Image();
      pending.src = asset(sample.image);
      await pending.decode();
      if (token !== request) return;
      image.src = pending.src;
      image.width = record.width;
      image.height = record.height;
      image.alt = `Nereus annotated ${object.concept} specimen`;
      await image.decode();
      if (token !== request) return;
      plane.style.aspectRatio = `${record.width}/${record.height}`;
      const [x, y, w, h] = object.bbox;
      Object.assign(box.style, {
        left: `${(x / record.width) * 100}%`,
        top: `${(y / record.height) * 100}%`,
        width: `${(w / record.width) * 100}%`,
        height: `${(h / record.height) * 100}%`,
      });
      ready = true;
      image.style.visibility = "visible";
      get("specimen-loading").hidden = true;
      root!.setAttribute("aria-busy", "false");
      paint();
      if (visible && perspective === -1 && !reduced.matches) start();
    } catch {
      if (token !== request) return;
      get("specimen-loading").textContent = "Image unavailable";
      get("specimen-retry").hidden = false;
      root!.setAttribute("aria-busy", "false");
    }
  }
  root.querySelectorAll<HTMLButtonElement>("[data-specimen]").forEach((b, i) =>
    b.addEventListener("click", () => {
      started = false;
      void load(i);
    }),
  );
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => {
      stop();
      started = true;
      perspective = Number(tab.dataset.perspective);
      time = 14;
      renderFields();
      paint();
    });
    tab.addEventListener("keydown", (e) => {
      const direction = ["ArrowRight", "ArrowDown"].includes(e.key)
        ? 1
        : ["ArrowLeft", "ArrowUp"].includes(e.key)
          ? -1
          : 0;
      const next = direction
        ? (i + direction + tabs.length) % tabs.length
        : e.key === "Home"
          ? 0
          : e.key === "End"
            ? tabs.length - 1
            : -1;
      if (next < 0) return;
      e.preventDefault();
      tabs[next].click();
      tabs[next].focus();
    });
  });
  play.addEventListener("click", () => (playing ? stop() : start()));
  get("attribute-replay").addEventListener("click", () => {
    time = 0;
    perspective = -1;
    zoomed = false;
    renderFields();
    paint();
    start();
  });
  get("attribute-all").addEventListener("click", () => {
    stop();
    started = true;
    time = 14;
    perspective = -2;
    zoomed = false;
    renderFields();
    paint();
  });
  zoom.addEventListener("click", () => {
    stop();
    started = true;
    time = 14;
    zoomed = !zoomed;
    paint();
  });
  progress.addEventListener("input", () => {
    stop();
    started = true;
    time = Number(progress.value);
    zoomed = false;
    paint();
  });
  get("specimen-retry").addEventListener("click", () => {
    void load(selected);
  });
  new IntersectionObserver(
    (entries) => {
      visible = entries[0].isIntersecting;
      if (!visible) stop();
      else if (ready && !started && perspective === -1 && !reduced.matches)
        start();
    },
    { threshold: 0.15 },
  ).observe(root);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
  });
  reduced.addEventListener("change", () => {
    stop();
    time = 14;
    zoomed = false;
    paint();
  });
  void load(0);
}
