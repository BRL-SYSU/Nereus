import { readFile, readdir, stat, access } from "node:fs/promises";
import { createRequire } from "node:module";
import { join, resolve, extname } from "node:path";
import assert from "node:assert/strict";
const require = createRequire(
  new URL("../website/package.json", import.meta.url),
);
const { parse } = require("parse5");
const root = new URL("../website/dist/", import.meta.url).pathname;
const html = await readFile(join(root, "index.html"), "utf8");
const tree = parse(html),
  nodes = [];
function visit(node) {
  nodes.push(node);
  for (const child of node.childNodes || []) visit(child);
}
visit(tree);
const ids = new Set(),
  links = [];
for (const node of nodes) {
  const a = Object.fromEntries(
    (node.attrs || []).map((x) => [x.name, x.value]),
  );
  if (a.id) {
    assert.ok(!ids.has(a.id), `Duplicate ID ${a.id}`);
    ids.add(a.id);
  }
  for (const key of ["src", "href", "poster"]) if (a[key]) links.push(a[key]);
  if (a.srcset)
    for (const entry of a.srcset.split(","))
      links.push(entry.trim().split(/\s+/)[0]);
  if (node.tagName === "img")
    assert.ok("alt" in a, "Image is missing alt text");
}
let internal = 0;
for (const link of links) {
  if (link.startsWith("#")) {
    assert.ok(ids.has(link.slice(1)), `Unknown anchor ${link}`);
    continue;
  }
  if (/^(https?:|mailto:|data:)/.test(link)) continue;
  assert.ok(link.startsWith("/Nereus/"), `Missing base path: ${link}`);
  const file = resolve(
    root,
    decodeURIComponent(
      link.slice("/Nereus/".length).split(/[?#]/)[0] || "index.html",
    ),
  );
  assert.ok(file.startsWith(root), `Path escape ${link}`);
  await access(file);
  internal++;
}
const files = [];
async function walk(dir) {
  for (const f of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, f.name);
    if (f.isDirectory()) await walk(p);
    else files.push(p);
  }
}
await walk(root);
let bytes = 0;
for (const path of files) {
  const info = await stat(path);
  bytes += info.size;
  assert.ok(info.size < 5 * 1024 * 1024, `Oversized file ${path}`);
  assert.ok(
    !/\.(pdf|npy|pt|pth|ckpt|tar|log|env)$/i.test(path),
    `Restricted artifact ${path}`,
  );
  if ([".html", ".css", ".js", ".json", ".txt"].includes(extname(path))) {
    const content = await readFile(path, "utf8");
    assert.doesNotMatch(
      content,
      /\/mnt\/nvme|\/data\/Jinzh|\/home\/jinzh|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|亮亮|周旗|乐乐/,
      `Private content in ${path}`,
    );
  }
}
assert.ok(bytes < 12 * 1024 * 1024, "Static site exceeds 12 MiB budget");
console.log(
  JSON.stringify(
    {
      files: files.length,
      bytes,
      internalResources: internal,
      anchors: ids.size,
      restrictedArtifacts: 0,
      privatePathMatches: 0,
    },
    null,
    2,
  ),
);
