import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
const file = new URL("../src/data/publication.json", import.meta.url);
test("author order, affiliation and contribution roles match owner confirmation", () => {
  assert.ok(existsSync(file), "publication metadata has not been added");
  const p = JSON.parse(readFileSync(file, "utf8"));
  assert.deepEqual(
    p.authors.map((a) => a.name),
    [
      "Jingtao Deng",
      "Zihao Jin",
      "Yuehe Chen",
      "Feixiang He",
      "Shuiwang Li",
      "Dan Zeng",
    ],
  );
  assert.deepEqual(
    p.authors.map((a) => a.affiliation),
    [1, 1, 1, 2, 3, 1],
  );
  assert.deepEqual(
    p.authors.filter((a) => a.equalContribution).map((a) => a.name),
    ["Jingtao Deng", "Zihao Jin"],
  );
  assert.deepEqual(
    p.authors.filter((a) => a.corresponding).map((a) => a.name),
    ["Dan Zeng"],
  );
  assert.deepEqual(
    p.affiliations.map((a) => a.name),
    [
      "Sun Yat-sen University",
      "Central South University",
      "Guilin University of Technology",
    ],
  );
  assert.equal(p.paperUrl, null);
  assert.equal(p.arxivUrl, null);
});
