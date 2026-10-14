import publication from "../data/publication.json";
import science from "../data/science.json";

export const bibtex = `@misc{deng2026nereus,
  title = {${science.title}},
  author = {${publication.authors.map((a) => `${a.family}, ${a.given}`).join(" and ")}},
  year = {${publication.year}},
  url = {${publication.codeUrl}},
  note = {Research manuscript; final publication details forthcoming}
}`;
