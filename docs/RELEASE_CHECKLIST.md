# Owner Review and Release Checklist

## Scientific and Permission Decisions

- [ ] Approve visible venue/acceptance statement (owner-supplied).
- [x] Supply approved author order, affiliations and contribution roles (2026-10-13).
- [ ] Supply a public research contact address, if desired; none is inferred.
- [ ] Supply authorized camera-ready PDF or public paper URL; do NOT use anonymous draft.
- [ ] Replace provisional manuscript BibTeX with verified publication/DOI/arXiv metadata when available.
- [ ] Reconcile paper65K points versus IOCfish659,024, and136K objects versus Figure3.
- [ ] Confirm Table3/11 InternVL camouflage discrepancies and Gemini Full protocol.
- [ ] Reconcile new-attribute CC BY4.0 versus release CC BY-NC4.0 scope.
- [ ] Approve three IOCfish5K noncommercial sample exports and attribution.
- [x] Match three real attribute image/annotation pairs for local interactive preview.
- [x] Owner confirmed public-display permission for the three CC-BY-NC-ND-4.0
  attribute examples and their current animated presentation on 2026-10-14.
  Original attribution and license terms remain; no blanket downstream grant.
- [ ] Provide verified model prediction/GT examples before adding qualitative prediction comparisons.
- [ ] Approve original QA error disclosure and remaining dataset-version notes.
- [ ] Confirm external dataset/code links and actual evaluation package availability.
- [ ] Choose a license for newly written website code; none is invented here.

## Technical Review

- [ ] Review scientific audit, asset manifest, build/test report and screenshots.
- [ ] Run npm ci, npm test, npm run check, npm run build, npm run verify:build in website/.
- [ ] Preview /Nereus/ on desktop and mobile; run browser tests.
- [ ] Inspect git status and intended staged files; reject secrets, private paths,
  manuscript PDFs, datasets, model files, local caches or unexpectedly large files.
- [x] Owner authorized upload and public deployment of the full page on 2026-10-14.
- [ ] After approved push/merge to main, set Pages source to GitHub Actions.
- [ ] Manually run Deploy research homepage with publication approval checked.
- [ ] Verify https://brl-sysu.github.io/Nereus/ after the deployment finishes.

Publication is intentionally not automatic on push. The workflow is manual-only,
requires main and an explicit approval input, and uses the github-pages environment.
