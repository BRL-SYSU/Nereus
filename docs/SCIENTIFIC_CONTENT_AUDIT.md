# Scientific Content Audit

Review date: 2026-10-13. Primary source: owner-supplied Nereus manuscript,
28 PDF pages, including supplement and checklist. This is not a release census.
Exact server mappings are in ignored `LOCAL_SOURCE_PATHS.md`, not website output.

## Source and Publication Boundaries

The PDF is anonymous and marked "Do not distribute". It was read internally;
neither PDF nor page/figure screenshots are exported. Acceptance at NeurIPS 2026
Datasets & Benchmarks is owner-provided, not independently established by this
anonymous submission. Authors, affiliations, DOI, arXiv, BibTeX, research contact
and authorized public PDF are not invented. The public paper remains pending.

Original figures were sought in the bounded `AQUACOUNT/test_dataset/paper_sample/figures`
directory: four pre-existing 0001 composite figures are available, but are not
blindly reused because layouts/annotations may differ from the supplied manuscript.
The site uses fresh HTML scientific diagrams and original licensed sample exports.
The academic-project-astro-template was inspected as an architectural reference;
no source code, styling or assets were copied from it.

## Quantitative Claim Register

| Website claim | Manuscript source | Treatment |
| --- | --- | --- |
| 88,555 images | Section 3.1 and Figure 3, p. 4 | Main statistic |
| 2.68M QA pairs | Abstract, p. 1 | Rounded paper statistic |
| 136 attribute dimensions | Section 3.1 p. 4; Figure 7 p. 14 | Schema-wide, not all fields on each object |
| 2,690 species; 1,515 genera | Abstract, p. 1 | Manuscript version |
| 136K objects | Abstract, p. 1 | Version note, unresolved figure discrepancy |
| 65K points | Abstract, p. 1 | Only verbatim abstract/version note, explicitly disputed |
| 67K grounded QA | Abstract, p. 1 | Version note |
| Source image counts: 67,248 / 13,480 / 5,637 / 2,190 | Figure 3 p. 4 | FathomNet / iNaturalist / IOCfish5K / CoralVQA; sum 88,555 |
| Attribute benchmark values | Table 2 p. 7 | Overall accuracy and SIM distinct; no new evaluation |
| RGB grounded counting values | Table 3A p. 8 | 18 rows; main-table values retained |
| Density evidence values | Table 3B p. 8 | 11 rows; supplied GT-derived evidence |
| Transfer values and 1,129 samples | Table 9 p. 18; Table 3C p. 8 | Nautilus supervision into Nereus |
| Supervision ablations | Table 4 p. 9 | Qwen2.5-VL-7B, selected overall values |
| Degradation ablations | Table 8 p. 18 | Original-image baseline same as structured attributes |
| Quality 9.26/10 train, 9.14/10 test; 200 training images | Section 3.2 p. 5, Appendix A.1 | Ratings, NOT accuracy estimates |
| 90 / 270 / 20 sample point counts and visibility counts | Exact selected final-annotation files | Checked only selected samples, not aggregate census |
| 4 / 10 / 1 selected density hotspots | Same final-annotation records | Existing hotspot entries, not predicted counts |

The 659,024 IOCfish5K original point total is from the upstream official README,
not substituted into Nereus paper statistics. A previous release audit found
658,803 valid points and 659,024 raw objects. That historical total was not
recomputed by traversing the dataset for this website task.

## Task Families and Annotation Process

Fine-Grained Object Attribute Understanding uses five conceptual categories:
identity; structural morphology; appearance and surface; condition and environment;
pose, visibility and interaction. Benchmark tables instead label the last two
Ecological Health and Pose and Interaction. Figure 7 organizes organism-specific
fields for fish, coral, cephalopods, crustaceans, echinoderms, other mollusks,
sponges, other creatures and other cnidaria. Do not equate a field list with a
universal per-instance schema. Figure 6 text excerpts are real published examples,
not linked released records or independently human-verified target boxes.

Attribute pipeline (Section 4.1): GroundingDINO localization where boxes are
missing; random 20% manual box review/correction for that source; object crops
with 15% margin and contextual image; biological knowledge priors; independent
Qwen3-VL-8B and InternVL3-8B annotations; three-turn discussion; confidence below
0.6 to Qwen2.5-VL-72B. Quality assessment uses GPT-4o and three marine-biology
master's students for scale calibration, followed by expert review. Test scores
94.8% >=8; 63 objects arbitrated and 43 attributes revised. This selected
low-score review is not an unbiased estimate of whole-dataset error rate.

Grounded Counting Perception has three capabilities (Section 4.2):
region-aware counting; camouflage-state counting; density-hotspot reasoning.
IOCfish points feed Qwen3-VL-8B semantic localization, SAM refinement and spatial
assignment, grid/region camouflage labels, point-derived adaptive density maps,
local hotspot attribution and Qwen3.5-9B QA. Numerical templates and language
generation have different guarantees. Website region mode renders stored scene
bounding boxes, NOT segmentation masks. Visibility states are dataset labels,
not new human annotations. Hotspot ecology is model-assisted interpretation.

## Benchmark Interpretation

- Track A is RGB-only. MAE lower is better; exact accuracy, relative-tolerance
  accuracy, densest-cell accuracy and camouflage F1 higher is better.
- Acc@5/10 means relative +/-5%/10%, not +/-5/10 objects.
- Track B gives the model RGB AND GT-derived density heatmaps. Do not describe
  it as RGB-only density prediction. Hotspot, density F1 and grounding F1 differ.
- Table 9 compares training sources on the same Nereus test set, not evaluation
  of Nereus-trained models on a held-out external dataset.
- Count/QAS/D-QA/Full describe the manuscript training settings for open models.
  Gemini Full does not establish Gemini fine-tuning; its implementation is unclear.
- Table 2 overall accuracy: LLaVA 48.91, InternVL 48.10, Qwen 41.55 for Nereus.
  Corresponding NautData values: 12.06, 16.08, 17.09. No significance claim.
- InternVL QAS MAE 25.99 versus Full 30.48. More supervision is not universally better.
- Qwen D-QA density F1 45.26 versus ZS 18.13, but hotspot accuracy drops 41.75 to 37.39.
- Qwen structured/original accuracy 41.55 versus enhanced+prior 37.36.
- QA-pair ecological health 77.90 exceeds structured 39.92 (Table 4), despite
  structured winning overall. Do not claim every category improves.

## Figure and Table Map

Figures: 1 overview p. 1; 2 motivation p. 2; 3 statistics p. 4;
4 construction p. 5; 5 degradation p. 9; 6 applications and 7 taxonomy p. 14;
8-9 counting/camouflage/hotspots p. 20; 10 density interpretation p. 21.
Tables: 1 dataset comparison p. 3; 2 attributes p. 7; 3 counting p. 8;
4 supervision p. 9; 5-7 quality/similarity scales p. 15;
8 degradation and 9 transfer p. 18; 10 transfer differences and 11 counting
ablation p. 19. Appendix A.2 training details p. 15, A.3 rights pp. 15-16.

## Conflicts and Limits

1. 65K points conflicts with upstream/raw ~659K. Never silently replace a paper total.
2. Figure 3 biological-group counts sum to 166,586, not 136K. Semantic-region
   counts are overlapping assignments and must not be shown as disjoint objects.
3. Table 3 InternVL P-ZS camouflage F1 18.98 conflicts with Table 11's 0.00;
   Table 3 ZS 21.26 and Count 14.72 are unreported in Table 11. Website follows T3.
4. Previous audit found release records labeled all-test despite split manifests
   3,945/563/1,129. No source metadata is corrected by this task. IOCfish originals
   3,137/500/2,000 live in a separate namespace; do not conflate them.
5. 0001 region QA pair 3 says top-right70 is densest, while grid top-center87 is
   larger. The site discloses this real annotation error and does not edit the QA.
6. Local supplementary prediction artifacts are NOT reproductions of all Table 3
   backbones/settings; the site does not claim to rerun/reproduce experiments.
7. Model-prediction-versus-GT qualitative pairs with verified provenance were not
   available in the selected evidence. No such comparisons are fabricated.
8. Taxa, geography, depth and device coverage are uneven; LMM/detector noise
   remains. Image labels do not establish water quality or hidden biological states.
9. No error bars/significance tests. Exact training hyperparameters remain
   incomplete in Appendix A.2. Ecological causation requires external review.

## Rights

IOCfish5K official README section 7 verified CC BY-NC 4.0 on 2026-10-13:
https://github.com/GuoleiSun/Indiscernible-Object-Counting#7-dataset-license
Owner-provided release documents agree. Original watermarks remain intact.
New attribute annotations are described as CC BY 4.0 by paper Appendix A.3,
but local release default is CC BY-NC 4.0; scope needs owner reconciliation.
Three FathomNet-derived Nereus test images are now exported for local preview.
Their source records specify CC-BY-NC-ND-4.0 (images), CC-BY-NC-4.0 (boxes),
and UNVERIFIED bbox review status. Image dimensions and bbox coordinates match;
this is not independent validation of biological attributes. Public deployment
of the interactive presentation requires rights review. iNaturalist/CoralVQA
have no verified per-item rights here. No blanket collection license is inferred.
