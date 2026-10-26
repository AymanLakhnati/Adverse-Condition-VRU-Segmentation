# Assessment criteria and evidence

This map follows the current Applied Vision Transformer Report rubric. It identifies verifiable evidence rather than predicting a mark. The technical report and presentation are private submission artifacts; this public map references their section/slide numbers without publishing the student's identifier.

| Criterion | Weight | Report coverage | Code and measured evidence | Presentation |
|---|---:|---|---|---|
| Dataset justification and preprocessing | 15% | Sections 1–2 justify ACDC against the adverse-weather question and describe alignment, normalization, crop/flip strategy and ignore ID 255. | `roadseg/download_acdc.py`, `roadseg/train_dinov3.py`; provenance, split audits and manifests under `results/reproducibility/` and `outputs/dinov3_tuning/`. | Slides 2–3 and 5 |
| Model selection and implementation | 25% | Sections 3–4 separate released HRDA/MIC inference from fresh DINOv3 head training and explain compute-driven choices and exact configuration. | `src/hrda_inference.py`, `src/mit_backbone.py`, `roadseg/tune_dinov3.py`, `roadseg/train_dinov3.py`; strict loading, dense-feature and grid tests; actual three-trial search, 2,000-step fit and trained head. | Slides 4 and 10–11, software demonstration on slide 13 |
| Evaluation and performance analysis | 20% | Sections 5–6 report the full 406-image validation evaluation, condition/class metrics, hypothesis margins, uncertainty and specific failures. | `roadseg/metrics.py`, `scripts/verify_results.py`; saved per-image counts, aggregate JSON/CSV, paired bootstrap, selected prediction examples and exploratory recording-family sensitivity. | Slides 6–9 and 14 |
| Ethical and societal considerations | 20% | Section 7 covers small-class and acquisition bias, pseudo-label errors, privacy, licensing, oversight and limits of fairness/safety claims. | `docs/ETHICS_AND_MODEL_CARD.md`, retained licenses, restricted-data exclusions in packaging and the sparse-class support analysis. | Slides 8 and 12, limitations on slide 14 |
| Report structure, clarity and referencing | 20% | Named sections, captions, primary references, distinctions between published and new results, and a counted maximum of 2,500 words. | `README.md`, executable notebook, `docs/REFERENCES.md`, software usage and verification commands, reproducibility receipts and submission manifest. | Editable tables/charts, presenter notes with citations, sources on slide 15 |

## Deliverables

- Technical report: PDF plus editable Markdown source in the classroom ZIP.
- GitHub code: modular Python, notebook, tests, documented commands and small trained semantic head.
- Visualizations/sample outputs: measured plots, selected abstract predicted masks and reproducible selection rules. Original RGB and GT are excluded under ACDC terms.
- Presentation: the final PowerPoint and corresponding Canva copy, with speaker notes.

## Scientific boundaries that remain

The released primary checkpoints have different seeds and prior author validation use. Their comparison is descriptive rather than a matched causal ablation. DINOv3 uses labeled target data, so it belongs to a separate learning setting. The point estimates meet the proposed thresholds, while their intervals do not establish the full joint improvement/no-harm margins. Recording-family omission is exploratory and is not a confidence interval. Sparse snow riders, a short tuning budget, no hidden-test evaluation and no measured demographic fairness remain limitations. Documentation and software features do not remove these boundaries.
