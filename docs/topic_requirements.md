# Topic and submission requirements

The project topic is **Transformer Segmentation of Vulnerable Road Users in Adverse Conditions**. The implementation retains person, rider and bicycle segmentation across fog, nighttime, rain and snow, while adapting the method to available compute.

## Supplied proposal

The uploaded Week 4 DOCX and PPTX concern person, rider and bicycle semantic segmentation across fog, nighttime, rain and snow. They propose unsupervised Cityscapes-to-ACDC adaptation: HRDA versus MIC(HRDA), with a MiT-B5 encoder. The exact proposed hypothesis is an increase of at least 2.0 percentage points in W, the minimum condition-specific three-class mean IoU, while no condition loses over 1.0 point. W is computed independently for each model; the weakest condition may switch.

The proposal explicitly says no model was trained and no dataset was access-approved. Its cited 68.0 and 70.4 ACDC benchmark scores are published literature values, not student results. They cannot establish the condition-by-class hypothesis. Preserve the earlier research question, or transparently state the narrowed experimental question if methods change. Changing both model families or supervision regimes cannot isolate MIC's causal contribution; do not label such a comparison HRDA versus MIC.

The original ACDC semantic benchmark has 1,600 training, 406 validation and 2,000 test images. Original validation conditions contain 100 fog, 106 nighttime, 100 rain and 100 snow images. Official test labels are hidden. Evaluating 406 labeled validation images is locally feasible but must be called validation evaluation, not an official held-out benchmark test result. Access and model checkpoint availability remain separate feasibility issues. Target training labels must not supervise a purported unsupervised adaptation experiment. Validation labels used for hyperparameter selection cannot simultaneously support an untouched-evaluation claim.

## Evaluation contract

`python -m roadseg.metrics manifest.csv --output results/ --require-official-validation` evaluates aligned HRDA/MIC prediction files. Manifest columns are `image_id,condition,gt_path,hrda_pred_path,mic_pred_path`; relative paths resolve from the manifest directory by default. For manifests generated with repository-relative inference paths, pass `--path-root .` from the repository root. Canonical conditions are `fog,night,rain,snow`; `nighttime` is accepted as an alias. Every mask filename must begin with its image ID. Ground truth must explicitly identify `labelTrainIds` in its filename. The evaluator does not infer label encodings or remap original label IDs.

Train IDs are 0 through 18, with person 11, rider 12 and bicycle 18. Ground-truth 255 alone is ignored. A prediction outside 0..18 at any valid GT pixel is an error, including predicted 255. Masks must have matching 2D shapes; no evaluation-time mask resizing occurs. Palette masks are read as integer indices; RGB visualizations are rejected.

Confusion rows are GT, columns predicted. Counts are pooled across images within each condition before deriving TP/(TP+FP+FN). All-class pooled metrics do not replace condition-specific VRU metrics. A class with zero union has undefined IoU (`null`), not perfect accuracy. V(c) requires all three VRU IoUs; W requires all four conditions. Missing evidence makes the proposed hypothesis unresolved. Precision and recall with zero denominators likewise remain undefined. `miou_present` explicitly averages defined classes; `miou_19` is provided only when all 19 unions exist.

The optional paired image bootstrap resamples aligned predictions within condition, recomputes pooled counts and model-specific minima, and reports percentile intervals for each condition's MIC-minus-HRDA V difference and W difference in percentage points. Undefined resamples are counted. These intervals assume image independence and do not establish driving safety, account for scene clustering or measure repeated-training uncertainty. Default is 1,000 draws; use `--bootstrap 2000` for 2,000 draws or `--bootstrap 0` to disable.

## Current deliverables and rubric

The current Applied Vision Transformer Report screenshot specifies a technical report of **at most 2,500 words**, code (notebook or Python; the user requests GitHub), and visualizations/sample outputs. The user gives submission as 26 October 2026 at 18:00 and presentation as 27 October 2026 at 18:30, Africa/Casablanca. Prepare slides consistent with actual measured results.

| Current criterion | Weight | Concrete evidence |
|---|---:|---|
| Dataset justification and preprocessing | 15% | Original split, provenance, image/label alignment, resizing and normalization, ignore-label semantics, access conditions. |
| Model selection and implementation | 25% | Exact transformer/checkpoints, configuration, reproducible inference/training, justified feasible method, truthful adaptation description. |
| Evaluation and performance analysis | 20% | Pooled condition-by-class IoUs, W, actual hypothesis outcome, error analysis, quantitative plots, representative outputs. |
| Ethics and societal considerations | 20% | Worst-condition/class disparities, pseudo-label/annotation uncertainty, geography and capture bias, privacy, human oversight, limits of safety inference. |
| Structure, clarity and references | 20% | Professional report within limit, coherent methods/results, correct primary citations and reproducible commands. |

The generic workshop rubric screenshot on page 5 does not override this task-specific page-6 rubric.

## Historical workshop instructions

The May Research Framing PDF asks for exactly five base papers, an 8–12-page DOCX and PPTX, and excludes methodology. These apply to the historical proposal stage. The five supplied base papers are SegFormer, ACDC, DAFormer, HRDA and MIC. Their reasoning remains useful; the page limit and prohibition of methodology do not govern the current applied report.

The May Methodology PDF continues that proposal and asks for design, dataset, preprocessing, training strategy, baseline, metrics, ethics and reproducibility. It excludes final findings for that planning workshop. These sections guide technical completeness, but the current applied assignment requires real implementation and results. Preserve source distinctions: historical document instructions are context; the user's current request and topic correction govern current work.


## Separate modern-transformer tuning and training

The supervised DINOv3 extension retains semantic segmentation while using a different supervision setting from the primary HRDA/MIC checkpoint analysis. `python -m roadseg.tune_dinov3 --refit` performs three one-epoch learning-rate trials (0.0003, 0.001, 0.003), using recording-family-disjoint internal fitting/development groups drawn solely from the original 1,600 training images. Conservative grouping keeps GOPR#### and GPxx#### chapters of the same recording together within condition. The subset nearest 20% of frames is chosen without labels; equal-distance count ties favor more development frames, then seeded subset order resolves equal counts. Report actual development counts because families may be large. Selection uses pooled person/rider/bicycle IoU on deterministic 512-pixel development center crops, with lower-rate ties. Disclose the declared present-class mIoU fallback only if all VRU scores are undefined and it actually occurs.

Trial weights are discarded. A fresh seed-42 7,315-parameter head then fits all 1,600 training images for five epochs using the selected LR, with the dense DINOv3 encoder frozen. Official validation/test data never select the LR. Report actual split counts, trial table, metric and selected LR, plus the limitation of one-epoch center-crop selection. This is limited tuning, not exhaustive optimization. Final official-validation inference remains a separate stage. Keep supervised DINO results distinct from the MIC hypothesis and retain differing-seed/supervision caveats.
