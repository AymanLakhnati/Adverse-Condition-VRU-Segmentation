# Ethics and model card

## Purpose and model scope

Educational research on19-class semantic segmentation in adverse driving scenes, emphasizing person, rider and bicycle (Cityscapes train IDs11,12,18). The primary comparison evaluates released MiT-B5 HRDA and MIC(HRDA) student checkpoints. The separate DINOv3 ViT-S/16 experiment freezes a2025 foundation encoder and trains a new19-class linear segmentation head with ACDC training labels. It is supervised target-domain learning; HRDA/MIC were trained using a source-supervised/target-unlabeled UDA protocol. Different supervision, architectures, pretraining, decoders and inference prevent attributing their differences to a single cause.

These research outputs are unsuitable for autonomous vehicle control, collision avoidance, law enforcement, biometric identification or individual tracking. A higher mean IoU does not establish that pedestrians will be detected reliably enough for safe driving.

## Dataset coverage and privacy

[Official ACDC overview](https://acdc.vision.ee.ethz.ch/overview) describes Swiss road recordings, mainly urban but also rural/highway scenes, using a1080p camera, and four adverse domains: fog, nighttime, rain and snow. Camera placement changes across conditions; condition differences can combine weather, lighting, scene and capture effects. Geographic split separation helps limit immediate scene leakage but cannot establish global coverage. Other countries, road designs, camera systems, mixed conditions and rare hazards may behave differently.

Use the provided anonymized RGB images locally. Anonymization reduces identity exposure but does not justify reidentification attempts or claims that every identifying contextual detail is absent. Roads, buildings, license plates, faces and scene context deserve care. Do not infer age, ethnicity, disability or other sensitive attributes from images. No demographic labels are available in this evaluation; class- or condition-based analysis measures performance disparity, not demographic fairness.

## Annotation uncertainty and metrics

ACDC supplies correspondence-assisted semantic annotations and a separate invalid-region mask because some adverse-image content is not visually discernible. Labels inferred with clear-weather correspondence can be more informative than the adverse RGB alone. Ground truth is a reference annotation, not proof that each pixel is directly observable.

This implementation evaluates standard train-ID semantics, ignores GT ID255 and does not implement the uncertainty-aware benchmark unless explicitly recorded otherwise.255 is an ignore/void ID, not a background category: all19 valid semantic classes remain supervised/evaluated. An absent class union yields an undefined IoU rather than an invented zero/perfect score. Pixel-pooled results and equal-weight class means answer different questions; report per-class supports and condition coverage.

Small, distant, occluded road users and narrow bicycle structures can vanish after downsampling or stride16 tokenization. Pixel IoU can also hide missed instances and severity of errors: a large nearby person dominates many tiny distant people. The person/rider/bicycle subset is an explicit operational proxy and omits motorcycle as a separate class; it is not a universal definition of all vulnerable road users. Inspect false negatives and boundaries, without treating a selected error montage as a random sample.

## Responsible interpretation

Report every prespecified condition, including regressions and sparse rider/bicycle support. A worst-condition gain can conceal class-specific harm, and a minimum over only four domains cannot bound unseen hazards. Paired image bootstrap intervals do not cover correlated scenes, training-seed variation or deployment shift. The released HRDA/MIC artifacts have different seeds and were evaluated by their authors on ACDC validation; this is an external-checkpoint comparison, not an untouched-test causal ablation. No local evaluation may be described as an official test-server submission unless one actually occurred.

Reusing released weights and freezing a compact encoder reduce local training work; record measured runtime/device rather than fabricating energy or carbon savings. Stronger future validation would include independent geographic/camera data, explicit small-instance recall, diverse conditions, uncertainty/rejection and human safety review. These are future requirements, not completed checks.

## License and distribution

The obtained ACDC agreement grants noncommercial research/teaching use and requires attribution. Section2.3 prohibits distributing the dataset **or modified versions**; abstract derivatives are permitted only when they cannot reconstruct the dataset or something similar in character. Therefore keep RGB images, GT masks, image crops, overlays and reconstructive galleries out of public GitHub and distributed archives. Provide official-download instructions, code, trained permissible adapters, aggregate metrics/charts and provenance. Local-only examples require the viewer to obtain ACDC directly. No private classroom exception to redistribution is asserted here.

MiT/SegFormer source has NVIDIA research/evaluation restrictions; HRDA/MIC adaptation code has Apache2.0 attribution requirements. DINOv3 has a custom license requiring acknowledgement and covered redistribution terms. Included license copies apply to their respective components; an original-code license cannot override upstream restrictions. No foundation or full upstream segmentation weights are bundled for public submission.

## Completion status

The prespecified DINOv3 plan uses five epochs and1600 training images. A plan or synthetic smoke test is not completed dataset training. Final status, actual epochs/images, hashes, losses and unchanged-encoder verification must come from local `training_config.json`, `training_steps.csv` and checkpoint artifacts. If interrupted or limited, label it explicitly.
