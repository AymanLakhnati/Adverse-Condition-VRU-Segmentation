"""Build the private final technical report only from completed real experiments.

Use the bundled runtime Python with ReportLab. Before the first authoring run,
execute the PDF skill operation marker; this script does not substitute for it.
Pass the student ID locally with --student-id; never publish the private outputs.
No synthetic/example result mode or fallback result values are provided.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from pathlib import Path

TITLE = "Transformer Segmentation of Vulnerable Road Users in Adverse Conditions"
CONDITIONS = ("fog", "night", "rain", "snow")
COUNTS = {"fog": 100, "night": 106, "rain": 100, "snow": 100}
LABELS = {"fog": "Fog", "night": "Night", "rain": "Rain", "snow": "Snow"}
FIGURES = ("condition_vru.png", "class_condition_heatmap.png", "sample_predictions_report.png",
           "dinov3_training_curve.png")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def pct(value):
    require(finite(value), "Required numeric score is undefined")
    return f"{100 * value:.2f}%"


def pp(value):
    require(finite(value), "Required numeric difference is undefined")
    return f"{value:+.2f} pp"


def load_verified(root):
    primary = read_json(root / "results/summary.json")
    run = read_json(root / "results/inference_run.json")
    dino = read_json(root / "results/dinov3/summary.json")
    train = read_json(root / "outputs/dinov3_dense/training_config.json")
    tuning_path = root / "outputs/dinov3_tuning/tuning_results.json"
    tuning = read_json(tuning_path)
    trials = tuning.get("trials", [])
    require(tuning.get("status") == "completed" and tuning.get("trial_epochs") == 1 and
            tuning.get("seed") == 42 and tuning.get("batch_size") == 4 and
            tuning.get("official_validation_or_test_used") is False and len(trials) == 3 and
            sorted(t.get("learning_rate", -1) for t in trials) == [0.0003, 0.001, 0.003] and
            all(t.get("fit_epochs") == 1 and t.get("fit_optimizer_steps", 0) > 0 and
                t.get("encoder_unchanged") is True for t in trials),
            "Three completed train-only learning-rate trials required")
    split = tuning.get("split", {})
    require(split.get("recording_families_disjoint") is True and
            split.get("sequence_groups_disjoint") is True and split.get("frame_ids_disjoint") is True and
            split.get("fit_images", 0) + split.get("development_images", 0) == 1600 and
            not set(split.get("fit_groups", [])) & set(split.get("development_groups", [])),
            "Internal fitting/development split must separate GoPro recording families within the 1600 training images")
    eligible = [t for t in trials if finite(t.get("development_vru_mean_iou"))]
    metric = "development_vru_mean_iou" if eligible else "development_miou_present"
    eligible = eligible or [t for t in trials if finite(t.get(metric))]
    require(eligible and tuning.get("selected_metric") == metric and
            bool(tuning.get("selection_fallback")) == (metric == "development_miou_present"),
            "Tuning selection metric/fallback disclosure mismatch")
    winner = sorted(eligible, key=lambda t: (-t[metric], t["learning_rate"]))[0]
    require(tuning.get("selected_learning_rate") == winner["learning_rate"] and
            math.isclose(tuning.get("selected_score", -1), winner[metric], abs_tol=1e-12) and
            train.get("learning_rate") == winner["learning_rate"] and
            train.get("internal_training_development_used_for_learning_rate_selection") is True and
            train.get("official_validation_used_for_selection") is False and
            train.get("tuning_results_sha256") == hashlib.sha256(tuning_path.read_bytes()).hexdigest() and
            train.get("head_before_sha256") == tuning.get("initial_head_sha256") and
            train.get("encoder_state_sha256") == tuning.get("encoder_state_sha256"),
            "Fresh full refit must match the actual train-only tuning selection and initialization")
    require(run.get("status") == "completed" and run.get("processed") == 406,
            "Primary inference must be completed on all 406 images")
    dataset = primary.get("dataset", {})
    require(dataset.get("images") == 406 and dataset.get("condition_counts") == COUNTS and
            dataset.get("matches_original_validation_counts") is True,
            "Primary evaluation must match the complete official validation counts")
    require(dino.get("validation_images") == 406 and dino.get("condition_image_counts") == COUNTS and
            dino.get("learning_setting") == "supervised ACDC train labels",
            "Complete separate supervised DINOv3 validation evaluation required")
    require(train.get("status") == "completed" and train.get("completed_fixed_full_protocol") is True and
            train.get("completed_epochs") == 5 and train.get("epochs") == 5 and
            train.get("training_images") == 1600 and train.get("available_train_images") == 1600 and
            train.get("batch_size") == 4 and train.get("seed") == 42 and
            train.get("head_parameters") == 7315 and train.get("limited_smoke_run") is False and
            train.get("encoder_state_unchanged") is True and train.get("head_state_changed") is True,
            "Completed full five-epoch, 1600-image DINOv3 protocol required")
    require(train.get("encoder_after_sha256") == train.get("encoder_state_sha256"),
            "Frozen-encoder hashes must agree")
    require(train.get("head_after_sha256") != train.get("head_before_sha256"),
            "Trained head must differ from initialization")
    with (root / "outputs/dinov3_dense/training_steps.csv").open(encoding="utf-8", newline="") as handle:
        steps = list(csv.DictReader(handle))
    require(len(steps) == train.get("optimizer_steps") and len(steps) > 0,
            "Logged optimizer-step count mismatch")
    require({int(row["epoch"]) for row in steps} == set(range(1, 6)), "All five epoch logs required")
    require(all(finite(float(row["loss"])) and float(row["gradient_norm"]) > 0 and
                finite(float(row["gradient_norm"])) and int(row["valid_pixels"]) > 0 for row in steps),
            "Training log contains invalid losses/gradients")
    epoch_loss = {}
    for epoch in range(1, 6):
        rows = [r for r in steps if int(r["epoch"]) == epoch]
        epoch_loss[epoch] = sum(float(r["loss"]) * int(r["valid_pixels"]) for r in rows) / sum(int(r["valid_pixels"]) for r in rows)
        require(sum(int(r["images"]) for r in rows) == 1600,
                f"Epoch {epoch} must log exactly 1600 images; all-ignore skips need explicit report revision")
    for model in ("hrda", "mic"):
        obj = primary["models"][model]
        require(set(obj["conditions"]) == set(CONDITIONS), "Missing primary condition")
        for condition in CONDITIONS:
            require(finite(obj["conditions"][condition]["vru_mean_iou"]), "Undefined primary VRU result")
            require(len(obj["conditions"][condition]["per_class"]) == 19, "19-class results required")
    require(primary["hypothesis"].get("status") in ("supported", "not_supported"), "Hypothesis unresolved")
    boot = primary.get("bootstrap", {})
    require(boot.get("repeats") == 2000 and boot.get("seed") == 42,
            "Required stratified paired bootstrap: 2000 draws, seed42")
    for endpoint in (*CONDITIONS, "delta_W_pp"):
        require(boot.get("endpoints", {}).get(endpoint, {}).get("percentile_95_pp") is not None,
                "Bootstrap interval missing")
    for figure in FIGURES:
        require((root / "results/figures" / figure).is_file(), f"Required actual figure missing: {figure}")
    return primary, run, dino, train, steps, epoch_loss, tuning


def content(primary, run, dino, train, steps, epoch_loss, tuning, student_id):
    """Return eight explicit pages; Markdown and PDF share every text/table value."""
    h, m, hyp = primary["models"]["hrda"], primary["models"]["mic"], primary["hypothesis"]
    endpoint = primary["bootstrap"]["endpoints"]["delta_W_pp"]
    ci = endpoint["percentile_95_pp"]
    verdict = "supported" if hyp["supported"] else "not supported"
    direction = "increased" if hyp["delta_W_pp"] >= 0 else "decreased"
    weak_h = ", ".join(LABELS[c] for c in h["weakest_conditions"])
    weak_m = ", ".join(LABELS[c] for c in m["weakest_conditions"])
    cells = [(model, condition, row) for model, obj in (("HRDA", h), ("MIC", m))
             for condition in CONDITIONS for row in obj["conditions"][condition]["per_class"]
             if row["class_id"] in (11, 12, 18) and row["iou"] is not None]
    worst_model, worst_condition, worst = min(cells, key=lambda x: x[2]["iou"])
    m_person = m["pooled"]["per_class"][11]
    own_gpu = str(run.get("gpu", "unspecified device"))
    pptext = pp(hyp["delta_W_pp"])
    selection_label = ("Development VRU IoU" if tuning["selected_metric"] == "development_vru_mean_iou"
                       else "Development present-class mIoU")
    fallback_note = (" All three VRU trial means were undefined, so selection used the declared present-class mIoU fallback."
                     if tuning.get("selection_fallback") else "")
    refs = [
        '[1] E. Xie et al., "SegFormer: Simple and efficient design for semantic segmentation with transformers," NeurIPS, vol. 34, pp. 12077-12090, 2021.',
        '[2] C. Sakaridis, D. Dai, and L. Van Gool, "ACDC: The adverse conditions dataset with correspondences for semantic driving scene understanding," ICCV, pp. 10765-10775, 2021.',
        '[3] L. Hoyer, D. Dai, and L. Van Gool, "DAFormer: Improving network architectures and training strategies for domain-adaptive semantic segmentation," CVPR, pp. 9924-9935, 2022.',
        '[4] L. Hoyer, D. Dai, and L. Van Gool, "HRDA: Context-aware high-resolution domain-adaptive semantic segmentation," ECCV, 2022.',
        '[5] L. Hoyer, D. Dai, H. Wang, and L. Van Gool, "MIC: Masked image consistency for context-enhanced domain adaptation," CVPR, pp. 11721-11732, 2023.',
        '[6] O. Simeoni et al., "DINOv3," arXiv:2508.10104, 2025.',
        '[7] ETH Zurich, "ACDC dataset overview," https://acdc.vision.ee.ethz.ch/overview, accessed 25 Oct. 2026.',
        '[8] ETH Zurich, "ACDC end user license agreement," https://acdc.vision.ee.ethz.ch/download, accessed 25 Oct. 2026.',
        '[9] PyTorch Image Models, "vit_small_patch16_dinov3.lvd1689m," https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m, accessed 25 Oct. 2026.',
        '[10] Meta, "DINOv3 license agreement," https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md, 2025.',
    ]
    pages = [
      [
        ("title", TITLE),
        ("meta", f"Ayman Lakhnati | Student ID {student_id}\nUniversity of Wollongong in Dubai\nMAAI901 | Dr Milan Dordevic | 25 October 2026"),
        ("heading", "Abstract"),
        ("p", f"Adverse visibility challenges semantic segmentation of vulnerable road users (VRUs). This study evaluates released HRDA and MIC(HRDA) transformer checkpoints on all 406 original ACDC validation images and separately trains a modern DINOv3 dense segmentation head. The prespecified endpoint is the minimum, across four conditions, of the person/rider/bicycle mean intersection-over-union (IoU). MIC {direction} this endpoint from {pct(h['W'])} to {pct(m['W'])}, a difference of {pptext}; the original threshold hypothesis was {verdict}. A frozen DINOv3 ViT-S/16 with a newly trained 7,315-parameter classifier achieved a separate supervised endpoint of {pct(dino['W'])}. Its target-label access prevents a controlled comparison with the UDA checkpoints. Results quantify conditional performance, not driving safety; reproducible artifacts distinguish published training, local inference and fresh supervised learning."),
        ("heading", "1. Research question and motivation"),
        ("p", "Pedestrians and riders occupy fewer pixels than roads and buildings, while fog, darkness, rain and snow reduce visual evidence. Overall segmentation averages can therefore conceal failures affecting people. The research question asks whether masked image consistency improves the weakest adverse condition without materially harming another. We retain that question and its three-class VRU definition."),
        ("p", "SegFormer motivates a hierarchical transformer encoder [1]; DAFormer couples transferable transformer features with multilevel context and adaptation strategies [3]. HRDA combines low-resolution context and high-resolution detail through scale attention [4]. MIC encourages contextual prediction from masked inputs using teacher supervision [5]. These established, task-specific systems support the original comparison. DINOv3 provides a recent, 2025 state-of-the-art foundation-model family [6]; our additional lightweight dense head tests a feasible modern transformer adaptation, without claiming that the simple decoder achieves the family's published segmentation records."),
      ],
      [
        ("heading", "2. Dataset choice and preprocessing"),
        ("p", "ACDC is appropriate because it contains real Swiss driving scenes, four adverse domains and 19 Cityscapes-compatible classes [2], [7]. We use the original adverse splits: 1,600 training images, 406 validation images and 2,000 test images. Training contributes 400 images per condition; validation contains 100 fog, 106 night, 100 rain and 100 snow images. Official test labels are withheld and were not accessed. Corresponding normal-condition images and uncertainty masks exist, but are not additional supervision in our local experiments."),
        ("p", "The downloader obtains official archives; manifests preserve image IDs, condition, split, RGB/label hashes and matched filenames. Integer labelTrainIds use 0-18, with person 11, rider 12 and bicycle 18; GT 255 alone is ignored. We reject incompatible shapes, RGB label visualizations, reused paths and illegal predictions at valid GT pixels. A matching count alone cannot prove provenance, so paths and hashes accompany split checks."),
        ("table", (["System", "Label access / local work", "Spatial processing"], [
            ["HRDA", "Published source-supervised, target-unlabeled UDA; local inference", "1024 outer / 512 detail crops"],
            ["MIC(HRDA)", "Same UDA setting; local inference", "Same official nested grids"],
            ["DINOv3 + head", "ACDC train labels; fresh supervised head", "512 crops / stride 384 inference"],
        ])),
        ("caption", "Table 1. Supervision and preprocessing differ between studies; this is not a controlled architecture ranking."),
        ("p", "For HRDA/MIC we preserve RGB normalization (means 123.675, 116.28, 103.53; standard deviations 58.395, 57.12, 57.375), the 1920-by-1080 input and published sliding-window scale fusion. No flip or multiscale test augmentation is added. Masks retain original resolution. DINOv3 uses the same channel normalization, synchronized random training crops/flips and a distinct patch-token decoder; there is no image-level CLS or average-pooled classification task."),
        ("heading", "3. Released models and implementation"),
        ("p", "Each primary inference network has 85,685,990 parameters. We strictly load all 1,126 student tensors, excluding teacher training state. A standalone PyTorch port sequences detail crops to fit the local 4 GB GPU while retaining the official fusion mathematics. Reference tests found zero maximum backbone CPU difference and approximately 7.08e-8 synthetic sequential/batched logit difference. These establish targeted implementation agreement, not complete equivalence across modern kernels or every dataset image."),
      ],
      [
        ("heading", "4. Training and fresh transformer adaptation"),
        ("p", "The authors trained MiT-B5/HRDA checkpoints for 40,000 iterations using AdamW, base LR 6e-5, decoder multiplier 10, weight decay 0.01, 1,500-step warmup and polynomial decay [4], [5]. MIC uses block-64 masking at ratio 0.7; its released separate mode masks one source and one target image. These are upstream runs, not locally repeated training. HRDA seed 2 versus MIC seed 0 prevents a matched-seed causal ablation; full settings accompany the code."),
        ("p", f"We performed fresh target-supervised learning on all 1,600 ACDC training images. DINOv3 ViT-S/16 stays frozen and in evaluation mode. From each 512-square crop we extract 1,024 final normalized 384-dimensional patch tokens, excluding the class token and four registers, and reshape them to 384-by-32-by-32. A newly initialized 1-by-1 convolution maps 384 channels to 19 classes; bilinear upsampling returns dense crop logits. Only its 7,315 weights/bias parameters train [6], [9]."),
        ("p", f"Train-only tuning conservatively grouped GoPro chapters into recording families, then held out the seeded group prefix closest to 20% per condition: {tuning['split']['fit_images']} fitting and {tuning['split']['development_images']} development images. Three one-epoch AdamW trials used identical fresh initialization and crop/flip RNGs. Selection maximized pooled VRU IoU on deterministic 512-pixel development center crops, with lower-LR ties. Selected LR was {tuning['selected_learning_rate']:g} using {selection_label.lower()}.{fallback_note} Large recording families can shift the holdout fraction. This short center-crop proxy is limited, not exhaustive tuning; official validation/test remained unused."),
        ("table", (["Learning rate", selection_label, "Selected"], [
            [f"{trial['learning_rate']:g}", pct(trial[tuning['selected_metric']]) if finite(trial[tuning['selected_metric']]) else "Undefined",
             "Yes" if trial['learning_rate'] == tuning['selected_learning_rate'] else "No"] for trial in tuning['trials']
        ])),
        ("caption", "Table 2. Measured one-epoch internal training-development trials. Scores use the actual selection metric; official validation remains separate."),
        ("p", f"Discarding trial weights, we freshly refit all 1,600 training images for five epochs: batch 4, seed 42, selected LR held fixed, random 512 crops/flips 0.5, weight decay 0.01 and cross entropy ignoring 255. No official-validation tuning or early stopping occurred. The run logged {len(steps):,} steps in {train['elapsed_seconds']/60:.1f} minutes on {train.get('gpu') or own_gpu}. Encoder hashes matched; the head changed. Epoch loss moved from {epoch_loss[1]:.4f} to {epoch_loss[5]:.4f}, which alone does not establish generalization."),
        ("figure", ("dinov3_training_curve.png", 200)),
        ("caption", "Figure 1. Actual supervised DINOv3 head-training losses from the saved local step log; no original-author curve is presented as student training."),
        ("p", "DINOv3 inference averages overlapping 512-square tile logits at stride 384, then predicts all 19 classes at original image resolution. Its recent pretraining and target-label supervision answer a separate transfer question; differences from UDA cannot isolate backbone superiority or MIC's effect."),
      ],
      [
        ("heading", "5. Evaluation protocol and primary results"),
        ("p", "For class k, IoU is TP/(TP+FP+FN), calculated after pooling integer confusion counts within condition; we do not average per-image IoUs. V(c) is the equal mean for person, rider and bicycle, and W is the minimum V(c) over four conditions, computed independently for each model. An absent union is undefined. The original hypothesis requires MIC-minus-HRDA W at least +2.0 percentage points and every condition difference at least -1.0 point."),
        ("table", (["Condition", "HRDA V", "MIC V", "Difference"], [
            [LABELS[c], pct(h['conditions'][c]['vru_mean_iou']), pct(m['conditions'][c]['vru_mean_iou']), pp(hyp['condition_delta_pp'][c])] for c in CONDITIONS
        ] + [["Worst W", pct(h['W']), pct(m['W']), pptext]])),
        ("caption", "Table 3. Local inference on the same 406 validation images; differences are percentage points, not relative percent changes."),
        ("figure", ("condition_vru.png", 169)),
        ("caption", "Figure 2. Condition-specific three-class mean IoU. The weakest condition can differ between systems."),
        ("p", f"HRDA's weakest condition was {weak_h}; MIC's was {weak_m}. The threshold hypothesis was {verdict}. The paired, condition-stratified 2,000-draw bootstrap (seed 42) gave a 95% percentile interval [{ci[0]:+.2f}, {ci[1]:+.2f}] pp for the W difference. This interval concerns fixed-model image resampling, not repeated training or a significance test of the prespecified joint threshold. All condition intervals and undefined-resample counts remain in the machine-readable summary."),
      ],
      [
        ("heading", "6. Performance disparities and error analysis"),
        ("p", f"Across all validation pixels, 19-class mIoU was {pct(h['pooled']['miou_19'])} for HRDA and {pct(m['pooled']['miou_19'])} for MIC; these aggregate values cannot replace V(c). The lowest class-condition IoU among primary VRU results was {worst_model} {worst['class_name']} in {LABELS[worst_condition]} ({pct(worst['iou'])}; its support was {worst['support']:,} pixels, with {worst['fn']:,} false-negative and {worst['fp']:,} false-positive pixels. These counts substantiate errors without inventing their visual cause."),
        ("figure", ("class_condition_heatmap.png", 310)),
        ("caption", "Figure 3. Class-by-condition IoUs expose variation hidden by the three-class mean. Class support is retained in CSV."),
        ("p", "The companion visualization pack provides full-size VRU confusion matrices with all 19 prediction columns. Read row-normalized displays with absolute counts because semantic classes differ in support."),
        ("p", f"MIC's pooled person precision was {pct(m_person['precision'])} and recall {pct(m_person['recall'])}, illustrating the distinction between false alarms and missed pixels. The separate supervised DINOv3 head reached 19-class mIoU {pct(dino['pooled']['miou_19'])}, W {pct(dino['W'])}, and weakest condition {', '.join(LABELS[c] for c in dino['weakest_conditions'])}. All three systems use the same labeled validation scenes, but only DINOv3's new head accessed target training labels. A stride 16 linear representation and simple decoder may limit fine structure; observed scores alone cannot identify that mechanism."),
      ],
      [
        ("heading", "7. Ethical and societal implications"),
        ("p", "Road-user misses are not interchangeable with errors on buildings: a pedestrian false negative can matter to a downstream safety system even when mean IoU improves. This research therefore reports the minimum condition score and individual class errors. Nevertheless, pixel IoU is not instance recall, detection range, collision risk or a safety guarantee. Large nearby people contribute more pixels than small distant ones, so acceptable averages can conceal dangerous missed instances. Independent operational testing and human oversight would be essential before deployment."),
        ("p", "Swiss geography, road design, a limited camera configuration and deliberately selected adverse scenes constrain coverage. Conditions may differ in scene composition and camera placement as well as visibility. Findings cannot automatically transfer to Morocco, other vehicles, combined weather or unfamiliar road users. Our semantic classes and condition groups contain no validated demographic attributes: performance disparity here is not a measurement of fairness across ethnicity, age, disability or gender. Inferring those attributes from images would create unsupported claims and privacy risks."),
        ("p", "Context can help reconstruct occluded structure but can also reinforce shortcuts. A teacher may propagate confident mistaken pseudo-labels; frequent classes and familiar scene arrangements can dominate learning. Color, clothing, scale and background correlations deserve targeted future audits, without asserting unmeasured subgroup bias. ACDC's correspondence-assisted labels and invalid-region masks acknowledge ambiguous visibility; annotation uncertainty is different from calibrated predictive confidence. Our standard semantic evaluation ignores 255 and does not claim to implement uncertainty-aware segmentation."),
        ("p", "We use anonymized data without identifying or tracking individuals. Anonymization does not make reidentification appropriate or eliminate every contextual clue. ACDC's noncommercial agreement prohibits redistributing its images or modified versions [8]; raw RGB, GT and image overlays stay local. Public deliverables contain code, abstract predictions and aggregate evidence, with attribution and model-specific licenses [9], [10]. No commercial driving authorization follows. Reusing checkpoints reduces local training work, but no unmeasured carbon saving is claimed."),
      ],
      [
        ("heading", "Qualitative prediction examples"),
        ("figure", ("sample_predictions_report.png", 560)),
        ("caption", "Figure 4. HRDA/MIC predictions for the image with greatest GT VRU pixel support in each condition; ties choose the lowest image ID. Each crop spans all GT person/rider/bicycle pixels with a 64-pixel margin, clipped to image bounds. Only our predicted masks appear, without RGB or GT pixels. This deliberate sample does not estimate representative error rates. The companion pack includes the separate supervised DINOv3 predictions."),
      ],
      [
        ("heading", "8. Reproducibility, limits and conclusion"),
        ("p", "The repository provides official download instructions, source checkpoint hashes, exact configurations, matched manifests, strict-loading tests, per-image confusion matrices and reproducible evaluation commands. New head artifacts include sampled crop/flip logs, losses, nonzero gradients and frozen-state verification. Original NVIDIA/ETH source attribution is retained for adapted inference code. Large foundation weights and restricted raw data are excluded. The report source and PDF derive from completed JSON/CSV results through a gated builder."),
        ("p", "Validation is not an untouched official test: the original authors used this split during model development/checkpoint evaluation. Comparing different released seeds does not isolate the masking loss. Bootstrap intervals assume independent images and omit scene clustering, training variability and geographic shift. Target-supervised DINOv3 results must remain separate from the UDA hypothesis; no hyperparameters were chosen from this local validation outcome."),
        ("p", f"On this bounded validation experiment, MIC changed the worst-condition VRU endpoint by {pptext} and the original hypothesis was {verdict}. The freshly trained DINOv3 dense head provides an additional measured modern-transformer adaptation, not a replacement causal test. Future work should evaluate independent scenes and small-instance recall while preserving adverse-condition and class disaggregation."),
        ("heading", "References"),
        *[("reference", ref) for ref in refs],
        ("small", "Supplementary source provenance: docs/REFERENCES.md. Reproducible code: https://github.com/AymanLakhnati/Adverse-Condition-VRU-Segmentation."),
      ],
    ]
    return pages


def plain_text(pages):
    values = []
    for page in pages:
        for kind, value in page:
            if kind == "table":
                header, rows = value
                values.extend(str(x) for row in [header, *rows] for x in row)
            elif kind != "figure":
                values.append(value)
    return "\n".join(values)


def markdown(pages, root, output):
    lines = []
    for number, page in enumerate(pages, 1):
        if number > 1:
            lines.extend(["", "<!-- PDF page break -->", ""])
        for kind, value in page:
            if kind == "title":
                lines.extend(["# " + value, ""])
            elif kind == "heading":
                lines.extend(["## " + value, ""])
            elif kind == "table":
                head, rows = value
                lines.append("| " + " | ".join(head) + " |")
                lines.append("| " + " | ".join("---" for _ in head) + " |")
                lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
                lines.append("")
            elif kind == "figure":
                name, _ = value
                absolute = (root / "results/figures" / name).resolve().as_posix()
                lines.extend([f"![{name}]({absolute})", ""])
            else:
                lines.extend([value, ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_pdf(pages, root, output):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
    from PIL import Image as PILImage

    ink, teal, grey = colors.HexColor("#172B3A"), colors.HexColor("#116A73"), colors.HexColor("#53616B")
    styles = {
        "p": ParagraphStyle("Body", fontName="Times-Roman", fontSize=11, leading=14.5,
                            textColor=ink, spaceAfter=8),
        "title": ParagraphStyle("Title", fontName="Times-Bold", fontSize=20, leading=23,
                                textColor=ink, alignment=TA_CENTER, spaceAfter=12),
        "meta": ParagraphStyle("Meta", fontName="Times-Roman", fontSize=11, leading=15,
                               textColor=grey, alignment=TA_CENTER, spaceAfter=16),
        "heading": ParagraphStyle("Heading", fontName="Times-Bold", fontSize=13, leading=16,
                                  textColor=teal, spaceBefore=5, spaceAfter=9),
        "caption": ParagraphStyle("Caption", fontName="Times-Italic", fontSize=9.4, leading=11.8,
                                  textColor=grey, spaceBefore=4, spaceAfter=8),
        "reference": ParagraphStyle("Reference", fontName="Times-Roman", fontSize=9.6, leading=12,
                                    textColor=ink, spaceAfter=5),
        "small": ParagraphStyle("Small", fontName="Times-Roman", fontSize=9.4, leading=12,
                                textColor=grey, spaceBefore=7, spaceAfter=5),
        "cell": ParagraphStyle("Cell", fontName="Times-Roman", fontSize=9.6, leading=12, textColor=ink),
    }
    width, height = A4
    content_width = width - 92
    output_pdf = output / "MAAI901_Technical_Report_Ayman_Lakhnati.pdf"
    document = SimpleDocTemplate(str(output_pdf), pagesize=A4, rightMargin=46, leftMargin=46,
                                topMargin=42, bottomMargin=45, title=TITLE,
                                author="Ayman Lakhnati", subject="MAAI901 applied vision transformer report")
    story = []
    def paragraph(value, kind):
        return Paragraph(html.escape(str(value)).replace("\n", "<br/>"), styles[kind])
    for page_index, page in enumerate(pages):
        if page_index:
            story.append(PageBreak())
        for kind, value in page:
            if kind == "table":
                head, rows = value
                data = [[paragraph(x, "cell") for x in row] for row in [head, *rows]]
                widths = [95, 226, content_width - 321] if len(head) == 3 else [content_width * .28] + [content_width * .24] * 3
                table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5EFF0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F7")]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LINEBELOW", (0, 0), (-1, 0), .6, teal),
                    ("LINEBELOW", (0, -1), (-1, -1), .4, colors.HexColor("#CED9DD")),
                ]))
                story.append(table)
            elif kind == "figure":
                name, max_height = value
                path = root / "results/figures" / name
                with PILImage.open(path) as source:
                    iw, ih = source.size
                ratio = min(content_width / iw, max_height / ih)
                story.append(Image(str(path), width=iw * ratio, height=ih * ratio, hAlign="CENTER"))
            else:
                story.append(paragraph(value, kind))
    def page_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D6DEE1"))
        canvas.line(46, 34, width - 46, 34)
        canvas.setFont("Times-Roman", 8.6)
        canvas.setFillColor(grey)
        canvas.drawString(46, 22, "MAAI901 | Ayman Lakhnati | 25 October 2026")
        canvas.drawRightString(width - 46, 22, str(doc.page))
        canvas.restoreState()
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return output_pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--student-id", required=True, help="Private local metadata; do not publish report artifacts")
    parser.add_argument("--replace", action="store_true", help="Replace a prior locally rendered report revision")
    args = parser.parse_args()
    root = args.root.resolve()
    primary, run, dino, train, steps, epoch_loss, tuning = load_verified(root)
    pages = content(primary, run, dino, train, steps, epoch_loss, tuning, args.student_id)
    # Count body, abstract, metadata, tables, captions and references conservatively.
    text = plain_text(pages)
    words = len(re.findall(r"\S+", text))
    require(words <= 2500, f"Report exceeds conservative word limit: {words}")
    output = root / "outputs/report"
    if output.exists() and any(output.iterdir()) and not args.replace:
        raise FileExistsError("Report output exists; use --replace only for intentional local revision")
    output.mkdir(parents=True, exist_ok=True)
    markdown(pages, root, output)
    pdf = build_pdf(pages, root, output)
    from pypdf import PdfReader
    count = len(PdfReader(str(pdf)).pages)
    require(count == 8, f"Layout produced {count} pages; revise page geometry and visually inspect")
    files = [root / "results/summary.json", root / "results/dinov3/summary.json",
             root / "results/inference_run.json", root / "outputs/dinov3_dense/training_config.json",
             root / "outputs/dinov3_dense/training_steps.csv",
             root / "outputs/dinov3_tuning/tuning_results.json", root / "outputs/dinov3_tuning/tuning_trials.csv"]
    receipt = {"word_count_including_captions_references_tables": words, "pages": count,
               "private_student_identifier_included": True, "pdf": str(pdf),
               "inputs_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
               "status": "rendered; visual review still required"}
    (output / "build_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
