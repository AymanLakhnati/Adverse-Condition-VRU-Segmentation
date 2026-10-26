"""Exploratory recording-family influence from saved semantic confusion counts.

No images, model checkpoints, GPU computation, confidence intervals or training.
Does not change the prespecified full-validation hypothesis or its result files.
Run from any directory: python scripts/analyze_recording_sensitivity.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

CONDITIONS = ("fog", "night", "rain", "snow")
MODELS = ("hrda", "mic")
VRU_IDS = (11, 12, 18)
CLASS_NAMES = {11: "person", 12: "rider", 18: "bicycle"}
EXPECTED = {"fog": 100, "night": 106, "rain": 100, "snow": 100}


def recording_family(sequence):
    """Same conservative expression/rule as roadseg.tune_dinov3.recording_family.

    GOPR0376/GP010376/GP020376 become recording_0376. Unknown full names
    remain unchanged. This filename grouping does not establish scene identity.
    """
    match = re.fullmatch(r"(?:GOPR|GP\d{2})(\d{4})", sequence)
    return f"recording_{match.group(1)}" if match else sequence


def image_sequence(image_id):
    # Explicit frame suffix avoids accidentally accepting an arbitrary filename.
    match = re.fullmatch(r"(.+)_frame_\d+", image_id)
    if not match:
        raise ValueError(f"Expected ACDC sequence_frame_number ID: {image_id}")
    return match.group(1)


def vru_score(matrix):
    """Independent scalar-count IoU derivation; absent unions remain undefined."""
    classes = {}
    for k in VRU_IDS:
        tp = int(matrix[k, k])
        support = int(matrix[k, :].sum())
        predicted = int(matrix[:, k].sum())
        union = support + predicted - tp
        classes[CLASS_NAMES[k]] = tp / union if union else None
    values = list(classes.values())
    return {"vru_mean_iou": sum(values) / 3 if all(x is not None for x in values) else None,
            "class_iou": classes}


def endpoint(condition_totals):
    out = {}
    for j, model in enumerate(MODELS):
        scores = {c: vru_score(condition_totals[c][j]) for c in CONDITIONS}
        values = [scores[c]["vru_mean_iou"] for c in CONDITIONS]
        w = min(values) if all(value is not None for value in values) else None
        out[model] = {"conditions": scores, "W": w,
                      "weakest_conditions": [c for c in CONDITIONS if scores[c]["vru_mean_iou"] == w] if w is not None else []}
    first, second = out["hrda"]["W"], out["mic"]["W"]
    return {"models": out, "delta_W_pp": 100 * (second - first) if first is not None and second is not None else None}


def difference(second, first):
    return 100 * (second - first) if first is not None and second is not None else None


def interval_range(values):
    defined = [float(value) for value in values if value is not None]
    return {"minimum": min(defined) if defined else None, "maximum": max(defined) if defined else None,
            "defined_omissions": len(defined), "undefined_omissions": len(values) - len(defined)}


def analyze(root):
    path = root / "results/per_image_confusions.npz"
    source_summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        counts, conditions = archive["confusion"], archive["condition"].astype(str)
        ids, model_names = archive["image_id"].astype(str), archive["models"].astype(str).tolist()
    if model_names != list(MODELS) or counts.shape != (406, 2, 19, 19):
        raise ValueError("Expected complete primary two-model validation count array")
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0) or len(set(ids)) != 406:
        raise ValueError("Invalid confusion counts or repeated frame IDs")
    if {c: int(np.sum(conditions == c)) for c in CONDITIONS} != EXPECTED:
        raise ValueError("Validation conditions/counts differ from the original split")
    # Same GT support for each model, independently checked at every image/class.
    if not np.array_equal(counts[:, 0].sum(axis=2), counts[:, 1].sum(axis=2)):
        raise ValueError("Primary models have different reference-pixel supports")
    sequences = np.array([image_sequence(i) for i in ids])
    families = np.array([recording_family(s) for s in sequences])
    totals = {c: counts[conditions == c].sum(axis=0) for c in CONDITIONS}
    baseline = endpoint(totals)
    # Verify independent scalar arithmetic against the primary saved summary.
    for model in MODELS:
        if not math.isclose(baseline["models"][model]["W"], source_summary["models"][model]["W"], abs_tol=1e-14):
            raise ValueError("Independent full-data W does not match saved summary")
        for c in CONDITIONS:
            actual = baseline["models"][model]["conditions"][c]
            original = source_summary["models"][model]["conditions"][c]
            if not math.isclose(actual["vru_mean_iou"], original["vru_mean_iou"], abs_tol=1e-14):
                raise ValueError("Independent condition VRU score differs")
            for k, name in CLASS_NAMES.items():
                if not math.isclose(actual["class_iou"][name], original["per_class"][k]["iou"], abs_tol=1e-14):
                    raise ValueError("Independent per-class IoU differs")
    support = {}
    omissions = []
    for condition in CONDITIONS:
        mask = conditions == condition
        support[condition] = {"images": int(mask.sum()), "recording_families": len(set(families[mask])), "classes": {}}
        for k, name in CLASS_NAMES.items():
            per_image_support = counts[mask, 0, k, :].sum(axis=1)
            positive = per_image_support > 0
            support[condition]["classes"][name] = {
                "gt_pixels": int(per_image_support.sum()), "gt_positive_images": int(positive.sum()),
                "gt_positive_recording_families": len(set(families[mask][positive]))}
        for family in sorted(set(families[mask])):
            removed = mask & (families == family)
            omitted = counts[removed].sum(axis=0)
            remaining = {c: value.copy() for c, value in totals.items()}
            remaining[condition] -= omitted
            if np.any(remaining[condition] < 0):
                raise ValueError("Subtracted group exceeds the condition total")
            # Independent subtraction check: re-sum retained images from scratch.
            if not np.array_equal(remaining[condition], counts[mask & ~removed].sum(axis=0)):
                raise ValueError("Direct retained-image and subtractive count arithmetic differ")
            scores = endpoint(remaining)
            row = {"omitted_condition": condition, "omitted_family": family,
                   "omitted_sequence_aliases": sorted(set(sequences[removed])),
                   "omitted_images": int(removed.sum()), "remaining_condition_images": int((mask & ~removed).sum()),
                   "remaining_validation_images": 406 - int(removed.sum()),
                   "omitted_rider_gt_pixels": int(omitted[0, 12, :].sum()),
                   "result": scores,
                   "changes_from_full_validation_pp": {model: {
                       "affected_condition_V": difference(scores["models"][model]["conditions"][condition]["vru_mean_iou"], baseline["models"][model]["conditions"][condition]["vru_mean_iou"]),
                       "W": difference(scores["models"][model]["W"], baseline["models"][model]["W"])} for model in MODELS}}
            row["change_in_delta_W_pp"] = (scores["delta_W_pp"] - baseline["delta_W_pp"]) if scores["delta_W_pp"] is not None else None
            for model in MODELS:
                undefined = [c for c in CONDITIONS if scores["models"][model]["conditions"][c]["vru_mean_iou"] is None]
                row.setdefault("undefined_conditions", {})[model] = undefined
            omissions.append(row)
    summaries = {}
    for c in CONDITIONS:
        rows = [r for r in omissions if r["omitted_condition"] == c]
        summaries[c] = {"omissions": len(rows), "delta_W_pp_after_omission": interval_range([r["result"]["delta_W_pp"] for r in rows]),
                        "change_in_delta_W_pp": interval_range([r["change_in_delta_W_pp"] for r in rows]),
                        "affected_condition_V_change_pp": {m: interval_range([r["changes_from_full_validation_pp"][m]["affected_condition_V"] for r in rows]) for m in MODELS}}
    return {"scope": "Exploratory leave-one-recording-family-out influence on completed fixed-model validation counts; no inference",
            "not_a_confidence_interval": True, "prespecified_hypothesis_unchanged": True,
            "grouping": {"same_rule_as": "roadseg.tune_dinov3.recording_family",
                         "expression": r"(?:GOPR|GP\d{2})(\d{4})",
                         "rule": "Parse sequence prefix before _frame_; fullmatch alias expression; condition/recording_#### is group. Unmatched sequence prefixes remain unchanged.",
                         "limitation": "Conservative filename heuristic; recording family does not prove geographic/scene independence or temporal adjacency"},
            "source_counts_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_summary_sha256": hashlib.sha256((root / "results/summary.json").read_bytes()).hexdigest(),
            "score_unit": "fraction", "difference_unit": "percentage_points",
            "baseline": baseline, "ground_truth_support": support,
            "omission_summary": summaries, "omissions": omissions,
            "arithmetic_checks": {"full_scores_match_saved_primary": True, "GT_support_matches_between_models": True,
                                   "every_subtraction_matches_direct_retained_image_sum": True},
            "limitations": ["Post hoc descriptive sensitivity, not a new hypothesis test",
                            "Each omission changes the evaluated image mixture and sample size",
                            "Few recording families and sparse VRU support limit interpretation",
                            "Undefined scores are retained as null, not replaced by zero",
                            "No inference about retraining, demographic fairness or driving safety"]}


def number(value):
    return "undefined" if value is None else f"{value:+.2f}"


def markdown(result):
    lines = ["# Exploratory recording-family sensitivity", "",
             "Computed from saved per-image HRDA/MIC confusion counts only. No model was rerun. This post hoc influence analysis neither changes the prespecified hypothesis nor supplies a confidence interval.", "",
             "## Grouping and support", "",
             "The rule matches the tuning helper exactly: GOPR#### and GPxx#### share recording_#### within a condition. We parse the sequence prefix of each image ID. Unknown naming conventions remain separate. This conservative filename heuristic does not establish geographic independence, duplicate frames or temporal adjacency.", "",
             "| Condition | Images | Families | Rider-positive images | Rider-positive families | Rider GT pixels |",
             "|---|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        s = result["ground_truth_support"][c]
        rider = s["classes"]["rider"]
        lines.append(f"| {c.title()} | {s['images']} | {s['recording_families']} | {rider['gt_positive_images']} | {rider['gt_positive_recording_families']} | {rider['gt_pixels']:,} |")
    lines.extend(["", "The snow rider class occupies only two annotated images. Counts are semantic pixels and positive frames, not rider-instance counts. Full person/bicycle support is retained in the JSON.", "",
                  "## Leave-one-family-out influence", "",
                  "For each condition/recording family, remove all its images, re-pool the remaining counts, then recompute both models' V(c), their independent minima W, and MIC-minus-HRDA delta W. Other conditions remain unchanged. No model or threshold is selected using these results. The full-data delta W remains +4.13 percentage points.", "",
                  "| Omitted condition | Families tried | Delta W range after omission (pp) | Undefined delta W cases | HRDA affected-condition V change range (pp) | MIC affected-condition V change range (pp) |",
                  "|---|---:|---:|---:|---:|---:|"])
    for c in CONDITIONS:
        summary = result["omission_summary"][c]
        w = summary["delta_W_pp_after_omission"]
        h, m = (summary["affected_condition_V_change_pp"][model] for model in MODELS)
        lines.append(f"| {c.title()} | {summary['omissions']} | {number(w['minimum'])} to {number(w['maximum'])} | {w['undefined_omissions']} | {number(h['minimum'])} to {number(h['maximum'])} | {number(m['minimum'])} to {number(m['maximum'])} |")
    lines.extend(["", "These minima/maxima describe the enumerated omissions; they are not percentile intervals, confidence bounds or estimates of unseen-scene performance.", "",
                  "| Omitted family | Removed / remaining condition images | HRDA V after omission | MIC V after omission | Delta W after omission (pp) |",
                  "|---|---:|---:|---:|---:|"])
    for r in result["omissions"]:
        c = r["omitted_condition"]
        values = [r["result"]["models"][m]["conditions"][c]["vru_mean_iou"] for m in MODELS]
        formatted = ["undefined" if value is None else f"{100*value:.2f}%" for value in values]
        lines.append(f"| {c}/{r['omitted_family']} | {r['omitted_images']} / {r['remaining_condition_images']} | {formatted[0]} | {formatted[1]} | {number(r['result']['delta_W_pp'])} |")
    undefined = [r for r in result["omissions"] if r["result"]["delta_W_pp"] is None]
    lines.extend(["", f"There are {len(result['omissions'])} total omissions and {len(undefined)} undefined delta W cases. JSON retains all class IoUs, affected conditions, aliases and changes from the full-data baseline.", "",
                  "## Interpretation and checks", "",
                  "The analysis shows how heavily a condition estimate can depend on one recording family. Removing a family also changes image composition and class support, so differences cannot isolate a causal weather or recording effect. Few families make a conventional recording-cluster confidence interval unreliable; none is supplied here. A changed or undefined omitted-data endpoint does not overwrite the original complete-validation result.", "",
                  "Independent scalar TP/union arithmetic matched every saved primary class/condition VRU score and W. For every omission, subtracting group counts was checked against a direct re-sum of retained images. Per-image GT supports also matched between HRDA and MIC. Source SHA256 hashes are recorded in results/recording_sensitivity.json.", "",
                  "Reproduce from repository root: `python scripts/analyze_recording_sensitivity.py`. Inputs are results/per_image_confusions.npz and results/summary.json; outputs are results/recording_sensitivity.json and this document. No raw RGB or GT images are redistributed."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    result = analyze(root)
    (root / "results/recording_sensitivity.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (root / "docs/RECORDING_SENSITIVITY.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"omissions": len(result["omissions"]), "summary": result["omission_summary"],
                      "arithmetic_checks": result["arithmetic_checks"]}, indent=2))


if __name__ == "__main__":
    main()
