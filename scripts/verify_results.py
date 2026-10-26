"""Verify saved results without raw ACDC data, model inference, torch or a GPU.

Run from any working directory: python /path/to/scripts/verify_results.py
or supply --root /path/to/project. Nothing is written unless --output is set.
This checks artifact consistency, not dataset authenticity or deployment safety.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CONDITIONS = ("fog", "night", "rain", "snow")
EXPECTED = {"fog": 100, "night": 106, "rain": 100, "snow": 100}
MODELS = ("hrda", "mic")
VRU = (11, 12, 18)
NAMES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
         "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
         "truck", "bus", "train", "motorcycle", "bicycle")
MODEL = "vit_small_patch16_dinov3.lvd1689m"


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant {value}: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _equal(actual, expected, where):
    if expected is None:
        _check(actual is None, f"{where}: expected undefined/null, got {actual!r}")
    elif isinstance(expected, bool):
        _check(actual is expected, f"{where}: boolean mismatch")
    elif isinstance(expected, int):
        _check(type(actual) is int and actual == expected, f"{where}: integer mismatch ({actual!r} != {expected})")
    elif isinstance(expected, float):
        _check(type(actual) in (int, float) and math.isfinite(actual) and
               math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
               f"{where}: numeric mismatch ({actual!r} != {expected!r})")
    else:
        _check(actual == expected, f"{where}: value mismatch")


def _ratio(a, b):
    return a / b if b else None


def _derive(cm):
    """Independent scalar TP/FP/FN formulas, including undefined denominators."""
    rows = []
    for k, name in enumerate(NAMES):
        tp = int(cm[k, k])
        support, predicted = int(sum(cm[k, :])), int(sum(cm[:, k]))
        fp, fn = predicted - tp, support - tp
        union = tp + fp + fn
        rows.append(dict(class_id=k, class_name=name, tp=tp, fp=fp, fn=fn,
                         support=support, predicted=predicted, union=union,
                         precision=_ratio(tp, tp + fp), recall=_ratio(tp, tp + fn),
                         f1=_ratio(2 * tp, 2 * tp + fp + fn), iou=_ratio(tp, union)))
    ious = [r["iou"] for r in rows]
    defined = [v for v in ious if v is not None]
    vru = [ious[k] for k in VRU]
    total = int(cm.sum())
    return dict(valid_pixels=total, pixel_accuracy=_ratio(sum(r["tp"] for r in rows), total),
                miou_present=sum(defined) / len(defined) if defined else None,
                miou_19=sum(defined) / 19 if len(defined) == 19 else None,
                defined_iou_classes=len(defined),
                vru_mean_iou=sum(vru) / 3 if all(v is not None for v in vru) else None,
                per_class=rows)


def _metrics(saved, expected, where):
    for key, value in expected.items():
        _check(key in saved, f"{where}: missing {key}")
        if key == "per_class":
            _check(isinstance(saved[key], list) and len(saved[key]) == 19, f"{where}: expected 19 class rows")
            for k, row in enumerate(value):
                for name, item in row.items():
                    _check(name in saved[key][k], f"{where}.class{k}: missing {name}")
                    _equal(saved[key][k][name], item, f"{where}.class{k}.{name}")
        else:
            _equal(saved[key], value, f"{where}.{key}")


def _strings(array, count, name):
    _check(array.ndim == 1 and len(array) == count and array.dtype.kind in "US",
           f"{name}: expected {count} non-object strings")
    result = array.astype(str)
    _check(all(x for x in result), f"{name}: empty string")
    return result


def _counts(array, shape, where):
    _check(array.shape == shape and array.dtype.kind in "iu", f"{where}: expected integer shape {shape}")
    _check(bool(np.all(array >= 0)) and bool(np.all(array <= 1920 * 1080)), f"{where}: invalid pixel counts")
    result = array.astype(np.int64)
    _check(bool(np.all(result.sum(axis=(-2, -1)) <= 1920 * 1080)), f"{where}: exceeds original image pixels")
    return result


def _coverage(ids, conditions, where):
    _check(len(set(ids)) == 406, f"{where}: duplicate image IDs")
    _check(all(re.fullmatch(r".+_frame_\d+", x) for x in ids), f"{where}: malformed image ID")
    _check(dict(Counter(conditions)) == EXPECTED, f"{where}: official condition counts mismatch")


def _model_scores(saved, stack, conditions, where):
    _check(set(saved["conditions"]) == set(CONDITIONS), f"{where}: missing/extra condition")
    expected = {c: _derive(stack[conditions == c].sum(axis=0)) for c in CONDITIONS}
    for c in CONDITIONS:
        _metrics(saved["conditions"][c], expected[c], f"{where}.{c}")
    _metrics(saved["pooled"], _derive(stack.sum(axis=0)), f"{where}.pooled")
    vru = [expected[c]["vru_mean_iou"] for c in CONDITIONS]
    w = min(vru) if all(x is not None for x in vru) else None
    _equal(saved["W"], w, f"{where}.W")
    weakest = [c for c in CONDITIONS if expected[c]["vru_mean_iou"] == w] if w is not None else []
    _check(saved["weakest_conditions"] == weakest, f"{where}: weakest conditions mismatch")
    return w, expected


def _hash_value(value, where):
    _check(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"{where}: invalid SHA256")
    return value


def _content_hash(rows):
    values = sorted((r["image_id"], r["image_sha256"], r["label_sha256"]) for r in rows)
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _train_rows(rows, where):
    _check(bool(rows), f"{where}: empty training manifest")
    seen = set()
    for row in rows:
        _check(row["split"] == "train" and row["condition"] in CONDITIONS, f"{where}: non-training row")
        _check(row["image_id"] not in seen, f"{where}: duplicate frame ID")
        seen.add(row["image_id"])
        for key in ("image_sha256", "label_sha256"):
            _hash_value(row[key], f"{where}.{key}")
        for key in ("image_path", "label_path"):
            # Treat recorded Windows paths as metadata even when verifying elsewhere.
            parts = re.split(r"[\\/]", row[key])
            _check("train" in parts and "val" not in parts and "test" not in parts,
                   f"{where}: path is not exclusively training")
    return seen


def _family(sequence):
    match = re.fullmatch(r"(?:GOPR|GP\d{2})(\d{4})", sequence)
    return f"recording_{match.group(1)}" if match else sequence


def _verify_training(root, validation_ids):
    base = root / "outputs/dinov3_dense"
    tuning_base = root / "outputs/dinov3_tuning"
    train, tuning = _json(base / "training_config.json"), _json(tuning_base / "tuning_results.json")
    _check(train["status"] == "completed" and train["completed_fixed_full_protocol"] is True and
           train["limited_smoke_run"] is False, "Training: completed full protocol required")
    for name, expected in {"epochs": 5, "completed_epochs": 5, "training_images": 1600,
                           "available_train_images": 1600, "batch_size": 4, "seed": 42,
                           "head_parameters": 7315, "optimizer_steps": 2000, "skipped_all_ignore_batches": 0}.items():
        _equal(train[name], expected, f"Training.{name}")
    _check(train["model_name"] == tuning["model_name"] == MODEL, "Training/tuning: model mismatch")
    _check(train["encoder_state_unchanged"] is True and train["encoder_frozen"] is True and
           train["head_state_changed"] is True, "Training: frozen/head-change flags invalid")
    enc = _hash_value(train["encoder_state_sha256"], "Training.encoder hash")
    _check(train["encoder_after_sha256"] == tuning["encoder_state_sha256"] == enc, "Training/tuning: encoder hash mismatch")
    initial = _hash_value(train["head_before_sha256"], "Training.initial head")
    _check(tuning["initial_head_sha256"] == initial, "Training/tuning: fresh initialization mismatch")
    _check(_hash_value(train["head_after_sha256"], "Training.final head") != initial,
           "Training: final head equals initialization")
    _check(train["official_validation_used_for_selection"] is False and
           train["internal_training_development_used_for_learning_rate_selection"] is True and
           tuning["official_validation_or_test_used"] is False, "Tuning: official evaluation used for selection")
    _check(train["tuning_results_sha256"] == _sha(tuning_base / "tuning_results.json"), "Training/tuning: file hash linkage mismatch")
    _check(str(train["tuning_results_path"]).replace("\\", "/").endswith("outputs/dinov3_tuning/tuning_results.json"),
           "Training: unexpected tuning artifact reference")
    _check(tuning["status"] == "completed" and tuning["trial_epochs"] == 1 and tuning["batch_size"] == 4 and
           tuning["seed"] == 42 and tuning["encoder_frozen"] is True, "Tuning: incomplete/changed protocol")
    trials = tuning["trials"]
    _check(len(trials) == 3 and sorted(t["learning_rate"] for t in trials) == [0.0003, 0.001, 0.003],
           "Tuning: expected exactly three prescribed learning rates")
    eligible = [t for t in trials if t["development_vru_mean_iou"] is not None]
    metric = "development_vru_mean_iou" if eligible else "development_miou_present"
    if not eligible:
        eligible = [t for t in trials if t[metric] is not None]
    _check(bool(eligible), "Tuning: no eligible trial")
    for trial in trials:
        _check(trial["fit_epochs"] == 1 and trial["encoder_unchanged"] is True and trial["fit_optimizer_steps"] > 0,
               "Tuning: incomplete/non-frozen trial")
        for name in ("development_vru_mean_iou", "development_miou_present"):
            score = trial[name]
            _check(score is None or (type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 1),
                   "Tuning: invalid selection score")
    winner = min(eligible, key=lambda t: (-t[metric], t["learning_rate"]))
    _equal(tuning["selected_learning_rate"], float(winner["learning_rate"]), "Tuning.selected rate")
    _equal(tuning["selected_score"], float(winner[metric]), "Tuning.selected score")
    _check(tuning["selected_metric"] == metric and bool(tuning["selection_fallback"]) ==
           (metric != "development_vru_mean_iou"), "Tuning: selection metric/fallback mismatch")
    _equal(train["learning_rate"], float(winner["learning_rate"]), "Training.selected learning rate")
    fit, dev, full = (_csv(tuning_base / "fit_manifest.csv"), _csv(tuning_base / "development_manifest.csv"),
                      _csv(base / "training_manifest.csv"))
    fit_ids, dev_ids, full_ids = (_train_rows(fit, "Fitting manifest"), _train_rows(dev, "Development manifest"),
                                 _train_rows(full, "Refit manifest"))
    _check(not fit_ids & dev_ids and fit_ids | dev_ids == full_ids and len(full_ids) == 1600,
           "Tuning/refit: frame partitions do not cover the same 1600 unique training images")
    _check(not full_ids & set(validation_ids), "Training/validation frame ID overlap")
    _check(Counter(r["condition"] for r in full) == {c: 400 for c in CONDITIONS}, "Refit: expected 400 train images per condition")
    groups = []
    for rows in (fit, dev):
        values = set()
        for row in rows:
            sequence = row["image_id"].rsplit("_frame_", 1)[0]
            family = _family(sequence)
            _check(row["sequence_directory"] == sequence and row["recording_family"] == family and
                   row["sequence_group"] == row["condition"] + "/" + family, "Tuning: recording alias metadata mismatch")
            values.add((row["condition"], family))
        groups.append(values)
    _check(not groups[0] & groups[1], "Tuning: recording-family overlap")
    split = tuning["split"]
    _check(split["fit_images"] == len(fit) and split["development_images"] == len(dev) and
           all(split[k] is True for k in ("recording_families_disjoint", "sequence_groups_disjoint", "frame_ids_disjoint")),
           "Tuning: split-count/disjointness metadata mismatch")
    for filename, key in (("fit_manifest.csv", "fit_manifest_sha256"), ("development_manifest.csv", "development_manifest_sha256")):
        _check(_sha(tuning_base / filename) == tuning[key], f"Tuning: {filename} hash mismatch")
    content = _content_hash(full)
    _check(content == _content_hash(fit + dev) == train["all_training_content_sha256"] == tuning["all_training_content_sha256"],
           "Tuning/refit: training image/label content hash mismatch")
    ids_hash = hashlib.sha256("\n".join(sorted(full_ids)).encode()).hexdigest()
    _check(tuning["all_training_ids_sha256"] == ids_hash, "Tuning: all-training ID hash mismatch")
    steps = _csv(base / "training_steps.csv")
    _check(len(steps) == 2000, "Training log: expected 2000 optimizer steps")
    for index, row in enumerate(steps, 1):
        _check(int(row["global_step"]) == index and int(row["epoch"]) == (index - 1) // 400 + 1,
               "Training log: step/epoch sequence mismatch")
        _check(int(row["images"]) == 4 and int(row["valid_pixels"]) > 0, "Training log: invalid batch coverage")
        for name in ("loss", "gradient_norm", "learning_rate"):
            value = float(row[name])
            _check(math.isfinite(value) and value >= 0 and (name != "gradient_norm" or value > 0),
                   f"Training log: invalid {name}")
        _equal(float(row["learning_rate"]), float(train["learning_rate"]), "Training log.fixed learning rate")
        image_ids = row["image_ids"].split()
        _check(len(image_ids) == 4 and set(image_ids) <= full_ids, "Training log: non-training image ID")
    for epoch in range(1, 6):
        ids = [i for r in steps if int(r["epoch"]) == epoch for i in r["image_ids"].split()]
        _check(len(ids) == 1600 and set(ids) == full_ids, f"Training epoch {epoch}: missing/repeated frame coverage")
    identity = _json(root / "results/dinov3/prediction_identity.json")
    _check(identity["encoder_state_sha256"] == enc and identity["head_sha256"] == _sha(base / "head.pt"),
           "DINO evaluation: evaluated head/encoder identity mismatch")
    _check(identity["tile_size"] == train["inference_tile_size"] == 512 and
           identity["stride"] == train["inference_stride"] == 384, "DINO evaluation: tiling mismatch")
    return {"completed_epochs": 5, "training_images": 1600, "optimizer_steps": len(steps),
            "selected_learning_rate": train["learning_rate"], "fit_images": len(fit), "development_images": len(dev),
            "hash_linkage_verified": True, "recording_family_disjoint_internal_split": True}


def verify_results(root: Path) -> dict:
    """Return verification receipt, or ValueError on any missing/inconsistent artifact."""
    root = Path(root).resolve()
    try:
        primary = _json(root / "results/summary.json")
        dino = _json(root / "results/dinov3/summary.json")
        run = _json(root / "results/inference_run.json")
        with np.load(root / "results/per_image_confusions.npz", allow_pickle=False) as data:
            counts = _counts(data["confusion"], (406, 2, 19, 19), "Primary counts")
            ids = _strings(data["image_id"], 406, "Primary IDs")
            conditions = _strings(data["condition"], 406, "Primary conditions")
            _check(data["models"].dtype.kind in "US" and data["models"].tolist() == list(MODELS), "Primary: model order must be hrda,mic")
            _check(data["class_names"].dtype.kind in "US" and data["class_names"].tolist() == list(NAMES), "Primary: class-name order mismatch")
        _coverage(ids, conditions, "Primary")
        _check(np.array_equal(counts[:, 0].sum(axis=2), counts[:, 1].sum(axis=2)), "Primary: GT supports differ between models")
        _check(run["status"] == "completed" and run["processed"] == 406, "Primary inference: incomplete")
        _check(primary["dataset"]["images"] == 406 and primary["dataset"]["condition_counts"] == EXPECTED and
               primary["dataset"]["matches_original_validation_counts"] is True, "Primary summary: coverage mismatch")
        _check(primary["score_unit"] == "fraction" and primary["delta_unit"] == "percentage_points", "Primary: score-unit mismatch")
        scores = {m: _model_scores(primary["models"][m], counts[:, j], conditions, m) for j, m in enumerate(MODELS)}
        condition_delta = {c: 100 * (scores["mic"][1][c]["vru_mean_iou"] - scores["hrda"][1][c]["vru_mean_iou"]) for c in CONDITIONS}
        delta_w = 100 * (scores["mic"][0] - scores["hrda"][0])
        hypothesis = primary["hypothesis"]
        _equal(hypothesis["delta_W_pp"], delta_w, "Hypothesis.delta_W_pp")
        for c in CONDITIONS:
            _equal(hypothesis["condition_delta_pp"][c], condition_delta[c], f"Hypothesis.{c}")
        _equal(hypothesis["threshold_delta_W_pp"], 2.0, "Hypothesis.W threshold")
        _equal(hypothesis["threshold_each_condition_pp"], -1.0, "Hypothesis.condition threshold")
        passed = delta_w >= 2 - 1e-10 and min(condition_delta.values()) >= -1 - 1e-10
        _equal(hypothesis["supported"], passed, "Hypothesis.supported")
        _equal(hypothesis["status"], "supported" if passed else "not_supported", "Hypothesis.status")
        with np.load(root / "results/dinov3/per_image_confusions.npz", allow_pickle=False) as data:
            dc = _counts(data["confusions"], (406, 19, 19), "DINO counts")
            di = _strings(data["image_ids"], 406, "DINO IDs")
            conditions_d = _strings(data["conditions"], 406, "DINO conditions")
        _coverage(di, conditions_d, "DINO")
        _check(set(di) == set(ids), "DINO/primary: validation frame IDs differ")
        ordering = {name: j for j, name in enumerate(di)}
        indices = [ordering[name] for name in ids]
        _check(np.array_equal(conditions_d[indices], conditions), "DINO/primary: per-ID condition differs")
        _check(np.array_equal(dc[indices].sum(axis=2), counts[:, 0].sum(axis=2)), "DINO/primary: per-ID GT supports differ")
        _check(dino["validation_images"] == 406 and dino["condition_image_counts"] == EXPECTED and
               dino["learning_setting"] == "supervised ACDC train labels" and dino["score_unit"] == "fraction",
               "DINO summary: coverage/supervision mismatch")
        _model_scores(dino, dc, conditions_d, "DINO")
        training = _verify_training(root, ids)
        return {"status": "passed", "validation_images": 406, "condition_counts": EXPECTED,
                "primary_model_order": list(MODELS), "independent_metric_formulas": True,
                "all_class_condition_pooled_metrics_match": True, "cross_model_GT_support_matches": True,
                "hypothesis_status": hypothesis["status"], "delta_W_pp": delta_w, "training": training,
                "numeric_tolerance": {"relative": 1e-12, "absolute": 1e-12},
                "limitations": "Artifact consistency only; no raw-data authenticity, pixel-identical upstream, scene independence or safety claim. Bootstrap intervals are not recomputed."}
    except ValueError:
        raise
    except (OSError, KeyError, TypeError, IndexError, AttributeError, OverflowError, StopIteration) as error:
        raise ValueError(f"Result verification failed: {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Optional JSON receipt; otherwise read-only")
    args = parser.parse_args()
    try:
        result = verify_results(args.root)
    except ValueError as error:
        result = {"status": "failed", "error": str(error)}
        code = 1
    else:
        code = 0
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
