"""Exact pooled semantic-segmentation metrics; GT rows, prediction columns.

Run ``python -m roadseg.metrics manifest.csv --output results/``.
Inputs must be integer train-ID masks, not original Cityscapes label IDs.
Only ground-truth 255 is ignored. Scores use [0,1]; differences use pp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

N_CLASSES = 19
IGNORE = 255
CONDITIONS = ("fog", "night", "rain", "snow")
VRU = (11, 12, 18)
CLASS_NAMES = ("road", "sidewalk", "building", "wall", "fence", "pole",
               "traffic light", "traffic sign", "vegetation", "terrain", "sky",
               "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
MODELS = ("hrda", "mic")


def confusion_matrix(gt, prediction):
    """Count valid pixels; reject illegal GT and predictions on valid GT.

    Predictions at ignored GT are deliberately unconstrained and uncounted.
    Both masks must have matching 2D integer shapes; no implicit resizing.
    """
    gt, prediction = np.asarray(gt), np.asarray(prediction)
    if gt.ndim != 2 or prediction.ndim != 2 or gt.shape != prediction.shape:
        raise ValueError("Ground-truth and prediction masks must have identical 2D shapes")
    if not np.issubdtype(gt.dtype, np.integer) or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError("Masks must contain integer train IDs")
    valid = gt != IGNORE
    if np.any((gt[valid] < 0) | (gt[valid] >= N_CLASSES)):
        raise ValueError("GT must use train IDs 0..18 or ignore 255; labelIds are not remapped")
    if np.any((prediction[valid] < 0) | (prediction[valid] >= N_CLASSES)):
        raise ValueError("Prediction must use IDs 0..18 wherever GT is not 255")
    encoded = N_CLASSES * gt[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def metrics_from_confusion(confusion):
    """Derive scores from pooled integer counts; absent unions remain null."""
    cm = np.asarray(confusion)
    if cm.shape != (N_CLASSES, N_CLASSES) or not np.issubdtype(cm.dtype, np.integer) or np.any(cm < 0):
        raise ValueError("Expected a nonnegative integer 19x19 confusion matrix")
    tp = np.diag(cm)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    rows = []
    for c in range(N_CLASSES):
        union = int(support[c] + predicted[c] - tp[c])
        rows.append(dict(class_id=c, class_name=CLASS_NAMES[c], tp=int(tp[c]),
                         fp=int(predicted[c] - tp[c]), fn=int(support[c] - tp[c]),
                         support=int(support[c]), predicted=int(predicted[c]), union=union,
                         precision=_ratio(tp[c], predicted[c]), recall=_ratio(tp[c], support[c]),
                         f1=_ratio(2 * tp[c], support[c] + predicted[c]), iou=_ratio(tp[c], union)))
    ious = [r["iou"] for r in rows]
    defined = [v for v in ious if v is not None]
    vru_values = [ious[c] for c in VRU]
    return dict(valid_pixels=int(cm.sum()), pixel_accuracy=_ratio(tp.sum(), cm.sum()),
                miou_present=float(np.mean(defined)) if defined else None,
                miou_19=float(np.mean(ious)) if len(defined) == N_CLASSES else None,
                defined_iou_classes=len(defined),
                vru_mean_iou=float(np.mean(vru_values)) if all(v is not None for v in vru_values) else None,
                per_class=rows)


def summarize(condition_confusions):
    """Compare MIC minus HRDA; minima are calculated independently per model."""
    output = {"models": {}, "conditions": list(CONDITIONS), "score_unit": "fraction",
              "delta_unit": "percentage_points", "vru_train_ids": list(VRU)}
    for model in MODELS:
        groups = condition_confusions.get(model, {})
        unexpected = set(groups) - set(CONDITIONS)
        if unexpected:
            raise ValueError(f"Unknown conditions: {unexpected}")
        scores = {c: metrics_from_confusion(groups[c]) for c in CONDITIONS if c in groups}
        values = {c: scores[c]["vru_mean_iou"] for c in scores}
        complete = set(values) == set(CONDITIONS) and all(v is not None for v in values.values())
        worst = min(values.values()) if complete else None
        pooled = sum((np.asarray(cm) for cm in groups.values()), np.zeros((19, 19), dtype=np.int64))
        output["models"][model] = dict(conditions=scores, pooled=metrics_from_confusion(pooled),
                  W=worst, weakest_conditions=[c for c, v in values.items() if v == worst] if complete else [],
                  missing_conditions=[c for c in CONDITIONS if c not in scores])
    delta = {}
    for c in CONDITIONS:
        first = output["models"]["hrda"]["conditions"].get(c, {}).get("vru_mean_iou")
        second = output["models"]["mic"]["conditions"].get(c, {}).get("vru_mean_iou")
        delta[c] = 100 * (second - first) if first is not None and second is not None else None
    h, m = (output["models"][model]["W"] for model in MODELS)
    dw = 100 * (m - h) if h is not None and m is not None else None
    resolved = dw is not None and all(v is not None for v in delta.values())
    passes = bool(dw >= 2.0 - 1e-10 and all(v >= -1.0 - 1e-10 for v in delta.values())) if resolved else None
    output["hypothesis"] = dict(status="supported" if passes else "not_supported" if resolved else "unresolved",
          supported=passes, delta_W_pp=dw, condition_delta_pp=delta,
          threshold_delta_W_pp=2.0, threshold_each_condition_pp=-1.0,
          interpretation="Descriptive threshold result, not proof of safety or statistical significance")
    return output


def paired_bootstrap(confusions, conditions, repeats=1000, seed=42):
    """Within-condition paired image bootstrap; aggregate counts before IoU.

    confusions shape: (n_images, 2_models, 19, 19). Each draw keeps both
    models paired and preserves each condition's sample size. Undefined
    replicates are counted and omitted from that endpoint's interval.
    """
    matrices = np.asarray(confusions)
    conditions = np.asarray(conditions)
    if matrices.shape != (len(conditions), 2, 19, 19) or repeats < 1:
        raise ValueError("Invalid paired bootstrap input or repeat count")
    rng = np.random.default_rng(seed)
    indices = {c: np.flatnonzero(conditions == c) for c in CONDITIONS}
    draws = {c: [] for c in CONDITIONS}
    draws["delta_W_pp"] = []
    for _ in range(repeats):
        aggregate = {model: {} for model in MODELS}
        for c, ids in indices.items():
            if len(ids):
                sample = rng.choice(ids, size=len(ids), replace=True)
                totals = matrices[sample].sum(axis=0)
                for j, model in enumerate(MODELS):
                    aggregate[model][c] = totals[j]
        hypothesis = summarize(aggregate)["hypothesis"]
        draws["delta_W_pp"].append(hypothesis["delta_W_pp"])
        for c in CONDITIONS:
            draws[c].append(hypothesis["condition_delta_pp"][c])
    endpoints = {}
    for name, values in draws.items():
        defined = [v for v in values if v is not None]
        endpoints[name] = {"percentile_95_pp": np.quantile(defined, [0.025, 0.975]).tolist() if defined else None,
                           "defined_replicates": len(defined), "undefined_replicates": repeats - len(defined)}
    return dict(repeats=repeats, seed=seed, resampling="paired images stratified by condition",
                endpoints=endpoints, limitations="Descriptive conditional intervals; image independence assumed; no scene clustering, training variability or domain shift coverage")


def _read_mask(path):
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Expected 2D integer ID mask, not RGB visualization: {path}")
    return array


def _aligned_name(path, image_id):
    stem = path.stem
    return stem == image_id or (stem.startswith(image_id) and stem[len(image_id):len(image_id) + 1] in ("_", "-", "."))


def _sha256_file(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_manifest(manifest_path, output, bootstrap_repeats=1000, seed=42, require_official_validation=False, path_root=None):
    """Evaluate explicit path pairs relative to manifest directory by default.

    Names must start with their image_id. GT names must contain
    labelTrainIds, making the label encoding declaration explicit.
    Full original ACDC validation counts can be required with a CLI flag.
    An explicit path_root instead resolves relative mask paths against that
    directory, e.g. the project root used by run_inference.py manifests.
    """
    manifest_path, output = Path(manifest_path).resolve(), Path(output)
    path_root = manifest_path.parent if path_root is None else Path(path_root).resolve()
    if not path_root.is_dir():
        raise ValueError(f"Mask path root must be an existing directory: {path_root}")
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {"image_id", "condition", "gt_path", "hrda_pred_path", "mic_pred_path"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Nonempty manifest required with columns {sorted(required)}")
    if (frame[list(required)] == "").any().any() or frame.image_id.duplicated().any():
        raise ValueError("Manifest has blank values or duplicate image IDs")
    frame["condition"] = frame.condition.str.lower().replace({"nighttime": "night"})
    if not set(frame.condition).issubset(CONDITIONS):
        raise ValueError(f"Conditions must be {CONDITIONS}")
    counts = frame.condition.value_counts().to_dict()
    expected = dict(fog=100, night=106, rain=100, snow=100)
    if require_official_validation and counts != expected:
        raise ValueError(f"Original ACDC validation counts required: {expected}; got {counts}")
    records, matrices = [], []
    verified_hashes=0
    used = {column: set() for column in required if column.endswith("path")}
    for row in frame.to_dict("records"):
        paths = {}
        for column in used:
            candidate = Path(row[column])
            path = candidate.resolve() if candidate.is_absolute() else (path_root / candidate).resolve()
            if not _aligned_name(path, row["image_id"]):
                raise ValueError(f"Filename/image_id alignment failed: {row['image_id']} / {path.name}")
            if path in used[column]:
                raise ValueError(f"Mask reused for multiple images: {path}")
            used[column].add(path)
            expected_hash=row.get(column.removesuffix("_path")+"_sha256","")
            if expected_hash:
                if _sha256_file(path)!=expected_hash:
                    raise ValueError(f"Manifest SHA256 mismatch for {column}: {path}")
                verified_hashes+=1
            paths[column] = path
        if "labelTrainIds" not in paths["gt_path"].stem:
            raise ValueError("GT filename must explicitly identify labelTrainIds; no labelIds remapping is performed")
        gt = _read_mask(paths["gt_path"])
        pair = [confusion_matrix(gt, _read_mask(paths[f"{model}_pred_path"])) for model in MODELS]
        matrices.append(pair)
        records.append(dict(image_id=row["image_id"], condition=row["condition"],
                            valid_pixels=int((gt != IGNORE).sum()), ignored_pixels=int((gt == IGNORE).sum()),
                            height=gt.shape[0], width=gt.shape[1]))
    matrices = np.asarray(matrices, dtype=np.int64)
    aggregate = {model: {c: matrices[frame.condition.to_numpy() == c, j].sum(axis=0)
                       for c in CONDITIONS if c in counts} for j, model in enumerate(MODELS)}
    summary = summarize(aggregate)
    summary["dataset"] = dict(images=len(frame), condition_counts=counts,
          matches_original_validation_counts=counts == expected,
          caveat="Counts alone cannot verify official split membership or provenance",
          gt_encoding="trainIds 0..18; ignore 255", ground_truth_rows=True,
          mask_path_root=str(path_root),
          verified_manifest_mask_sha256_count=verified_hashes,
          manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    if bootstrap_repeats:
        summary["bootstrap"] = paired_bootstrap(matrices, frame.condition.to_numpy(), bootstrap_repeats, seed)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "per_image_confusions.npz", confusion=matrices,
            image_id=frame.image_id.to_numpy(dtype=str), condition=frame.condition.to_numpy(dtype=str),
            models=np.asarray(MODELS), class_names=np.asarray(CLASS_NAMES))
    pd.DataFrame(records).to_csv(output / "image_audit.csv", index=False)
    frame.to_csv(output / "evaluated_manifest.csv", index=False)
    per_class, condition_rows = [], []
    for model in MODELS:
        for c, values in {**summary["models"][model]["conditions"], "pooled": summary["models"][model]["pooled"]}.items():
            per_class.extend({"model": model, "condition": c, **r} for r in values["per_class"])
            condition_rows.append({"model": model, "condition": c, **{k: v for k, v in values.items() if k != "per_class"}})
        for c, cm in aggregate[model].items():
            pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output / f"confusion_{model}_{c}.csv", index_label="ground_truth")
    pd.DataFrame(per_class).to_csv(output / "per_class.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(output / "condition_metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Road segmentation evaluation summary",
       "type": "object", "required": ["models", "hypothesis", "dataset", "score_unit", "delta_unit"],
       "properties": {"score_unit": {"const": "fraction"}, "delta_unit": {"const": "percentage_points"},
          "models": {"type": "object", "required": list(MODELS)},
          "hypothesis": {"type": "object", "required": ["status", "supported", "delta_W_pp", "condition_delta_pp"],
                         "properties": {"status": {"enum": ["supported", "not_supported", "unresolved"]}, "supported": {"type": ["boolean", "null"]}}}},
       "$comment": "All undefined metric denominators serialize as null. NPZ confusion axes are image, model(hrda,mic), GT trainID, predicted trainID; shape (N,2,19,19), int64. Per-class CSV includes support, TP, FP, FN, union, precision, recall, F1, IoU. Group scores are computed from pooled pixel counts."}
    (output / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap", type=int, default=1000, help="Paired bootstrap repeats; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-official-validation", action="store_true")
    parser.add_argument("--path-root",type=Path,help="Resolve relative mask paths from this directory (default: manifest directory)")
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    result = evaluate_manifest(args.manifest, args.output, args.bootstrap, args.seed, args.require_official_validation, path_root=args.path_root)
    print(json.dumps(result["hypothesis"], indent=2))


if __name__ == "__main__":
    main()
