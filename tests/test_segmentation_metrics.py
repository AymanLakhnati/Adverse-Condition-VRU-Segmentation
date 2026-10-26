"""Hand-computed regression tests for the actual segmentation endpoint."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from roadseg.metrics import (CONDITIONS, VRU, confusion_matrix, metrics_from_confusion,
                             summarize, paired_bootstrap, evaluate_manifest)


def class_quality(correct, total=100):
    """Three VRU classes each with recall/IoU correct/total, no VRU FP."""
    cm = np.zeros((19, 19), dtype=np.int64)
    for c in VRU:
        cm[c, c] = correct
        cm[c, 0] = total - correct
    return cm


def test_ignore_only_gt_and_orientation():
    gt = np.array([[11, 11, 12, 255]])
    pred = np.array([[11, 12, 12, 999]])
    cm = confusion_matrix(gt, pred)
    assert cm.sum() == 3
    assert cm[11, 12] == 1
    assert cm[12, 11] == 0
    values = metrics_from_confusion(cm)
    person, rider = values["per_class"][11:13]
    assert person["precision"] == 1
    assert person["recall"] == .5
    assert person["iou"] == .5
    assert person["f1"] == pytest.approx(2 / 3)
    assert rider["precision"] == .5
    assert rider["recall"] == 1
    assert rider["iou"] == .5
    assert values["vru_mean_iou"] is None  # absent bicycle is not 100% correct
    assert values["per_class"][18]["iou"] is None
    assert values["per_class"][18]["f1"] is None
    assert values["miou_19"] is None


@pytest.mark.parametrize("pred", [255, 19, -1])
def test_invalid_prediction_on_valid_gt_rejected(pred):
    with pytest.raises(ValueError, match="Prediction"):
        confusion_matrix(np.array([[11]]), np.array([[pred]]))


def test_invalid_gt_shapes_and_noninteger_rejected():
    with pytest.raises(ValueError, match="GT must"):
        confusion_matrix(np.array([[24]]), np.array([[11]]))
    with pytest.raises(ValueError, match="identical 2D"):
        confusion_matrix(np.array([[11]]), np.array([[11, 12]]))
    with pytest.raises(ValueError, match="integer"):
        confusion_matrix(np.array([[11.0]]), np.array([[11]]))


def test_predicted_only_class_is_zero_not_undefined():
    cm = confusion_matrix(np.array([[0]]), np.array([[18]]))
    row = metrics_from_confusion(cm)["per_class"][18]
    assert row["iou"] == 0
    assert row["precision"] == 0
    assert row["recall"] is None
    assert row["f1"] == 0


def test_iou_pools_pixels_not_image_ious():
    first = confusion_matrix(np.array([[11]]), np.array([[11]]))
    second = confusion_matrix(np.full((1, 9), 11), np.zeros((1, 9), dtype=int))
    pooled = metrics_from_confusion(first + second)["per_class"][11]["iou"]
    image_mean = np.mean([metrics_from_confusion(c)["per_class"][11]["iou"] for c in (first, second)])
    assert pooled == .1
    assert image_mean == .5


def test_worst_condition_can_switch_and_thresholds_are_pp():
    h = dict(zip(CONDITIONS, map(class_quality, (50, 70, 80, 90))))
    m = dict(zip(CONDITIONS, map(class_quality, (75, 69, 81, 90))))
    result = summarize({"hrda": h, "mic": m})
    assert result["models"]["hrda"]["weakest_conditions"] == ["fog"]
    assert result["models"]["mic"]["weakest_conditions"] == ["night"]
    assert result["hypothesis"]["delta_W_pp"] == pytest.approx(19)
    assert result["hypothesis"]["condition_delta_pp"]["night"] == pytest.approx(-1)
    assert result["hypothesis"]["supported"] is True
    # A large minimum improvement cannot excuse a condition loss over 1 pp.
    m["snow"] = class_quality(88)
    assert summarize({"hrda": h, "mic": m})["hypothesis"]["status"] == "not_supported"


def test_missing_conditions_or_vru_union_are_unresolved():
    complete = {c: class_quality(80) for c in CONDITIONS}
    partial = dict(complete)
    del partial["snow"]
    result = summarize({"hrda": complete, "mic": partial})
    assert result["hypothesis"]["supported"] is None
    assert result["hypothesis"]["status"] == "unresolved"
    assert result["models"]["mic"]["W"] is None
    complete["fog"] = np.zeros((19, 19), dtype=np.int64)
    assert summarize({"hrda": complete, "mic": complete})["hypothesis"]["status"] == "unresolved"


def test_exact_two_percentage_point_boundary():
    h = {c: class_quality(50) for c in CONDITIONS}
    m = {c: class_quality(52) for c in CONDITIONS}
    assert summarize({"hrda": h, "mic": m})["hypothesis"]["supported"] is True
    m["snow"] = class_quality(51)
    assert summarize({"hrda": h, "mic": m})["hypothesis"]["supported"] is False


def test_bootstrap_pairs_images_and_recomputes_minima():
    paired = np.asarray([[class_quality(50), class_quality(60)] for _ in CONDITIONS])
    result = paired_bootstrap(paired, CONDITIONS, repeats=20, seed=42)
    assert result["endpoints"]["delta_W_pp"]["percentile_95_pp"] == pytest.approx([10, 10])
    assert result["endpoints"]["fog"]["defined_replicates"] == 20
    assert paired_bootstrap(paired, CONDITIONS, 20, 42) == result


def test_manifest_cli_artifacts_and_no_labelid_guessing(tmp_path):
    rows = []
    for c in CONDITIONS:
        image_id = f"sample_{c}"
        gt = np.array([[11, 12, 18, 255]], dtype=np.uint8)
        for suffix, mask in (("gt_labelTrainIds", gt), ("hrda", gt.copy()), ("mic", gt.copy())):
            # Model 255 at ignored GT is permitted; 255 on any valid pixel is not.
            Image.fromarray(mask).save(tmp_path / f"{image_id}_{suffix}.png")
        rows.append(dict(image_id=image_id, condition=c,
                         gt_path=f"{image_id}_gt_labelTrainIds.png",
                         hrda_pred_path=f"{image_id}_hrda.png", mic_pred_path=f"{image_id}_mic.png"))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    result = evaluate_manifest(manifest, tmp_path / "out", bootstrap_repeats=5)
    assert result["models"]["hrda"]["W"] == 1
    assert result["hypothesis"]["supported"] is False  # no +2 point improvement
    assert result["dataset"]["matches_original_validation_counts"] is False
    with np.load(tmp_path / "out/per_image_confusions.npz", allow_pickle=False) as saved:
        assert saved["confusion"].shape == (4, 2, 19, 19)
        assert saved["confusion"].sum() == 24
    parsed = json.loads((tmp_path / "out/summary.json").read_text())
    assert parsed["models"]["hrda"]["conditions"]["fog"]["per_class"][0]["iou"] is None
    assert (tmp_path / "out/schema.json").exists()
    with pytest.raises(ValueError, match="validation counts"):
        evaluate_manifest(manifest, tmp_path / "bad", 0, require_official_validation=True)
    rows[0]["gt_path"] = "sample_fog_gt_labelIds.png"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="labelTrainIds"):
        evaluate_manifest(manifest, tmp_path / "bad", 0)


def test_manifest_rejects_misaligned_names(tmp_path):
    pd.DataFrame([dict(image_id="foo", condition="fog", gt_path="bar_gt_labelTrainIds.png",
                       hrda_pred_path="foo_hrda.png", mic_pred_path="foo_mic.png")]).to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="alignment"):
        evaluate_manifest(tmp_path / "manifest.csv", tmp_path / "out", 0)


def test_explicit_project_path_root_for_nested_inference_manifest(tmp_path):
    project=tmp_path/"project"
    (project/"data").mkdir(parents=True)
    (project/"results"/"predictions").mkdir(parents=True)
    image_id="sample_fog"
    gt=np.array([[11,12,18,255]],dtype=np.uint8)
    Image.fromarray(gt).save(project/"data"/f"{image_id}_gt_labelTrainIds.png")
    for model in ("hrda","mic"):
        Image.fromarray(gt).save(project/"results"/"predictions"/f"{image_id}_{model}.png")
    manifest=project/"results"/"inference_manifest.csv"
    pd.DataFrame([dict(image_id=image_id,condition="fog",
        gt_path=f"data/{image_id}_gt_labelTrainIds.png",
        hrda_pred_path=f"results/predictions/{image_id}_hrda.png",
        mic_pred_path=f"results/predictions/{image_id}_mic.png")]).to_csv(manifest,index=False)
    # Preserve the documented default, without guessing a parent directory.
    with pytest.raises(FileNotFoundError):
        evaluate_manifest(manifest,project/"default_out",bootstrap_repeats=0)
    result=evaluate_manifest(manifest,project/"explicit_out",bootstrap_repeats=0,path_root=project)
    assert result["dataset"]["images"]==1
    assert result["models"]["mic"]["pooled"]["vru_mean_iou"]==1
    assert result["models"]["hrda"]["pooled"]["valid_pixels"]==3
    assert result["dataset"]["mask_path_root"]==str(project.resolve())


def test_manifest_checks_provided_gt_hash_before_scoring(tmp_path):
    import hashlib
    gt=tmp_path/"example_gt_labelTrainIds.png"
    pred=tmp_path/"example_mic.png"
    mask=np.array([[11,12,18]],dtype=np.uint8)
    Image.fromarray(mask).save(gt)
    Image.fromarray(mask).save(pred)
    row=dict(image_id="example",condition="fog",gt_path=gt.name,
             hrda_pred_path=pred.name,mic_pred_path=pred.name,
             gt_sha256=hashlib.sha256(gt.read_bytes()).hexdigest())
    manifest=tmp_path/"manifest.csv"
    pd.DataFrame([row]).to_csv(manifest,index=False)
    result=evaluate_manifest(manifest,tmp_path/"valid",bootstrap_repeats=0)
    assert result["dataset"]["verified_manifest_mask_sha256_count"]==1
    Image.fromarray(np.array([[11,11,18]],dtype=np.uint8)).save(gt)
    with pytest.raises(ValueError,match="Manifest SHA256 mismatch"):
        evaluate_manifest(manifest,tmp_path/"changed",bootstrap_repeats=0)
