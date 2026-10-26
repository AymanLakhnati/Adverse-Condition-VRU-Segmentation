import numpy as np
import pytest
from PIL import Image

from roadseg.run_inference import sha,validate_cached_row


def cached_row(tmp_path,sealed=True):
    image=tmp_path/"example_rgb_anon.png"
    gt=tmp_path/"example_gt_labelTrainIds.png"
    prediction=tmp_path/"example_mic.png"
    Image.fromarray(np.zeros((1,4,3),dtype=np.uint8)).save(image)
    mask=np.array([[11,12,18,255]],dtype=np.uint8)
    Image.fromarray(mask).save(gt)
    Image.fromarray(mask).save(prediction)
    row=dict(image_id="example",image_path=str(image),gt_path=str(gt),
             rgb_sha256=sha(image),gt_sha256=sha(gt),mic_pred_path=str(prediction))
    if sealed:
        row["mic_pred_sha256"]=sha(prediction)
    return row


def test_resume_rejects_changed_source_bytes_under_same_paths(tmp_path):
    row=cached_row(tmp_path)
    assert validate_cached_row(row,["mic"])
    Image.fromarray(np.ones((1,4,3),dtype=np.uint8)).save(row["image_path"])
    with pytest.raises(ValueError,match="input SHA256 mismatch"):
        validate_cached_row(row,["mic"])


def test_resume_rejects_altered_valid_mask_when_hash_is_recorded(tmp_path):
    row=cached_row(tmp_path)
    Image.fromarray(np.array([[11,11,18,255]],dtype=np.uint8)).save(row["mic_pred_path"])
    with pytest.raises(ValueError,match="prediction SHA256 mismatch"):
        validate_cached_row(row,["mic"])


def test_legacy_cache_still_checks_shape_and_valid_pixel_ids(tmp_path):
    row=cached_row(tmp_path,sealed=False)
    assert validate_cached_row(row,["mic"])
    Image.fromarray(np.array([[11,255,18,255]],dtype=np.uint8)).save(row["mic_pred_path"])
    with pytest.raises(ValueError,match="Invalid cached prediction"):
        validate_cached_row(row,["mic"])


def test_missing_cached_prediction_is_recomputed(tmp_path):
    row=cached_row(tmp_path)
    from pathlib import Path
    Path(row["mic_pred_path"]).unlink()
    assert validate_cached_row(row,["mic"]) is False


def test_failed_resume_preserves_existing_run_metadata(tmp_path,monkeypatch):
    import argparse
    import pandas as pd
    import roadseg.run_inference as inference
    monkeypatch.setattr(inference,"load_model",lambda *args:object())
    checkpoint=tmp_path/"checkpoint.pth"
    checkpoint.write_bytes(b"fixture checkpoint, loader mocked")
    output=tmp_path/"output"
    args=argparse.Namespace(hrda=checkpoint,mic=checkpoint,adapted=None,
        output=output,device="cpu",data=tmp_path,expected=0,wait_for_data=False)
    inference.run(args)
    meta_path=output/"inference_run.json"
    before=meta_path.read_bytes()
    row=cached_row(tmp_path)
    row["hrda_pred_path"]=row["mic_pred_path"]
    pd.DataFrame([row]).to_csv(output/"manifest.csv",index=False)
    Image.fromarray(np.ones((1,4,3),dtype=np.uint8)).save(row["image_path"])
    args.expected=1
    with pytest.raises(ValueError,match="input SHA256 mismatch"):
        inference.run(args)
    assert meta_path.read_bytes()==before
