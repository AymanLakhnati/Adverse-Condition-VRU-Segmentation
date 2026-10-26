import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import roadseg.predict as predict


def make_image(path,value=0):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.full((6,8,3),value,dtype=np.uint8)).save(path)
    return path


@pytest.fixture
def fake_model(tmp_path,monkeypatch):
    checkpoint=tmp_path/"head.pt"
    checkpoint.write_bytes(b"fixture model; actual model loader mocked")
    loads,calls=[],[]
    model=object()
    def load(*args):
        loads.append(args)
        return model
    def infer(selected,image,model_name,device):
        assert selected is model
        calls.append(Path(image).name)
        with Image.open(image) as source:
            return np.full((source.height,source.width),11,dtype=np.uint8)
    monkeypatch.setattr(predict,"load_prediction_model",load)
    monkeypatch.setattr(predict,"predict_mask",infer)
    return checkpoint,loads,calls


def test_directory_prediction_is_sorted_loads_once_and_keeps_stem_collisions(tmp_path,fake_model):
    checkpoint,loads,calls=fake_model
    source=tmp_path/"inputs"
    for name in ("b.png","A.jpg","same.jpg","same.png","nested/c.png"):
        make_image(source/name)
    output=tmp_path/"batch"
    result=predict.run_prediction(source,checkpoint=checkpoint,output=output,recursive=True)
    assert len(loads)==1
    assert calls==["A.jpg","b.png","c.png","same.jpg","same.png"]
    manifest=json.loads(Path(result["manifest"]).read_text())
    assert manifest["status"]=="completed" and manifest["images"]==5
    assert [row["source_relative_path"] for row in manifest["outputs"]]==[
        "A.jpg","b.png","nested/c.png","same.jpg","same.png"]
    assert (output/"same.jpg_dinov3.png").is_file()
    assert (output/"same.png_dinov3.png").is_file()
    for row in manifest["outputs"]:
        metadata=json.loads((output/row["metadata"]).read_text())
        assert metadata["checkpoint_sha256"]==predict.sha256(checkpoint)
        assert metadata["labels_sha256"]==predict.sha256(output/row["labels"])
        assert metadata["predicted_pixel_counts"]["person"]==48
        assert metadata["output_shape"]==[6,8]


def test_single_image_cli_naming_preserves_dotted_stems(tmp_path,fake_model):
    checkpoint,loads,calls=fake_model
    source=make_image(tmp_path/"road.scene.v1.png")
    output=tmp_path/"single"
    result=predict.run_prediction(source,checkpoint=checkpoint,output=output)
    assert Path(result["labels"]).name=="road.scene.v1_dinov3.png"
    info=json.loads(Path(result["metadata"]).read_text())
    assert "elapsed_seconds_including_model_load" in info
    assert info["shared_batch_model"] is False
    assert len(loads)==1


def test_batch_refuses_overwrite_before_loading_or_publishing(tmp_path,fake_model):
    checkpoint,loads,calls=fake_model
    source=tmp_path/"images"
    make_image(source/"a.png");make_image(source/"z.png")
    output=tmp_path/"output";output.mkdir()
    sentinel=output/"z.png_dinov3.json"
    sentinel.write_text("user metadata")
    with pytest.raises(FileExistsError,match="Refusing to overwrite"):
        predict.run_prediction(source,checkpoint=checkpoint,output=output)
    assert loads==calls==[]
    assert sentinel.read_text()=="user metadata"
    assert list(output.iterdir())==[sentinel]


def test_publication_race_rolls_back_own_files_and_preserves_existing_file(tmp_path):
    first,second=tmp_path/"staged1",tmp_path/"staged2"
    first.write_text("new1");second.write_text("new2")
    out1,out2=tmp_path/"out1",tmp_path/"out2"
    out2.write_text("concurrent user file")
    with pytest.raises(FileExistsError):
        predict.publish_bundle([(first,out1),(second,out2)])
    assert not out1.exists()
    assert out2.read_text()=="concurrent user file"


def test_later_failure_preserves_completed_image_but_never_claims_complete_batch(tmp_path,fake_model,monkeypatch):
    checkpoint,loads,calls=fake_model
    source=tmp_path/"inputs"
    make_image(source/"a.png");make_image(source/"b.png")
    original=predict.predict_mask
    def fail_second(*args):
        if Path(args[1]).name=="b.png":
            raise RuntimeError("inference failed")
        return original(*args)
    monkeypatch.setattr(predict,"predict_mask",fail_second)
    output=tmp_path/"output"
    with pytest.raises(RuntimeError,match="inference failed"):
        predict.run_prediction(source,checkpoint=checkpoint,output=output)
    assert (output/"a.png_dinov3.json").is_file()
    assert not (output/"b.png_dinov3.png").exists()
    assert not (output/"batch_manifest.json").exists()
    assert not list(output.glob(".predict-*"))


def test_all_inputs_preflight_before_loading_and_output_subtree_excluded(tmp_path,fake_model):
    checkpoint,loads,calls=fake_model
    source=tmp_path/"inputs"
    make_image(source/"a.png")
    output=source/"generated"
    make_image(output/"old.png")
    discovered,is_batch=predict.discover_images(source,output,recursive=True)
    assert is_batch and discovered==[source/"a.png"]
    (source/"z_corrupt.png").write_bytes(b"not an image")
    with pytest.raises(OSError):
        predict.run_prediction(source,checkpoint=checkpoint,output=output,recursive=True)
    assert loads==calls==[]
    assert not (output/"batch_manifest.json").exists()


def test_output_ancestor_does_not_hide_input_images(tmp_path):
    image=make_image(tmp_path/"inputs"/"a.png")
    discovered,_=predict.discover_images(image.parent,tmp_path)
    assert discovered==[image]


def test_changed_source_is_rejected_before_output_publication(tmp_path,fake_model,monkeypatch):
    checkpoint,loads,calls=fake_model
    source=make_image(tmp_path/"a.png")
    def modify_source(*args):
        make_image(source,value=200)
        return np.full((6,8),11,dtype=np.uint8)
    monkeypatch.setattr(predict,"predict_mask",modify_source)
    output=tmp_path/"out"
    with pytest.raises(ValueError,match="changed during inference"):
        predict.run_prediction(source,checkpoint=checkpoint,output=output)
    assert not output.exists()
