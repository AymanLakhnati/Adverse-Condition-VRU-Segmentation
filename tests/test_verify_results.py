"""CPU-only consistency/tampering tests with tiny synthetic count artifacts.

Fixtures are deliberately perfect diagonal predictions with hand-known metrics;
they are not measurements and never touch the real saved experiment artifacts.
"""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.verify_results import CONDITIONS, EXPECTED, NAMES, _derive, _family, verify_results


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def perfect(n):
    # Every image has exactly one correctly predicted pixel of each class.
    classes = [dict(class_id=k, class_name=name, tp=n, fp=0, fn=0, support=n,
                    predicted=n, union=n, precision=1.0, recall=1.0, f1=1.0, iou=1.0)
               for k, name in enumerate(NAMES)]
    return dict(valid_pixels=n * 19, pixel_accuracy=1.0, miou_present=1.0, miou_19=1.0,
                defined_iou_classes=19, vru_mean_iou=1.0, per_class=classes)


@pytest.fixture(scope="module")
def original_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("consistent_results")
    result_dir, dino_dir = root / "results", root / "results/dinov3"
    result_dir.mkdir()
    dino_dir.mkdir()
    conditions = np.array([c for c in CONDITIONS for _ in range(EXPECTED[c])])
    ids = np.array([f"GOPR9000_frame_{i:06d}" for i in range(406)])
    counts = np.tile(np.eye(19, dtype=np.int64), (406, 2, 1, 1))
    np.savez_compressed(result_dir / "per_image_confusions.npz", confusion=counts,
                        condition=conditions, image_id=ids, models=np.array(["hrda", "mic"]),
                        class_names=np.array(NAMES))
    # Opposite image order tests ID-based cross-model alignment rather than a
    # false assumption that independent evaluators iterate images identically.
    np.savez_compressed(dino_dir / "per_image_confusions.npz", confusions=counts[::-1, 0],
                        conditions=conditions[::-1], image_ids=ids[::-1])
    scores = dict(conditions={c: perfect(EXPECTED[c]) for c in CONDITIONS}, pooled=perfect(406),
                  W=1.0, weakest_conditions=list(CONDITIONS))
    primary = dict(models={"hrda": scores, "mic": scores}, score_unit="fraction", delta_unit="percentage_points",
                   dataset=dict(images=406, condition_counts=EXPECTED, matches_original_validation_counts=True),
                   hypothesis=dict(delta_W_pp=0.0, condition_delta_pp={c: 0.0 for c in CONDITIONS},
                                   threshold_delta_W_pp=2.0, threshold_each_condition_pp=-1.0,
                                   supported=False, status="not_supported"))
    dump(result_dir / "summary.json", primary)
    dump(result_dir / "inference_run.json", dict(status="completed", processed=406))
    dump(dino_dir / "summary.json", dict(**scores, validation_images=406, condition_image_counts=EXPECTED,
                                       learning_setting="supervised ACDC train labels", score_unit="fraction"))
    fitting, development = [], []
    for j, condition in enumerate(CONDITIONS):
        for target, count, offset in ((fitting, 300, 0), (development, 100, 1)):
            sequence = f"GOPR{(j+1)*100+offset:04d}"
            for i in range(count):
                image_id = f"{sequence}_frame_{i:06d}"
                image_hash = hashlib.sha256(image_id.encode()).hexdigest()
                target.append(dict(image_id=image_id, condition=condition, split="train",
                                   image_path=f"data/acdc/rgb_anon/{condition}/train/{sequence}/{image_id}.png",
                                   label_path=f"data/acdc/gt/{condition}/train/{sequence}/{image_id}_gt_labelTrainIds.png",
                                   image_sha256=image_hash, label_sha256=hashlib.sha256((image_id+"label").encode()).hexdigest(),
                                   sequence_directory=sequence, recording_family=_family(sequence),
                                   sequence_group=condition+"/"+_family(sequence)))
    full = fitting + development
    tuning_base, train_base = root / "outputs/dinov3_tuning", root / "outputs/dinov3_dense"
    csv_write(tuning_base / "fit_manifest.csv", fitting)
    csv_write(tuning_base / "development_manifest.csv", development)
    csv_write(train_base / "training_manifest.csv", full)
    content = hashlib.sha256(json.dumps(sorted((r["image_id"], r["image_sha256"], r["label_sha256"]) for r in full),
                                        separators=(",", ":")).encode()).hexdigest()
    tuning = dict(status="completed", model_name="vit_small_patch16_dinov3.lvd1689m", trial_epochs=1,
                  batch_size=4, seed=42, encoder_frozen=True, official_validation_or_test_used=False,
                  encoder_state_sha256="a"*64, initial_head_sha256="b"*64,
                  all_training_content_sha256=content,
                  all_training_ids_sha256=hashlib.sha256("\n".join(sorted(r["image_id"] for r in full)).encode()).hexdigest(),
                  fit_manifest_sha256=sha(tuning_base / "fit_manifest.csv"),
                  development_manifest_sha256=sha(tuning_base / "development_manifest.csv"),
                  split=dict(fit_images=1200, development_images=400, recording_families_disjoint=True,
                             sequence_groups_disjoint=True, frame_ids_disjoint=True),
                  trials=[dict(learning_rate=rate, fit_epochs=1, fit_optimizer_steps=300, encoder_unchanged=True,
                               development_vru_mean_iou=score, development_miou_present=score)
                          for rate, score in [(0.0003, 0.0), (0.001, 0.1), (0.003, 0.2)]],
                  selected_learning_rate=0.003, selected_score=0.2, selected_metric="development_vru_mean_iou",
                  selection_fallback=None)
    dump(tuning_base / "tuning_results.json", tuning)
    training = dict(status="completed", completed_fixed_full_protocol=True, limited_smoke_run=False,
                    epochs=5, completed_epochs=5, training_images=1600, available_train_images=1600,
                    batch_size=4, seed=42, head_parameters=7315, optimizer_steps=2000, skipped_all_ignore_batches=0,
                    model_name=tuning["model_name"], encoder_state_unchanged=True, encoder_frozen=True,
                    head_state_changed=True, encoder_state_sha256="a"*64, encoder_after_sha256="a"*64,
                    head_before_sha256="b"*64, head_after_sha256="c"*64, official_validation_used_for_selection=False,
                    internal_training_development_used_for_learning_rate_selection=True,
                    tuning_results_sha256=sha(tuning_base / "tuning_results.json"),
                    tuning_results_path="outputs\\dinov3_tuning\\tuning_results.json", learning_rate=0.003,
                    all_training_content_sha256=content, inference_tile_size=512, inference_stride=384)
    dump(train_base / "training_config.json", training)
    steps = [dict(global_step=(epoch-1)*400+batch+1, epoch=epoch, images=4, valid_pixels=76,
                  loss=0.5, gradient_norm=1.0, learning_rate=0.003,
                  image_ids=" ".join(r["image_id"] for r in full[batch*4:batch*4+4]))
             for epoch in range(1, 6) for batch in range(400)]
    csv_write(train_base / "training_steps.csv", steps)
    (train_base / "head.pt").write_bytes(b"synthetic test identity only; no model")
    dump(dino_dir / "prediction_identity.json", dict(encoder_state_sha256="a"*64,
         head_sha256=sha(train_base / "head.pt"), tile_size=512, stride=384))
    return root


@pytest.fixture
def fixture_root(original_fixture, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(original_fixture, target)
    return target


def mutate_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    dump(path, data)


def mutate_npz(path, change):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    change(data)
    np.savez_compressed(path, **data)


def test_verified_perfect_results_are_portable(original_fixture, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = verify_results(original_fixture)
    assert result["status"] == "passed"
    assert result["delta_W_pp"] == 0
    assert result["hypothesis_status"] == "not_supported"
    assert result["training"]["optimizer_steps"] == 2000


def test_independent_formulas_count_false_positive_and_false_negative():
    cm = np.zeros((19, 19), dtype=np.int64)
    cm[11, 11], cm[11, 12], cm[12, 11] = 6, 2, 3
    person = _derive(cm)["per_class"][11]
    assert person["tp"] == 6 and person["fp"] == 3 and person["fn"] == 2
    assert person["iou"] == pytest.approx(6/11)
    assert person["precision"] == pytest.approx(6/9)
    assert person["recall"] == pytest.approx(6/8)
    assert _derive(cm)["vru_mean_iou"] is None  # bicycle union is absent


@pytest.mark.parametrize("field", ["iou", "fp", "class_name"])
def test_class_metric_tampering_rejected(fixture_root, field):
    value = {"iou": 0.5, "fp": 1, "class_name": "wrong"}[field]
    mutate_json(fixture_root / "results/summary.json",
                lambda x: x["models"]["mic"]["conditions"]["fog"]["per_class"][11].__setitem__(field, value))
    with pytest.raises(ValueError, match="mic.fog.class11"):
        verify_results(fixture_root)


@pytest.mark.parametrize("issue", ["duplicate_ids", "float_counts", "wrong_model_order"])
def test_primary_array_schema_tampering_rejected(fixture_root, issue):
    def change(x):
        if issue == "duplicate_ids": x["image_id"][0] = x["image_id"][1]
        elif issue == "float_counts": x["confusion"] = x["confusion"].astype(float)
        else: x["models"] = x["models"][::-1]
    mutate_npz(fixture_root / "results/per_image_confusions.npz", change)
    with pytest.raises(ValueError):
        verify_results(fixture_root)


def test_dino_reference_support_mismatch_rejected(fixture_root):
    mutate_npz(fixture_root / "results/dinov3/per_image_confusions.npz",
               lambda x: x["confusions"].__setitem__((0, 11, 11), 0))
    with pytest.raises(ValueError, match="GT supports differ"):
        verify_results(fixture_root)


def test_training_log_missing_step_rejected(fixture_root):
    path = fixture_root / "outputs/dinov3_dense/training_steps.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1])+"\n")
    with pytest.raises(ValueError, match="2000 optimizer steps"):
        verify_results(fixture_root)


def test_changed_tuning_file_breaks_refit_hash(fixture_root):
    mutate_json(fixture_root / "outputs/dinov3_tuning/tuning_results.json",
                lambda x: x.__setitem__("selected_score", 0.9))
    with pytest.raises(ValueError, match="file hash linkage"):
        verify_results(fixture_root)


def test_wrong_lr_selection_rejected_even_after_rehash(fixture_root):
    path = fixture_root / "outputs/dinov3_tuning/tuning_results.json"
    mutate_json(path, lambda x: x.__setitem__("selected_learning_rate", 0.001))
    mutate_json(fixture_root / "outputs/dinov3_dense/training_config.json",
                lambda x: x.__setitem__("tuning_results_sha256", sha(path)))
    with pytest.raises(ValueError, match="Tuning.selected rate"):
        verify_results(fixture_root)


def test_evaluated_head_bytes_must_match_identity(fixture_root):
    (fixture_root / "outputs/dinov3_dense/head.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evaluated head/encoder identity"):
        verify_results(fixture_root)


def test_nonfinite_json_and_missing_artifacts_fail_closed(fixture_root):
    mutate_json(fixture_root / "results/summary.json", lambda x: x.__setitem__("unexpected", float("nan")))
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        verify_results(fixture_root)
    (fixture_root / "results/summary.json").unlink()
    with pytest.raises(ValueError, match="Result verification failed"):
        verify_results(fixture_root)


def test_cli_receipt_from_another_working_directory(original_fixture, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/verify_results.py"
    output = tmp_path / "receipt.json"
    result = subprocess.run([sys.executable, str(script), "--root", str(original_fixture), "--output", str(output)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text())["status"] == "passed"


def test_gopro_alias_rule_is_conservative():
    assert _family("GOPR0376") == _family("GP010376") == _family("GP020376") == "recording_0376"
    assert _family("other_name_0376") == "other_name_0376"
