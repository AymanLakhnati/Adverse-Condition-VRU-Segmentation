from collections import Counter

import pytest

from roadseg.tune_dinov3 import choose_trial,sequence_holdout


def fixture_rows():
    rows=[]
    for condition in ("fog","night","rain","snow"):
        for sequence,size in (("sequence_a",9),("sequence_b",5),("sequence_c",3)):
            for frame in range(size):
                image_id=f"{condition}_{sequence}_{frame}"
                rows.append({"image_id":image_id,"condition":condition,"split":"train",
                             "image_path":f"data/acdc/rgb_anon/{condition}/train/{sequence}/{image_id}.png"})
    return rows


def test_sequence_holdout_is_reproducible_and_preserves_complete_groups():
    rows=fixture_rows()
    fit,development=sequence_holdout(rows,seed=42)
    other_fit,other_dev=sequence_holdout(list(reversed(rows)),seed=42)
    fit_ids={r["image_id"] for r in fit}
    dev_ids={r["image_id"] for r in development}
    assert fit_ids.isdisjoint(dev_ids)
    assert fit_ids|dev_ids=={r["image_id"] for r in rows}
    assert fit_ids=={r["image_id"] for r in other_fit}
    assert dev_ids=={r["image_id"] for r in other_dev}
    assert {r["sequence_group"] for r in fit}.isdisjoint({r["sequence_group"] for r in development})
    assert set(Counter(r["condition"] for r in fit))=={"fog","night","rain","snow"}
    assert set(Counter(r["condition"] for r in development))=={"fog","night","rain","snow"}


def test_tuning_rejects_official_validation_rows():
    rows=fixture_rows()
    rows[0]["split"]="val"
    with pytest.raises(ValueError,match="training"):
        sequence_holdout(rows)


def test_gopro_chapters_never_cross_fitting_development_boundary():
    rows=[]
    for condition in ("fog","night","rain","snow"):
        for sequence in ("GOPR0376","GP010376","GP020376","GOPR0351"):
            for frame in range(2):
                image_id=f"{condition}_{sequence}_{frame}"
                rows.append({"image_id":image_id,"condition":condition,"split":"train",
                             "image_path":f"data/acdc/rgb_anon/{condition}/train/{sequence}/{image_id}.png"})
    fit,development=sequence_holdout(rows,seed=42)
    assignments={row["image_id"]:partition for partition,values in (("fit",fit),("dev",development)) for row in values}
    for condition in ("fog","night","rain","snow"):
        chapter_rows=[row for row in fit+development if row["condition"]==condition and row["recording_family"]=="recording_0376"]
        assert len(chapter_rows)==6
        assert {row["sequence_directory"] for row in chapter_rows}=={"GOPR0376","GP010376","GP020376"}
        assert len({assignments[row["image_id"]] for row in chapter_rows})==1


def test_selection_uses_vru_objective_and_predeclared_fallback():
    trials=[{"learning_rate":rate,"development_vru_mean_iou":score,"development_miou_present":other}
            for rate,score,other in ((0.003,0.3,0.7),(0.001,0.4,0.5),(0.0003,0.4,0.4))]
    selected,metric,fallback=choose_trial(trials)
    assert selected["learning_rate"]==0.0003
    assert metric=="development_vru_mean_iou" and fallback is None
    for trial in trials:
        trial["development_vru_mean_iou"]=None
    selected,metric,fallback=choose_trial(trials)
    assert selected["learning_rate"]==0.003
    assert metric=="development_miou_present" and fallback is not None
