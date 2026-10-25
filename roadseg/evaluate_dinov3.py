"""Evaluate the newly trained DINOv3 dense head on all official ACDC val images."""
import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from roadseg.metrics import CONDITIONS, confusion_matrix, metrics_from_confusion
from roadseg.train_dinov3 import image_tensor, load_trained_model, predict_logits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", type=Path, default=Path("outputs/dinov3_dense/head.pt"))
    p.add_argument("--data-root", type=Path, default=Path("data/acdc"))
    p.add_argument("--output", type=Path, default=Path("results/dinov3"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    paths = sorted(args.data_root.glob("rgb_anon/*/val/*/*_rgb_anon.png"))
    counts = {c: sum(p.relative_to(args.data_root).parts[1] == c for p in paths) for c in CONDITIONS}
    if counts != {"fog": 100, "night": 106, "rain": 100, "snow": 100}:
        raise ValueError(f"Official validation image counts not available: {counts}")
    config = json.loads((args.head.parent / "training_config.json").read_text())
    if config.get("status") != "completed" or not config.get("completed_fixed_full_protocol"):
        raise ValueError("Evaluation requires completed full prespecified DINOv3 training")
    model = load_trained_model(args.head, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"head_sha256": hashlib.sha256(args.head.read_bytes()).hexdigest(),
                "encoder_state_sha256": config["encoder_state_sha256"], "tile_size": 512,
                "stride": 384, "precision": "float32", "torch": str(torch.__version__),
                "inference_source_sha256": hashlib.sha256(Path("roadseg/train_dinov3.py").read_bytes()).hexdigest()}
    identity_path = args.output / "prediction_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing predictions use different head/encoder/settings; use a fresh --output")
    if not identity_path.exists() and any((args.output / "predictions").rglob("*.png")):
        raise ValueError("Existing predictions have no model provenance; use a fresh --output")
    identity_path.write_text(json.dumps(identity, indent=2))
    confusions, rows, start = [], [], time.monotonic()
    for index, path in enumerate(paths):
        parts = path.relative_to(args.data_root).parts
        condition = parts[1]
        image_id = path.name.removesuffix("_rgb_anon.png")
        gt_path = args.data_root / "gt" / condition / "val" / parts[3] / (image_id + "_gt_labelTrainIds.png")
        pred_path = args.output / "predictions" / condition / (image_id + "_dinov3.png")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        before = time.monotonic()
        if not pred_path.exists():
            with torch.inference_mode():
                tensor = image_tensor(path, args.device)
                logits = predict_logits(model, tensor, batch_size=args.batch_size)
                pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                del tensor, logits
            Image.fromarray(pred).save(pred_path)
        else:
            with Image.open(pred_path) as image:
                pred = np.asarray(image)
        with Image.open(gt_path) as image:
            gt = np.asarray(image)
        confusions.append(confusion_matrix(gt, pred))
        rows.append(dict(image_id=image_id, condition=condition, gt_path=str(gt_path), image_path=str(path),
                         pred_path=str(pred_path), seconds=time.monotonic() - before))
        pd.DataFrame(rows).to_csv(args.output / "manifest.csv", index=False)
        logging.info("DINOv3 validation %d/406 %s %.2fs", index + 1, condition, rows[-1]["seconds"])
    stack = np.stack(confusions)
    condition_array = np.array([r["condition"] for r in rows])
    grouped = {c: metrics_from_confusion(stack[condition_array == c].sum(0)) for c in CONDITIONS}
    values = {c: grouped[c]["vru_mean_iou"] for c in CONDITIONS}
    w = min(values.values()) if all(v is not None for v in values.values()) else None
    summary = {"model": "DINOv3 ViT-S/16 dense linear head", "learning_setting": "supervised ACDC train labels",
               "conditions": grouped, "pooled": metrics_from_confusion(stack.sum(0)), "W": w,
               "weakest_conditions": [c for c, v in values.items() if v == w] if w is not None else [],
               "validation_images": 406, "condition_image_counts": counts, "score_unit": "fraction",
               "training_config_path": str(args.head.parent / "training_config.json"),
               "elapsed_seconds": time.monotonic() - start,
               "comparison_limit": "Different supervision and pretraining from HRDA/MIC; not an isolated architecture comparison"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    np.savez_compressed(args.output / "per_image_confusions.npz", confusions=stack, conditions=condition_array,
                        image_ids=np.array([r["image_id"] for r in rows]))
    class_rows = []
    for c, metrics in grouped.items():
        for row in metrics["per_class"]:
            class_rows.append(dict(condition=c, **row))
    pd.DataFrame(class_rows).to_csv(args.output / "per_class.csv", index=False)
    logging.info("DINOv3 full validation completed in %.1f minutes", summary["elapsed_seconds"] / 60)


if __name__ == "__main__":
    main()
