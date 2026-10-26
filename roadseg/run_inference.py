"""Resume full official ACDC validation inference while data streams locally."""
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

from src.hrda_inference import load_model, preprocess
from roadseg.metrics import confusion_matrix

LOG = logging.getLogger("roadseg-inference")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def validate_cached_row(row, model_names):
    """Reuse a row only when its inputs and cached masks remain consistent.

    Older manifests lack prediction-file hashes, so their masks still receive
    geometry/ID checks. New writes additionally bind each cached PNG by SHA256.
    Missing predictions return False for recomputation. Changed inputs or
    corrupt cached files raise instead of silently mixing experiment states.
    """
    destinations={name:Path(row[f"{name}_pred_path"]) for name in model_names}
    if not all(path.is_file() for path in destinations.values()):
        return False
    for path_key,hash_key in (("image_path","rgb_sha256"),("gt_path","gt_sha256")):
        if not row.get(path_key) or not row.get(hash_key):
            raise ValueError("Cached row lacks input SHA256 provenance; use a fresh output directory")
        source=Path(row[path_key])
        if not source.is_file() or sha(source)!=row[hash_key]:
            raise ValueError(f"Cached input SHA256 mismatch: {source}; use a fresh output directory")
    try:
        with Image.open(row["gt_path"]) as source:
            gt=np.asarray(source)
        for name,path in destinations.items():
            expected=row.get(f"{name}_pred_sha256")
            if expected and sha(path)!=expected:
                raise ValueError(f"Cached prediction SHA256 mismatch: {path}")
            with Image.open(path) as source:
                prediction=np.asarray(source)
            confusion_matrix(gt,prediction)
    except (OSError,ValueError) as error:
        raise ValueError(f"Invalid cached prediction for {row.get('image_id')}: {error}") from error
    return True


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    checkpoints = {"hrda": args.hrda, "mic": args.mic}
    if args.adapted:
        checkpoints = {"adapted": args.adapted}
    args.output.mkdir(parents=True, exist_ok=True)
    models = {name: load_model(path, args.device) for name, path in checkpoints.items()}
    metadata = {"checkpoints": {k: {"path": str(p), "sha256": sha(p)} for k, p in checkpoints.items()},
                "models": list(models), "precision": "float32", "seed": 42,
                "full_validation_expected": 406, "inference": "official 1024/512 outer and 512/256 inner sliding grids",
                "input": "1920x1080 RGB; official mean/std; no flip or multiscale test augmentation",
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "CPU",
                "code_sha256": {str(p): sha(p) for p in [Path("src/hrda_inference.py"), Path("src/mit_backbone.py")]},
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "running"}
    meta_path = args.output / ("adapted_run.json" if args.adapted else "inference_run.json")
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        for key in ("checkpoints", "precision", "inference", "input", "code_sha256", "torch"):
            if previous.get(key) != metadata[key]:
                raise ValueError(f"Existing predictions have different {key}; use a fresh --output directory")
    done, rows, start = set(), [], time.monotonic()
    manifest_path = args.output / ("adapted_manifest.csv" if args.adapted else "manifest.csv")
    if manifest_path.exists():
        rows = pd.read_csv(manifest_path,keep_default_na=False).to_dict("records")
        if len({row["image_id"] for row in rows})!=len(rows):
            raise ValueError("Existing inference manifest has duplicate image IDs")
        for row in rows:
            if validate_cached_row(row,models):
                done.add(row["image_id"])
        rows = [r for r in rows if r["image_id"] in done]
    # Preserve prior run evidence when any cached row fails validation.
    meta_path.write_text(json.dumps(metadata, indent=2))
    while len(done) < args.expected:
        available = sorted(args.data.glob("rgb_anon/*/val/*/*_rgb_anon.png"))
        progress = False
        for path in available:
            image_id = path.name.removesuffix("_rgb_anon.png")
            if image_id in done:
                continue
            parts = path.relative_to(args.data).parts
            condition = parts[1]
            gt = args.data / "gt" / condition / "val" / parts[3] / (image_id + "_gt_labelTrainIds.png")
            if not gt.exists():
                raise FileNotFoundError(gt)
            try:
                tensor = preprocess(path, args.device)
            except (OSError, ValueError):
                # Downloader writes a member in one operation; retry if observed midway.
                continue
            row = dict(image_id=image_id, condition=condition, gt_path=str(gt), image_path=str(path),
                       rgb_sha256=sha(path), gt_sha256=sha(gt))
            for name, model in models.items():
                destination = args.output / "predictions" / name / condition / (image_id + f"_{name}.png")
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.cuda.reset_peak_memory_stats() if args.device.startswith("cuda") else None
                before = time.monotonic()
                with torch.inference_mode():
                    logits = model(tensor)
                    labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    del logits
                with Image.open(gt) as mask:
                    if labels.shape != (mask.height, mask.width):
                        raise ValueError("Prediction and original GT shape mismatch")
                Image.fromarray(labels).save(destination)
                row[f"{name}_pred_path"] = str(destination)
                row[f"{name}_pred_sha256"] = sha(destination)
                row[f"{name}_seconds"] = time.monotonic() - before
                row[f"{name}_peak_gpu_bytes"] = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
                LOG.info("%s %s %.2fs", name, image_id, row[f"{name}_seconds"])
            del tensor
            rows.append(row)
            done.add(image_id)
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            LOG.info("Completed %d/%d validation images (%s), elapsed %.1f min", len(done), args.expected,
                     condition, (time.monotonic() - start) / 60)
            progress = True
            if len(done) >= args.expected:
                break
        if len(done) >= args.expected:
            break
        if not args.wait_for_data:
            raise ValueError(f"Found only {len(done)}/{args.expected} complete validation images")
        if not progress:
            LOG.info("Waiting for dataset download; %d/%d images processed", len(done), args.expected)
            time.sleep(15)
    metadata.update(status="completed", processed=len(done), elapsed_seconds=time.monotonic() - start,
                    finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    meta_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/acdc"))
    p.add_argument("--output", type=Path, default=Path("results"))
    p.add_argument("--hrda", type=Path, default=Path("checkpoints/hrda/csHR2acdcHR_hrda_d3a68/iter_40000_relevant.pth"))
    p.add_argument("--mic", type=Path, default=Path("checkpoints/mic/csHR2acdcHR_mic_hrda_b52f1/iter_40000_relevant.pth"))
    p.add_argument("--adapted", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--expected", type=int, default=406)
    p.add_argument("--wait-for-data", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args)
