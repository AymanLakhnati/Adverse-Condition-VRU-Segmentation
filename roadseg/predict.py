"""Predict an image with a released UDA model or our trained DINOv3 head."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from roadseg.metrics import CLASS_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("--model", choices=["hrda", "mic", "dinov3"], default="dinov3")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/sample_output"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    defaults = {"hrda": "checkpoints/hrda/csHR2acdcHR_hrda_d3a68/iter_40000_relevant.pth",
                "mic": "checkpoints/mic/csHR2acdcHR_mic_hrda_b52f1/iter_40000_relevant.pth",
                "dinov3": "outputs/dinov3_dense/head.pt"}
    checkpoint = args.checkpoint or Path(defaults[args.model])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Download or train the selected checkpoint first: {checkpoint}")
    with Image.open(args.image) as source:
        original_shape = (source.height, source.width)
    started = time.perf_counter()
    if args.model == "dinov3":
        from roadseg.train_dinov3 import load_trained_model, image_tensor, predict_logits
        model = load_trained_model(checkpoint, args.device)
        tensor = image_tensor(args.image, args.device)
        with torch.inference_mode():
            logits = predict_logits(model, tensor)
    else:
        from src.hrda_inference import load_model, preprocess
        model = load_model(checkpoint, args.device)
        tensor = preprocess(args.image, args.device)
        resized_shape = tensor.shape[-2:]
        # The official ACDC size already aligns to eight. Padding allows other
        # aspect ratios while preserving low/high decoder grid alignment.
        tensor = F.pad(tensor, (0, (-tensor.shape[-1]) % 8, 0, (-tensor.shape[-2]) % 8))
        with torch.inference_mode():
            logits = model(tensor)[:, :, :resized_shape[0], :resized_shape[1]]
    with torch.inference_mode():
        if logits.shape[-2:] != original_shape:
            logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)
        labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.output / (args.image.stem + "_" + args.model)
    label_path = prefix.with_suffix(".png")
    Image.fromarray(labels).save(label_path)
    palette = np.array([[55 + 7 * i] * 3 for i in range(19)], dtype=np.uint8)
    palette[11], palette[12], palette[18] = [239, 94, 85], [247, 181, 65], [37, 184, 201]
    Image.fromarray(palette[labels]).save(prefix.with_name(prefix.name + "_preview").with_suffix(".png"))
    counts = np.bincount(labels.ravel(), minlength=19)
    info = {"model": args.model, "checkpoint": str(checkpoint), "source_image": str(args.image),
            "source_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
            "output_shape": list(labels.shape), "train_id_labels": {i: name for i, name in enumerate(CLASS_NAMES)},
            "predicted_pixel_counts": {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)},
            "elapsed_seconds_including_model_load": time.perf_counter() - started,
            "interpretation": "Semantic pixel labels, not individual detections, demographic labels or a driving-safety assessment"}
    prefix.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps({"labels": str(label_path), "metadata": str(prefix.with_suffix('.json'))}, indent=2))


if __name__ == "__main__":
    main()
