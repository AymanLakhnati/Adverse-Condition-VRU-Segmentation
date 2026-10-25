"""Plot measured segmentation evidence without redistributing ACDC pixels."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

from roadseg.metrics import CLASS_NAMES, CONDITIONS, VRU

ROOT = Path("results")
FIG = ROOT / "figures"
LABELS = ["Fog", "Nighttime", "Rain", "Snow"]
COLORS = ["#9ba9ad", "#087f8c", "#bf7045"]


def save(name):
    plt.savefig(FIG / (name + ".png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(FIG / (name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    primary = json.loads((ROOT / "summary.json").read_text())
    dino = json.loads((ROOT / "dinov3/summary.json").read_text())
    train = json.loads(Path("outputs/dinov3_dense/training_config.json").read_text())
    if primary["dataset"]["images"] != 406 or dino["validation_images"] != 406 or train["status"] != "completed":
        raise ValueError("Figures require completed, measured full protocols")
    systems = {"HRDA (UDA)": primary["models"]["hrda"], "MIC (UDA)": primary["models"]["mic"],
               "DINOv3 (supervised)": dino}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, (ax, delta_ax) = plt.subplots(1, 2, figsize=(10, 3.7), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(4)
    for j, name in enumerate(("hrda", "mic")):
        values = [primary["models"][name]["conditions"][c]["vru_mean_iou"] * 100 for c in CONDITIONS]
        ax.bar(x + (j - .5) * .34, values, .34, color=COLORS[j], label="HRDA" if j == 0 else "MIC(HRDA)")
    ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 100), ylabel="Mean IoU of person, rider and bicycle (%)",
           title="Released UDA checkpoints on ACDC validation")
    ax.legend(frameon=False)
    point = [primary["hypothesis"]["condition_delta_pp"][c] for c in CONDITIONS]
    intervals = [primary["bootstrap"]["endpoints"][c]["percentile_95_pp"] for c in CONDITIONS]
    delta_ax.errorbar(point, x, xerr=np.array([[p - interval[0] for p, interval in zip(point, intervals)],
                                              [interval[1] - p for p, interval in zip(point, intervals)]]),
                      fmt="o", color=COLORS[1], capsize=4)
    delta_ax.axvline(0, color="#888888", linewidth=.8)
    delta_ax.axvline(-1, color="#bf7045", linestyle="--", linewidth=1)
    delta_ax.set(yticks=x, yticklabels=LABELS, xlabel="MIC minus HRDA (percentage points)",
                 title="Paired image bootstrap 95% intervals")
    delta_ax.invert_yaxis()
    fig.tight_layout()
    save("condition_vru")

    values, rows = [], []
    for name, system in systems.items():
        for class_id in VRU:
            values.append([system["conditions"][c]["per_class"][class_id]["iou"] for c in CONDITIONS])
            rows.append(f"{name}: {CLASS_NAMES[class_id]}")
    array = np.array([[np.nan if v is None else v * 100 for v in row] for row in values])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(array, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    for r in range(9):
        for c in range(4):
            ax.text(c, r, "n/a" if np.isnan(array[r, c]) else f"{array[r,c]:.1f}", ha="center", va="center",
                    color="white" if array[r, c] > 55 else "#183644")
    ax.set(xticks=range(4), xticklabels=LABELS, yticks=range(9), yticklabels=rows,
           title="Class and condition IoU (%)\nDINOv3 uses target labels; learning settings differ")
    fig.colorbar(im, ax=ax, label="IoU (%)")
    save("class_condition_heatmap")

    data = np.load(ROOT / "per_image_confusions.npz")
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.8), constrained_layout=True)
    for index, ((name, system), ax) in enumerate(zip(systems.items(), axes)):
        if index < 2:
            cm = data["confusion"][:, index].sum(0)
        else:
            third = np.load(ROOT / "dinov3/per_image_confusions.npz")
            cm = third["confusions"].sum(0)
        matrix = cm[np.array(VRU)].astype(float)
        matrix /= np.maximum(matrix.sum(1, keepdims=True), 1)
        im = ax.imshow(matrix * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set(yticks=range(3), yticklabels=[CLASS_NAMES[i] for i in VRU], xticks=range(19),
               xticklabels=CLASS_NAMES, title=name)
        ax.tick_params(axis="x", labelrotation=50, labelsize=8)
    fig.colorbar(im, ax=list(axes), label="Fraction of true-class pixels (%)", shrink=.8)
    save("confusion_vru")

    steps = pd.read_csv("outputs/dinov3_dense/training_steps.csv")
    fig, ax = plt.subplots(figsize=(8, 3.1))
    ax.plot(steps.global_step, steps.loss, color="#9daeb2", alpha=.25, linewidth=.6, label="Batch loss")
    smooth = steps.loss.rolling(50, min_periods=1).mean()
    ax.plot(steps.global_step, smooth, color=COLORS[1], linewidth=1.6, label="50-step moving mean")
    for epoch in range(2, 6):
        ax.axvline(steps[steps.epoch == epoch].global_step.min(), color="#c7d3d5", linestyle="--", linewidth=.7)
    ax.set(xlabel="Optimizer step", ylabel="Cross entropy on random training crops",
           title="New DINOv3 segmentation head: five fixed training epochs")
    ax.legend(frameon=False)
    save("dinov3_training_curve")

    manifest = pd.read_csv(ROOT / "manifest.csv")
    third_manifest = pd.read_csv(ROOT / "dinov3/manifest.csv").set_index("image_id")
    # Select the largest GT VRU support image in each condition, deterministically.
    support = data["confusion"][:, 0, np.array(VRU), :].sum((1, 2))
    supports = dict(zip(data["image_id"].astype(str), support))
    manifest["vru_gt_pixels"] = manifest.image_id.map(supports)
    selected = []
    for condition in CONDITIONS:
        selected.append(manifest[manifest.condition == condition].sort_values(
            ["vru_gt_pixels", "image_id"], ascending=[False, True]).iloc[0])
    palette = np.array([[55 + i * 7] * 3 for i in range(19)], dtype=np.uint8)
    palette[11], palette[12], palette[18] = [239, 94, 85], [247, 181, 65], [37, 184, 201]
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 8.2))
    selection_records = []
    for r, row in enumerate(selected):
        with Image.open(row.gt_path) as image:
            gt = np.asarray(image)
        yy, xx = np.where(np.isin(gt, VRU))
        box = (max(0, yy.min() - 64), min(gt.shape[0], yy.max() + 65),
               max(0, xx.min() - 64), min(gt.shape[1], xx.max() + 65))
        paths = [row.hrda_pred_path, row.mic_pred_path, third_manifest.loc[row.image_id, "pred_path"]]
        for j, path in enumerate(paths):
            with Image.open(path) as image:
                mask = np.asarray(image)
            axes[r, j].imshow(palette[mask[box[0]:box[1], box[2]:box[3]]])
            axes[r, j].axis("off")
            if r == 0:
                axes[r, j].set_title(list(systems)[j], fontsize=11)
            if j == 0:
                axes[r, j].text(-.04, .5, LABELS[r], rotation=90, va="center", ha="right",
                                transform=axes[r, j].transAxes, fontsize=11)
            axes[r, j].text(.5, -.03, row.image_id, transform=axes[r, j].transAxes,
                            ha="center", va="top", fontsize=7)
        selection_records.append({"image_id": row.image_id, "condition": row.condition,
                                  "gt_vru_pixels": int(row.vru_gt_pixels), "crop_y1_y2_x1_x2": list(map(int, box))})
    fig.legend(handles=[Patch(color=palette[i] / 255, label=CLASS_NAMES[i]) for i in VRU],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Own predicted labels around selected VRU regions\nLargest labeled VRU support per condition; no RGB or GT pixels shown", fontsize=12)
    fig.tight_layout(rect=[.025, .045, 1, .94], h_pad=1.5)
    save("sample_predictions")
    (FIG / "sample_selection.json").write_text(json.dumps(selection_records, indent=2))
    print(FIG.resolve())


if __name__ == "__main__":
    main()
