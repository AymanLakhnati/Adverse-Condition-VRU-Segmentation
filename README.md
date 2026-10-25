# Transformer Segmentation of Vulnerable Road Users in Adverse Conditions

MAAI901 applied vision transformer project by Ayman Lakhnati.

[Automated reproducibility workflow](https://github.com/AymanLakhnati/Adverse-Condition-VRU-Segmentation/actions/workflows/tests.yml). Local test evidence is saved under `results/reproducibility/`. The GitHub-hosted job could not start because of an account billing lock; it is not reported as a passing hosted run.

[Open the executable notebook in Colab](https://colab.research.google.com/github/AymanLakhnati/Adverse-Condition-VRU-Segmentation/blob/main/notebooks/Adverse_Condition_VRU_Segmentation.ipynb). Select a GPU runtime before running it. The notebook can also run locally or in Kaggle.

This project studies **person, rider, and bicycle segmentation across fog, nighttime, rain, and snow** in the official ACDC benchmark. It retains the subject of the preceding research proposal.

The implemented study has two clearly separated parts:

1. **Original proposal analysis:** independently evaluate the authors' released HRDA and MIC(HRDA) MiT-B5 checkpoints on all 406 ACDC validation images. Test whether MIC increases the weakest condition's three-class mean IoU by at least 2 percentage points, while no condition loses more than 1 point.
2. **New supervised experiment:** tune the learning rate on recording-family-disjoint internal groups drawn only from ACDC's 1,600 training images, then freshly train a 7,315-parameter segmentation head on all training images using frozen DINOv3 ViT-S/16 dense patch features. This is target-supervised transfer learning, distinct from unsupervised domain adaptation.

The DINOv3 model adds a recent transformer family and fresh training to the applied assignment. Its training labels, pretraining, and decoder differ from HRDA/MIC; the models do not form a controlled architecture ablation. The released HRDA and MIC checkpoints also use different training seeds, so their comparison is descriptive rather than a matched causal estimate of MIC's contribution.

## Reproduce

Python 3.12 is tested. A CUDA GPU is recommended. The sequential inference port was designed to fit a 4 GB GTX 1650 Ti.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\Activate.ps1
```

For another CUDA driver/platform, install the appropriate official PyTorch wheel. The activation command makes the following `python` commands use the project environment. CPU execution is supported but substantially slower; replace `--device cuda` with `--device cpu`. Run every command from the repository root. The executable [notebook](notebooks/Adverse_Condition_VRU_Segmentation.ipynb) preserves cloud cloning and dependency installation for Colab/Kaggle.

First download the shared data and published checkpoints:

```powershell
python -m roadseg.download_checkpoints
python -m roadseg.download_acdc --include-train
```

Run the original HRDA/MIC checkpoint analysis:

```powershell
python -m roadseg.run_inference --device cuda --expected 406
python -m roadseg.metrics results/manifest.csv --path-root . --output results --bootstrap 2000 --require-official-validation
```

Run the separate supervised DINOv3 search and fresh full refit, then evaluate its fixed final head:

```powershell
python -m roadseg.tune_dinov3 --device cuda --refit
python -m roadseg.evaluate_dinov3 --device cuda --batch-size 4
```

After both evaluations complete, generate the measured visualizations:

```powershell
python -m roadseg.visualize
```

After training, predict a single downloaded ACDC image with `python -m roadseg.predict IMAGE_PATH --model dinov3`. Select `--model hrda` or `--model mic` for a released checkpoint. The command writes a numeric label mask, an abstract color preview and a JSON sidecar with class pixel counts. Pixel counts are not object detections or counts of people.

The official RGB package is approximately 15.6 GB and its endpoint does not support range requests. The streaming downloader reads the package while retaining only the adverse training and validation images, checks each retained ZIP member's CRC32, and verifies the label archive's official MD5. It does **not** claim to verify the full RGB archive MD5. Raw data remain local. The downloader uses OS certificate roots without disabling TLS validation.

Tuning and training refuse to overwrite nonempty output directories. The notebook reuses a completed search/refit only after checking their provenance. If the search completed without a refit, launch `roadseg.train_dinov3` with the actual saved `--learning-rate` and `--tuning-results outputs/dinov3_tuning/tuning_results.json`; trial weights are never resumed as the final model. Preserve incomplete runs and choose new output directories when restarting them. Inference can resume using unchanged model/settings provenance. Never use official validation outcomes to select the DINOv3 learning rate or alter the protocol while retaining an untouched evaluation claim.

## Methods and outputs

The primary metric is condition-specific VRU mean IoU: the arithmetic mean of the person, rider, and bicycle IoUs calculated from **pooled pixel confusion counts** in each condition. `W` is the minimum across the four conditions, recomputed independently for each model. Train IDs are 11, 12, and 18; only GT ID 255 is ignored. An absent class with zero union has undefined IoU and cannot silently become perfect or disappear from the three-class hypothesis.

The HRDA/MIC inference port preserves the official dual-scale attention, 1024-pixel outer windows with stride 512, and 512-pixel detail windows with stride 256. It processes detail crops sequentially to reduce memory. The MiT backbone was checked numerically against the official source; checkpoint loading rejects missing and unexpected student tensors. Modern PyTorch kernels can still differ numerically from the original environment. Published training logs provide an additional aggregate validation reference, not new experimental results.

DINOv3 uses frozen normalized patch features with stride 16 and a learned 1x1 convolution into 19 classes. Before the full fit, three one-epoch AdamW trials compare learning rates **0.0003, 0.001 and 0.003**. Within each weather condition, matching GoPro chapters such as `GOPR0376` and `GP010376` stay together as one conservative recording family. A seed-42 ordered recording-family subset targets the nearest feasible 20% development holdout. Equal-distance count ties favor more development frames; equal-count subsets follow seeded family order. Large families can shift the actual fraction; the saved manifests report exact counts. Fitting and development recording families remain disjoint. Trials share the same fresh head initialization and fitting crop/flip randomness.

Selection maximizes pooled person/rider/bicycle mean IoU on one deterministic 512-pixel center crop per development image; exact ties choose the lower LR. Only if all three VRU means are undefined does the declared present-class mIoU fallback apply, and the saved results explicitly disclose it. This one-epoch, center-crop search is a limited proxy rather than exhaustive tuning. **Official validation and test images never enter selection.**

All trial weights are discarded. A fresh seeded head then fits all 1,600 training images for **five fixed epochs**, using the selected LR held fixed, AdamW weight decay 0.01, batch size 4, random 512-pixel crops/flips and cross-entropy ignoring label 255. There is no official-validation tuning, early stopping or official-validation checkpoint selection. Final evaluation averages overlapping 512-pixel tile logits at stride 384 at original image resolution. Report the measured selected LR and selection metric from the tuning JSON, rather than assuming the default 0.001 won.

Important generated outputs:

| Path | Contents |
|---|---|
| `results/summary.json` | HRDA/MIC condition and pooled metrics, hypothesis and paired bootstrap intervals |
| `results/manifest.csv` | Exact evaluated image IDs, hashes, model outputs and timing |
| `results/per_image_confusions.npz` | Paired pixel confusion counts for independent reanalysis |
| `results/dinov3/summary.json` | Separate supervised DINOv3 segmentation results |
| `outputs/dinov3_tuning/tuning_results.json` | Actual internal split, three trials, selection metric, selected LR and any fallback |
| `outputs/dinov3_tuning/tuning_trials.csv` | Measured one-epoch trial scores and timing |
| `outputs/dinov3_tuning/fit_manifest.csv`, `development_manifest.csv` | Train-only recording families, original chapter directories, image IDs and hashes |
| `outputs/dinov3_tuning/training_steps.csv` | Per-trial losses, gradients and crop/flip records |
| `outputs/dinov3_dense/training_config.json` | Actual training configuration, status and unchanged encoder hash |
| `outputs/dinov3_dense/training_steps.csv` | Measured losses and optimization record |
| `outputs/dinov3_dense/head.pt` | Small trained semantic head and foundation-model identity |
| `results/figures/` | Charts and abstract model prediction visualizations |

The official validation split has 100 fog, 106 night, 100 rain, and 100 snow images. It is **validation**, not the private benchmark test set. The released checkpoints were selected using validation information by their authors. Image bootstrap intervals are conditional summaries and do not capture scene dependence, model training variability, or deployment shift.

## Sources and responsible use

See [references](docs/REFERENCES.md), [training provenance](docs/TRAINING_PROVENANCE.md), [model card](docs/ETHICS_AND_MODEL_CARD.md), and [proposal alignment](docs/topic_requirements.md).

ACDC data are governed by the official noncommercial research license. This repository excludes source images, ground-truth masks, raw-image overlays and foundation checkpoints. Code obtains them directly from the official sources. Public illustrations use aggregate statistics and abstract model predictions. Neither segmentation scores nor confidence establish driving safety, collision risk, human demographic fairness, or universal weather robustness.

The backbone/decoder adaptations retain their upstream attribution and licenses in `licenses/`. DINOv3 uses Meta's custom DINOv3 terms. No blanket project license overrides third-party rights.

AI tools assisted implementation, literature checking, analysis and document preparation. The student should verify the artifacts, understand the experimental distinctions, and follow the course's disclosure rules.
