# Transformer Segmentation of Vulnerable Road Users in Adverse Conditions

MAAI901 applied vision transformer project by Ayman Lakhnati.

This project studies **person, rider, and bicycle segmentation across fog, nighttime, rain, and snow** in the official ACDC benchmark. It retains the subject of the preceding research proposal.

The implemented study has two clearly separated parts:

1. **Original proposal analysis:** independently evaluate the authors' released HRDA and MIC(HRDA) MiT-B5 checkpoints on all 406 ACDC validation images. Test whether MIC increases the weakest condition's three-class mean IoU by at least 2 percentage points, while no condition loses more than 1 point.
2. **New supervised experiment:** extract dense patch features from a frozen DINOv3 ViT-S/16 and train a new 7,315-parameter pixel classifier using only ACDC's 1,600 training images and labels. This is target-supervised transfer learning, distinct from unsupervised domain adaptation.

The DINOv3 model adds a recent transformer family and fresh training to the applied assignment. Its training labels, pretraining, and decoder differ from HRDA/MIC; the models do not form a controlled architecture ablation. The released HRDA and MIC checkpoints also use different training seeds, so their comparison is descriptive rather than a matched causal estimate of MIC's contribution.

## Reproduce

Python 3.12 is tested. A CUDA GPU is recommended. The sequential inference port was designed to fit a 4 GB GTX 1650 Ti.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

For another CUDA driver/platform, install the appropriate official PyTorch wheel. CPU execution is supported but substantially slower. Run every command from the repository root. See the executable [notebook](notebooks/Adverse_Condition_VRU_Segmentation.ipynb).

```powershell
python -m roadseg.download_checkpoints
python -m roadseg.download_acdc --include-train
python -m roadseg.run_inference
python -m roadseg.metrics results/manifest.csv --output results --bootstrap 2000 --require-official-validation
python -m roadseg.train_dinov3 --epochs 5 --batch-size 4 --seed 42
python -m roadseg.evaluate_dinov3
python -m roadseg.visualize
```

The official RGB package is approximately 15.6 GB and its endpoint does not support range requests. The streaming downloader reads the package while retaining only the adverse training and validation images, checks each retained ZIP member's CRC32, and verifies the label archive's official MD5. It does **not** claim to verify the full RGB archive MD5. Raw data remain local. The downloader uses OS certificate roots without disabling TLS validation.

Training refuses to overwrite a nonempty output directory. Inference can resume using unchanged model/settings provenance. Use a separate output directory for a different experiment. Never use observed validation outcomes to alter the prespecified protocol while retaining an untouched evaluation claim.

## Methods and outputs

The primary metric is condition-specific VRU mean IoU: the arithmetic mean of the person, rider, and bicycle IoUs calculated from **pooled pixel confusion counts** in each condition. `W` is the minimum across the four conditions, recomputed independently for each model. Train IDs are 11, 12, and 18; only GT ID 255 is ignored. An absent class with zero union has undefined IoU and cannot silently become perfect or disappear from the three-class hypothesis.

The HRDA/MIC inference port preserves the official dual-scale attention, 1024-pixel outer windows with stride 512, and 512-pixel detail windows with stride 256. It processes detail crops sequentially to reduce memory. The MiT backbone was checked numerically against the official source; checkpoint loading rejects missing and unexpected student tensors. Modern PyTorch kernels can still differ numerically from the original environment. Published training logs provide an additional aggregate validation reference, not new experimental results.

DINOv3 uses 512-pixel random crops and horizontal flips, frozen normalized patch features with stride 16, and a learned 1x1 convolution into 19 classes. Five fixed epochs use AdamW, learning rate 0.001, weight decay 0.01, batch size 4, and cross-entropy ignoring label 255. No early stopping or evaluation-based hyperparameter selection is used. Evaluation averages overlapping 512-pixel tile logits with stride 384 at the original image resolution.

Important generated outputs:

| Path | Contents |
|---|---|
| `results/summary.json` | HRDA/MIC condition and pooled metrics, hypothesis and paired bootstrap intervals |
| `results/manifest.csv` | Exact evaluated image IDs, hashes, model outputs and timing |
| `results/per_image_confusions.npz` | Paired pixel confusion counts for independent reanalysis |
| `results/dinov3/summary.json` | Separate supervised DINOv3 segmentation results |
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
