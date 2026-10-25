# Training provenance and adaptation protocol

## Published checkpoint training: performed by the original authors

The two40,000-iteration runs below were **not trained locally for this assignment**. Values were read directly from the released MIC Python configuration, HRDA JSON configuration and accompanying original JSON logs under `checkpoints/`. Their plots must be labeled “original-author training logs.” Local inference timing and newly trained heads belong in separate tables.

| Setting | HRDA ACDC release | MIC(HRDA) ACDC release |
|---|---|---|
| Artifact directory | `csHR2acdcHR_hrda_d3a68` | `csHR2acdcHR_mic_hrda_b52f1` |
| Original run/log date |26 July2022 |1 November2022 |
| Seed |2 |0 |
| Iterations |40,000 |40,000 |
| Original hardware |1 NVIDIA TITAN RTX |1 NVIDIA TITAN RTX |
| Original software |Python3.8.5; torch1.7.1+cu110; torchvision0.8.2+cu110; MMCV1.3.7; MMSegmentation0.16 |same versions reported in original log metadata |
| Encoder/decoder |ImageNet-initialized MiT-B5; HRDA with DAFormer multilevel separable ASPP fusion |same family/configuration |
| Source supervision |Cityscapes train semantic labels |Cityscapes train semantic labels |
| Target access |ACDC adverse train RGB, no target labels collected in pipeline |same |
| Batch |`samples_per_gpu=2`, one GPU; UDADataset supplies a batch of2 source and2 target images |same; extra masked branch described below |
| Train crop |1024 square; source resize2048×1024, target1920×1080 |same |
| Geometry |Random crop; horizontal flip probability0.5 |same |
| Optimizer |AdamW, base LR0.00006; betas0.9/0.999; weight decay0.01 |same |
| Parameter groups |head LR multiplier10; positional/norm decay multiplier0 |same |
| Schedule |linear warmup1500 iterations, initial ratio0.000001; polynomial power1 to minimum0 |same |
| EMA teacher alpha |0.999 |0.999 |
| Pseudo-label confidence threshold |0.968 |0.968 |
| Feature distance |ImageNet feature-distance weight0.005 on configured thing IDs6,7,11–18; min scale ratio0.75 |same |
| Rare-class sampling |min_pixels3000; temperature0.01; min_crop_ratio2 |same |
| Mixing/augmentation |class mixing; blur enabled; color-jitter strength/probability0.2 |same base protocol |
| HRDA fusion |scales1/0.5,512 detail crop, feature scale0.5, classwise attention, HR loss weight0.1 |same |
| Author evaluation |mIoU every4000 iterations on ACDC val |same |

MIC adds `mask_mode='separate'`, block size64, masked ratio0.7, mask loss weight1, `mask_alpha='same'`, `mask_pseudo_threshold='same'`, mask feature-distance weight0, and `mask_blur=True`. The original `masking_consistency_module.py` explicitly stacks **one source and one target image** in this mode, supervising the source with its label and target with teacher pseudo-label/weight. Thus this particular clear-to-adverse release applies masking to both domains. The general paper's explanation of target masking must not be simplified into an incorrect target-only description of this checkpoint.

The released configurations use source label padding255 and normalize RGB using mean123.675/116.28/103.53 and std58.395/57.12/57.375. Source random crops impose `cat_max_ratio=0.75`; target pipeline collects only images. No local access to Cityscapes is needed merely to evaluate these weights.

## Artifact identity and original-log checks

`checkpoints/provenance.json` records original author download links and SHA256:

- HRDA `iter_40000_relevant.pth`: `85e809060244684e7d85f9aa0aa4b0bdacff2f7b7fce92e90e2917ff5c47923d` (verify against machine-readable provenance; never retype hashes for loading).
- MIC `iter_40000_relevant.pth`: `02a9eba7209f1911fd17c61200de5bf60e2ce389b0bc79d1b1319ba61a9b8b61`.

The last original log records include validation-like entries with `iter=406` despite `mode='train'`: these contain aAcc/mIoU and class IoUs, and406 is the validation image count, not proof of a406-step training run. Do not parse every row with mode=train as a training loss step. Some upstream feature-distance log values are NaN; preserve missing/nonfinite values in provenance, omit gaps rather than inventing interpolated losses. These source logs do not establish a locally converged training run.

HRDA's final logged mIoU0.6524 and MIC's0.6960 are **original-author logged values**, not local assignment results and not necessarily the paper's seed-averaged values. Different seeds and original validation use limit causal attribution and independence.

## Local standalone inference port

`src/mit_backbone.py` retains NVIDIA copyright and license; `src/hrda_inference.py` identifies its HRDA/DAFormer adaptation. The port preserves official parameter names and two-scale logic but sequences high-resolution detail crops to fit4GB memory. `working/hrda_port_validation.json` records strict loading of1126 student tensors for each model, exact backbone CPU reference agreement and a synthetic sequential/batched maximum absolute difference about7.08×10^-8. These are implementation checks, not ACDC evaluation results. They do not establish full official-framework equivalence on every image or backend.

The official inference configuration uses1024 outer crops/512 stride,512 detail crops and RGB normalization; any local differences must be recorded with actual prediction manifests. No MIC training loss is executed by this inference port.

## Fresh DINOv3 supervised head: prespecified local work

`docs/DINOV3_DENSE_PLAN.md` and `roadseg/train_dinov3.py` specify: frozen timm DINOv3 ViT-S/16; final normalized patch tokens excluding1CLS+4register tokens;384 channels on a32×32 grid for512-pixel crops; newly initialized1×1 convolution384→19, **7315 trainable parameters**. Bilinear upsampling uses `align_corners=False`.

Plan:1600 adverse ACDC **training** images, five fixed epochs, batch4, seed42, shuffled512×512 random crops and horizontal flips0.5; AdamW LR0.001, weight decay0.01, fixed LR, cross entropy ignoring255, no class weights. The encoder remains eval/frozen with detached features and byte-hash verification; only head weights/bias update. All19 semantic classes are trained;255 is void/ignore, not background. No early stopping, validation checkpoint selection or validation tuning. Inference uses512 tiles/384 stride and averaged overlapping logits at original resolution.

**Status in this document: protocol/code verified; dataset training completion not asserted.** Actual status comes from the run's `training_config.json` and step log. Record partial runs/limits, completed epochs, source hashes, sampled geometry, loss/gradient values, time, device, peak memory, final head hash and unchanged encoder verification. The tiny head is fresh supervised target learning, not retraining DINOv3, reproducing MIC training or an unsupervised-adaptation ablation.

An additional optional MIC-initialized supervised classifier adaptation is specified separately in `docs/SUPERVISED_EXTENSION_PLAN.md`:4883 trainable classifier parameters, one epoch, batch1, LR0.0001, weight decay0.01, train labels only. Do not imply it was run unless actual completed artifacts exist; keep it separate from both primary UDA checkpoint comparison and fresh DINOv3 head.
