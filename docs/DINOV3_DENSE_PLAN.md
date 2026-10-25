# Prespecified modern-transformer segmentation extension

Protocol fixed on 25 October 2026 before examining local ACDC validation scores. This experiment studies road-user semantic segmentation using only the prescribed ACDC splits.

The original hypothesis concerns the released HRDA and MIC(HRDA) checkpoints. This separate experiment applies the recent DINOv3 ViT-S/16 model family and trains a new target-supervised segmentation head on ACDC training labels. The settings differ in supervision, pretraining and architecture, so it cannot isolate a transformer architecture effect or serve as a MIC-only ablation. DINOv3's 2025 model-family results motivate the backbone choice, but this simple classifier does not claim state-of-the-art segmentation performance.

Load `timm/vit_small_patch16_dinov3.lvd1689m`, enable dynamic input size, and freeze every encoder parameter. At 512-by-512 resolution, use the final normalized patch tokens from `forward_features`, excluding the one class token and four register tokens. Reshape the 1,024 remaining 384-dimensional vectors to a 384-by-32-by-32 feature map. Train a newly initialized 1-by-1 convolution from 384 features to 19 Cityscapes-compatible classes, totaling 7,315 parameters. Bilinearly upsample its logits to the crop size with `align_corners=False`.

Use all 1,600 adverse-weather ACDC training images for five fixed epochs, batch size four, shuffled each epoch and seed 42. Each occurrence uses one random 512-by-512 crop and horizontal flip with probability 0.5, synchronizing geometry with its label mask. Normalize RGB using ImageNet means and standard deviations. Train with AdamW, fixed learning rate 0.001 and weight decay 0.01. Optimize cross entropy, ignore label 255, and use no class weights. Keep the backbone in evaluation mode with no gradient tracking. No validation-based tuning, early stopping or checkpoint selection is permitted. A partial `--limit` run or changed epoch/batch settings must be explicitly distinguished from this full protocol.

Evaluate the final fixed-epoch head on the same 406 adverse validation images. Preserve each image's original resolution, infer 512-by-512 tiles at stride 384 with border-aligned final windows, bilinearly upsample tile logits and average overlaps before argmax. Predict all 19 classes. Report the original person's, rider's and bicycle's per-class IoU and the same primary VRU aggregation, alongside all-class metrics, in an explicitly labeled supervised-extension table.

Save a small head checkpoint, the exact encoder state SHA256, source model configuration, package versions, training IDs with image/label hashes, sampled crop coordinates and flips, per-step losses and gradient norms, actual duration and peak GPU allocation. Verify the encoder's parameters and buffers remain byte-identical. Do not distribute foundation weights or dataset images. The stride-16 representation and linear decoder can lose small road-user detail, and no result is guaranteed before evaluation.

Synthetic-label smoke tests establish shape, gradient and memory feasibility only. They are not dataset training or evaluation results.

Primary sources: [DINOv3 paper](https://arxiv.org/abs/2508.10104), [timm checkpoint and configuration](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m), [official ACDC dataset](https://acdc.vision.ee.ethz.ch/).
