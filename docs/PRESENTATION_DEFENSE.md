# Presentation defense notes

Use these explanations in your own words. For measured values, read `results/summary.json` and `results/dinov3/summary.json` after their full runs complete. Training completion and actual settings come from `outputs/dinov3_dense/training_config.json`. A planned protocol or synthetic smoke test is not a completed experiment.

## What is the task?

Assign a semantic class to every image pixel, emphasizing person, rider and bicycle under fog, nighttime, rain and snow. This is dense semantic segmentation, not image classification, identity recognition, instance tracking or collision prediction. Those three classes are the proposal's operational VRU subset; they do not cover every possible vulnerable road user.

## Why IoU rather than pixel accuracy?

IoU measures overlap between predicted and reference pixels of one class: true positives divided by true positives plus false positives plus false negatives. Pixel accuracy can look high when roads and buildings dominate while small people are missed. We therefore report individual VRU IoUs, their equal mean and supporting precision/recall/counts. IoU still cannot tell us how many whole people were missed or whether a vehicle is safe.

## Why the minimum across conditions?

An average can hide one weak domain. First compute the person/rider/bicycle mean within each condition, then take the minimum of those four means. This endpoint is W. Each model gets its own minimum; its weakest condition can change. The original hypothesis requires at least a two-percentage-point improvement in W and no condition losing more than one point. These are prespecified descriptive thresholds, not an independently validated safety standard.

## Do you average each image's IoU?

No. We sum pixel confusion counts over images within a condition, then calculate class IoUs. Confusion rows are ground truth and columns are prediction. An absent class union is undefined, not automatically zero or perfect. GT 255 is ignored; it is a void ID, not a twentieth background class. Valid semantic classes 0-18 remain part of training and evaluation.

## How are HRDA and MIC different?

HRDA combines broad low-resolution context with smaller high-resolution crops using learned scale attention. MIC is an additional training consistency objective: a student predicts masked images using supervision from labels or an EMA teacher's predictions of the complete image. The released Cityscapes-to-ACDC MIC configuration applies its separate masked branch to one source and one target image. Masking is a training technique; we do not mask images during normal checkpoint inference.

## Did you train the released HRDA and MIC models yourself?

No. Their full adaptation training was performed by the original authors. Our contribution is downloading verified checkpoints, implementing memory-bounded inference, testing the port, evaluating the same validation images and analyzing condition/class errors. The checkpoint configurations and original logs explain prior training, but their curves are not our own training results.

## What fresh training did you perform?

The separate DINOv3 experiment trains a new dense linear segmentation head using ACDC training labels while freezing the pretrained encoder. It first compares three learning rates using an internal fitting/development split of the official training data. The selected rate then feeds a fresh five-epoch fit on all 1,600 training images; trial weights are discarded. The full fit uses batch four, random 512-pixel crops/flips, AdamW and cross entropy ignoring 255. Only 7,315 head parameters update; this is supervised target-domain learning, not full encoder fine-tuning or UDA. Verify the completed tuning and full-refit artifacts before saying either stage finished.

## What are DINOv3 dense patch features?

A 512-square image yields a 32-by-32 grid of stride-16 patch features. Each feature has 384 channels. We discard the class token and four register tokens, retain the spatial grid, and apply a 1-by-1 convolution to predict 19 class logits per grid location. Bilinear upsampling produces pixel-resolution logits. There is no image-level average pooling or CLS-based classification. Overlapping inference tiles are combined by averaging logits before argmax.

## Why use DINOv3?

It supplies a recent 2025 transformer foundation-model family and a small checkpoint feasible on the available GPU. Frozen features let us train a new segmentation head with a modest memory budget. The family has strong published dense-vision results; our simple linear head does not automatically achieve those results or establish a new state of the art. Patch stride and decoder simplicity can restrict fine detail.

## Is DINOv3 versus MIC a fair architecture experiment?

It is a comparison of different complete learning settings, not an isolated architecture experiment. The new DINOv3 head sees ACDC training labels; the released UDA models used target training images without target labels. Pretraining data, decoders and inference also differ. Report the supervised extension separately. Even HRDA versus MIC is not a matched-seed causal ablation: these released artifacts use seeds two and zero respectively.

## How did you control leakage?

Learning-rate tuning is permitted, but it stays entirely inside the official ACDC training partition. The plan compares 0.0003, 0.001 and 0.003 for one fitting epoch each, resetting to the same fresh head initialization and random sequence. An internal development subset is held out by recording family rather than individual frame. GOPR/GP camera-file aliases belonging to the same recording family must remain together; verify this disjointness in the completed tuning provenance before claiming it was enforced. Group assignment targets approximately 20% development images within each condition without inspecting label content.

Selection uses pooled person/rider/bicycle mean IoU on one fixed 512-pixel center crop per development image; exact ties prefer the lower rate. This center-crop proxy is a limited tuning criterion, not the full-image validation endpoint. An undefined VRU union makes a trial ineligible; if all VRU means are undefined, the declared present-class mean-IoU fallback must be explicitly reported. After selection, discard all trial weights and initialize a fresh head for five fixed epochs on all 1,600 training images. There is no early stopping or checkpoint selection from the official 406-image validation split, and official validation never chooses the learning rate.

Manifests preserve identities, recording groups, split paths and file hashes; mask alignment is checked. The released primary models' authors nevertheless used ACDC validation during development/evaluation. Therefore call our 406-image evaluation validation, not untouched test performance. Official test labels were not accessed. Foundation-model pretraining overlap also cannot be independently excluded. Until the actual tuning/refit artifacts confirm completion, describe these steps as the planned protocol rather than completed work.

## What does the bootstrap interval mean?

It describes variation from resampling the observed validation images while keeping both primary models paired and preserving condition sample sizes. Each draw recomputes pooled class IoUs and each model's minimum. It does not measure variability from retraining, account for correlated scenes, establish generalization to another country or prove safety. Read the interval and undefined-draw count from the saved summary rather than guessing them.

## How do you know the inference port is correct?

Strict loading checks tensor names and dimensions. Reference-backbone and sequential-versus-batched synthetic comparisons test selected numerical behavior; mask tests check metric calculations and invalid encodings. These are stronger than checking that an image looks plausible. They still do not prove bit-identical predictions to every version of the original framework or eliminate all implementation uncertainty. Modern kernels and checkpoint-selection differences must remain limitations.

## What are the main ethical limitations?

Small or occluded people can remain difficult despite better averages. Teacher mistakes can reinforce themselves, and context can become a shortcut. Swiss geography, road layouts, camera placement and selected adverse scenes limit coverage. ACDC annotation uncertainty is not model confidence. We have no validated demographic labels and cannot claim demographic fairness. Predictions do not identify individuals, and this classroom evaluation cannot authorize autonomous driving.

## Why are original RGB and GT absent from public files?

ACDC's research agreement prohibits redistributing the dataset or modified versions. We provide official download instructions, code, aggregate charts and nonreconstructive abstract outputs; original RGB, GT and image overlays stay local. Anonymization does not override the license or justify reidentification. Adapted NVIDIA/ETH code and DINOv3 materials retain their respective license and attribution requirements.

## What if the proposed hypothesis is not supported?

Report that outcome plainly and show the condition and class evidence. A negative result still answers the prespecified question and can reveal a harmful tradeoff hidden by an aggregate score. Do not change thresholds, choose a favorable subset, substitute published scores or present supervised DINOv3 gains as evidence that MIC alone caused an improvement. Explain what the evidence supports and what remains uncertain.
