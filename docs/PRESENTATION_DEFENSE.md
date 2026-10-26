# Presentation defense notes

Preparation for the 27 October 2026 presentation. Use these explanations in your own words. The primary evaluation, train-only tuning, fresh supervised fit and DINOv3 validation evaluation are complete and independently checked. The measured sources are `results/summary.json`, `results/dinov3/summary.json`, `outputs/dinov3_tuning/tuning_results.json` and `outputs/dinov3_dense/training_config.json`. Check those files when quoting more precision; synthetic smoke tests are separate from these completed dataset experiments.

## Numbers to know

| Item | Verified result |
|---|---|
| Official validation coverage | 406 images: fog 100, night 106, rain 100, snow 100 |
| Primary worst-condition W | HRDA 34.45%; MIC(HRDA) 38.58%; both weakest at night |
| W difference | +4.13 percentage points; point-estimate hypothesis supported |
| Condition changes, MIC minus HRDA | Fog +0.52 pp; night +4.13 pp; rain -0.41 pp; snow +2.58 pp |
| W 95% bootstrap interval | [+0.17, +8.56] pp; includes effects below the required +2 pp |
| Rain 95% bootstrap interval | [-2.71, +1.17] pp; includes losses beyond the allowed -1 pp |
| Sparse snow rider | Only 1,867 GT pixels in two images; HRDA 3.04%, MIC 2.96%, DINOv3 0% rider IoU |
| Train-only tuning split | 1,198 fitting / 402 development images, recording-family disjoint |
| Learning-rate trials | 0.0003, 0.001, 0.003; selected 0.003 using development crop VRU IoU |
| Fresh full-data fit | 1,600 training images, five epochs, batch four, 2,000 optimizer steps, 16.42 minutes |
| New head | 7,315 trainable parameters; encoder hash unchanged |
| Separate supervised DINOv3 validation | Pooled 19-class mIoU 62.75%; W 37.33%, weakest at night |
| Local implementation tests | 54 passed after integrity and feature improvements; all saved primary and DINOv3 aggregate metrics independently recomputed |

The hypothesis passes the stated thresholds at the observed point estimates. The confidence intervals do **not** establish the full joint improvement/no-harm thresholds, and no result proves driving safety. Snow and W intervals omit 13 undefined draws out of 2,000; this is disclosed in the summary. DINOv3 uses target training labels, so its scores belong to a separate learning setting.

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

We completed a separate DINOv3 experiment that trained a new dense linear segmentation head using ACDC training labels while freezing the pretrained encoder. Three learning rates were compared on an internal fitting/development split of the official training data. The selected rate, 0.003, was used in a fresh five-epoch fit on all 1,600 training images; trial weights were discarded. The full fit used batch four, random 512-pixel crops/flips, AdamW with weight decay 0.01 and cross entropy ignoring 255. It completed 2,000 optimizer steps in 16.42 minutes. Only 7,315 head parameters updated; the encoder hash remained identical. This is supervised target-domain learning, not full encoder fine-tuning or UDA. The completed tuning and full-refit configurations and logs provide the evidence.

## What are DINOv3 dense patch features?

A 512-square image yields a 32-by-32 grid of stride-16 patch features. Each feature has 384 channels. We discard the class token and four register tokens, retain the spatial grid, and apply a 1-by-1 convolution to predict 19 class logits per grid location. Bilinear upsampling produces pixel-resolution logits. There is no image-level average pooling or CLS-based classification. Overlapping inference tiles are combined by averaging logits before argmax.

## Why use DINOv3?

It supplies a recent 2025 transformer foundation-model family and a small checkpoint feasible on the available GPU. Frozen features let us train a new segmentation head with a modest memory budget. The family has strong published dense-vision results; our simple linear head does not automatically achieve those results or establish a new state of the art. Patch stride and decoder simplicity can restrict fine detail.

## Is DINOv3 versus MIC a fair architecture experiment?

It is a comparison of different complete learning settings, not an isolated architecture experiment. The new DINOv3 head sees ACDC training labels; the released UDA models used target training images without target labels. Pretraining data, decoders and inference also differ. Report the supervised extension separately. Even HRDA versus MIC is not a matched-seed causal ablation: these released artifacts use seeds two and zero respectively.

## How did you control leakage?

Learning-rate tuning stayed entirely inside the official ACDC training partition. We compared 0.0003, 0.001 and 0.003 for one fitting epoch each, resetting to the same fresh head initialization and random sequence. The internal development subset was held out by recording family rather than individual frame. GOPR/GP camera-file aliases belonging to the same recording family remained together; the fitting and development manifests have no shared frame IDs or condition/recording-family keys. Group assignment targeted approximately 20% development images within each condition without inspecting label content. The actual split was 1,198 fitting and 402 development images: large recording groups made the overall development share 25.125%, with 154 of 400 rain images in development.

Selection used pooled person/rider/bicycle mean IoU on one fixed 512-pixel center crop per development image, with lower-rate tie breaking. This center-crop proxy is a limited tuning criterion, not the full-image validation endpoint. The two lower rates produced 0% development VRU IoU; 0.003 produced 8.42% and was selected. No undefined-score fallback was needed. After selection, all trial weights were discarded and a fresh head completed five fixed epochs on all 1,600 training images. There was no early stopping or checkpoint selection from the official 406-image validation split; official validation never chose the learning rate.

Recording-family disjointness is an internal fitting/development safeguard, not an independently established property of the official training/validation split. A filename heuristic finds family names shared across official splits even though frame IDs are disjoint. That does not establish duplicate frames or geographic leakage: distinct road segments can still satisfy official geographic separation. Treat recording dependence as an evaluation limitation.

Manifests preserve identities, recording groups, split paths and file hashes; mask alignment was checked. The released primary models' authors nevertheless used ACDC validation during development/evaluation. Therefore call our 406-image evaluation validation, not untouched test performance. Official test labels were not accessed. Foundation-model pretraining overlap also cannot be independently excluded. Completion, selected rate, content hashes, encoder invariance and actual step counts are verified in the saved tuning/refit artifacts.

## What does the bootstrap interval mean?

It describes variation from resampling the observed validation images while keeping both primary models paired and preserving condition sample sizes. Each draw recomputes pooled class IoUs and each model's minimum. It does not measure variability from retraining, account for correlated scenes, establish generalization to another country or prove safety. Read the interval and undefined-draw count from the saved summary rather than guessing them.

## How do you know the inference port is correct?

Strict loading checks tensor names and dimensions. Reference-backbone and sequential-versus-batched synthetic comparisons test selected numerical behavior; mask tests check metric calculations and invalid encodings. These are stronger than checking that an image looks plausible. They still do not prove bit-identical predictions to every version of the original framework or eliminate all implementation uncertainty. Modern kernels and checkpoint-selection differences must remain limitations.

## What are the main ethical limitations?

Small or occluded people can remain difficult despite better averages. Teacher mistakes can reinforce themselves, and context can become a shortcut. Swiss geography, road layouts, camera placement and selected adverse scenes limit coverage. ACDC annotation uncertainty is not model confidence. We have no validated demographic labels and cannot claim demographic fairness. Predictions do not identify individuals, and this classroom evaluation cannot authorize autonomous driving.

## Why are original RGB and GT absent from public files?

ACDC's research agreement prohibits redistributing the dataset or modified versions. We provide official download instructions, code, aggregate charts and nonreconstructive abstract outputs; original RGB, GT and image overlays stay local. Anonymization does not override the license or justify reidentification. Adapted NVIDIA/ETH code and DINOv3 materials retain their respective license and attribution requirements.

## What did the hypothesis show, and how would you handle a negative result?

The actual point-estimate hypothesis was supported: W improved by 4.13 pp and the only condition regression was rain at -0.41 pp, within the declared -1 pp allowance. The intervals are wider than those thresholds, so we do not claim statistical confirmation of the joint requirement. Snow rider remained poor despite a higher snow average. Had the hypothesis failed, we would report that outcome plainly; a negative result still answers the prespecified question. Do not change thresholds, choose a favorable subset, substitute published scores or present supervised DINOv3 gains as evidence that MIC alone caused an improvement.

## Software demonstration

Slide 13 introduces folder prediction and the CPU-only independent results verifier. Use the commands in `SOFTWARE_GUIDE.md`. A real two-image DINOv3 batch (one fog, one night) produced pixel-identical masks to the saved full-evaluation predictions; this is a software equivalence check, not another accuracy experiment.

The cover and context/detail picture are conceptual AI-generated illustrations. Say so when needed; only the measured prediction panels and scientific charts represent experimental evidence.
