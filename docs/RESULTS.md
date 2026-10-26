# Measured results and independent audit

Audit date: 26 October 2026. These are completed local measurements from `results/summary.json`, `results/dinov3/summary.json`, `outputs/dinov3_tuning/tuning_results.json` and `outputs/dinov3_dense/training_config.json`. Published scores are identified separately. No official test-server submission is claimed.

## Primary checkpoint comparison

Both released checkpoints were evaluated on the same complete original ACDC validation split: 100 fog, 106 night, 100 rain and 100 snow images. Confusion counts were pooled within condition before computing class IoUs. V is the equal mean for person, rider and bicycle; W is the minimum V across conditions. All scores below are percentages; differences are percentage points.

| Condition | HRDA V | MIC(HRDA) V | MIC minus HRDA | Paired bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Fog | 69.7298 | 70.2513 | +0.5214 | [-1.0465, +2.5501] |
| Night | 34.4458 | 38.5758 | +4.1300 | [+1.9476, +8.6046] |
| Rain | 66.2862 | 65.8794 | -0.4069 | [-2.7066, +1.1744] |
| Snow | 48.0236 | 50.6018 | +2.5782 | [+0.1206, +7.5494] |
| W | 34.4458 | 38.5758 | +4.1300 | [+0.1683, +8.5626] |

The prespecified **point-estimate hypothesis is supported**: W improves by at least 2 pp and every condition changes by at least -1 pp. Night is the weakest condition for both systems. This does not statistically establish the joint threshold: the W interval includes improvements below 2 pp, and the rain interval includes losses beyond 1 pp. Intervals use 2,000 paired image draws stratified by condition, seed 42. Snow and W each have 1,987 defined and 13 undefined draws; the other endpoints have 2,000 defined draws. Undefined draws are omitted from the corresponding percentile calculation and disclosed, not silently converted to zero.

## Numerical reproduction of the original author logs

| System | Local pooled 19-class mIoU | Last original-author logged mIoU | Local minus logged |
|---|---:|---:|---:|
| HRDA | 65.242980% | 65.24% | +0.002980 pp |
| MIC(HRDA) | 69.600597% | 69.60% | +0.000597 pp |

These agree at the four-decimal fraction precision of the released logs. Independent recomputation from the `(406, 2, 19, 19)` per-image confusion array exactly reproduced every primary condition/class metric and the hypothesis JSON. This is strong evidence against a substantive aggregate inference/metric discrepancy, but it does not prove pixel-identical equivalence to the authors' framework. The author log entries are not our own training. Different original seeds (HRDA 2, MIC 0) and prior author use of ACDC validation prevent a matched-seed causal or untouched-test claim.

## Completed train-only DINOv3 tuning

The internal recording-family split contained 1,198 fitting and 402 development images, with no shared frame IDs or condition/recording-family keys. GOPR/GP chapter aliases were merged. Development counts were fog 81, night 93, rain 154 and snow 74. Coarse recording groups make this 25.125% overall, rather than an exact 20%; the rain fraction is particularly large. Recording-family disjointness does not establish independent geography or scene content.

Each fresh seeded trial trained for one epoch/300 optimizer steps. Selection used pooled VRU IoU on fixed 512-square development center crops; official validation/test did not select the learning rate.

| Learning rate | Development crop VRU mean IoU | Development crop 19-class mIoU |
|---:|---:|---:|
| 0.0003 | 0.0000% | 17.5457% |
| 0.001 | 0.0000% | 31.6023% |
| **0.003** | **8.4212%** | **41.2893%** |

The selected rate was 0.003; no undefined-score fallback was used. Tuning took 597.64 seconds (9.96 minutes). These are internal crop-proxy scores, not official full-image validation scores. Low one-epoch VRU results and the small grid limit how much can be concluded about optimization. Trial weights were discarded.

## Fresh supervised full-data fit and separate evaluation

A freshly initialized 7,315-parameter dense head trained on all 1,600 ACDC training images for five epochs, batch four, AdamW LR 0.003 and weight decay 0.01. The log contains exactly 2,000 optimizer steps and 1,600 images in every epoch, with no skipped all-ignore batch. Training took 985.19 seconds (16.42 minutes), with peak allocated CUDA memory 427.43 MiB on an NVIDIA GeForce GTX 1650 Ti. The frozen encoder's before/after SHA256 matches exactly; the head hash changed. The training configuration's tuning-artifact hash was independently checked.

| Epoch | Valid-pixel-weighted training cross entropy |
|---|---:|
| 1 | 0.45834 |
| 2 | 0.26089 |
| 3 | 0.23804 |
| 4 | 0.23349 |
| 5 | 0.22652 |

The final head was evaluated on all 406 original-resolution validation images, with 512-square tiles and stride 384. This target-supervised result must remain separate from the released UDA comparison.

| Condition | DINOv3 dense-head V |
|---|---:|
| Fog | 48.2471% |
| Night | 37.3262% |
| Rain | 67.6080% |
| Snow | 44.3159% |
| W (night) | 37.3262% |
| Pooled 19-class mIoU | 62.7492% |

Target-label access, foundation pretraining, architecture and decoder differences prevent an isolated architecture ranking or interpretation as a MIC-loss ablation. The modern backbone does not guarantee strong small-class performance with a linear decoder.

## Failures, sparsity and operational limits

Snow rider has only **1,867 GT pixels across two validation images**. Rider IoU there is 3.0418% for HRDA, 2.9647% for MIC and 0% for DINOv3; the new head predicts no snow-rider pixels. Thus the improved snow VRU mean does not imply improvement in every VRU class. In night, rider IoUs are also low: 11.1667%, 13.8809% and 11.3561%, respectively. These are pixel/class measurements, not counts of detected people or demographic fairness estimates.

Recorded primary end-to-end elapsed time was 46,761.19 seconds, including an extended period of laptop inactivity. Per-image processing-time sums were 18,185.10 seconds for HRDA and 28,031.65 for MIC; those timings also reflect interruptions and changing desktop load. They are not controlled speed benchmarks or evidence that MIC's training-only objective adds inference architecture cost; both use the same deployed architecture. DINOv3 evaluation took 393.59 seconds. Do not extrapolate these figures to real-time driving.

No material aggregate implementation discrepancy was found. Remaining limits include sparse-class instability, the omitted undefined bootstrap draws, image rather than scene resampling, different released seeds, source validation-selection bias, unknown foundation pretraining overlap, and no official hidden-test evaluation. A higher W or aggregate mIoU is not proof of safe road-user perception. ACDC raw RGB/GT/overlays remain restricted; public figures should use permissible abstract predictions and aggregate statistics.

## Local checks

An independent `python -m pytest -q` run on 26 October 2026 completed with **24 passed in 17.99 seconds**. The suite covers metrics/encoding, inference-grid logic and train-only tuning/provenance behavior. Tests establish selected implementation properties rather than independent deployment safety. The result tables above were audited against completed artifacts rather than inferred from tests or literature.
