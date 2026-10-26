# Exploratory recording-family sensitivity

Computed from saved per-image HRDA/MIC confusion counts only. No model was rerun. This post hoc influence analysis neither changes the prespecified hypothesis nor supplies a confidence interval.

## Grouping and support

The rule matches the tuning helper exactly: GOPR#### and GPxx#### share recording_#### within a condition. We parse the sequence prefix of each image ID. Unknown naming conventions remain separate. This conservative filename heuristic does not establish geographic independence, duplicate frames or temporal adjacency.

| Condition | Images | Families | Rider-positive images | Rider-positive families | Rider GT pixels |
|---|---:|---:|---:|---:|---:|
| Fog | 100 | 2 | 10 | 2 | 20,814 |
| Night | 106 | 2 | 25 | 2 | 21,587 |
| Rain | 100 | 4 | 8 | 3 | 23,556 |
| Snow | 100 | 6 | 2 | 2 | 1,867 |

The snow rider class occupies only two annotated images. Counts are semantic pixels and positive frames, not rider-instance counts. Full person/bicycle support is retained in the JSON.

## Leave-one-family-out influence

For each condition/recording family, remove all its images, re-pool the remaining counts, then recompute both models' V(c), their independent minima W, and MIC-minus-HRDA delta W. Other conditions remain unchanged. No model or threshold is selected using these results. The full-data delta W remains +4.13 percentage points.

| Omitted condition | Families tried | Delta W range after omission (pp) | Undefined delta W cases | HRDA affected-condition V change range (pp) | MIC affected-condition V change range (pp) |
|---|---:|---:|---:|---:|---:|
| Fog | 2 | +4.13 to +4.13 | 0 | -21.66 to -1.46 | -18.99 to -3.09 |
| Night | 2 | +3.85 to +7.46 | 0 | -9.66 to +2.02 | -6.33 to +1.73 |
| Rain | 4 | +4.13 to +4.13 | 0 | -21.36 to +4.04 | -21.36 to +4.67 |
| Snow | 6 | +4.13 to +4.13 | 0 | -2.08 to +6.61 | -1.75 to +10.34 |

These minima/maxima describe the enumerated omissions; they are not percentile intervals, confidence bounds or estimates of unseen-scene performance.

| Omitted family | Removed / remaining condition images | HRDA V after omission | MIC V after omission | Delta W after omission (pp) |
|---|---:|---:|---:|---:|
| fog/recording_0475 | 22 / 78 | 68.27% | 67.16% | +4.13 |
| fog/recording_0476 | 78 / 22 | 48.07% | 51.26% | +4.13 |
| night/recording_0351 | 56 / 50 | 24.79% | 32.24% | +7.46 |
| night/recording_0356 | 50 / 56 | 36.46% | 40.31% | +3.85 |
| rain/recording_0400 | 18 / 82 | 44.92% | 44.52% | +4.13 |
| rain/recording_0402 | 67 / 33 | 70.32% | 70.55% | +4.13 |
| rain/recording_0571 | 2 / 98 | 65.44% | 64.20% | +4.13 |
| rain/recording_0572 | 13 / 87 | 63.12% | 62.98% | +4.13 |
| snow/recording_0122 | 33 / 67 | 48.88% | 50.32% | +4.13 |
| snow/recording_0176 | 10 / 90 | 45.95% | 48.85% | +4.13 |
| snow/recording_0604 | 23 / 77 | 46.82% | 49.57% | +4.13 |
| snow/recording_0605 | 3 / 97 | 48.24% | 50.97% | +4.13 |
| snow/recording_0606 | 14 / 86 | 54.64% | 60.94% | +4.13 |
| snow/recording_0607 | 17 / 83 | 48.58% | 50.79% | +4.13 |

There are 14 total omissions and 0 undefined delta W cases. JSON retains all class IoUs, affected conditions, aliases and changes from the full-data baseline.

## Interpretation and checks

The analysis shows how heavily a condition estimate can depend on one recording family. Removing a family also changes image composition and class support, so differences cannot isolate a causal weather or recording effect. Few families make a conventional recording-cluster confidence interval unreliable; none is supplied here. A changed or undefined omitted-data endpoint does not overwrite the original complete-validation result.

Independent scalar TP/union arithmetic matched every saved primary class/condition VRU score and W. For every omission, subtracting group counts was checked against a direct re-sum of retained images. Per-image GT supports also matched between HRDA and MIC. Source SHA256 hashes are recorded in results/recording_sensitivity.json.

Reproduce from repository root: `python scripts/analyze_recording_sensitivity.py`. Inputs are results/per_image_confusions.npz and results/summary.json; outputs are results/recording_sensitivity.json and this document. No raw RGB or GT images are redistributed.
