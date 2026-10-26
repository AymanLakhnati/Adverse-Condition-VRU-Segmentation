# Software demonstration and evidence verification

Run the commands below from the repository root in the environment described in `README.md`. Dataset and foundation checkpoints must be obtained from their official sources. Code does not bundle restricted ACDC images or labels.

## Verify the completed results without a GPU

```powershell
python -m pip install numpy==2.2.6
python scripts/verify_results.py
python scripts/verify_results.py --root . --output working/evidence_verification.json
```

The verifier uses saved per-image confusion counts to independently derive the reported condition/class and pooled metrics. It checks validation coverage and reference supports, the worst-condition endpoint, and the recorded tuning/refit chain. It does not load model weights or source images. A successful check verifies consistency of the supplied evidence; it cannot establish that the original labels are flawless or that the models are safe to deploy. It does not convert validation results into hidden-test results.

To invoke it from another directory, supply absolute paths to the script and repository with `--root`. Only Python and NumPy are needed for the verifier itself. The command exits unsuccessfully with an error if the evidence is missing or inconsistent. `--output` optionally saves a machine-readable receipt.

## Predict one image

```powershell
python -m roadseg.predict IMAGE_PATH --model dinov3 --output working/single_demo --device cuda
```

Models are `dinov3`, `hrda` and `mic`; `--checkpoint` overrides the documented checkpoint default. Use `--device cpu` where CUDA is unavailable, allowing for much longer inference. Each successful image produces an integer train-ID PNG, an abstract color preview and a metadata JSON. The labels are 0–18, including person 11, rider 12 and bicycle 18.

## Predict an image directory

```powershell
python -m roadseg.predict INPUT_DIRECTORY --model dinov3 --output working/folder_demo --device cuda
python -m roadseg.predict INPUT_DIRECTORY --recursive --model dinov3 --output working/recursive_demo --device cuda
```

The model loads once per command. Directory mode sorts relative paths deterministically. It accepts PNG, JPEG, BMP, TIFF and WebP images; without `--recursive`, it visits only the immediate directory. Recursive mode preserves nested paths, excludes the output subtree, and avoids symbolic-link traversal. Batch output names preserve the input extension to distinguish inputs such as `scene.png` and `scene.jpg`. Single-image invocation preserves the earlier output naming scheme, including dotted stems.

Metadata records source/checkpoint/output hashes, shape, device, processing time and class pixel counts. Pixel counts are not object counts, detections, confidence, identity or fairness measurements. Timings from ordinary desktop execution are not controlled speed benchmarks.

## Output integrity and failures

Choose a fresh output directory for each demonstration. The command refuses to overwrite any planned output or an existing `batch_manifest.json` before loading the model. Per-image outputs are staged and published with exclusive hard links, so the destination filesystem must support hard links. This works on the tested NTFS workspace and common local Linux filesystems; unsupported filesystems fail rather than silently weaken overwrite protection.

The final batch manifest has `status: completed` only after all images succeed. If a later image fails, earlier completed image bundles remain valid, and the command does not create a completed batch manifest. Per-image publication rolls back only files created by the failed bundle. It does not promise a globally atomic transaction over the entire directory or automatic continuation of a partial batch. Preserve partial evidence and choose a new output directory when retrying.

## Research outputs versus software checks

The full 406-image evaluations, train-only learning-rate search and five-epoch head fit are completed experiments. Tiny mocked unit tests and the two-image feature smoke check validate software behavior, not new model accuracy. `results/reproducibility/` contains local verification evidence. The workflow defines the same CPU checks, but no successful hosted execution is claimed without an actual run.
