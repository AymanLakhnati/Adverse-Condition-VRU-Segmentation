# Prespecified supervised classifier adaptation extension

Protocol fixed on 25 October 2026 before reading any local ACDC validation scores.

The primary hypothesis remains the comparison between the released HRDA and MIC(HRDA) Cityscapes-to-ACDC checkpoints. This separate extension uses **ACDC training labels**, so it is supervised target-domain adaptation and must never be described as unsupervised domain adaptation or evidence that the MIC training loss alone caused a change.

Initialize from the released MIC student checkpoint. Freeze the MiT-B5 backbone, DAFormer feature-fusion layers and scale-attention decoder. Update only `decode_head.head.conv_seg.weight` and `.bias`, totaling 4,883 parameters. Keep the whole model in evaluation mode throughout optimization to freeze batch-normalization buffers and dropout behavior.

Use exactly one epoch over all 1,600 adverse-weather ACDC training images, batch size one and seed 42. For each shuffled training image, choose one random 512-by-512 crop and flip it horizontally with probability 0.5, applying identical geometry to its training mask. Use the original RGB normalization. Optimize cross entropy with ignore ID 255, no class weights, and AdamW with learning rate 0.0001 and weight decay 0.01. No validation-based checkpoint selection, early stopping, learning-rate selection or hyperparameter search is permitted. A limited smoke run must carry an explicit partial-run flag and cannot support a completed-training claim.

After the single fixed epoch, evaluate the resulting head on the same 406-image adverse validation split, using the primary experiment's unchanged inference and metrics. Report it in a separate supervised-extension table. Do not replace the primary HRDA/MIC comparison or reinterpret supervised gains as an isolated MIC effect. The released initialization was selected by its original authors using validation performance, which limits the independence of this validation evaluation.

Save the training IDs, image and label hashes, crop coordinates, flips, per-step losses, trainable gradient norms, source checkpoint hash, wall-clock time and package/device information. Compare all frozen parameters and buffers before and after training by SHA256, and require that they remain unchanged. Save both the complete adapted student checkpoint for local evaluation and a small classifier adapter for reproduction from the original checkpoint. Exclude large checkpoints, dataset images and labels from GitHub.

Synthetic gradient checks validate the implementation only. Their artificial labels and loss values are never experimental results.
