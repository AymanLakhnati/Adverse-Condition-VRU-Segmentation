# Applicable third-party terms

- `segformer.txt`: NVIDIA Source Code License for adapted MiT/SegFormer backbone in `src/mit_backbone.py`. Original NVIDIA copyright retained in the source header; standalone PyTorch changes are identified there.
- `hrda.txt`, `daformer.txt`, `mic.txt`: original authors' Apache2.0 terms for adapted segmentation/inference components. Copyright: ETH Zurich/Lukas Hoyer and respective upstream contributors. `src/hrda_inference.py` identifies the original source and memory-management changes. These copies do not claim the authors endorse this implementation.
- `dinov3.txt`: custom DINOv3 model/materials license from the official Meta repository, fetched25 October2026. Backbone weights are downloaded separately and are not included in submission source.
- `acdc_dataset.pdf`: obtained official ACDC agreement; data are not included. Its noncommercial use, attribution and redistribution restrictions also matter for derived presentation imagery.

The project does not redistribute an entire upstream training repository. Licenses for unused detection/classification/adversarial features are not copied here. Additional third-party components, if introduced later, require their own applicable notices. Runtime Python dependencies retain their respective package licenses.
