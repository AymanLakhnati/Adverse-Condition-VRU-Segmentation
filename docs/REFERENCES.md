# References

IEEE-style bibliography; primary sources verified 25 October 2026. The 2021 ACDC conference paper below is distinct from the later expanded arXiv version with additional authors and panoptic annotations.

[1] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “SegFormer: Simple and efficient design for semantic segmentation with transformers,” in *Advances in Neural Information Processing Systems*, vol. 34, 2021, pp. 12077–12090. [Official proceedings](https://papers.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html).

[2] C. Sakaridis, D. Dai, and L. Van Gool, “ACDC: The adverse conditions dataset with correspondences for semantic driving scene understanding,” in *Proc. IEEE/CVF International Conference on Computer Vision*, 2021, pp. 10765–10775. [Official conference paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.pdf).

[3] L. Hoyer, D. Dai, and L. Van Gool, “DAFormer: Improving network architectures and training strategies for domain-adaptive semantic segmentation,” in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022, pp. 9924–9935. [Official conference paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.pdf).

[4] L. Hoyer, D. Dai, and L. Van Gool, “HRDA: Context-aware high-resolution domain-adaptive semantic segmentation,” in *Proc. European Conference on Computer Vision*, 2022. [Official proceedings paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136900370.pdf); [authors' implementation and checkpoints](https://github.com/lhoyer/HRDA).

[5] L. Hoyer, D. Dai, H. Wang, and L. Van Gool, “MIC: Masked image consistency for context-enhanced domain adaptation,” in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023, pp. 11721–11732. [Official proceedings and BibTeX](https://openaccess.thecvf.com/content/CVPR2023/html/Hoyer_MIC_Masked_Image_Consistency_for_Context-Enhanced_Domain_Adaptation_CVPR_2023_paper.html); [authors' code](https://github.com/lhoyer/MIC).

[6] O. Siméoni *et al.*, “DINOv3,” arXiv:2508.10104, 2025. [Paper metadata](https://arxiv.org/abs/2508.10104); [version 1 full text](https://arxiv.org/html/2508.10104v1). Backbone-family reference for the separate supervised linear segmentation-head experiment; no claim that this simple head inherits published dense-task scores.

[7] ETH Zurich, “ACDC dataset: Overview,” [official dataset description](https://acdc.vision.ee.ethz.ch/overview), accessed 25 October 2026. Switzerland recordings; four adverse conditions; geographically separated splits; 19 Cityscapes-compatible semantic classes and uncertainty masks.

[8] ETH Zurich, “ACDC end user license agreement,” [official download portal](https://acdc.vision.ee.ethz.ch/download), accessed 25 October 2026. The obtained agreement is preserved as `licenses/acdc_dataset.pdf`. Sections 1.1 and 2.3 cover noncommercial use and redistribution restrictions.

[9] PyTorch Image Models maintainers, “vit_small_patch16_dinov3.lvd1689m,” [checkpoint card](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m), accessed 25 October 2026; [timm 1.0.22 implementation](https://github.com/huggingface/pytorch-image-models/blob/v1.0.22/timm/models/eva.py). Checkpoint hash and actual runtime settings are recorded with the trained head.

[10] Meta, “DINOv3 license,” updated 19 August 2025. [Official terms](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Included in `licenses/dinov3.txt`.

## Attribution boundaries

The standalone MiT backbone is adapted from NVIDIA's SegFormer/MixTransformer implementation, under its NVIDIA Source Code License. The standalone HRDA/DAFormer inference components are adaptations of the authors' Apache 2.0 source. Source headers retain attribution and describe changes; they are not wholly original architectures. Local download, memory management, evaluation, manifests, metrics, supervised-head training and analysis are the assignment implementation. Checkpoint/config/log files are upstream artifacts; their training curves are original-author training, never locally performed training. Local prediction metrics must be labeled separately from published results.
