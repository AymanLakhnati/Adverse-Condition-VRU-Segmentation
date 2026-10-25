"""Memory-bounded PyTorch inference for the released MiT-B5 HRDA/MIC models.

Adapted from the official HRDA/MIC DAFormer and HRDA decode heads and segmentors:
https://github.com/lhoyer/MIC/tree/master/seg/mmseg/models
Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. Apache License, Version 2.0.
The MiT backbone has its separate upstream NVIDIA license (see mit_backbone.py).

This port evaluates published student weights. It performs no MIC training.
It keeps the official two-scale attention fusion and overlapping crop grids,
but forwards one detail crop at a time instead of materializing all crops.
Eval-mode batch-normalization and disabled dropout make the batching change
mathematically equivalent, subject to floating-point implementation differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from src.mit_backbone import mit_b5

LOG = logging.getLogger(__name__)


def resize(x, size=None, scale_factor=None):
    return F.interpolate(x, size=size, scale_factor=scale_factor,
                         mode="bilinear", align_corners=False)


class ConvModule(nn.Module):
    """The exact conv -> BN -> ReLU subset used by the released decoder."""
    def __init__(self, inp, out, kernel_size, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, out, kernel_size, padding=padding,
                              dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out, eps=1e-5, momentum=0.1)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, inp, out, dilation):
        super().__init__()
        self.depthwise_conv = ConvModule(inp, inp, 3, padding=dilation,
                                          dilation=dilation, groups=inp)
        self.pointwise_conv = ConvModule(inp, out, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class ASPPWrapper(nn.Module):
    def __init__(self, inp=1024, channels=256):
        super().__init__()
        self.aspp_modules = nn.ModuleList([
            ConvModule(inp, channels, 1),
            *[DepthwiseSeparableConvModule(inp, channels, d) for d in (6, 12, 18)],
        ])
        self.bottleneck = ConvModule(4 * channels, channels, 3, padding=1)

    def forward(self, x):
        return self.bottleneck(torch.cat([layer(x) for layer in self.aspp_modules], dim=1))


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


class DAFormerHead(nn.Module):
    def __init__(self, attention=False):
        super().__init__()
        self.embed_layers = nn.ModuleDict({str(i): MLP(c) for i, c in enumerate((64,128,320,512))})
        self.fuse_layer = ConvModule(1024, 256, 1) if attention else ASPPWrapper()
        self.dropout = nn.Dropout2d(0.1)
        self.conv_seg = nn.Conv2d(256, 19, 1)

    def forward(self, features):
        size = features[0].shape[-2:]
        embeddings = []
        for i, feature in enumerate(features):
            n, _, h, w = feature.shape
            x = self.embed_layers[str(i)](feature).permute(0,2,1).contiguous().reshape(n,256,h,w)
            if x.shape[-2:] != size:
                x = resize(x, size=size)
            embeddings.append(x)
        return self.conv_seg(self.dropout(self.fuse_layer(torch.cat(embeddings, dim=1))))


class HRDAHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = DAFormerHead()
        self.scale_attention = DAFormerHead(attention=True)


def crop_boxes(height, width, crop_size, stride):
    """Official HRDA edge-aligned overlap grid, including final border crops."""
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride
    h_grids = max(height - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(width - w_crop + w_stride - 1, 0) // w_stride + 1
    for hi in range(h_grids):
        for wi in range(w_grids):
            y2, x2 = min(hi*h_stride+h_crop, height), min(wi*w_stride+w_crop, width)
            yield max(y2-h_crop,0), y2, max(x2-w_crop,0), x2


class HRDA(nn.Module):
    def __init__(self, outer_crop=1024, detail_crop=512):
        super().__init__()
        self.backbone = mit_b5()
        self.decode_head = HRDAHead()
        self.outer_crop, self.detail_crop = outer_crop, detail_crop

    def encode_decode(self, image):
        """Two-scale prediction for a single official outer crop."""
        if self.training:
            raise RuntimeError("Inference port requires model.eval(); no training implementation")
        if image.shape[0] != 1:
            raise ValueError("Sequential inference requires exactly one image")
        low_features = self.backbone(resize(image, scale_factor=0.5))
        low = self.decode_head.head(low_features)
        attention = torch.sigmoid(self.decode_head.scale_attention(low_features))
        del low_features
        h, w = image.shape[-2:]
        # Source model assumes crop boxes align to the decoder's stride four.
        if h % 4 or w % 4:
            raise ValueError("Outer crop dimensions must be divisible by four")
        high = image.new_zeros((1,19,h//4,w//4))
        counts = image.new_zeros((1,1,h//4,w//4))
        for y1,y2,x1,x2 in crop_boxes(h,w,(self.detail_crop,)*2,(self.detail_crop//2,)*2):
            if any(v % 4 for v in (y1,y2,x1,x2)):
                raise ValueError("Detail grid must align to the decoder's stride four")
            detail = self.decode_head.head(self.backbone(image[:,:,y1:y2,x1:x2]))
            high[:,:,y1//4:y2//4,x1//4:x2//4] += detail
            counts[:,:,y1//4:y2//4,x1//4:x2//4] += 1
            del detail
        if not torch.all(counts > 0):
            raise RuntimeError("Uncovered detail pixels")
        high /= counts
        # Preserve original fusion order: weight LR before interpolation.
        low = resize((1 - attention)*low, scale_factor=2)
        attention = resize(attention, scale_factor=2)
        if high.shape[-2:] != low.shape[-2:]:
            raise ValueError("Scale grids do not align")
        fused = attention * high + low
        return resize(fused, size=image.shape[-2:])

    def forward(self, image):
        """Sequential official outer sliding-window inference, logits output."""
        h,w = image.shape[-2:]
        if image.shape[0] != 1:
            raise ValueError("Exactly one full image per inference call")
        prediction = image.new_zeros((1,19,h,w))
        counts = image.new_zeros((1,1,h,w))
        for y1,y2,x1,x2 in crop_boxes(h,w,(self.outer_crop,)*2,(self.outer_crop//2,)*2):
            logits = self.encode_decode(image[:,:,y1:y2,x1:x2])
            prediction[:,:,y1:y2,x1:x2] += logits
            counts[:,:,y1:y2,x1:x2] += 1
            del logits
        if not torch.all(counts > 0):
            raise RuntimeError("Uncovered outer pixels")
        return prediction / counts


def load_model(checkpoint_path, device="cuda"):
    """Load ONLY student weights and reject missing/unexpected model tensors."""
    # Official archives are trusted project checkpoints, not arbitrary user pickles.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {k.removeprefix("module."):v for k,v in state.items()}
    student = {k.removeprefix("model."):v for k,v in state.items() if k.startswith("model.")}
    if not student:
        student = {k:v for k,v in state.items() if k.startswith(("backbone.","decode_head."))}
    if not student:
        raise ValueError("No student model weights in checkpoint")
    model = HRDA()
    # Older MiT classification head is irrelevant and sometimes retained.
    ignored = [key for key in student if key.startswith("backbone.head.")]
    for key in ignored:
        del student[key]
    model.load_state_dict(student, strict=True)
    model.eval().requires_grad_(False).to(device)
    LOG.info("Loaded %d student tensors, %d parameters",len(student),sum(p.numel() for p in model.parameters()))
    return model


def preprocess(image_path, device="cuda"):
    """Official ACDC test input uses unchanged 1920×1080 resolution and RGB norm."""
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        # Official Resize keep_ratio with img_scale=(1920,1080).
        ratio = min(1920/rgb.width,1080/rgb.height)
        target = (int(rgb.width*ratio+0.5),int(rgb.height*ratio+0.5))
        if target != rgb.size:
            # cv2's bilinear implementation is used by the evaluator when available.
            import cv2
            pixels = cv2.resize(np.asarray(rgb),target,interpolation=cv2.INTER_LINEAR)
        else:
            pixels = np.asarray(rgb)
        array = pixels.astype(np.float32)
        array = (array - np.asarray([123.675,116.28,103.53],dtype=np.float32))/np.asarray([58.395,57.12,57.375],dtype=np.float32)
    return torch.from_numpy(array.transpose(2,0,1).copy()).unsqueeze(0).to(device)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("image",type=Path)
    parser.add_argument("--output",type=Path,default=Path("outputs/segmentation/prediction.png"))
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    model=load_model(args.checkpoint,args.device)
    with torch.inference_mode():
        logits=model(preprocess(args.image,args.device))
        labels=logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(labels).save(args.output)
    print(args.output)


if __name__=="__main__":
    main()
