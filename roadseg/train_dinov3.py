"""Train a new dense linear segmentation head on frozen DINOv3 patch features.

Target-supervised extension on ACDC train labels. This model does not isolate
architecture from pretraining or supervision and does not replace the primary
released HRDA/MIC unsupervised-adaptation checkpoint comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from roadseg.train_head import collect_training_rows, file_hash, json_write, tensor_hash
from roadseg.train_head import TrainingCrops
from src.hrda_inference import crop_boxes

MODEL = "vit_small_patch16_dinov3.lvd1689m"
LOG = logging.getLogger("dinov3-dense-training")


def training_content_hash(rows):
    """Bind refitting to the same image AND label bytes used during tuning."""
    values=sorted((row["image_id"],row["image_sha256"],row["label_sha256"]) for row in rows)
    return hashlib.sha256(json.dumps(values,separators=(",",":")).encode()).hexdigest()


class DenseDINOv3(nn.Module):
    """Frozen normalized patch tokens plus a newly initialized 19-class head."""
    def __init__(self, pretrained=True):
        super().__init__()
        # Honor the Windows OS trust store without disabling HTTPS validation.
        try:
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        import timm
        self.encoder=timm.create_model(MODEL,pretrained=pretrained,num_classes=0,dynamic_img_size=True)
        self.encoder.eval().requires_grad_(False)
        if self.encoder.num_features!=384 or self.encoder.num_prefix_tokens!=5:
            raise ValueError("Unexpected DINOv3 feature configuration")
        if tuple(self.encoder.patch_embed.patch_size)!=(16,16):
            raise ValueError("Expected16pixel patches")
        self.head=nn.Conv2d(384,19,kernel_size=1)
        if sum(p.numel() for p in self.head.parameters())!=7315:
            raise ValueError("Dense classifier must contain7315parameters")

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def patch_features(self, image):
        n,_,h,w=image.shape
        if h%16 or w%16:
            raise ValueError("DINOv3 input dimensions must be divisible by16")
        # no_grad yields ordinary detached tensors that Conv2d can save for its
        # weight gradient; inference_mode here would forbid those saved tensors.
        with torch.no_grad():
            tokens=self.encoder.forward_features(image)
        if not isinstance(tokens,torch.Tensor) or tokens.ndim!=3:
            raise ValueError("Expected normalized NLC feature tokens")
        patches=tokens[:,self.encoder.num_prefix_tokens:,:]
        if patches.shape!=(n,(h//16)*(w//16),384):
            raise ValueError(f"Patch token shape mismatch: {patches.shape}")
        return patches.transpose(1,2).reshape(n,384,h//16,w//16).contiguous()

    def forward(self,image):
        logits=self.head(self.patch_features(image))
        return F.interpolate(logits,size=image.shape[-2:],mode="bilinear",align_corners=False)


def load_trained_model(head_path, device="cuda"):
    """Fetch the public backbone, verify its hash and apply the small trained head."""
    package=torch.load(head_path,map_location="cpu",weights_only=True)
    if package.get("model_name")!=MODEL:
        raise ValueError("Unexpected foundation model in dense-head checkpoint")
    model=DenseDINOv3(pretrained=True)
    if tensor_hash(model.encoder.state_dict())!=package["encoder_state_sha256"]:
        raise ValueError("Downloaded DINOv3 encoder SHA256 differs from trained-head provenance")
    model.head.load_state_dict(package["head_state_dict"],strict=True)
    return model.eval().requires_grad_(False).to(device)


def image_tensor(path,device="cuda"):
    """Full original RGB image with ImageNet normalization, no resizing/crop."""
    with Image.open(path) as source:
        array=np.asarray(source.convert("RGB"),dtype=np.float32)
    array=(array-np.asarray([123.675,116.28,103.53],dtype=np.float32))/np.asarray([58.395,57.12,57.375],dtype=np.float32)
    return torch.from_numpy(array.transpose(2,0,1).copy()).unsqueeze(0).to(device)


@torch.inference_mode()
def predict_logits(model,image,tile_size=512,stride=384,batch_size=4):
    """Average overlapping full-resolution tile logits, keeping original size."""
    if image.ndim!=4 or image.shape[:2]!=(1,3):
        raise ValueError("Expected one normalized RGB image")
    if tile_size%16 or stride<1 or stride>tile_size or batch_size<1:
        raise ValueError("Invalid tiling configuration")
    h,w=image.shape[-2:]
    padded=F.pad(image,(0,max(0,tile_size-w),0,max(0,tile_size-h)))
    height,width=padded.shape[-2:]
    boxes=list(crop_boxes(height,width,(tile_size,tile_size),(stride,stride)))
    output=image.new_zeros((1,19,height,width))
    counts=image.new_zeros((1,1,height,width))
    for start in range(0,len(boxes),batch_size):
        group=boxes[start:start+batch_size]
        tiles=torch.cat([padded[:,:,y1:y2,x1:x2] for y1,y2,x1,x2 in group],dim=0)
        predictions=model(tiles)
        for j,(y1,y2,x1,x2) in enumerate(group):
            output[:,:,y1:y2,x1:x2]+=predictions[j:j+1]
            counts[:,:,y1:y2,x1:x2]+=1
    if not bool(torch.all(counts>0)):
        raise RuntimeError("Uncovered DINOv3 inference pixels")
    return (output/counts)[:,:,:h,:w]


def _save_head(model,path,encoder_hash,config):
    torch.save({"model_name":MODEL,"encoder_state_sha256":encoder_hash,
                "head_state_dict":{name:value.detach().cpu().clone() for name,value in model.head.state_dict().items()},
                "head_parameters":7315,"training_kind":"supervised ACDC train-only dense linear head",
                "crop_size":512,"inference_tile_size":512,"inference_stride":384,
                "seed":config["seed"],"completed_epochs":config.get("completed_epochs",0)},path)


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output directory must be empty. Select a new --output to preserve prior runs.")
    args.output.mkdir(parents=True,exist_ok=True)
    if args.epochs<1 or args.batch_size<1:
        raise ValueError("Epochs and batch size must be positive")
    learning_rate=float(getattr(args,"learning_rate",0.001))
    if not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError("Learning rate must be finite and positive")
    tuning_path=getattr(args,"tuning_results",None)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    rows=collect_training_rows(args.data_root)
    available=len(rows)
    if args.limit is None and available!=1600:
        raise ValueError(f"Full protocol requires1600ACDC training images, found{available}")
    if args.limit is not None:
        if args.limit<1:
            raise ValueError("Positive --limit required")
        choice=np.random.default_rng(args.seed).permutation(len(rows))[:args.limit]
        rows=[rows[int(i)] for i in choice]
    if not rows:
        raise ValueError("No training images found")
    tuning=None
    if tuning_path is not None:
        tuning=json.loads(Path(tuning_path).read_text(encoding="utf-8"))
        ids_hash=hashlib.sha256("\n".join(sorted(row["image_id"] for row in rows)).encode()).hexdigest()
        if (tuning.get("status")!="completed" or
            not np.isclose(tuning.get("selected_learning_rate",-1),learning_rate,rtol=0,atol=1e-12) or
            tuning.get("all_training_ids_sha256")!=ids_hash or
            tuning.get("official_validation_or_test_used") is not False):
            raise ValueError("Tuning provenance does not match this completed train-only refit")
    for row in rows:
        row["image_sha256"]=file_hash(row["image_path"])
        row["label_sha256"]=file_hash(row["label_path"])
    content_hash=training_content_hash(rows)
    if tuning is not None and tuning.get("all_training_content_sha256")!=content_hash:
        raise ValueError("Training image or label bytes differ from the tuning dataset")
    with open(args.output/"training_manifest.csv","w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    model=DenseDINOv3(pretrained=True).to(args.device)
    encoder_hash=tensor_hash(model.encoder.state_dict())
    initial_head_hash=tensor_hash(model.head.state_dict())
    if tuning is not None and (tuning["encoder_state_sha256"]!=encoder_hash or
                              tuning["initial_head_sha256"]!=initial_head_hash):
        raise ValueError("Refit must use the identical frozen encoder and fresh seeded head initialization")
    trainable_names=[name for name,param in model.named_parameters() if param.requires_grad]
    if set(trainable_names)!={"head.weight","head.bias"}:
        raise ValueError("Unexpected trainable encoder parameters")
    config={"status":"running","kind":"supervised target-domain segmentation head training, NOT UDA",
            "primary_hrda_mic_hypothesis_unchanged":True,"model_name":MODEL,
            "encoder_state_sha256":encoder_hash,"encoder_frozen":True,
            "encoder_provenance":model.encoder.pretrained_cfg,
            "encoder_parameters":sum(p.numel() for p in model.encoder.parameters()),
            "head_parameters":7315,"trainable_parameter_names":trainable_names,
            "feature_source":"Final normalized patch tokens from forward_features, excluding1CLS+4register tokens",
            "feature_dimension":384,"patch_stride":16,"patch_feature_shape_at512":[384,32,32],
            "pretrained_default_resolution":256,"experiment_crop_resolution":512,"dynamic_image_size":True,
            "normalization_mean_rgb":[123.675,116.28,103.53],"normalization_std_rgb":[58.395,57.12,57.375],
            "seed":args.seed,"epochs":args.epochs,"batch_size":args.batch_size,
            "learning_rate":learning_rate,"optimizer":"AdamW","weight_decay":0.01,
            "learning_rate_selection":"train-only sequence-held-out search" if tuning else "fixed default, no search",
            "tuning_results_path":str(tuning_path) if tuning_path is not None else None,
            "tuning_results_sha256":file_hash(tuning_path) if tuning_path is not None else None,
            "refit_initialization":f"Fresh seed{args.seed} head, not selected trial weights" if tuning else "Fresh seeded head",
            "learning_rate_schedule":"fixed","loss":"CrossEntropyLoss(ignore_index=255), no class weights",
            "train_split":"Only ACDC adverse train labels","training_images":len(rows),"available_train_images":available,
            "all_training_content_sha256":content_hash,
            "limited_smoke_run":args.limit is not None,"crop_size":512,"flip_probability":0.5,
            "official_validation_used_for_selection":False,
            "internal_training_development_used_for_learning_rate_selection":bool(tuning),
            "inference_tile_size":512,"inference_stride":384,
            "inference_aggregation":"Mean overlapping bilinearly-upsampled logits",
            "encoder_mode":"eval throughout, detached features; classifier in train mode during optimization",
            "precision":"float32","head_before_sha256":initial_head_hash,
            "torch_version":str(torch.__version__),"timm_version":importlib.metadata.version("timm"),
            "device":args.device,"gpu":torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"completed_epochs":0}
    json_write(args.output/"training_config.json",config)
    loader=DataLoader(TrainingCrops(rows),batch_size=args.batch_size,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed),drop_last=False)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=learning_rate,weight_decay=0.01)
    fields=["epoch","global_step","images","loss","valid_pixels","gradient_norm","seconds","image_ids","conditions","crop_coordinates","flips","learning_rate"]
    started=time.perf_counter();step=0;skipped=0
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with open(args.output/"training_steps.csv","w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for epoch in range(1,args.epochs+1):
            model.train()
            for batch in loader:
                batch_started=time.perf_counter()
                images=batch["image"].to(args.device)
                labels=batch["label"].to(args.device)
                valid=int((labels!=255).sum())
                if not valid:
                    skipped+=1;continue
                optimizer.zero_grad(set_to_none=True)
                logits=model(images)
                loss=F.cross_entropy(logits,labels,ignore_index=255)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite dense-head loss")
                loss.backward()
                grad=float(torch.sqrt(sum((param.grad.detach()**2).sum() for param in model.head.parameters())))
                if not np.isfinite(grad) or grad<=0:
                    raise FloatingPointError("Expected finite nonzero head gradients")
                if any(p.grad is not None for p in model.encoder.parameters()):
                    raise RuntimeError("Frozen encoder received gradient")
                optimizer.step();step+=1
                record={"epoch":epoch,"global_step":step,"images":len(batch["image_id"]),
                        "loss":float(loss.detach()),"valid_pixels":valid,"gradient_norm":grad,
                        "seconds":time.perf_counter()-batch_started,"image_ids":" ".join(batch["image_id"]),
                        "conditions":" ".join(batch["condition"]),"crop_coordinates":json.dumps(list(zip(batch["crop_y"].tolist(),batch["crop_x"].tolist()))),
                        "flips":json.dumps(batch["flipped"].tolist()),"learning_rate":learning_rate}
                writer.writerow(record);handle.flush()
                LOG.info("DINOv3 epoch%d/%d step%d loss%.4f grad%.4f",epoch,args.epochs,step,record["loss"],grad)
            config["completed_epochs"]=epoch
            # Persist progress for interruption recovery without treating it as
            # validation-selected or fully completed training.
            _save_head(model,args.output/"head_latest.pt",encoder_hash,config)
            json_write(args.output/"training_config.json",config)
    after_hash=tensor_hash(model.encoder.state_dict())
    if after_hash!=encoder_hash:
        raise RuntimeError("Frozen DINOv3 encoder state changed")
    head_hash=tensor_hash(model.head.state_dict())
    if head_hash==initial_head_hash:
        raise RuntimeError("Dense classifier did not change")
    config.update(status="completed",encoder_after_sha256=after_hash,encoder_state_unchanged=True,
                  head_after_sha256=head_hash,head_state_changed=True,optimizer_steps=step,
                  skipped_all_ignore_batches=skipped,elapsed_seconds=time.perf_counter()-started,
                  completed_fixed_full_protocol=(args.limit is None and available==1600 and args.epochs==5 and args.batch_size==4 and learning_rate in (0.0003,0.001,0.003)),
                  peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if args.device.startswith("cuda") else None)
    _save_head(model,args.output/"head.pt",encoder_hash,config)
    json_write(args.output/"training_config.json",config)
    LOG.info("Completed dense-head training: %s",args.output)
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,default=Path("data/acdc"))
    p.add_argument("--output",type=Path,default=Path("outputs/dinov3_dense"))
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--learning-rate",type=float,default=0.001)
    p.add_argument("--tuning-results",type=Path,help="Completed train-only search JSON, verified before fresh full refit")
    p.add_argument("--limit",type=int,help="Partial training smoke run only")
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    train(args)


if __name__=="__main__":
    main()
