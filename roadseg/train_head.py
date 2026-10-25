"""Optional supervised ACDC-training-label adaptation of the released MIC head.

This extension uses labels and is NOT unsupervised domain adaptation. It does
not change the primary published-checkpoint HRDA/MIC comparison. The protocol
fixes one epoch and all hyperparameters before validation outcomes are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.hrda_inference import load_model

LOG = logging.getLogger("supervised-head-adaptation")
TRAINABLE = ("decode_head.head.conv_seg.weight", "decode_head.head.conv_seg.bias")
CONDITIONS = ("fog", "night", "rain", "snow")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(state, exclude=()):
    """Include frozen buffers as well as parameters, so BN changes are detected."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name not in exclude:
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=str),encoding="utf-8")


def collect_training_rows(root):
    """Only the explicit adverse-weather TRAIN paths are eligible."""
    root = Path(root)
    rows=[]
    for condition in CONDITIONS:
        for image in sorted((root/"rgb_anon"/condition/"train").rglob("*_rgb_anon.png")):
            relative=image.relative_to(root/"rgb_anon")
            label=root/"gt"/relative.parent/relative.name.replace("_rgb_anon.png","_gt_labelTrainIds.png")
            if not label.exists():
                raise FileNotFoundError(f"Missing training label: {label}")
            if relative.parts[1] != "train":
                raise ValueError("Non-training split entered adaptation")
            rows.append({"image_id":relative.name.removesuffix("_rgb_anon.png"),
                         "condition":condition,"image_path":str(image),"label_path":str(label),
                         "split":"train"})
    if len({r["image_id"] for r in rows})!=len(rows):
        raise ValueError("Repeated training image IDs")
    return rows


class TrainingCrops(Dataset):
    def __init__(self, rows, crop_size=512):
        self.rows,self.crop_size=rows,crop_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row["image_path"]) as source:
            image=np.array(source.convert("RGB"),copy=True)
        with Image.open(row["label_path"]) as source:
            label=np.array(source,copy=True)
        if image.shape[:2]!=label.shape or label.ndim!=2:
            raise ValueError(f"Mismatched training image/label shapes: {row['image_id']}")
        if not np.all((label<19)|(label==255)):
            raise ValueError("Training labels must use Cityscapes train IDs0..18 or255")
        height,width=label.shape
        size=self.crop_size
        if min(height,width)<size:
            raise ValueError("Training image smaller than fixed512crop")
        y=random.randrange(height-size+1)
        x=random.randrange(width-size+1)
        image=image[y:y+size,x:x+size]
        label=label[y:y+size,x:x+size]
        flipped=random.random()<0.5
        if flipped:
            image,label=image[:,::-1],label[:,::-1]
        image=image.astype(np.float32)
        image=(image-np.array([123.675,116.28,103.53],dtype=np.float32))/np.array([58.395,57.12,57.375],dtype=np.float32)
        return {"image":torch.from_numpy(image.transpose(2,0,1).copy()),
                "label":torch.from_numpy(label.astype(np.int64).copy()),
                "image_id":row["image_id"],"condition":row["condition"],
                "crop_y":y,"crop_x":x,"flipped":flipped}


def configure_trainable_head(model):
    model.eval().requires_grad_(False)
    for name,param in model.named_parameters():
        if name in TRAINABLE:
            param.requires_grad_(True)
    names=[name for name,param in model.named_parameters() if param.requires_grad]
    count=sum(param.numel() for param in model.parameters() if param.requires_grad)
    if tuple(names)!=TRAINABLE or count!=4883:
        raise ValueError(f"Unexpected trainable head: {names}, {count} parameters")
    return count


def load_adapted_model(source_checkpoint, adapter_path, device="cuda"):
    """Reproduce the adapted model from released weights plus the small adapter."""
    adapter=torch.load(adapter_path,map_location="cpu",weights_only=True)
    if adapter.get("source_checkpoint_sha256")!=file_hash(source_checkpoint):
        raise ValueError("Adapter source checkpoint SHA256 does not match")
    state=adapter.get("state_dict",{})
    if set(state)!=set(TRAINABLE):
        raise ValueError("Adapter must contain exactly the two selected classifier tensors")
    model=load_model(source_checkpoint,device)
    result=model.load_state_dict(state,strict=False)
    expected=set(model.state_dict())-set(TRAINABLE)
    if set(result.missing_keys)!=expected or result.unexpected_keys:
        raise ValueError("Unexpected adapter state mismatch")
    return model.eval().requires_grad_(False)


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Adaptation output directory is nonempty. Choose a new --output for a fresh experiment.")
    args.output.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    rows=collect_training_rows(args.data_root)
    available=len(rows)
    if args.limit is None and available!=1600:
        raise ValueError(f"Full fixed-epoch protocol requires1600 training images, found{available}")
    if args.limit is not None:
        if args.limit<1:
            raise ValueError("--limit must be positive")
        selection=np.random.default_rng(args.seed).permutation(len(rows))[:args.limit]
        rows=[rows[int(i)] for i in selection]
    if not rows:
        raise ValueError("No labeled ACDC training images available")
    for row in rows:
        row["image_sha256"]=file_hash(row["image_path"])
        row["label_sha256"]=file_hash(row["label_path"])
    with open(args.output/"training_manifest.csv","w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    model=load_model(args.checkpoint,args.device)
    count=configure_trainable_head(model)
    frozen_before=tensor_hash(model.state_dict(),exclude=TRAINABLE)
    trainable_before=tensor_hash({k:v for k,v in model.state_dict().items() if k in TRAINABLE})
    config={"status":"running","kind":"supervised target-domain head adaptation, NOT UDA",
            "primary_comparison_unchanged":"Released HRDA vs released MIC checkpoints",
            "source_checkpoint":str(args.checkpoint),"source_checkpoint_sha256":file_hash(args.checkpoint),
            "seed":args.seed,"epochs":1,"selected_images":len(rows),"available_train_images":available,
            "limited_smoke_run":args.limit is not None,"train_split":"ACDC adverse train only",
            "validation_used_for_training_or_selection":False,"crop_size":512,"horizontal_flip_probability":0.5,
            "optimizer":"AdamW","learning_rate":1e-4,"weight_decay":0.01,"batch_size":1,
            "loss":"CrossEntropyLoss(ignore_index=255), no class weights",
            "trainable_parameters":count,"trainable_parameter_names":list(TRAINABLE),
            "frozen_state_before_sha256":frozen_before,"trainable_before_sha256":trainable_before,
            "module_mode":"eval throughout, including frozen BN and dropout",
            "precision":"float32","device":args.device,
            "gpu":torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "torch_version":torch.__version__,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    json_write(args.output/"training_config.json",config)
    loader=DataLoader(TrainingCrops(rows),batch_size=1,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(args.seed))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.01)
    fields=["epoch","step","image_id","condition","loss","valid_pixels","gradient_norm","seconds","crop_y","crop_x","flipped","learning_rate"]
    total_start=time.perf_counter()
    processed,skipped=0,0
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with open(args.output/"training_steps.csv","w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for step,batch in enumerate(loader,1):
            started=time.perf_counter()
            image=batch["image"].to(args.device)
            label=batch["label"].to(args.device)
            valid=int((label!=255).sum())
            if not valid:
                skipped+=1
                LOG.warning("All-ignore crop skipped: %s",batch["image_id"][0])
                continue
            optimizer.zero_grad(set_to_none=True)
            logits=model.encode_decode(image)
            loss=F.cross_entropy(logits,label,ignore_index=255)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite adaptation loss")
            loss.backward()
            gradient_norm=float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None)))
            if not np.isfinite(gradient_norm) or gradient_norm<=0:
                raise FloatingPointError("Expected finite nonzero classifier gradient")
            if any(p.grad is not None for name,p in model.named_parameters() if name not in TRAINABLE):
                raise RuntimeError("A frozen parameter received a gradient")
            optimizer.step()
            record={"epoch":1,"step":step,"image_id":batch["image_id"][0],"condition":batch["condition"][0],
                    "loss":float(loss.detach()),"valid_pixels":valid,"gradient_norm":gradient_norm,
                    "seconds":time.perf_counter()-started,"crop_y":int(batch["crop_y"][0]),
                    "crop_x":int(batch["crop_x"][0]),"flipped":bool(batch["flipped"][0]),"learning_rate":1e-4}
            writer.writerow(record);handle.flush();processed+=1
            LOG.info("Supervised head %d/%d loss%.4f grad%.4f",step,len(rows),record["loss"],gradient_norm)
    frozen_after=tensor_hash(model.state_dict(),exclude=TRAINABLE)
    if frozen_after!=frozen_before:
        raise RuntimeError("Frozen model state changed during head adaptation")
    adapter={name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name in TRAINABLE}
    trainable_after=tensor_hash(adapter)
    if trainable_after==trainable_before:
        raise RuntimeError("Classifier weights did not change")
    torch.save({"state_dict":adapter,"source_checkpoint_sha256":config["source_checkpoint_sha256"],
                "kind":"Supervised ACDC train-only classifier adapter","trainable_parameters":4883},args.output/"head_adapter.pt")
    torch.save({"state_dict":{"model."+name:value.detach().cpu() for name,value in model.state_dict().items()},
                "meta":{"adaptation_config":config,"kind":"supervised extension, NOT UDA"}},args.output/"student_head.pth")
    config.update(status="completed",processed_steps=processed,skipped_all_ignore_crops=skipped,
                  completed_fixed_full_epoch=(args.limit is None and available==1600 and processed+skipped==1600),
                  frozen_state_after_sha256=frozen_after,frozen_state_unchanged=True,
                  trainable_after_sha256=trainable_after,trainable_state_changed=True,
                  elapsed_seconds=time.perf_counter()-total_start,
                  peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if args.device.startswith("cuda") else None)
    json_write(args.output/"training_config.json",config)
    LOG.info("Completed supervised extension: %s",args.output)
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,default=Path("checkpoints/mic/csHR2acdcHR_mic_hrda_b52f1/iter_40000_relevant.pth"))
    p.add_argument("--data-root",type=Path,default=Path("data/acdc"))
    p.add_argument("--output",type=Path,default=Path("outputs/adaptation"))
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--limit",type=int,help="Explicit partial smoke run, never reported as full training")
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    train(args)


if __name__=="__main__":
    main()
