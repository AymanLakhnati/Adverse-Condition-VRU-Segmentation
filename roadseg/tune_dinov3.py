"""Train-only, sequence-held-out learning-rate tuning for the dense DINOv3 head.

The official ACDC validation and test partitions are never opened here.
Development scoring uses a declared center-crop proxy rather than full images.
The selected rate can then initialize a fresh five-epoch fit on all1600 train
images with ``python -m roadseg.tune_dinov3 --refit``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

from roadseg.metrics import CONDITIONS,confusion_matrix,metrics_from_confusion
from roadseg.train_dinov3 import DenseDINOv3,MODEL
from roadseg.train_head import TrainingCrops,collect_training_rows,file_hash,json_write,tensor_hash

RATES=(0.0003,0.001,0.003)
LOG=logging.getLogger("dinov3-train-only-tuning")


def recording_family(sequence):
    """Conservatively keep GoPro chapter aliases in the same recording group.

    GOPR0376, GP010376 and GP020376 share the trailing recording number.
    Unknown naming conventions retain the complete original directory name.
    This prevents an avoidable chapter split, but cannot prove scene independence.
    """
    match=re.fullmatch(r"(?:GOPR|GP\d{2})(\d{4})",sequence)
    return f"recording_{match.group(1)}" if match else sequence


def sequence_holdout(rows,seed=42,fraction=0.2):
    """Choose a shuffled group prefix nearest20% images within each condition.

    A group is condition/recording-family, conservatively merging GoPro chapters.
    At least one development and one fitting group are required per condition.
    No label content participates in the split.
    """
    if not 0<fraction<1:
        raise ValueError("Development fraction must lie strictly between0and1")
    groups=defaultdict(list)
    for row in rows:
        if row.get("split")!="train":
            raise ValueError("Only declared training rows may enter tuning")
        if row["condition"] not in CONDITIONS:
            raise ValueError("Unknown weather condition")
        parts=Path(row["image_path"]).parts
        if "train" not in parts or "val" in parts or "test" in parts:
            raise ValueError("Tuning path must identify train, never val/test")
        sequence=Path(row["image_path"]).parent.name
        family=recording_family(sequence)
        groups[(row["condition"],family)].append({**row,"sequence_directory":sequence,
                                                 "recording_family":family})
    rng=np.random.default_rng(seed)
    development_groups=set()
    for condition in CONDITIONS:
        keys=sorted(key for key in groups if key[0]==condition)
        if len(keys)<2:
            raise ValueError(f"At least two training recording families needed for {condition}")
        keys=[keys[int(i)] for i in rng.permutation(len(keys))]
        cumulative=np.cumsum([len(groups[key]) for key in keys])
        target=sum(len(groups[key]) for key in keys)*fraction
        count=min(range(1,len(keys)),key=lambda n:abs(float(cumulative[n-1])-target))
        development_groups.update(keys[:count])
    fit,development=[],[]
    for key in sorted(groups):
        destination=development if key in development_groups else fit
        for original in groups[key]:
            destination.append({**original,"sequence_group":"/".join(key)})
    fit_groups={row["sequence_group"] for row in fit}
    dev_groups={row["sequence_group"] for row in development}
    if fit_groups & dev_groups:
        raise RuntimeError("Recording-family group leakage")
    if {row["image_id"] for row in fit}&{row["image_id"] for row in development}:
        raise ValueError("Repeated image IDs across tuning partitions")
    return fit,development


class CenterCrops(Dataset):
    """Deterministic512center crops, selected without ground-truth inspection."""
    def __init__(self,rows):
        self.rows=rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(row["image_path"]) as source:
            image=np.asarray(source.convert("RGB"))
        with Image.open(row["label_path"]) as source:
            label=np.asarray(source)
        if image.shape[:2]!=label.shape or label.ndim!=2 or min(label.shape)<512:
            raise ValueError("Development image/label must align and permit512center crop")
        if not np.all((label<19)|(label==255)):
            raise ValueError("Development masks require trainIds0..18and255")
        y,x=(label.shape[0]-512)//2,(label.shape[1]-512)//2
        pixels=image[y:y+512,x:x+512].astype(np.float32)
        pixels=(pixels-np.asarray([123.675,116.28,103.53],dtype=np.float32))/np.asarray([58.395,57.12,57.375],dtype=np.float32)
        return {"image":torch.from_numpy(pixels.transpose(2,0,1).copy()),
                "label":torch.from_numpy(label[y:y+512,x:x+512].astype(np.int64).copy()),
                "condition":row["condition"],"image_id":row["image_id"]}


def choose_trial(trials):
    """Predeclared VRU objective, lower-LR ties, explicit all-undefined fallback."""
    valid=[t for t in trials if t.get("development_vru_mean_iou") is not None]
    metric="development_vru_mean_iou"
    fallback=None
    if not valid:
        metric="development_miou_present"
        valid=[t for t in trials if t.get(metric) is not None]
        fallback="All three development VRU means undefined; used predeclared mean-IoU-over-present-unions fallback"
    if not valid:
        raise ValueError("All development selection scores are undefined")
    winner=sorted(valid,key=lambda t:(-t[metric],t["learning_rate"]))[0]
    return winner,metric,fallback


def _manifest(path,rows):
    with open(path,"w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def tune(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Tuning output must be empty. Choose a new --output to preserve evidence.")
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    rows=collect_training_rows(args.data_root)
    if len(rows)!=1600:
        raise ValueError(f"Tuning requires all1600 ACDC training images, found{len(rows)}")
    for row in rows:
        row["image_sha256"]=file_hash(row["image_path"])
        row["label_sha256"]=file_hash(row["label_path"])
    fit,development=sequence_holdout(rows,args.seed)
    _manifest(args.output/"fit_manifest.csv",fit)
    _manifest(args.output/"development_manifest.csv",development)
    split={"fit_images":len(fit),"development_images":len(development),
           "fit_groups":sorted({row["sequence_group"] for row in fit}),
           "development_groups":sorted({row["sequence_group"] for row in development}),
           "condition_counts":{c:{"fit":sum(r["condition"]==c for r in fit),"development":sum(r["condition"]==c for r in development),
                                  "fit_recording_families":len({r["recording_family"] for r in fit if r["condition"]==c}),
                                  "development_recording_families":len({r["recording_family"] for r in development if r["condition"]==c})} for c in CONDITIONS},
           "sequence_groups_disjoint":True,"recording_families_disjoint":True,"frame_ids_disjoint":True,
           "original_sequence_directories":{c:sorted({r["sequence_directory"] for r in fit+development if r["condition"]==c}) for c in CONDITIONS},
           "recording_family_aliases":{group:sorted({r["sequence_directory"] for r in fit+development if r["sequence_group"]==group})
                                       for group in sorted({r["sequence_group"] for r in fit+development})},
           "recording_family_rule":"Within condition, GOPR#### and GPxx#### share recording_####; unmatched names remain unchanged",
           "split_rule":"Within each condition, conservatively merge GoPro chapter directories by recording number, then choose seeded shuffled recording-family prefix nearest20%image count; retain at least1family in each partition",
           "split_limitation":"Few large recording families can make the development fraction differ substantially from20%; recording grouping cannot prove geographic or scene independence"}
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=DenseDINOv3(pretrained=True).to(args.device)
    encoder_hash=tensor_hash(model.encoder.state_dict())
    initial={name:value.detach().cpu().clone() for name,value in model.head.state_dict().items()}
    initial_hash=tensor_hash(initial)
    config={"status":"running","model_name":MODEL,"learning_rates":list(RATES),"trial_epochs":1,
            "batch_size":4,"seed":args.seed,"optimizer":"AdamW","weight_decay":0.01,
            "crop_size":512,"fit_augmentation":"Random crop and horizontalflip0.5",
            "development_protocol":"One deterministic512center crop per development image, no flips, positions chosen without labels",
            "selection_metric":"Pooled development person/rider/bicycle meanIoU, lower LR wins exact ties",
            "undefined_score_rule":"Ignore undefined VRU trials; if every trial undefined, predeclared fallback to meanIoU over present class unions; disclose fallback",
            "official_validation_or_test_used":False,"encoder_state_sha256":encoder_hash,
            "torch_version":str(torch.__version__),"timm_version":importlib.metadata.version("timm"),
            "device":args.device,"gpu":torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
            "initial_head_sha256":initial_hash,"encoder_frozen":True,"head_parameters":7315,
            "split":split,"all_training_ids_sha256":hashlib.sha256("\n".join(sorted(row["image_id"] for row in rows)).encode()).hexdigest(),
            "fit_manifest_sha256":file_hash(args.output/"fit_manifest.csv"),
            "development_manifest_sha256":file_hash(args.output/"development_manifest.csv"),
            "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"trials":[]}
    json_write(args.output/"tuning_results.json",config)
    development_loader=DataLoader(CenterCrops(development),batch_size=4,shuffle=False,num_workers=0)
    total_start=time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with open(args.output/"training_steps.csv","w",newline="",encoding="utf-8") as handle:
        fields=["learning_rate","step","images","valid_pixels","loss","gradient_norm","image_ids","crop_coordinates","flips"]
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for rate in RATES:
            # Reset every RNG and the same initial head for matched trial crops.
            random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            model.head.load_state_dict(initial,strict=True);model.train()
            if tensor_hash(model.head.state_dict())!=initial_hash:
                raise RuntimeError("Head reset mismatch")
            optimizer=torch.optim.AdamW(model.head.parameters(),lr=rate,weight_decay=0.01)
            fit_loader=DataLoader(TrainingCrops(fit),batch_size=4,shuffle=True,num_workers=0,
                                  generator=torch.Generator().manual_seed(args.seed))
            trial_start=time.perf_counter();loss_sum=0.0;valid_sum=0;steps=0;skipped=0
            for batch in fit_loader:
                images=batch["image"].to(args.device);labels=batch["label"].to(args.device)
                valid=int((labels!=255).sum())
                if not valid:
                    skipped+=1;continue
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(images),labels,ignore_index=255)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite tuning loss")
                loss.backward()
                grad=float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.head.parameters())))
                if not np.isfinite(grad) or grad<=0 or any(p.grad is not None for p in model.encoder.parameters()):
                    raise RuntimeError("Invalid head-only tuning gradient")
                optimizer.step();steps+=1;loss_value=float(loss.detach())
                loss_sum+=loss_value*valid;valid_sum+=valid
                writer.writerow({"learning_rate":rate,"step":steps,"images":len(batch["image_id"]),
                    "valid_pixels":valid,"loss":loss_value,"gradient_norm":grad,"image_ids":" ".join(batch["image_id"]),
                    "crop_coordinates":json.dumps(list(zip(batch["crop_y"].tolist(),batch["crop_x"].tolist()))),
                    "flips":json.dumps(batch["flipped"].tolist())});handle.flush()
                if steps%25==0:
                    LOG.info("LR%g fittingstep%d loss%.4f",rate,steps,loss_value)
            model.eval();groups={c:np.zeros((19,19),dtype=np.int64) for c in CONDITIONS}
            with torch.inference_mode():
                for batch in development_loader:
                    predictions=model(batch["image"].to(args.device)).argmax(1).cpu().numpy()
                    labels=batch["label"].numpy()
                    for i,condition in enumerate(batch["condition"]):
                        groups[condition]+=confusion_matrix(labels[i],predictions[i])
            pooled=sum(groups.values(),np.zeros((19,19),dtype=np.int64))
            scores=metrics_from_confusion(pooled)
            after=tensor_hash(model.encoder.state_dict())
            if after!=encoder_hash:
                raise RuntimeError("Frozen encoder changed during tuning")
            trial={"learning_rate":rate,"fit_epochs":1,"fit_optimizer_steps":steps,"skipped_all_ignore_batches":skipped,
                   "training_mean_pixel_loss":loss_sum/valid_sum if valid_sum else None,
                   "development_vru_mean_iou":scores["vru_mean_iou"],
                   "development_miou_19":scores["miou_19"],"development_miou_present":scores["miou_present"],
                   "development_valid_pixels":scores["valid_pixels"],"development_per_class":scores["per_class"],
                   "development_conditions":{c:metrics_from_confusion(cm) for c,cm in groups.items()},
                   "seconds":time.perf_counter()-trial_start,"encoder_unchanged":True}
            np.savez_compressed(args.output/f"development_confusions_lr{rate:g}.npz",
                                **groups,pooled=pooled)
            config["trials"].append(trial);json_write(args.output/"tuning_results.json",config)
            LOG.info("LR%g developmentVRU=%s mIoU19=%s",rate,trial["development_vru_mean_iou"],trial["development_miou_19"])
    winner,metric,fallback=choose_trial(config["trials"])
    config.update(status="completed",selected_learning_rate=winner["learning_rate"],
                  selected_metric=metric,selected_score=winner[metric],selection_fallback=fallback,
                  elapsed_seconds=time.perf_counter()-total_start,
                  peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2 if args.device.startswith("cuda") else None,
                  refit_rule="Fresh seeded head initialization on all1600 training images for5epochs, batch4; discard all trial weights")
    json_write(args.output/"tuning_results.json",config)
    fields=["learning_rate","training_mean_pixel_loss","development_vru_mean_iou","development_miou_19",
            "development_miou_present","development_valid_pixels","fit_optimizer_steps","seconds"]
    with open(args.output/"tuning_trials.csv","w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({field:trial[field] for field in fields} for trial in config["trials"])
    if fallback:
        LOG.warning(fallback)
    LOG.info("SelectedLR%g using%s",winner["learning_rate"],metric)
    # Delete shared GPU model before creating the fresh full-refit instance.
    del model,optimizer
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    if args.refit:
        from roadseg.train_dinov3 import train
        train(argparse.Namespace(data_root=args.data_root,output=args.refit_output,device=args.device,
              seed=args.seed,batch_size=4,epochs=5,limit=None,learning_rate=winner["learning_rate"],
              tuning_results=args.output/"tuning_results.json"))
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=Path("data/acdc"))
    parser.add_argument("--output",type=Path,default=Path("outputs/dinov3_tuning"))
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--refit",action="store_true",help="Fresh5epoch full-train fit after search")
    parser.add_argument("--refit-output",type=Path,default=Path("outputs/dinov3_dense"))
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    tune(args)


if __name__=="__main__":
    main()
