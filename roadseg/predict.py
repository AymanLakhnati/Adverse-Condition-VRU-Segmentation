"""Predict one image or a deterministic directory batch without overwriting outputs.

python -m roadseg.predict image.png --model dinov3 --output results/sample_output
python -m roadseg.predict images/ --recursive --model mic --output results/new_batch

Directory batches load the model once. Successful image bundles remain valid if
a later image fails. batch_manifest.json appears only after complete success.
"""
import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from roadseg.metrics import CLASS_NAMES

IMAGE_EXTENSIONS={".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
DEFAULT_CHECKPOINTS={"hrda":"checkpoints/hrda/csHR2acdcHR_hrda_d3a68/iter_40000_relevant.pth",
                     "mic":"checkpoints/mic/csHR2acdcHR_mic_hrda_b52f1/iter_40000_relevant.pth",
                     "dinov3":"outputs/dinov3_dense/head.pt"}


def sha256(path):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def discover_images(source,output,recursive=False):
    """Enumerate once, excluding output subtrees and symbolic links."""
    source,output=Path(source),Path(output)
    if source.is_file():
        return [source],False
    if not source.is_dir():
        raise FileNotFoundError(source)
    source_root,output_root=source.resolve(),output.resolve()
    if source_root==output_root:
        raise ValueError("Batch output must differ from the input directory")
    candidates=source.rglob("*") if recursive else source.iterdir()
    output_inside_source=output_root.is_relative_to(source_root)
    images=[p for p in candidates if p.is_file() and not p.is_symlink()
            and p.suffix.lower() in IMAGE_EXTENSIONS and p.resolve().is_relative_to(source_root)
            and not (output_inside_source and p.resolve().is_relative_to(output_root))]
    images.sort(key=lambda p:(p.relative_to(source).as_posix().casefold(),p.relative_to(source).as_posix()))
    if not images:
        raise ValueError("Input directory has no supported images")
    return images,True


def output_paths(image,source,output,model_name,is_batch):
    if is_batch:
        relative=Path(image).relative_to(source)
        prefix=Path(output)/relative.parent/(relative.name+"_"+model_name)
    else:
        prefix=Path(output)/(Path(image).stem+"_"+model_name)
    # Append suffixes: with_suffix() truncates dotted source stems.
    return {"labels":Path(str(prefix)+".png"),
            "preview":Path(str(prefix)+"_preview.png"),
            "metadata":Path(str(prefix)+".json")}


def publish_bundle(staged_to_final):
    """Exclusive publication with rollback of only this bundle's own files.

    Same-directory staging keeps hard links on one filesystem. Link creation
    refuses existing destinations, including races after the initial preflight.
    """
    published=[]
    try:
        for staged,final in staged_to_final:
            os.link(staged,final)
            published.append((staged,final))
    except BaseException:
        for staged,final in reversed(published):
            try:
                if os.path.samefile(staged,final):
                    final.unlink()
            except FileNotFoundError:
                pass
        raise


def load_prediction_model(model_name,checkpoint,device):
    if model_name=="dinov3":
        from roadseg.train_dinov3 import load_trained_model
        return load_trained_model(checkpoint,device)
    from src.hrda_inference import load_model
    return load_model(checkpoint,device)


def predict_mask(model,image,model_name,device):
    """Keep the existing single-image inference mathematics and alignment."""
    with Image.open(image) as source:
        original_shape=(source.height,source.width)
    if model_name=="dinov3":
        from roadseg.train_dinov3 import image_tensor,predict_logits
        tensor=image_tensor(image,device)
        with torch.inference_mode():
            logits=predict_logits(model,tensor)
    else:
        from src.hrda_inference import preprocess
        tensor=preprocess(image,device)
        resized_shape=tensor.shape[-2:]
        tensor=F.pad(tensor,(0,(-tensor.shape[-1])%8,0,(-tensor.shape[-2])%8))
        with torch.inference_mode():
            logits=model(tensor)[:,:,:resized_shape[0],:resized_shape[1]]
    with torch.inference_mode():
        if logits.shape[-2:]!=original_shape:
            logits=F.interpolate(logits,size=original_shape,mode="bilinear",align_corners=False)
        return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def run_prediction(source,model_name="dinov3",checkpoint=None,output=Path("results/sample_output"),device="cpu",recursive=False):
    source,output=Path(source),Path(output)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if model_name not in DEFAULT_CHECKPOINTS:
        raise ValueError("Unknown segmentation model")
    checkpoint=Path(checkpoint) if checkpoint is not None else Path(DEFAULT_CHECKPOINTS[model_name])
    images,is_batch=discover_images(source,output,recursive)
    jobs=[(image,output_paths(image,source,output,model_name,is_batch)) for image in images]
    reserved=[path for _,paths in jobs for path in paths.values()]
    manifest_path=output/"batch_manifest.json" if is_batch else None
    if manifest_path is not None:
        reserved.append(manifest_path)
    keys=[str(path.resolve()).casefold() for path in reserved]
    if len(keys)!=len(set(keys)):
        raise ValueError("Input filenames produce colliding output paths")
    for path in reserved:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a fresh --output")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Download or train the selected checkpoint first: {checkpoint}")
    # Detect invalid inputs before loading a model or publishing any output.
    for image,_ in jobs:
        with Image.open(image) as opened:
            opened.verify()
    checkpoint_hash=sha256(checkpoint)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    load_started=time.perf_counter()
    model=load_prediction_model(model_name,checkpoint,device)
    model_load_seconds=time.perf_counter()-load_started
    if sha256(checkpoint)!=checkpoint_hash:
        raise ValueError("Checkpoint changed while the model was loading")
    completed=[]
    palette=np.array([[55+7*i]*3 for i in range(19)],dtype=np.uint8)
    palette[11],palette[12],palette[18]=[239,94,85],[247,181,65],[37,184,201]
    for image,paths in jobs:
        source_hash=sha256(image)
        started=time.perf_counter()
        try:
            labels=np.asarray(predict_mask(model,image,model_name,device))
        except Exception as error:
            raise RuntimeError(f"Prediction failed for {image} after {len(completed)} completed image bundles: {error}") from error
        with Image.open(image) as opened:
            expected_shape=(opened.height,opened.width)
        if (labels.ndim!=2 or labels.shape!=expected_shape or
            not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>18)):
            raise ValueError(f"Model produced invalid train-ID mask for {image}")
        if sha256(image)!=source_hash:
            raise ValueError(f"Source image changed during inference: {image}")
        labels=labels.astype(np.uint8)
        counts=np.bincount(labels.ravel(),minlength=19)
        paths["labels"].parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".predict-",dir=paths["labels"].parent) as staging:
            staging=Path(staging)
            label_file,preview_file,metadata_file=staging/"labels.png",staging/"preview.png",staging/"metadata.json"
            Image.fromarray(labels).save(label_file)
            Image.fromarray(palette[labels]).save(preview_file)
            info={"model":model_name,"checkpoint":str(checkpoint),"checkpoint_sha256":checkpoint_hash,
                  "source_image":str(image),"source_sha256":source_hash,"output_shape":list(labels.shape),
                  "train_id_labels":{i:name for i,name in enumerate(CLASS_NAMES)},
                  "predicted_pixel_counts":{name:int(counts[i]) for i,name in enumerate(CLASS_NAMES)},
                  "labels_sha256":sha256(label_file),"preview_sha256":sha256(preview_file),
                  "elapsed_seconds":time.perf_counter()-started,"model_load_seconds":model_load_seconds,
                  "shared_batch_model":is_batch,"device":device,"torch_version":str(torch.__version__),
                  "interpretation":"Semantic pixel labels, not individual detections, demographic labels or a driving-safety assessment"}
            if not is_batch:
                info["elapsed_seconds_including_model_load"]=model_load_seconds+info["elapsed_seconds"]
            metadata_file.write_text(json.dumps(info,indent=2,allow_nan=False),encoding="utf-8")
            publish_bundle([(label_file,paths["labels"]),(preview_file,paths["preview"]),(metadata_file,paths["metadata"])])
        completed.append({"source_relative_path":image.relative_to(source).as_posix() if is_batch else image.name,
                          **{key:str(path.relative_to(output)) for key,path in paths.items()},
                          "source_sha256":source_hash,"labels_sha256":info["labels_sha256"]})
    if is_batch:
        manifest={"status":"completed","model":model_name,"checkpoint_sha256":checkpoint_hash,
                  "images":len(completed),"recursive":recursive,"model_load_seconds":model_load_seconds,
                  "discovery_order":"Case-insensitive relative POSIX path, then exact relative path",
                  "outputs":completed}
        output.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".predict-",dir=output) as staging:
            staged=Path(staging)/"batch_manifest.json"
            staged.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
            publish_bundle([(staged,manifest_path)])
        return {"images":len(completed),"manifest":str(manifest_path)}
    return {"labels":str(jobs[0][1]["labels"]),"metadata":str(jobs[0][1]["metadata"])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image",type=Path,help="One image file or an image directory")
    parser.add_argument("--model",choices=list(DEFAULT_CHECKPOINTS),default="dinov3")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--output",type=Path,default=Path("results/sample_output"))
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recursive",action="store_true",help="Include nested image directories")
    args=parser.parse_args()
    print(json.dumps(run_prediction(args.image,args.model,args.checkpoint,args.output,args.device,args.recursive),indent=2))


if __name__=="__main__":
    main()
