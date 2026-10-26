"""Large, presentation-readable scientific panel from saved prediction masks.

Only our three predicted masks are displayed. The previously recorded sample
selection supplies the image ID and crop, so no new favorable example is chosen.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/"results/figures"
selection=json.loads((FIG/"sample_selection.json").read_text())
fog=next(row for row in selection if row["condition"]=="fog")
image_id=fog["image_id"]
box=fog["crop_y1_y2_x1_x2"]
primary=pd.read_csv(ROOT/"results/manifest.csv").set_index("image_id").loc[image_id]
third=pd.read_csv(ROOT/"results/dinov3/manifest.csv").set_index("image_id").loc[image_id]
paths=[primary.hrda_pred_path,primary.mic_pred_path,third.pred_path]
palette=np.array([[55+i*7]*3 for i in range(19)],dtype=np.uint8)
palette[11],palette[12],palette[18]=[239,94,85],[247,181,65],[37,184,201]
background="#F8F7F2"
plt.rcParams.update({"font.family":"Arial","font.size":14})
fig,axes=plt.subplots(1,3,figsize=(10,4),facecolor=background)
fig.subplots_adjust(left=.02,right=.98,bottom=.25,top=.79,wspace=.02)
for ax,file,label in zip(axes,paths,["HRDA\nUDA checkpoint","MIC(HRDA)\nUDA checkpoint","DINOv3 + head\nSupervised target labels"]):
    with Image.open(ROOT/file) as source:
        mask=np.asarray(source)
    if mask.ndim!=2 or np.any(mask>18):
        raise ValueError("Expected saved19class prediction-ID mask")
    crop=mask[box[0]:box[1],box[2]:box[3]]
    ax.imshow(palette[crop],interpolation="nearest")
    ax.axis("off")
    ax.set_title(label,fontsize=16,pad=10,color="#102B3F")
fig.legend(handles=[Patch(color=palette[i]/255,label=name) for i,name in [(11,"Person"),(12,"Rider"),(18,"Bicycle")]],
           loc="lower center",bbox_to_anchor=(.5,.12),ncol=3,frameon=False,fontsize=14)
fig.text(.5,.035,"Selected illustration, not random. Same saved crop for all models.",
         ha="center",va="bottom",fontsize=14,color="#536576")
fig.savefig(FIG/"presentation_fog_focus.png",dpi=180,facecolor=background)
plt.close(fig)

with np.load(ROOT/"results/per_image_confusions.npz",allow_pickle=False) as saved:
    support=saved["confusion"][:,0,12,:].sum(axis=1)
    mask=(saved["condition"]=="snow")&(support>0)
    snow={"gt_pixels":int(support[mask].sum()),"images":int(mask.sum()),
          "image_ids":saved["image_id"][mask].tolist()}
evidence={"focus":fog,"displayed":"Saved predictions only; no RGB or GT image pixels",
          "snow_rider":snow,"selection_rule":"Reuse previously recorded largest-GT-VRU-support fog image and crop"}
(FIG/"presentation_focus_evidence.json").write_text(json.dumps(evidence,indent=2))
print(FIG/"presentation_fog_focus.png")
