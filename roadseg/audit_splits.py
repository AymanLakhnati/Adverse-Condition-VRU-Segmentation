"""Audit official frame identities and conservative recording-name families."""
import argparse
import hashlib
import json
from pathlib import Path

from roadseg.metrics import CONDITIONS
from roadseg.tune_dinov3 import recording_family


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('data/acdc'))
    parser.add_argument('--output',type=Path,default=Path('results/reproducibility/split_audit.json'))
    parser.add_argument('--check-image-headers',action='store_true',help='Require all RGB/GT pairs and aligned original 1920x1080 headers')
    args=parser.parse_args()
    rows=[]
    for condition in CONDITIONS:
        for split in ('train','val'):
            for mask in sorted((args.data_root/'gt'/condition/split).glob('*/*_gt_labelTrainIds.png')):
                rows.append({'condition':condition,'split':split,
                             'image_id':mask.name.removesuffix('_gt_labelTrainIds.png'),
                             'chapter':mask.parent.name,'recording_name_family':recording_family(mask.parent.name)})
    ids={split:{r['image_id'] for r in rows if r['split']==split} for split in ('train','val')}
    counts={split:sum(r['split']==split for r in rows) for split in ('train','val')}
    expected={'train':1600,'val':406}
    if counts!=expected or any(len(ids[s])!=counts[s] for s in expected):
        raise ValueError(f'Complete official labels with unique frame identities required: {counts}')
    overlap=sorted(ids['train']&ids['val'])
    if overlap:
        raise ValueError('Identical frame IDs occur in training and validation')
    checked=0
    if args.check_image_headers:
        from PIL import Image
        for row in rows:
            parent=Path(row['condition'])/row['split']/row['chapter']
            rgb=args.data_root/'rgb_anon'/parent/(row['image_id']+'_rgb_anon.png')
            gt=args.data_root/'gt'/parent/(row['image_id']+'_gt_labelTrainIds.png')
            with Image.open(rgb) as image, Image.open(gt) as mask:
                if image.size!=(1920,1080) or mask.size!=image.size or image.mode!='RGB' or mask.mode not in ('L','P'):
                    raise ValueError(f'Unexpected RGB/GT header or spatial alignment: {row["image_id"]}')
            checked+=1
    families={c:{s:{r['recording_name_family'] for r in rows if r['condition']==c and r['split']==s}
                 for s in ('train','val')} for c in CONDITIONS}
    result={'scope':'Official train/validation label metadata, not perceptual duplicate detection',
            'counts':counts,'frame_ids_disjoint':True,'identical_frame_id_overlap':overlap,
            'image_id_list_sha256':{s:hashlib.sha256('\n'.join(sorted(ids[s])).encode()).hexdigest() for s in ids},
            'shared_recording_name_families':{c:sorted(families[c]['train']&families[c]['val']) for c in CONDITIONS},
            'interpretation':'Recording-name grouping is a conservative filename heuristic. Shared families do not prove duplicate frames, geographic overlap or temporal adjacency. Official splits are preserved. Internal tuning groups remain disjoint; official train/validation recording independence is not established.',
            'near_or_perceptual_duplicates_checked':False}
    result['aligned_original_rgb_gt_headers_checked']=checked
    result['header_check_limit']='Headers establish dimensions and modes, not perceptual duplicate absence or every pixel value'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
