#!/usr/bin/env python3
"""Overlay actual target binaries and observed processed-mask images in ORIGINAL coordinates.

Requires local 8 target .npy files and actual processed masks; neither is included in
this code package. NO alignment, shift, translation, or crop normalization applied.
Supports 3 observed masks from ideal_target_designer/ideal_target.json or explicit
hole IDs from the processed data root (h####/p####/target.npy).
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def resize_nearest(mask, size=256):
    h,w=mask.shape
    if (h,w)==(size,size):return mask.copy()
    yi=np.minimum((np.arange(size)*h/size).astype(int),h-1)
    xi=np.minimum((np.arange(size)*w/size).astype(int),w-1)
    return mask[np.ix_(yi,xi)]


def load_binary(path:Path,channel:int=1,threshold:float=.5,size:int=256):
    if path.suffix.lower()=='.npy':
        arr=np.asarray(np.load(path),dtype=np.float32)
        if arr.ndim==3:
            if not (0<=channel<arr.shape[0]):raise ValueError(f'channel {channel} missing: {path} {arr.shape}')
            arr=arr[channel]
        if arr.ndim!=2:raise ValueError(f'Expected mask 2D or C,H,W: {path} shape={arr.shape}')
        m=arr>=threshold
    else:
        from PIL import Image
        m=np.asarray(Image.open(path).convert('L'))>=128
    return resize_nearest(m.astype(np.uint8),size).astype(bool)


def fractions(mask,target):
    n=int(target.sum())
    if n<=0: raise ValueError('Empty target')
    overlap=int((mask&target).sum())
    over=int((mask&~target).sum())
    missing=n-overlap
    return {'target_area_px':n,'observed_processed_area_px':int(mask.sum()),
            'overlap_area_px':overlap,'over_area_px':over,'missing_area_px':missing,
            'overall':overlap/n,'over':over/n,'iou':overlap/(n+over)}


def rgb_overlay(t,m):
    # RGB values: target only blue, observed outside red, overlap green, empty dark.
    im=np.empty((*t.shape,3),dtype=np.uint8);im[:]=[245,246,249]
    im[t&~m]=[75,125,214]
    im[m&~t]=[227,95,76]
    im[t&m]=[75,158,112]
    return im


def source_json_map(root:Path|None):
    result={}
    if root and root.is_dir():
        for p in root.rglob('ideal_target.json'):
            data=json.loads(p.read_text(encoding='utf8'))
            tid=str(data.get('target_id') or p.parent.name)
            result[tid]=(p,data)
    return result


def source_options(tid:str,data_root:Path,holes:list[str],pulse:int,designer_map:dict,use_designer:bool):
    if use_designer and tid in designer_map:
        json_path,payload=designer_map[tid]
        options=[]
        for s in payload.get('sources',[]):
            raw=Path(s['path'])
            if not raw.is_absolute():raw=(json_path.parent/raw).resolve()
            options.append((str(raw),raw,int(payload.get('channel',1)),float(payload.get('threshold',.5)),
                            bool(payload.get('source_alignment_for_display',False))))
        return options[:3]
    return [(f'{h} p{pulse:04d}',data_root/h/f'p{pulse:04d}'/'target.npy',1,.5,False) for h in holes[:3]]


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(args.target_root)
    paths=sorted(root.glob('T_*.npy'))
    if not paths: raise FileNotFoundError(f'No T_*.npy in target-root: {root}')
    if args.target_ids:
        wanted=set(args.target_ids); paths=[p for p in paths if p.stem in wanted]
        if {p.stem for p in paths}!=wanted:raise ValueError('Missing target IDs: '+str(wanted-{p.stem for p in paths}))
    if len(paths)!=8 and not args.allow_subset:
        raise ValueError(f'Expected 8 target npy files; found {len(paths)}. Use --allow-subset for debug only.')
    designer=source_json_map(Path(args.designer_root) if args.designer_root else None)
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    metrics=[]; missing=[]
    fig,axes=plt.subplots(len(paths),4,figsize=(15,3*len(paths)),squeeze=False)
    for it,tp in enumerate(paths):
        t=load_binary(tp,channel=0,threshold=.5,size=256)
        tid=tp.stem
        axes[it,0].imshow(t,cmap='gray',vmin=0,vmax=1)
        axes[it,0].set_title(f'{tid}\nTarget {int(t.sum())} px',fontsize=9)
        for ax in axes[it]:ax.set_xticks([]);ax.set_yticks([])
        options=source_options(tid,Path(args.data_root),args.hole_uids,args.pulse,designer,args.use_designer_sources)
        for k in range(3):
            ax=axes[it,k+1]
            if k>=len(options): ax.axis('off');continue
            name,src,ch,threshold,had_display_alignment=options[k]
            if not src.exists():
                missing.append(str(src));ax.text(.5,.5,'NOT FOUND',ha='center',va='center',transform=ax.transAxes)
                continue
            obs=load_binary(src,channel=ch,threshold=threshold,size=256)
            vals=fractions(obs,t)
            ax.imshow(rgb_overlay(t,obs))
            ax.set_title(f'{Path(name).name[:32]}\nC={vals["overall"]:.3f} O={vals["over"]:.3f} IoU={vals["iou"]:.3f}',fontsize=9)
            metrics.append({'target_id':tid,'reference_mask':str(src),'source_display_alignment_flag':had_display_alignment,
                            'source_mode':'designer_original' if args.use_designer_sources and tid in designer else 'selected_observation',
                            **vals})
    fig.suptitle('Physical-coordinate overlays (NO horizontal/vertical alignment)\nBlue=target only  Red=observed outside  Green=intersection',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.96])
    fig.savefig(out/'target_observed_strict_overlays.png',dpi=160)
    plt.close(fig)
    if metrics:
        with (out/'target_observed_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    info={'target_root':str(root),'observed_data_root':str(args.data_root),'use_designer_sources':args.use_designer_sources,
          'n_targets':len(paths),'n_images':len(metrics),'missing_observed_mask_paths':missing,
          'caution':'No translation or registration is performed. Actual target designer may have used display-only X alignment when creating the target; review ideal_target.json source_alignment_for_display.'}
    (out/'overlay_provenance.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    print(f'Wrote: {out}; targets={len(paths)}, matched observed={len(metrics)}, missing={len(missing)}')
    if missing:print('Missing examples:',missing[:5])
    if args.use_designer_sources and not designer:print('WARNING: No ideal_target.json source metadata found. Used provided hole IDs as fallback.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('F:/260910_ml_data'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-ids',nargs='+')
    p.add_argument('--hole-uids',nargs='+',default=['h0001','h0008','h0020'])
    p.add_argument('--pulse',type=int,default=50)
    p.add_argument('--designer-root',type=Path)
    p.add_argument('--use-designer-sources',action='store_true')
    p.add_argument('--allow-subset',action='store_true')
    run(p.parse_args())

if __name__=='__main__':main()
