from __future__ import annotations
import numpy as np

def fractions(mask: np.ndarray, target: np.ndarray):
    m=np.asarray(mask,bool); t=np.asarray(target,bool); denom=max(int(t.sum()),1)
    overall=float(np.logical_and(m,t).sum())/denom
    over=float(np.logical_and(m,~t).sum())/denom
    return {'overall_removal_fraction':overall,'under_removal_fraction':1.0-overall,'over_removal_fraction':over}

def soft_dice_loss(pred: np.ndarray,target: np.ndarray,eps:float=1e-6):
    p=np.asarray(pred,np.float32); t=np.asarray(target,np.float32)
    return float(1.0-(2.0*(p*t).sum()+eps)/(p.sum()+t.sum()+eps))

def candidate_table(probabilities, target, actions, threshold, l1_weight, dice_weight, over_limit):
    out=[]
    for i,(p,a) in enumerate(zip(probabilities,actions)):
        l1=float(np.mean(np.abs(np.asarray(p,np.float32)-target)))
        dice=soft_dice_loss(p,target)
        cost=float(l1_weight*l1+dice_weight*dice)
        fr=fractions(np.asarray(p)>=threshold,target)
        out.append({'candidate_index':i,'action_id':a.action_id,'energy_uJ':a.energy_uJ,'action_y_um':a.action_y_um,'action_z_um':a.action_z_um,'shape_l1':l1,'shape_dice_loss':dice,'cost_j':cost,**{f'predicted_{k}':v for k,v in fr.items()},'is_safe':fr['over_removal_fraction']<over_limit})
    safe=[r for r in out if r['is_safe']]
    no_safe=not safe
    if safe: best=min(safe,key=lambda r:r['cost_j'])
    else: best=min(out,key=lambda r:(r['predicted_over_removal_fraction'],r['cost_j']))
    ordered=sorted(out,key=lambda r:(not r['is_safe'],r['cost_j']) if not no_safe else (r['predicted_over_removal_fraction'],r['cost_j']))
    for rank,r in enumerate(ordered,1): r['rank']=rank; r['selected']=r is best
    return out,best,len(safe),no_safe
