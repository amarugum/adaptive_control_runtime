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
    """Score all action candidates.

    When no safe candidate exists, ``best`` is still populated with the
    minimum-over-removal diagnostic row so offline tools can rank infeasible
    candidates.  V1.8 live runtime MUST NOT execute this diagnostic fallback;
    ``runtime.engine`` converts ``no_safe=True`` into a STOP decision.
    """
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


def candidate_table_policy_v1(
    probabilities, target, actions, threshold, current_fractions, lambda_over, over_limit
):
    """Live Policy v1.0 action scores from current and predicted binary masks.

    C = |S ∩ T| / |T|, O = |S \ T| / |T|.  As in the validated
    combined-stop analysis, each *increment* is clamped at zero to avoid
    rewarding nonphysical apparent shrinkage in model predictions.

    L(a) = lambda_over * max(O_next-O_now, 0)
           - max(C_next-C_now, 0).

    Only candidates with O_next < over_limit are eligible. A candidate with
    L >= 0 is NOT an executable action; the decision layer returns STOP.
    No unsafe diagnostic fallback is ever produced by this selector.
    """
    if not np.isfinite(lambda_over) or lambda_over <= 0:
        raise ValueError('lambda_over must be finite and > 0')
    if not np.isfinite(over_limit) or over_limit <= 0:
        raise ValueError('over_limit must be finite and > 0')
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError('threshold must be in (0, 1]')
    target_array = np.asarray(target)
    target_mask = target_array >= 0.5
    if target_mask.ndim != 2 or not target_mask.any():
        raise ValueError('Target must be a nonempty 2D binary mask')
    probabilities = np.asarray(probabilities)
    if (probabilities.ndim != 3 or probabilities.shape[1:] != target_mask.shape
            or probabilities.shape[0] != len(actions) or len(actions) == 0):
        raise ValueError('Candidate predictions and action registry shape/count mismatch')
    if not np.all(np.isfinite(probabilities)):
        raise ValueError('Nonfinite candidate predictions; fail closed')
    current_c = float(current_fractions['overall_removal_fraction'])
    current_o = float(current_fractions['over_removal_fraction'])
    if not (np.isfinite(current_c) and np.isfinite(current_o)):
        raise ValueError('Nonfinite current-state fractions; fail closed')
    out = []
    for i, (p, a) in enumerate(zip(probabilities, actions)):
        fr = fractions(p >= threshold, target_mask)
        delta_c_raw = fr['overall_removal_fraction'] - current_c
        delta_o_raw = fr['over_removal_fraction'] - current_o
        delta_c = max(delta_c_raw, 0.0)
        delta_o = max(delta_o_raw, 0.0)
        loss = float(lambda_over * delta_o - delta_c)
        out.append({
            'candidate_index': i,
            'action_id': a.action_id,
            'energy_uJ': a.energy_uJ,
            'action_y_um': a.action_y_um,
            'action_z_um': a.action_z_um,
            'delta_overall_raw': float(delta_c_raw),
            'delta_over_raw': float(delta_o_raw),
            'delta_overall': float(delta_c),
            'delta_over': float(delta_o),
            'cost_j': loss,
            'policy_loss': loss,
            'predicted_utility': -loss,
            **{f'predicted_{key}': value for key, value in fr.items()},
            'is_safe': fr['over_removal_fraction'] < over_limit,
        })
    safe = [row for row in out if row['is_safe']]
    # Stable action-registry ordering resolves exact score ties.
    best = min(safe, key=lambda row: (row['policy_loss'], row['candidate_index'])) if safe else None
    ordered = sorted(out, key=lambda row: (not row['is_safe'], row['policy_loss'], row['candidate_index']))
    for rank, row in enumerate(ordered, 1):
        row['rank'] = rank
        row['selected'] = row is best and row['policy_loss'] < 0.0
    return out, best, len(safe), not safe
