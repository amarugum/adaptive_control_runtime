#!/usr/bin/env python3
"""Offline two-stage STOP policy screening from val policy-candidate cache.

No runtime / PC1 changes. Historical fixed-plan observations are independent
states, NOT closed-loop adaptive trajectories. Oracle-current uses a measured
current mask followed by the same trained Transition; counterfactual future
masks are NOT ground truth.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np

REQUIRED = ('modes','target_ids','action_ids','hole_uids','current_pulses',
            'est_cur_overall','est_cur_over','oracle_cur_overall','oracle_cur_over',
            'est_cand_overall','est_cand_over','oracle_cand_overall','oracle_cand_over')
REASONS = ('current_over_limit','target_reached','no_safe_action','nonpositive_utility','continue')


def load_cache(path:Path, expected_split:str='val'):
    if path.suffix.lower()=='.zip':
        with ZipFile(path) as z:
            members=[x for x in z.namelist() if x.endswith('policy_candidate_cache.npz')]
            metas=[x for x in z.namelist() if x.endswith('policy_cache_metadata.json')]
            if len(members)!=1 or len(metas)!=1:raise ValueError('ZIP must contain one policy_candidate_cache.npz and policy_cache_metadata.json')
            meta=json.loads(z.read(metas[0]))
            with np.load(io.BytesIO(z.read(members[0])),allow_pickle=False) as f:
                d={k:f[k] for k in f.files}
    elif path.suffix.lower()=='.npz':
        meta=json.loads(path.with_name('policy_cache_metadata.json').read_text(encoding='utf8'))
        with np.load(path,allow_pickle=False) as f:
            d={k:f[k] for k in f.files}
    else: raise ValueError('Cache must be zip or npz')
    if meta.get('split') != expected_split:
        raise ValueError(f'Expected split={expected_split!r}; got {meta.get("split")!r}. Keep test held out.')
    missing=set(REQUIRED)-set(d)
    if missing:raise ValueError(f'Missing arrays {missing}')
    shape=(len(d['modes']),len(d['hole_uids']),len(d['target_ids']))
    na=len(d['action_ids'])
    if 0 in shape or na<1: raise ValueError('Empty cache dimensions')
    if d['current_pulses'].shape!=(shape[1],):raise ValueError('Invalid pulse shape')
    for k in REQUIRED[5:]:
        required=shape+(na,) if '_cand_' in k else shape
        if d[k].shape!=required or not np.all(np.isfinite(d[k])):raise ValueError(f'Bad {k} shape/finite')
    if list(map(str,d['target_ids']))!=meta.get('target_ids'):raise ValueError('Target ID mismatch')
    if list(map(str,d['action_ids']))!=meta.get('action_ids'):raise ValueError('Action ID mismatch')
    return d,meta


def policy(d:dict, side:str, lam:float, completion:float, over_limit:float):
    """Return shape [M,N,T] decision and best candidate, ties use first action."""
    if not (np.isfinite(lam) and lam >= 0 and 0 < completion <= 1 and np.isfinite(over_limit) and over_limit>0):
        raise ValueError('Invalid lambda/completion/over_limit')
    cc=np.asarray(d[side+'_cur_overall'],dtype=np.float64)
    co=np.asarray(d[side+'_cur_over'],dtype=np.float64)
    nc=np.asarray(d[side+'_cand_overall'],dtype=np.float64)
    no=np.asarray(d[side+'_cand_over'],dtype=np.float64)
    # Match runtime's strict < over-limit; no automatic laser authorization.
    safe=no < over_limit
    dc=np.maximum(nc-cc[...,None],0.)
    do=np.maximum(no-co[...,None],0.)
    u=dc-lam*do
    scored=np.where(safe,u,-np.inf)
    idx=np.argmax(scored,axis=-1)
    ub=np.take_along_axis(scored,idx[...,None],axis=-1)[...,0]
    safe_count=safe.sum(axis=-1)
    stop_over=co>=over_limit
    stop_reach=(cc>=completion)&~stop_over
    stop_nosafe=(safe_count==0)&~stop_over&~stop_reach
    stop_value=(ub<=0)&~stop_over&~stop_reach&~stop_nosafe
    reasons=np.full(cc.shape,'continue',dtype='<U24')
    reasons[stop_value]='nonpositive_utility'
    reasons[stop_nosafe]='no_safe_action'
    reasons[stop_reach]='target_reached'
    reasons[stop_over]='current_over_limit'
    cont=reasons=='continue'
    idx=np.where(cont,idx,-1)
    return {'reason':reasons,'continue':cont,'chosen':idx,'best_u':ub,
            'safe_count':safe_count,'current_c':cc,'current_o':co,
            'selected_next_c':np.take_along_axis(nc,np.maximum(idx,0)[...,None],axis=-1)[...,0],
            'selected_next_o':np.take_along_axis(no,np.maximum(idx,0)[...,None],axis=-1)[...,0]}


def stats(d,e,o,mask,lam,completion,over_limit,mode,target='',pulse=''):
    m=np.asarray(mask,bool);n=int(m.sum())
    if n==0:return None
    ec=e['continue'][m];oc=o['continue'][m]
    reason=e['reason'][m]
    out={'mode':str(mode),'target':str(target),'pulse':str(pulse),
         'lambda':lam,'completion_target':completion,'over_limit':over_limit,
         'n_cases':n,'continue_rate':float(ec.mean()),'stop_rate':float((~ec).mean()),
         'oracle_continue_rate':float(oc.mean()),
         'decision_match_rate':float(np.mean(ec==oc)),
         'false_continue_rate_all':float(np.mean(ec & ~oc)),
         'false_stop_rate_all':float(np.mean(~ec & oc)),
         'mean_selected_est_next_c_if_continue':float(e['selected_next_c'][m][ec].mean()) if ec.any() else float('nan'),
         'mean_selected_est_next_o_if_continue':float(e['selected_next_o'][m][ec].mean()) if ec.any() else float('nan'),
         'mean_est_current_c_at_stop':float(e['current_c'][m][~ec].mean()) if (~ec).any() else float('nan'),
         'mean_est_current_o_at_stop':float(e['current_o'][m][~ec].mean()) if (~ec).any() else float('nan')}
    for rs in REASONS:
        out['rate_'+rs]=float(np.mean(reason==rs))
    # Actual oracle-current outcomes for the estimated-selected action, not a physical GT.
    midx=e['chosen'][m][ec]
    if ec.any():
        o_n_o=d['oracle_cand_over'][m][ec]
        o_n_c=d['oracle_cand_overall'][m][ec]
        o_cur_o=d['oracle_cur_over'][m][ec]
        o_cur_c=d['oracle_cur_overall'][m][ec]
        sel_over=np.take_along_axis(o_n_o,midx[:,None],axis=-1)[:,0]
        sel_c=np.take_along_axis(o_n_c,midx[:,None],axis=-1)[:,0]
        out['selected_action_oracle_over_limit_rate_among_continue']=float(np.mean(sel_over>=over_limit))
        out['selected_action_oracle_over_limit_rate_all']=float(np.sum(sel_over>=over_limit)/n)
        out['selected_action_oracle_nonpositive_u_rate_among_continue']=float(np.mean((sel_c-o_cur_c)-lam*(sel_over-o_cur_o)<=0))
    else:
        out['selected_action_oracle_over_limit_rate_among_continue']=float('nan')
        out['selected_action_oracle_over_limit_rate_all']=0.
        out['selected_action_oracle_nonpositive_u_rate_among_continue']=float('nan')
    return out


def csv_write(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(args):
    d,meta=load_cache(Path(args.cache),args.expected_split)
    if len(set(zip(d['hole_uids'].astype(str),d['current_pulses'].astype(int))))!=len(d['hole_uids']):
        raise ValueError('Duplicate hole/pulse physical state')
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    lm=sorted(set(float(x) for x in args.lambdas))
    ct=sorted(set(float(x) for x in args.completion_thresholds))
    ol=sorted(set(float(x) for x in args.over_limits))
    model_rows=[];target_rows=[];pulse_rows=[];first_rows=[];first_summary=[]
    modes=list(map(str,d['modes']));targets=list(map(str,d['target_ids']))
    pulses=d['current_pulses'].astype(int);unique_pulses=sorted(set(pulses))
    holes=d['hole_uids'].astype(str);unique_holes=sorted(set(holes))
    m,n,t=map(len,(modes,holes,targets))
    for l in lm:
        for comp in ct:
            for olim in ol:
                e=policy(d,'est',l,comp,olim);o=policy(d,'oracle',l,comp,olim)
                for im,mode in enumerate(modes):
                    for it,target in enumerate(targets):
                        mask=np.zeros((m,n,t),bool);mask[im,:,it]=True
                        r=stats(d,e,o,mask,l,comp,olim,mode,target,'')
                        target_rows.append(r)
                        for pulse in unique_pulses:
                            pm=mask.copy();pm[:,pulses!=pulse,:]=False
                            pr=stats(d,e,o,pm,l,comp,olim,mode,target,pulse)
                            if pr:pulse_rows.append(pr)
                    mm=np.zeros((m,n,t),bool);mm[im]=True
                    model_rows.append(stats(d,e,o,mm,l,comp,olim,mode,'ALL','ALL'))
                    # Virtual first STOP along fixed historical sequences ONLY.
                    for it,target in enumerate(targets):
                        first_p=[]
                        for hole in unique_holes:
                            ix=np.where(holes==hole)[0];ix=ix[np.argsort(pulses[ix])]
                            stops=ix[~e['continue'][im,ix,it]]
                            if len(stops):
                                i=int(stops[0]); first_p.append(int(pulses[i]))
                                first_rows.append({'mode':mode,'target':target,'lambda':l,'completion_target':comp,
                                                   'over_limit':olim,'hole_uid':hole,'first_stop_pulse':int(pulses[i]),
                                                   'reason':str(e['reason'][im,i,it]),'est_current_c':float(e['current_c'][im,i,it]),
                                                   'est_current_o':float(e['current_o'][im,i,it])})
                            else:
                                first_rows.append({'mode':mode,'target':target,'lambda':l,'completion_target':comp,
                                                   'over_limit':olim,'hole_uid':hole,'first_stop_pulse':'not_observed',
                                                   'reason':'not_observed','est_current_c':'','est_current_o':''})
                        first_summary.append({'mode':mode,'target':target,'lambda':l,'completion_target':comp,
                                              'over_limit':olim,'n_holes':len(unique_holes),
                                              'historical_first_stop_fraction':len(first_p)/len(unique_holes),
                                              'median_first_stop_pulse_if_any':float(np.median(first_p)) if first_p else float('nan')})
    for nm,rows in [('summary_by_model.csv',model_rows),('summary_by_target.csv',target_rows),
                    ('summary_by_pulse_target.csv',pulse_rows),('historical_first_stop.csv',first_rows),
                    ('historical_first_stop_summary.csv',first_summary)]:csv_write(out/nm,rows)
    metadata={'source':str(args.cache),'split':meta['split'],'lambdas':lm,'completion_thresholds':ct,'over_limits':ol,
              'models':modes,'targets':targets,'n_states':n,'n_holes':len(unique_holes),
              'reason_priority':['current_over_limit','target_reached','no_safe_action','nonpositive_utility'],
              'notes':['No physical future ground truth for unexecuted actions',
                       'Historical fixed-plan states are not adaptive rollout',
                       'Oracle-current is measured current + trained Transition',
                       'The current model overestimates/underestimates can bias safety checks',
                       'This offline script never authorizes real laser firing']}
    (out/'analysis_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    print(f'PASS: val={meta["split"]}, states={n}, holes={len(unique_holes)}, modes={m}, targets={t}; output={out}')
    for row in model_rows:
        if row['mode'] in ['shock_metadata','shock_action_history'] and row['lambda']==3 and row['completion_target']==0.9 and row['over_limit'] in [0.15,0.2]:
            print(f"{row['mode']} l=3 c=.9 o={row['over_limit']}: "+
                  f"stop={row['stop_rate']:.3f} (cur_over={row['rate_current_over_limit']:.3f}, "+
                  f"reach={row['rate_target_reached']:.3f}, no_safe={row['rate_no_safe_action']:.3f}, "+
                  f"nonpos={row['rate_nonpositive_utility']:.3f}); oracle-selected-over-limit={row['selected_action_oracle_over_limit_rate_among_continue']:.3f}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-split',choices=['val','test'],default='val')
    p.add_argument('--lambdas',type=float,nargs='+',default=[2,3,5])
    p.add_argument('--completion-thresholds',type=float,nargs='+',default=[.85,.90,.95])
    p.add_argument('--over-limits',type=float,nargs='+',default=[.05,.10,.15,.20])
    args=p.parse_args();run(args)

if __name__=='__main__': main()
