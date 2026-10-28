#!/usr/bin/env python3
"""Offline lambda sweep for L = lambda*Delta(over) - Delta(overall).

Uses ONLY the already cached 27-action estimated/oracle-current Transition predictions.
No runtime, ML model, controller, or hardware configuration is changed.

Oracle-current is NOT physical ground truth for an unexecuted candidate action.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pathlib
import zipfile
from collections import Counter

import numpy as np

LAMBDA_DEFAULT = (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30, 50)
NEEDED = (
    'modes', 'target_ids', 'action_ids', 'hole_uids', 'current_pulses',
    'est_cur_overall', 'est_cur_over', 'oracle_cur_overall', 'oracle_cur_over',
    'est_cand_overall', 'est_cand_over', 'oracle_cand_overall', 'oracle_cand_over',
    'est_cost',
)


def load_cache(path: pathlib.Path):
    """Returns (materialized numpy arrays, optional original cache metadata)."""
    meta = None
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            npz_names = [n for n in z.namelist() if n.endswith('policy_candidate_cache.npz')]
            if len(npz_names) != 1:
                raise ValueError(f'Expected exactly one policy_candidate_cache.npz, found {len(npz_names)}')
            metadata_names = [n for n in z.namelist() if n.endswith('policy_cache_metadata.json')]
            if len(metadata_names) == 1:
                meta = json.loads(z.read(metadata_names[0]).decode('utf-8'))
            raw = io.BytesIO(z.read(npz_names[0]))
            with np.load(raw, allow_pickle=False) as f:
                d = {k: f[k] for k in f.files}
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
        metapath = path.with_name('policy_cache_metadata.json')
        if metapath.exists():
            meta = json.loads(metapath.read_text(encoding='utf-8'))
    else:
        raise ValueError('Input must be the policy_sensitivity .zip or policy_candidate_cache.npz')
    missing = set(NEEDED).difference(d)
    if missing:
        raise ValueError(f'Cache missing required arrays: {sorted(missing)}')
    return d, meta


def validate_cache(d, meta, expected_split: str):
    if meta is None or 'split' not in meta:
        raise ValueError('Cache split not verifiable. Provide policy_cache_metadata.json beside NPZ or use the source ZIP.')
    if meta['split'] != expected_split:
        raise ValueError(f'Cache split is {meta["split"]!r} but expected {expected_split!r}. '
                         'Do not choose lambda on test. To assess held-out test only, explicitly use --expected-split test.')
    modes, targets, actions = d['modes'], d['target_ids'], d['action_ids']
    h = d['hole_uids']
    base = (len(modes), len(h), len(targets))
    if len(set(map(str, actions))) != len(actions):
        raise ValueError('Duplicate action IDs')
    for key in ('est_cur_overall','est_cur_over','oracle_cur_overall','oracle_cur_over'):
        if d[key].shape != base:
            raise ValueError(f'{key} shape {d[key].shape} != {base}')
    for key in ('est_cand_overall','est_cand_over','oracle_cand_overall','oracle_cand_over','est_cost'):
        if d[key].shape != base + (len(actions),):
            raise ValueError(f'{key} shape mismatch: {d[key].shape}')
        if not np.all(np.isfinite(d[key])):
            raise ValueError(f'non-finite numbers in {key}')
    if d['current_pulses'].shape != (len(h),):
        raise ValueError('current_pulses shape mismatch')
    if meta.get('target_ids') != list(map(str, targets)):
        raise ValueError('target order differs between cache metadata and arrays')
    if meta.get('action_ids') != list(map(str, actions)):
        raise ValueError('action order differs between cache metadata and arrays')
    for prefix in ('est', 'oracle'):
        dc = d[f'{prefix}_cand_overall'].astype(np.float64) - d[f'{prefix}_cur_overall'][...,None]
        do = d[f'{prefix}_cand_over'].astype(np.float64) - d[f'{prefix}_cur_over'][...,None]
        if np.min(dc) < -1e-5 or np.min(do) < -1e-5:
            raise ValueError(f'{prefix}: negative processed-area increment! Check mask convention/monotonicity.')
    return len(h)


def make_deltas(d):
    arr = {}
    for prefix in ('est', 'oracle'):
        for dest, field in (('dc','overall'),('do','over')):
            values = d[f'{prefix}_cand_{field}'].astype(np.float64) - d[f'{prefix}_cur_{field}'][...,None]
            arr[f'{prefix}_{dest}'] = np.maximum(values, 0.0)
    return arr


def choose(deltas, lam: float):
    if lam < 0 or not np.isfinite(lam):
        raise ValueError('lambda must be finite and >= 0')
    # lower action index breaks exact ties deterministically
    return np.argmin(lam * deltas['est_do'] - deltas['est_dc'], axis=-1)


def choose_oracle(deltas, lam: float):
    return np.argmin(lam * deltas['oracle_do'] - deltas['oracle_dc'],axis=-1)


def take(arr, indices):
    return np.take_along_axis(arr, indices[...,None], axis=-1)[...,0]


def means_and_fractions(d, deltas, choices, legacy, lam, target_idx=None, pulse=None,
                        stagnant_threshold=0.01):
    """Compute summary per model; optional target or pulse restriction.

    All rows are unconditional 1-step comparisons (NO hard STOP, NO safety filter).
    """
    selector = np.ones((len(d['hole_uids']),len(d['target_ids'])),dtype=bool)
    if target_idx is not None:
        selector &= np.arange(len(d['target_ids']))[None,:] == target_idx
    if pulse is not None:
        selector &= d['current_pulses'][:,None] == pulse
    n = int(selector.sum())
    if n == 0:
        return None
    out = []
    for im, mode in enumerate(d['modes']):
        ix = choices[im]
        est_dc=take(deltas['est_dc'][im],ix)[selector]
        est_do=take(deltas['est_do'][im],ix)[selector]
        oracle_dc=take(deltas['oracle_dc'][im],ix)[selector]
        oracle_do=take(deltas['oracle_do'][im],ix)[selector]
        # compute oracle-selected action with the same lambda for comparison (diagnostic)
        if lam is not None:
            oracle_idx = choose_oracle({k:v[im] for k,v in deltas.items()},lam)
            oracle_best_score = take(lam*deltas['oracle_do'][im]-deltas['oracle_dc'][im],oracle_idx)[selector]
            chosen_score = lam*oracle_do-oracle_dc
            regret = np.maximum(chosen_score-oracle_best_score,0)
            match = float(np.mean(ix[selector] == oracle_idx[selector]))
        else:
            regret=np.empty((0,),dtype=float)
            match=float('nan')
        out.append({
            'mode':str(mode), 'n_cases':n,
            'mean_est_delta_overall':float(est_dc.mean()),
            'mean_est_delta_over':float(est_do.mean()),
            'mean_oracle_transition_delta_overall':float(oracle_dc.mean()),
            'mean_oracle_transition_delta_over':float(oracle_do.mean()),
            'fraction_oracle_delta_overall_below_threshold':float((oracle_dc < stagnant_threshold).mean()),
            'fraction_action_changed_from_legacy':float((ix[selector]!=legacy[im][selector]).mean()),
            'fraction_action_matches_oracle_optimum':match,
            'oracle_score_regret_mean':float(regret.mean()) if regret.size else float('nan'),
        })
    return out


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_difference_by_hole(d, deltas, choices, legacy, B: int, seed: int):
    """Paired bootstrap of lambda minus legacy, resampling hole UID, not state rows.

    Each hole contributes the mean over 9 control stages x 8 targets.
    Same resampled holes are used for the paired lambda-versus-legacy difference.
    """
    holes, hole_idx=np.unique(d['hole_uids'], return_inverse=True)
    rng=np.random.default_rng(seed)
    boot_indices = rng.integers(0,len(holes),size=(B,len(holes)),dtype=np.int32)
    count=np.bincount(hole_idx,minlength=len(holes)).astype(float)
    rows=[]
    for im,mode in enumerate(d['modes']):
        for metric in ('dc','do'):
            values=deltas[f'oracle_{metric}'][im]
            selected=take(values,choices[im]); old=take(values,legacy[im])
            delta=(selected-old).mean(axis=-1)
            per_hole=np.bincount(hole_idx,weights=delta,minlength=len(holes))/count
            sims=per_hole[boot_indices].mean(axis=1)
            rows.append({
                'mode':str(mode),'metric':('oracle_delta_overall' if metric=='dc' else 'oracle_delta_over'),
                'difference_lambda_minus_legacy':float(per_hole.mean()),
                'ci_2p5':float(np.percentile(sims,2.5)),
                'ci_97p5':float(np.percentile(sims,97.5)),
                'n_holes':len(holes),'bootstrap_iterations':B,
            })
    return rows


def plot_pareto(rows, out, enabled=True):
    if not enabled:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib missing: skipping plot (CSV results remain available)')
        return
    for mode in sorted(set(r['mode'] for r in rows)):
        subset=[r for r in rows if r['mode']==mode and r['strategy']!='legacy_L1_Dice']
        legacy=next(r for r in rows if r['mode']==mode and r['strategy']=='legacy_L1_Dice')
        x=[r['mean_oracle_transition_delta_over'] for r in subset]
        y=[r['mean_oracle_transition_delta_overall'] for r in subset]
        fig,ax=plt.subplots(figsize=(7.6,5.4))
        ax.plot(x,y,'o-',label='Lambda sweep')
        ax.scatter([legacy['mean_oracle_transition_delta_over']],[legacy['mean_oracle_transition_delta_overall']],
                   marker='x',s=100,label='Legacy L1 + Dice')
        for p in subset:
            lab=float(p['lambda'])
            if lab in (0,0.5,1,2,3,5,7.5,10,20,50):
                ax.annotate(f'{lab:g}',(p['mean_oracle_transition_delta_over'],p['mean_oracle_transition_delta_overall']),
                            textcoords='offset points',xytext=(4,4),fontsize=8)
        ax.set_xlabel('Mean oracle-current Transition Delta over (lower better)')
        ax.set_ylabel('Mean oracle-current Transition Delta overall (higher better)')
        ax.set_title(f'Validation 1-step lambda trade-off: {mode}')
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out/f'pareto_{mode}.png',dpi=170)
        plt.close(fig)


def run(args):
    d,meta=load_cache(pathlib.Path(args.cache))
    validate_cache(d,meta,args.expected_split)
    lambdas=sorted(set(float(i) for i in args.lambdas))
    if not lambdas or any(not np.isfinite(i) or i<0 for i in lambdas):
        raise ValueError('At least one finite lambda >= 0 required')
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    deltas=make_deltas(d)
    legacy=np.argmin(d['est_cost'],axis=-1)
    all_strategies=[('legacy_L1_Dice',None,legacy)] + [('lambda',lam,choose(deltas,lam)) for lam in lambdas]
    rows_model=[]; rows_target=[]; rows_pulse=[]; ci_rows=[]; action_rows=[]
    for typ,lam,chosen in all_strategies:
        label='legacy' if lam is None else f'{lam:g}'
        for row in means_and_fractions(d,deltas,chosen,legacy,lam,stagnant_threshold=args.small_progress_threshold):
            rows_model.append({'strategy':typ,'lambda':label,**row})
        for it,t in enumerate(d['target_ids']):
            for row in means_and_fractions(d,deltas,chosen,legacy,lam,target_idx=it,stagnant_threshold=args.small_progress_threshold):
                rows_target.append({'strategy':typ,'lambda':label,'target_id':str(t),**row})
        for p in sorted(set(map(int,d['current_pulses']))):
            for row in means_and_fractions(d,deltas,chosen,legacy,lam,pulse=p,stagnant_threshold=args.small_progress_threshold):
                rows_pulse.append({'strategy':typ,'lambda':label,'current_pulse':p,**row})
        if lam is not None and args.bootstrap_iterations>0:
            for row in bootstrap_difference_by_hole(d,deltas,chosen,legacy,args.bootstrap_iterations,args.seed):
                ci_rows.append({'strategy':typ,'lambda':label,**row})
        for im,m in enumerate(d['modes']):
            for ia,a in enumerate(d['action_ids']):
                action_rows.append({'strategy':typ,'lambda':label,'mode':str(m),'action_id':str(a),
                                    'n_selected':int(np.count_nonzero(chosen[im]==ia))})
    write_csv(out/'summary_by_model.csv',rows_model)
    write_csv(out/'summary_by_target.csv',rows_target)
    write_csv(out/'summary_by_pulse.csv',rows_pulse)
    write_csv(out/'paired_hole_bootstrap_vs_legacy.csv',ci_rows)
    write_csv(out/'action_selection_counts.csv',action_rows)
    plot_pareto(rows_model,out,enabled=not args.no_plots)
    metadata={
        'source_cache':str(pathlib.Path(args.cache).resolve()),'input_metadata':meta,
        'expected_split':args.expected_split,'lambda_values':lambdas,
        'n_states':len(d['hole_uids']),'n_holes':int(len(np.unique(d['hole_uids']))),
        'n_targets':int(len(d['target_ids'])),'n_modes':int(len(d['modes'])),
        'num_actions':int(len(d['action_ids'])),
        'formula':'L(a) = lambda * [O_next(a)-O_current] - [C_next(a)-C_current]',
        'ranking':'minimum L, ties by first action ID in cache order',
        'comparison':'legacy=min(est_cost) among all 27 candidates; NO hard STOP or over threshold',
        'oracle_definition':'measured current mask with the SAME trained Transition (and same learned z_shock). '
                            'NOT the physical measured future for counterfactual candidate actions.',
        'bootstrap':'paired hole-UID resampling, oracle-current model deltas, lambda minus legacy',
        'small_progress_threshold':args.small_progress_threshold,
        'limitations':['1-step counterfactual prediction only','No hard safety/stop policy applied',
                       'Do not choose lambda on held-out test','model-based oracle-current is not hardware outcome'],
    }
    (out/'resolved_lambda_analysis.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS: split={meta["split"]}, states={len(d["hole_uids"])}, holes={len(np.unique(d["hole_uids"]))}, modes={len(d["modes"])}, targets={len(d["target_ids"])}, lambdas={len(lambdas)}')
    print('Wrote:',out)
    for r in rows_model:
        if r['mode'] in ('shock_metadata','shock_action_history') and r['lambda'] in ('legacy','1','2','3','5','10'):
            print(f'{r["mode"]:22} lambda={r["lambda"]:6}  oracle_DeltaC={r["mean_oracle_transition_delta_overall"]:.5f}  oracle_DeltaO={r["mean_oracle_transition_delta_over"]:.5f}')


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache',required=True,help='policy sensitivity results ZIP OR policy_candidate_cache.npz')
    ap.add_argument('--output',required=True,help='output directory')
    ap.add_argument('--expected-split',choices=['val','test'],default='val',
                    help='Split must match metadata. Test is for independent evaluation AFTER freeze, not lambda tuning.')
    ap.add_argument('--lambdas',nargs='+',type=float,default=LAMBDA_DEFAULT)
    ap.add_argument('--bootstrap-iterations',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=20261008)
    ap.add_argument('--small-progress-threshold',type=float,default=0.01,
                    help='DIAGNOSTIC only, not an action-selection threshold')
    ap.add_argument('--no-plots',action='store_true')
    return ap.parse_args(argv)


if __name__=='__main__':
    run(parse_args())
