#!/usr/bin/env python3
"""Offline continuation-value / epsilon sensitivity (NOT a live STOP policy).

Score for each candidate a: U(a)=Delta C(a)-lambda*Delta O(a).
STOP if max_a U_est(a) <= epsilon, else CONTINUE with the max-score action.

Inputs: cached 27-action estimated-current and oracle-current Transition predictions
from validation/policy_sensitivity_analysis.py. No inference or hardware access.
Oracle-current is a model counterfactual, NOT the measured outcome for unexecuted actions.

All per-state/first-STOP metrics are *historical fixed-plan replay diagnostics*.
They do not simulate the dynamics of executing adaptively chosen actions.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

REQUIRED = [
    'modes', 'target_ids', 'action_ids', 'hole_uids', 'current_pulses',
    'est_cur_overall', 'est_cur_over', 'oracle_cur_overall', 'oracle_cur_over',
    'est_cand_overall', 'est_cand_over', 'oracle_cand_overall', 'oracle_cand_over',
]
DEFAULT_LAMBDAS = (2., 3., 5.)
DEFAULT_EPSILONS = (0., 0.001, 0.0025, 0.005, 0.01, 0.02)
DEFAULT_SMALL = (0.001, 0.005, 0.01)


def load_cache(path: Path):
    meta = None
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as archive:
            cache_files = [name for name in archive.namelist() if name.endswith('/policy_candidate_cache.npz') or name == 'policy_candidate_cache.npz']
            metadata_files = [name for name in archive.namelist() if name.endswith('/policy_cache_metadata.json') or name == 'policy_cache_metadata.json']
            if len(cache_files) != 1 or len(metadata_files) != 1:
                raise ValueError('ZIP needs exactly one policy_candidate_cache.npz and one policy_cache_metadata.json')
            meta = json.loads(archive.read(metadata_files[0]).decode('utf-8'))
            with np.load(io.BytesIO(archive.read(cache_files[0])), allow_pickle=False) as contents:
                arr = {name: contents[name] for name in contents.files}
    elif path.suffix.lower() == '.npz':
        metadata_path = path.with_name('policy_cache_metadata.json')
        if not metadata_path.exists():
            raise ValueError('Missing adjacent policy_cache_metadata.json; cannot verify split')
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        with np.load(path, allow_pickle=False) as contents:
            arr = {name: contents[name] for name in contents.files}
    else:
        raise ValueError('--cache must be policy_sensitivity ZIP or policy_candidate_cache.npz')
    missing = set(REQUIRED) - set(arr)
    if missing:
        raise ValueError(f'Missing cache arrays: {sorted(missing)}')
    return arr, meta


def validate_cache(d, meta, expected_split):
    if meta.get('split') != expected_split:
        raise ValueError(f'Input split is {meta.get("split")!r}; expected {expected_split!r}. '
                         'Do not tune parameters using test split.')
    m, n, t, a = map(len, (d['modes'], d['hole_uids'], d['target_ids'], d['action_ids']))
    if m < 1 or n < 1 or t < 1 or a < 1:
        raise ValueError('Empty dimensions are not allowed')
    if list(map(str, d['target_ids'])) != meta.get('target_ids'):
        raise ValueError('Target ID order mismatch')
    if list(map(str, d['action_ids'])) != meta.get('action_ids'):
        raise ValueError('Action ID order mismatch')
    if len(set(map(str, d['action_ids']))) != a:
        raise ValueError('Duplicate action IDs')
    if d['current_pulses'].shape != (n,):
        raise ValueError('current_pulses shape mismatch')
    for name in REQUIRED[5:]:
        expected = (m, n, t, a) if '_cand_' in name else (m, n, t)
        if d[name].shape != expected or not np.isfinite(d[name]).all():
            raise ValueError(f'{name}: expected finite shape {expected}, got {d[name].shape}')
    pairs = list(zip(map(str, d['hole_uids']), map(int, d['current_pulses'])))
    if len(set(pairs)) != n:
        raise ValueError('Duplicate (hole_uid, current_pulse) states')
    for side in ('est', 'oracle'):
        for metric in ('overall', 'over'):
            if (d[f'{side}_cand_{metric}'].astype(np.float64) -
                    d[f'{side}_cur_{metric}'][..., None]).min() < -1e-5:
                raise ValueError(f'{side} {metric}: violates monotonicity beyond tolerance')
    return m, n, t, a


def deltas(d, side):
    dc = np.maximum(0., d[f'{side}_cand_overall'].astype(np.float64) -
                    d[f'{side}_cur_overall'].astype(np.float64)[..., None])
    do = np.maximum(0., d[f'{side}_cand_over'].astype(np.float64) -
                    d[f'{side}_cur_over'].astype(np.float64)[..., None])
    return dc, do


def evaluate(d, lam, eps):
    if not np.isfinite(lam) or lam < 0 or not np.isfinite(eps) or eps < 0:
        raise ValueError('lambda and epsilon must both be finite and nonnegative')
    dc, do = deltas(d, 'est')
    oc, oo = deltas(d, 'oracle')
    eu = dc - lam * do
    ou = oc - lam * oo
    chosen = np.argmax(eu, axis=-1)  # deterministic first index for exact ties
    oracle_chosen = np.argmax(ou, axis=-1)
    est_best = np.take_along_axis(eu, chosen[..., None], axis=-1)[..., 0]
    oracle_best = np.take_along_axis(ou, oracle_chosen[..., None], axis=-1)[..., 0]
    oracle_utility_of_est_action = np.take_along_axis(ou, chosen[..., None], axis=-1)[..., 0]
    est_delta_c = np.take_along_axis(dc, chosen[..., None], axis=-1)[..., 0]
    est_delta_o = np.take_along_axis(do, chosen[..., None], axis=-1)[..., 0]
    oracle_delta_c_of_est = np.take_along_axis(oc, chosen[..., None], axis=-1)[..., 0]
    oracle_delta_o_of_est = np.take_along_axis(oo, chosen[..., None], axis=-1)[..., 0]
    return {
        'chosen': chosen, 'oracle_chosen': oracle_chosen,
        'est_best': est_best, 'oracle_best': oracle_best,
        'est_continue': est_best > eps, 'oracle_continue': oracle_best > eps,
        'oracle_selected_utility': oracle_utility_of_est_action,
        'est_selected_dc': est_delta_c, 'est_selected_do': est_delta_o,
        'oracle_selected_dc': oracle_delta_c_of_est,
        'oracle_selected_do': oracle_delta_o_of_est,
    }


def summarize(d, x, selector):
    # selector is (model, state, target), always nonempty
    sel = np.asarray(selector, dtype=bool)
    n = int(np.count_nonzero(sel))
    if not n:
        return None
    ec = x['est_continue'][sel]
    oc = x['oracle_continue'][sel]
    ub = x['est_best'][sel]
    est_cur_c = d['est_cur_overall'][sel]
    est_cur_o = d['est_cur_over'][sel]
    # STOP distributions refer to cross-sectional historical states, not sequential trajectories
    stopped = ~ec
    positive = ub > 0
    row = {
        'n_cases': n,
        'estimated_continue_fraction': float(ec.mean()),
        'estimated_stop_fraction': float(stopped.mean()),
        'oracle_continue_fraction': float(oc.mean()),
        'oracle_stop_fraction': float((~oc).mean()),
        'decision_match_fraction': float((ec == oc).mean()),
        'false_continue_fraction_all': float((ec & ~oc).mean()),
        'false_stop_fraction_all': float((~ec & oc).mean()),
        'n_estimated_stop': int(stopped.sum()),
        'nonpositive_utility_stop_fraction_all': float((~positive).mean()),
        'epsilon_only_additional_stop_fraction_all': float((positive & stopped).mean()),
        'oracle_negative_selected_action_fraction_among_est_continue': float(
            np.mean(x['oracle_selected_utility'][sel][ec] <= 0)) if ec.any() else float('nan'),
        'oracle_below_epsilon_selected_action_fraction_among_est_continue': float(
            np.mean(x['oracle_selected_utility'][sel][ec] <= x['epsilon'])) if ec.any() else float('nan'),
        'est_best_utility_mean': float(ub.mean()),
        'est_best_utility_p05': float(np.percentile(ub, 5)),
        'est_best_utility_median': float(np.median(ub)),
        'mean_est_selected_delta_overall_among_continue': float(x['est_selected_dc'][sel][ec].mean()) if ec.any() else float('nan'),
        'mean_est_selected_delta_over_among_continue': float(x['est_selected_do'][sel][ec].mean()) if ec.any() else float('nan'),
        'mean_oracle_selected_delta_overall_among_continue': float(x['oracle_selected_dc'][sel][ec].mean()) if ec.any() else float('nan'),
        'mean_oracle_selected_delta_over_among_continue': float(x['oracle_selected_do'][sel][ec].mean()) if ec.any() else float('nan'),
    }
    for label, vals in [('estimated_current_overall_at_stop', est_cur_c[stopped]),
                        ('estimated_current_over_at_stop', est_cur_o[stopped]),
                        ('oracle_current_overall_at_estimated_stop', d['oracle_cur_overall'][sel][stopped]),
                        ('oracle_current_over_at_estimated_stop', d['oracle_cur_over'][sel][stopped])]:
        row[label+'_mean'] = float(vals.mean()) if vals.size else float('nan')
        row[label+'_median'] = float(np.median(vals)) if vals.size else float('nan')
        row[label+'_p25'] = float(np.percentile(vals,25)) if vals.size else float('nan')
        row[label+'_p75'] = float(np.percentile(vals,75)) if vals.size else float('nan')
    return row


def first_stop_rows(d, x, mode_i, target_i):
    """One virtual first-stop record per historical hole-target sequence.

    Later fixed-plan observations are not treated as counterfactual adaptive states.
    """
    hole_uids = d['hole_uids'].astype(str)
    pulses = d['current_pulses'].astype(int)
    ans = []
    for hole in np.unique(hole_uids):
        rows = np.where(hole_uids == hole)[0]
        rows = rows[np.argsort(pulses[rows])]
        stops = rows[~x['est_continue'][mode_i, rows, target_i]]
        if stops.size:
            ix = int(stops[0])
            ans.append({
                'hole_uid':str(hole), 'target_id':str(d['target_ids'][target_i]),
                'mode':str(d['modes'][mode_i]),
                'first_stop_current_pulse':int(pulses[ix]),
                'first_stop_step': int(pulses[ix] // 5),
                'first_stop_est_overall':float(d['est_cur_overall'][mode_i,ix,target_i]),
                'first_stop_est_over':float(d['est_cur_over'][mode_i,ix,target_i]),
                'first_stop_oracle_overall':float(d['oracle_cur_overall'][mode_i,ix,target_i]),
                'first_stop_oracle_over':float(d['oracle_cur_over'][mode_i,ix,target_i]),
                'first_stop_est_best_utility':float(x['est_best'][mode_i,ix,target_i]),
                'first_stop_type':('nonpositive' if x['est_best'][mode_i,ix,target_i] <= 0 else 'epsilon_only'),
            })
        else:
            ans.append({
                'hole_uid':str(hole),'target_id':str(d['target_ids'][target_i]),
                'mode':str(d['modes'][mode_i]),
                'first_stop_current_pulse':'none_in_p5_to_p45', 'first_stop_step':'',
                'first_stop_est_overall':'','first_stop_est_over':'',
                'first_stop_oracle_overall':'','first_stop_oracle_over':'',
                'first_stop_est_best_utility':'','first_stop_type':'no_stop',
            })
    return ans


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows to write for {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        writer=csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def run(args):
    d, meta = load_cache(Path(args.cache))
    m, n, t, a = validate_cache(d, meta, args.expected_split)
    epsilons = sorted(set(float(x) for x in args.epsilons))
    lambdas = sorted(set(float(x) for x in args.lambdas))
    smalls = sorted(set(float(x) for x in args.small_utility_cutoffs))
    if (not epsilons or not lambdas or not smalls or
        any(not np.isfinite(x) or x < 0 for x in [*epsilons, *lambdas, *smalls])):
        raise ValueError('lambda, epsilon, and small cutoffs must be nonnegative finite lists')
    if 0.0 not in epsilons:
        raise ValueError('Include epsilon=0 so epsilon-only STOP is interpretable')
    out=Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    overall=[]; by_target=[]; by_pulse=[]; small=[]; small_progress=[]; first=[]; first_summary=[]
    all_mask=np.ones((m,n,t),dtype=bool)
    for lam in lambdas:
        base = evaluate(d,lam,0)
        for cutoff in smalls:
            for im, mode in enumerate(d['modes']):
                mask = np.zeros((m,n,t),dtype=bool);mask[im]=True
                positive = (base['est_best']>0) & (base['est_best']<=cutoff) & mask
                is_continue=base['est_best']>0
                small.append({
                    'mode':str(mode),'lambda':lam,'small_utility_cutoff':cutoff,
                    'n_cases':n*t,
                    'n_positive_but_small':int(positive.sum()),
                    'fraction_all_cases':float(positive.sum()/(n*t)),
                    'fraction_among_epsilon0_continue':float(positive.sum()/is_continue[mask].sum()),
                    'mean_est_delta_overall_selected_in_small':float(base['est_selected_dc'][positive].mean()) if positive.any() else float('nan'),
                    'mean_est_delta_over_selected_in_small':float(base['est_selected_do'][positive].mean()) if positive.any() else float('nan'),
                    'oracle_model_nonpositive_fraction_in_small':float((base['oracle_selected_utility'][positive] <= 0).mean()) if positive.any() else float('nan'),
                })
                # Distinguish tiny *net utility* from truly tiny Delta-overall.
                est_dc=base['est_selected_dc'][mask]
                util=base['est_best'][mask]
                small_dc=(est_dc<=cutoff) & (util>0)
                max_all_dc=np.max(deltas(d, 'est')[0][im],axis=-1)
                small_progress.append({
                    'mode':str(mode), 'lambda':lam, 'delta_overall_cutoff':cutoff,
                    'n_cases':n*t,
                    'n_continuing_with_selected_delta_overall_at_or_below_cutoff':int(small_dc.sum()),
                    'fraction_all_cases':float(small_dc.mean()),
                    'fraction_among_epsilon0_continue':float(small_dc.sum()/(util>0).sum()),
                    'n_cases_all_27_actions_delta_overall_at_or_below_cutoff':int((max_all_dc<=cutoff).sum()),
                    'fraction_all_actions_stagnant_by_delta_overall':float((max_all_dc<=cutoff).mean()),
                })
        for eps in epsilons:
            x = evaluate(d,lam,eps)
            x['epsilon'] = eps
            for im,mode in enumerate(d['modes']):
                mask=np.zeros((m,n,t),dtype=bool);mask[im]=True
                common={'mode':str(mode),'lambda':lam,'epsilon':eps}
                overall.append({**common, **summarize(d,x,mask)})
                for ti, target in enumerate(d['target_ids']):
                    q=np.zeros((m,n,t),dtype=bool);q[im,:,ti]=True
                    by_target.append({**common,'target_id':str(target),**summarize(d,x,q)})
                for p in sorted(set(map(int,d['current_pulses']))):
                    q=mask & (d['current_pulses'][None,:,None]==p)
                    by_pulse.append({**common,'current_pulse':p,**summarize(d,x,q)})
                one=[]
                for ti in range(t):
                    one.extend(first_stop_rows(d,x,im,ti))
                for r in one:
                    first.append({ 'lambda':lam, 'epsilon':eps, **r})
                stopped=[r for r in one if r['first_stop_type']!='no_stop']
                v_c=np.asarray([r['first_stop_est_overall'] for r in stopped], dtype=float)
                v_o=np.asarray([r['first_stop_est_over'] for r in stopped], dtype=float)
                first_summary.append({
                    **common,'n_hole_target_trajectories':len(one),
                    'n_with_virtual_stop':len(stopped),
                    'fraction_with_virtual_stop':len(stopped)/len(one),
                    'n_no_stop_observed_p5_to_p45':len(one)-len(stopped),
                    'first_stop_median_overall_est':float(np.median(v_c)) if v_c.size else float('nan'),
                    'first_stop_median_over_est':float(np.median(v_o)) if v_o.size else float('nan'),
                    'first_stop_mean_overall_est':float(v_c.mean()) if v_c.size else float('nan'),
                    'first_stop_mean_over_est':float(v_o.mean()) if v_o.size else float('nan'),
                    **{f'first_stop_at_p{p}':sum(r['first_stop_current_pulse']==p for r in stopped) for p in sorted(set(map(int,d['current_pulses'])))},
                })
    write_csv(out/'continuation_summary_by_model.csv',overall)
    write_csv(out/'continuation_summary_by_target.csv',by_target)
    write_csv(out/'continuation_summary_by_pulse.csv',by_pulse)
    write_csv(out/'small_positive_utility.csv',small)
    write_csv(out/'small_progress_analysis.csv',small_progress)
    write_csv(out/'historical_first_stop_summary.csv',first_summary)
    write_csv(out/'historical_first_stop_cases.csv',first)
    write_json(out/'resolved_continuation_sensitivity.json',{
        'source_cache':str(Path(args.cache).resolve()),'split':meta['split'],
        'num_states':n,'num_unique_holes':len(np.unique(d['hole_uids'])),
        'num_targets':t,'num_modes':m,'num_actions':a,
        'lambdas':lambdas,'epsilons':epsilons,'small_utility_cutoffs':smalls,
        'score':'U(a)=Delta C(a)-lambda*Delta O(a), Delta are fractions normalized by target area',
        'action':'argmax U_est, tie -> first candidate in cache action order',
        'estimated_stop':'max(U_est)<=epsilon',
        'oracle_stop':'max(U_oracle)<=epsilon, where oracle uses measured current and trained Transition',
        'epsilon_only':'0 < max(U_est) <= epsilon',
        'stop_at_time':'Cross-sectional states are independently evaluated. A separate first-stop diagnostic '
                       'uses fixed-plan historical hole sequences, NOT model-rolled-out closed-loop trajectories.',
        'excluded_policies':'No max-steps, overall completion threshold, over hard safety threshold, '
                            'stagnation, PC1 hardware gates. This output MUST NOT be used as an execution permission.',
        'estimated_over_underprediction':'Known issue from separate calibration/validation; STOP agreement is not a physical safety guarantee',
        'test_split':'Do not optimize lambda/epsilon from test data',
        'source_meta':meta,
    })
    print(f'PASS: split={meta["split"]}, states={n}, holes={len(np.unique(d["hole_uids"]))}, modes={m}, targets={t}, actions={a}')
    print('Written:',out)
    for row in overall:
        if row['mode'] in ('shock_metadata','shock_action_history') and row['epsilon'] in (0.,0.005,0.01):
            print(f'{row["mode"]:23} lambda={row["lambda"]:g} eps={row["epsilon"]:g} '
                  f'CONTINUE={row["estimated_continue_fraction"]:.4f} '
                  f'STOP={row["estimated_stop_fraction"]:.4f} '
                  f'match={row["decision_match_fraction"]:.4f} '
                  f'eps_only_STOP={row["epsilon_only_additional_stop_fraction_all"]:.4f}')
    return overall


def parse_args(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache',required=True,help='Policy sensitivity ZIP or NPZ with adjacent metadata JSON')
    ap.add_argument('--output',required=True,help='New output directory')
    ap.add_argument('--expected-split',choices=['val','test'],default='val',
                    help='Default VAL. Do not use test to select lambda/epsilon.')
    ap.add_argument('--lambdas',nargs='+',type=float,default=DEFAULT_LAMBDAS)
    ap.add_argument('--epsilons',nargs='+',type=float,default=DEFAULT_EPSILONS)
    ap.add_argument('--small-utility-cutoffs',nargs='+',type=float,default=DEFAULT_SMALL)
    return ap.parse_args(argv)


if __name__=='__main__':
    run(parse_args())
