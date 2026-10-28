"""Dependency-light tests for the offline continuation-value/epsilon analysis."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

FILE = Path(__file__).resolve().parents[1] / 'validation' / 'continuation_value_sensitivity.py'
spec = importlib.util.spec_from_file_location('continuation_value_sensitivity', FILE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def fixture_data():
    # 1 model, 3 physical states, 1 target, 2 actions, with nontrivial oracle disagreement
    d = {
        'modes':np.array(['shock_metadata']),
        'target_ids':np.array(['T1']),
        'action_ids':np.array(['A01','A02']),
        'hole_uids':np.array(['h001','h001','h002']),
        'current_pulses':np.array([5,10,5]),
        'est_cur_overall':np.array([[[0.2],[0.4],[0.5]]]).reshape(1,3,1).astype('f4'),
        'est_cur_over':np.zeros((1,3,1),dtype='f4'),
        'oracle_cur_overall':np.array([.25,.45,.55],dtype='f4').reshape(1,3,1),
        'oracle_cur_over':np.zeros((1,3,1),dtype='f4'),
    }
    d['est_cand_overall']=d['est_cur_overall'][...,None]+np.array([[[[.05,.02]],[[.01,.01]],[[.01,.03]]]],dtype='f4')
    d['est_cand_over']=np.array([[[[.01,0]],[[.01,.01]],[[.01,.03]]]],dtype='f4')
    d['oracle_cand_overall']=d['oracle_cur_overall'][...,None]+np.array([[[[.01,.04]],[[.01,.01]],[[.005,.01]]]],dtype='f4')
    d['oracle_cand_over']=np.array([[[[.01,0]],[[.005,.005]],[[.02,.01]]]],dtype='f4')
    return d, {'split':'val','target_ids':['T1'],'action_ids':['A01','A02']}


def test_cache_shape_and_split():
    d,meta=fixture_data()
    assert analysis.validate_cache(d,meta,'val') == (1,3,1,2)
    with pytest.raises(ValueError,match='split'):
        analysis.validate_cache(d,meta,'test')


def test_score_maximization_and_tie_breaking():
    d,_=fixture_data()
    x=analysis.evaluate(d,2,0)
    assert x['chosen'].shape==(1,3,1)
    # state 0: A01 score=.05 -2*.01=.03; A02 score=.02
    assert int(x['chosen'][0,0,0])==0
    # state 1: equal scores for both actions; first selected
    assert int(x['chosen'][0,1,0])==0
    # state 2: max score is negative, so STOP
    assert not bool(x['est_continue'][0,2,0])


def test_epsilon_additional_stop():
    d,_=fixture_data()
    # state1 max score = -.01, state2 = -.01, state0 = +.03
    x=analysis.evaluate(d,2,0.04)
    assert not x['est_continue'].any()
    x['epsilon']=0.04
    s=analysis.summarize(d,x,np.ones((1,3,1),dtype=bool))
    assert s['n_estimated_stop']==3
    assert abs(s['epsilon_only_additional_stop_fraction_all']-1/3)<1e-8
    assert abs(s['nonpositive_utility_stop_fraction_all']-2/3)<1e-8


def test_oracle_is_not_selected_action():
    d,_=fixture_data()
    x=analysis.evaluate(d,2,0)
    # oracle chooses A02 at state0; estimated chooses A01
    assert int(x['oracle_chosen'][0,0,0])==1
    assert int(x['chosen'][0,0,0])==0
    assert x['oracle_selected_utility'][0,0,0] < x['oracle_best'][0,0,0]


def test_virtual_first_stop_is_sorted_by_pulse():
    d,_=fixture_data()
    order=np.array([1,0,2])
    for name in ('hole_uids','current_pulses'):
        d[name]=d[name][order]
    for name in ('est_cur_overall','est_cur_over','oracle_cur_overall','oracle_cur_over',
                 'est_cand_overall','est_cand_over','oracle_cand_overall','oracle_cand_over'):
        d[name]=d[name][:,order]
    x=analysis.evaluate(d,2,0)
    rows=analysis.first_stop_rows(d,x,0,0)
    assert len(rows)==2
    assert next(r for r in rows if r['hole_uid']=='h001')['first_stop_current_pulse']==10
    assert next(r for r in rows if r['hole_uid']=='h002')['first_stop_current_pulse']==5


def test_negative_increment_rejected():
    d,meta=fixture_data()
    d['est_cand_overall'][0,0,0,0]=0.1
    with pytest.raises(ValueError,match='monotonicity'):
        analysis.validate_cache(d,meta,'val')


def test_nonfinite_and_invalid_epsilon():
    d,meta=fixture_data()
    with pytest.raises(ValueError,match='nonnegative'):
        analysis.evaluate(d,3,-0.01)
    d['est_cand_over'][0,0,0,0]=np.nan
    with pytest.raises(ValueError,match='expected finite'):
        analysis.validate_cache(d,meta,'val')


def test_io_zip_integration(tmp_path):
    d,meta=fixture_data()
    cache=tmp_path/'policy_candidate_cache.npz'
    np.savez_compressed(cache,**d)
    (tmp_path/'policy_cache_metadata.json').write_text(json.dumps(meta),encoding='utf-8')
    rows=analysis.run(SimpleNamespace(cache=str(cache), output=str(tmp_path/'out'),
        expected_split='val',lambdas=[2,3],epsilons=[0,0.005],small_utility_cutoffs=[.001,.005]))
    assert len(rows)==4
    assert (tmp_path/'out'/'historical_first_stop_cases.csv').is_file()
    with ZipFile(tmp_path/'policy.zip','w') as zipfile:
        zipfile.write(cache,'root/policy_candidate_cache.npz')
        zipfile.write(tmp_path/'policy_cache_metadata.json','root/policy_cache_metadata.json')
    loaded, loaded_meta=analysis.load_cache(tmp_path/'policy.zip')
    assert loaded_meta['split']=='val'
    assert loaded['est_cand_over'].shape==(1,3,1,2)
