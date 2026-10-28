"""No hardware/GPU: integration tests of both STOP stages and response artifacts."""
import csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from unittest.mock import patch

from runtime.action_registry import Action
from runtime.engine import AdaptiveEngine


class FakeRuntime:
    def __init__(self, current, candidates):
        self.current = current
        self.candidates = candidates
        self.transition_called = 0
        self.state_ckpt = Path('fake_estimator.pt')
        self.transition_ckpt = Path('fake_transition.pt')

    def infer_state(self, request, shock):
        return torch.from_numpy(self.current.astype(np.float32))[None,None], None

    def predict_candidates(self, current_binary, z, actions):
        self.transition_called += 1
        assert len(actions) == 27
        return self.candidates.astype(np.float32)

    def performance_settings(self):
        return {'candidate_chunk_size':6,'estimator_amp':False,'transition_amp':False,'device':'cpu'}


def fixture(tmp_path, k=1, request_max=20):
    target=np.zeros((256,256),dtype=np.uint8)
    target[20:30,20:30]=1
    current=np.zeros_like(target)
    current[20:23,20:30]=1  # 30% coverage
    preds=np.broadcast_to(current.copy(), (27,256,256)).copy()
    model=FakeRuntime(current,preds)
    engine=AdaptiveEngine.__new__(AdaptiveEngine)
    engine.output_root=tmp_path
    engine.cfg={'action_selection':{'candidate_threshold':.5,'lambda_over':3.0,'scoring_method':'delta_area_v1'},
                'stop_policy':{'overall_removal_target':.9,'over_removal_limit':.1,'max_control_steps':20}}
    engine.registry=SimpleNamespace(path=tmp_path/'actions.csv',sha256='somesha',actions=[Action(f'A{i+1:02d}',10.,0.,0.) for i in range(27)])
    engine.targets=SimpleNamespace(load=lambda tid:(target,'targethash'),path_for=lambda tid:tmp_path/'target.npy',registry_csv=tmp_path/'targets.csv')
    engine._runtime=lambda mode:model
    req={'request_id':'r1','experiment':{'experiment_id':'exp_v1','hole_uid':'h0001'},
         'model':{'mode':'metadata_only'},'target':{'target_id':'T001'},
         'step':{'completed_control_step':k,'pulse_end':k*5},
         'control':{'action_registry_sha256':'somesha','max_control_steps':request_max}}
    return target,current,preds,model,engine,req


def run(engine,req):
    with patch('runtime.engine.ensure_experiment_snapshot', return_value=Path('unused')):
        out=engine.process(req)
    return out


def test_engine_stage1_over_precedes_completion_and_skips_transition(tmp_path):
    target,current,preds,model,engine,req=fixture(tmp_path)
    current[20:30,20:30]=1 # perfect coverage
    current[30,20:30]=1    # 0.10 over
    out=run(engine,req)
    assert out['decision']=='stop' and out['stop_reason']=='over_removal'
    assert model.transition_called==0
    assert out['best_action'] is None
    assert out['candidate_evaluation'] is None
    assert out['policy']['over_removal_limit']==.1


def test_engine_stage1_completion_and_max_steps(tmp_path):
    target,current,preds,model,engine,req=fixture(tmp_path)
    current[20:29,20:30]=1
    out=run(engine,req)
    assert out['stop_reason']=='target_reached' and model.transition_called==0
    target,current,preds,model,engine,req=fixture(tmp_path/'next',k=21,request_max=50)
    out=run(engine,req)
    assert out['stop_reason']=='max_control_steps' and model.transition_called==0


def test_engine_stage2_no_safe_at_exact_limit(tmp_path):
    target,current,preds,model,engine,req=fixture(tmp_path)
    preds[:,30,20:30]=1 # O_next == 0.10 -> excluded
    out=run(engine,req)
    assert out['decision']=='stop' and out['stop_reason']=='no_safe_action'
    assert out['best_action'] is None and model.transition_called==1
    assert out['candidate_evaluation']['safe_candidate_count']==0
    assert len(list(csv.DictReader((tmp_path/'exp_v1'/'h0001'/'k01'/'candidate_scores.csv').open(encoding='utf-8-sig'))))==27


def test_engine_stage2_nonpositive_utility(tmp_path):
    target,current,preds,model,engine,req=fixture(tmp_path)
    out=run(engine,req)
    assert out['stop_reason']=='nonpositive_utility'
    assert out['candidate_evaluation']['best_policy_loss']==0
    assert out['best_action'] is None


def test_engine_stage2_skips_unsafe_high_gain_then_selects_safe(tmp_path):
    target,current,preds,model,engine,req=fixture(tmp_path)
    # High-gain A01 has next over==0.10 and MUST be excluded.
    preds[0,23:30,20:30]=1
    preds[0,30,20:30]=1
    # Safe A02, +20% current target coverage, no over
    preds[1,23:25,20:30]=1
    out=run(engine,req)
    assert out['decision']=='continue' and out['stop_reason'] is None
    assert out['best_action']['action_id']=='A02'
    assert out['best_action']['predicted_policy_loss']==pytest.approx(-.2)
    assert out['best_action']['predicted_delta_overall']==pytest.approx(.2)
    assert out['policy']['lambda_over']==3
    p=tmp_path/'exp_v1'/'h0001'/'k01'/'candidate_scores.csv'
    rows=list(csv.DictReader(p.open(encoding='utf-8-sig')))
    assert rows[0]['is_safe']=='False'
    assert rows[1]['selected']=='True'


def test_engine_fails_closed_on_nonfinite_state(tmp_path):
    target,current,preds,model,engine,req=fixture(tmp_path)
    model.current=current.astype(float)
    model.current[0,0]=np.nan
    with pytest.raises(ValueError,match='NONFINITE_STATE_PROBABILITY'):
        run(engine,req)
    assert model.transition_called==0
