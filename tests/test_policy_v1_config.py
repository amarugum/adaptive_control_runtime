from pathlib import Path
import math
import pytest
import yaml
from runtime.config import load_runtime_config


def base(tmp_path):
    return {
        'storage': {'session_root': str(tmp_path/'archive'), 'transfer_root': str(tmp_path/'transfer')},
        'resources': {'action_set_id':'action27_v1','target_set_id':'target_set_v1'},
        'models': {},
        'action_selection': {'scoring_method':'delta_area_v1','lambda_over':3.0,'candidate_threshold':.5,'no_safe_action_policy':'stop'},
        'stop_policy': {'overall_removal_target':.9,'over_removal_limit':.1,'max_control_steps':20},
    }


def load(tmp_path, obj):
    p=tmp_path/'runtime.yaml'; p.write_text(yaml.safe_dump(obj),encoding='utf-8')
    return load_runtime_config(p)


def test_expected_policy_v1_parameters(tmp_path):
    cfg=load(tmp_path,base(tmp_path))
    assert cfg['action_selection']['scoring_method']=='delta_area_v1'
    assert cfg['action_selection']['lambda_over']==3
    assert cfg['stop_policy']['overall_removal_target']==.9
    assert cfg['stop_policy']['over_removal_limit']==.1
    assert cfg['stop_policy']['max_control_steps']==20


def test_reject_old_l1_dice_config_not_silently_using_new_score(tmp_path):
    obj=base(tmp_path)
    obj['action_selection']={'l1_weight':1,'dice_weight':1,'candidate_threshold':.5,'no_safe_action_policy':'stop'}
    with pytest.raises(ValueError,match='Legacy L1\\+Dice'):
        load(tmp_path,obj)


@pytest.mark.parametrize('patch,pattern', [
    ({'scoring_method':'legacy'},'scoring_method'),
    ({'lambda_over':float('nan')},'lambda_over'),
    ({'lambda_over':0},'lambda_over'),
    ({'candidate_threshold':0},'candidate_threshold'),
    ({'candidate_threshold':1.1},'candidate_threshold'),
])
def test_reject_invalid_selection(tmp_path,patch,pattern):
    obj=base(tmp_path)
    obj['action_selection'].update(patch)
    with pytest.raises(ValueError,match=pattern):
        load(tmp_path,obj)


@pytest.mark.parametrize('patch,pattern', [
    ({'overall_removal_target':1.2},'overall_removal_target'),
    ({'over_removal_limit':0},'over_removal_limit'),
    ({'max_control_steps':0},'max_control_steps'),
])
def test_reject_invalid_stop_config(tmp_path,patch,pattern):
    obj=base(tmp_path)
    obj['stop_policy'].update(patch)
    with pytest.raises(ValueError,match=pattern):
        load(tmp_path,obj)
