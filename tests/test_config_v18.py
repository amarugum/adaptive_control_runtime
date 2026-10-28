from pathlib import Path
import pytest
import yaml

from runtime.config import load_runtime_config


def _base(tmp_path: Path):
    return {
        'storage': {
            'session_root': str(tmp_path/'session'),
            'transfer_root': str(tmp_path/'transfer'),
        },
        'resources': {'action_set_id':'action27_v1','target_set_id':'target_set_v1'},
        'paths': {'ml_repo':'../laser_processing_ml'},
        'models': {
            'action_history_only': {
                'state_estimator_run':'se',
                'transition_run':'tr',
                'performance': {
                    'estimator_amp': False,
                    'transition_amp': True,
                    'candidate_chunk_size': 12,
                },
            }
        },
        'action_selection': {'no_safe_action_policy':'stop'},
    }


def test_v18_performance_settings_are_loaded(tmp_path: Path):
    p=tmp_path/'adaptive_runtime_config.yaml'
    p.write_text(yaml.safe_dump(_base(tmp_path),sort_keys=False),encoding='utf-8')
    cfg=load_runtime_config(p)
    perf=cfg['models']['action_history_only']['performance']
    assert perf['estimator_amp'] is False
    assert perf['transition_amp'] is True
    assert perf['candidate_chunk_size']==12
    assert cfg['action_selection']['no_safe_action_policy']=='stop'


def test_v18_rejects_legacy_unsafe_fallback_policy(tmp_path: Path):
    obj=_base(tmp_path)
    obj['action_selection']['no_safe_action_policy']='min_over_then_cost'
    p=tmp_path/'adaptive_runtime_config.yaml'
    p.write_text(yaml.safe_dump(obj,sort_keys=False),encoding='utf-8')
    with pytest.raises(ValueError,match="must be 'stop'"):
        load_runtime_config(p)
