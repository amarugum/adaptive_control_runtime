from runtime.decision_policy import apply_live_candidate_decision


def _best(loss=-0.12):
    return {
        'action_id': 'A07',
        'energy_uJ': 15.0,
        'action_y_um': 0.0,
        'action_z_um': -5.0,
        'policy_loss': loss,
        'predicted_utility': -loss,
        'delta_overall': 0.15,
        'delta_over': 0.01,
        'predicted_overall_removal_fraction': 0.80,
        'predicted_over_removal_fraction': 0.06,
        'is_safe': True,
    }


def test_no_safe_action_stops_and_never_exposes_fallback_action():
    out = apply_live_candidate_decision({}, None, 0, True)
    assert out['decision'] == 'stop'
    assert out['stop_reason'] == 'no_safe_action'
    assert out['best_action'] is None
    assert out['candidate_evaluation']['no_safe_action'] is True
    assert out['candidate_evaluation']['no_safe_action_policy'] == 'stop'


def test_safe_candidate_continues_with_best_action():
    out = apply_live_candidate_decision({}, _best(), 5, False)
    assert out['decision'] == 'continue'
    assert out['stop_reason'] is None
    assert out['best_action']['action_id'] == 'A07'
    assert out['candidate_evaluation']['safe_candidate_count'] == 5
    assert out['best_action']['predicted_cost'] == -0.12
    assert out['best_action']['predicted_utility'] == 0.12


def test_safe_candidates_with_nonnegative_loss_must_stop():
    for loss in (0, 0.0001):
        out = apply_live_candidate_decision({}, _best(loss), 2, False)
        assert out['decision'] == 'stop'
        assert out['stop_reason'] == 'nonpositive_utility'
        assert out['best_action'] is None


def test_unsafe_best_rejected():
    import pytest
    best = _best()
    best['is_safe'] = False
    with pytest.raises(ValueError, match='Unsafe'):
        apply_live_candidate_decision({}, best, 1, False)
