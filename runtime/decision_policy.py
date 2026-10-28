from __future__ import annotations


def apply_live_candidate_decision(base: dict, best: dict | None, safe_count: int, no_safe: bool) -> dict:
    """Policy v1.0: fail closed on no-safe or non-beneficial actions.

    Only a safe action with L_min < 0 may be returned to PC1. The current
    state has already passed the estimator-stage STOP checks in engine.py.
    """
    if no_safe != (safe_count == 0):
        raise ValueError('Inconsistent safe candidate count')
    if not no_safe and best is None:
        raise ValueError('Missing best candidate despite safe candidates')

    base['candidate_evaluation'] = {
        'safe_candidate_count': int(safe_count),
        'no_safe_action': bool(no_safe),
        'selected_from_safe_set': bool(best is not None and best['policy_loss'] < 0.0),
        'no_safe_action_policy': 'stop',
        'best_policy_loss': float(best['policy_loss']) if best is not None else None,
        'best_predicted_utility': float(best['predicted_utility']) if best is not None else None,
    }
    base['best_action'] = None
    if no_safe:
        base['decision'] = 'stop'
        base['stop_reason'] = 'no_safe_action'
        return base

    if not best['is_safe']:
        raise ValueError('Unsafe action marked best; fail closed')
    if best['policy_loss'] >= 0.0:
        base['decision'] = 'stop'
        base['stop_reason'] = 'nonpositive_utility'
        return base

    base['decision'] = 'continue'
    base['stop_reason'] = None
    base['best_action'] = {
        'action_id': best['action_id'],
        'energy_uJ': best['energy_uJ'],
        'action_y_um': best['action_y_um'],
        'action_z_um': best['action_z_um'],
        # Backwards-compatible field name; now refers to Policy v1.0 loss L.
        'predicted_cost': best['policy_loss'],
        'predicted_policy_loss': best['policy_loss'],
        'predicted_utility': best['predicted_utility'],
        'predicted_delta_overall': best['delta_overall'],
        'predicted_delta_over': best['delta_over'],
        'predicted_overall_removal_fraction': best['predicted_overall_removal_fraction'],
        'predicted_over_removal_fraction': best['predicted_over_removal_fraction'],
    }
    return base
