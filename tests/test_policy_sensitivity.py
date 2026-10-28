import numpy as np

from validation.policy_sensitivity_analysis import _aggregate_slice, _branch_policy


def test_branch_policy_priority_and_no_safe():
    # [state,target] = [1,1], two candidate actions.
    cur_overall = np.array([[0.90]], dtype=np.float32)
    cur_over = np.array([[0.02]], dtype=np.float32)
    cost = np.array([[[0.2, 0.1]]], dtype=np.float32)
    cand_over = np.array([[[0.06, 0.07]]], dtype=np.float32)

    # With over_limit=0.05, current state is active but no candidate is safe -> STOP no-safe.
    out = _branch_policy(cur_overall, cur_over, cost, cand_over, 0.95, 0.05, np.array([5]), 20)
    assert not bool(out["continue"][0, 0])
    assert int(out["reason"][0, 0]) == 4

    # Current over-removal has priority over target-reached.
    out = _branch_policy(
        np.array([[0.99]], dtype=np.float32),
        np.array([[0.08]], dtype=np.float32),
        cost,
        cand_over,
        0.95,
        0.05,
        np.array([5]),
        20,
    )
    assert int(out["reason"][0, 0]) == 1


def test_aggregate_detects_false_continue_and_oracle_unsafe_action():
    # Estimated current looks safe and continues; oracle current is already over limit.
    est_cur_overall = np.array([[0.50]], dtype=np.float32)
    est_cur_over = np.array([[0.01]], dtype=np.float32)
    oracle_cur_overall = np.array([[0.50]], dtype=np.float32)
    oracle_cur_over = np.array([[0.08]], dtype=np.float32)

    est_cost = np.array([[[0.1, 0.2]]], dtype=np.float32)
    est_cand_overall = np.array([[[0.7, 0.8]]], dtype=np.float32)
    est_cand_over = np.array([[[0.02, 0.03]]], dtype=np.float32)
    oracle_cost = np.array([[[0.2, 0.1]]], dtype=np.float32)
    oracle_cand_overall = np.array([[[0.6, 0.7]]], dtype=np.float32)
    oracle_cand_over = np.array([[[0.06, 0.04]]], dtype=np.float32)

    row = _aggregate_slice(
        "m", 0.95, 0.05, np.array([5]),
        est_cur_overall, est_cur_over, oracle_cur_overall, oracle_cur_over,
        est_cost, est_cand_overall, est_cand_over,
        oracle_cost, oracle_cand_overall, oracle_cand_over,
        20,
    )
    assert row["false_continue_fraction"] == 1.0
    assert row["false_continue_due_oracle_over_fraction"] == 1.0
