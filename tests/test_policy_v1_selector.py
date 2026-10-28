"""Unit tests for the live Policy v1.0 selector (legacy offline selector is separate)."""
import numpy as np
import pytest
from runtime.selector import candidate_table_policy_v1, fractions
from runtime.action_registry import Action
from runtime.decision_policy import apply_live_candidate_decision


def base():
    target = np.zeros((10, 20), dtype=np.uint8)
    target[1:5, 1:11] = 1 # 40 pixels, so 4 extra pixels = over 0.10
    current = np.zeros_like(target)
    current[1:3, 1:11] = 1 # 0.5 coverage
    actions = [Action(f'A{i+1:02d}', 10+i, 0, 0) for i in range(3)]
    return target, current, actions


def evaluate(target, current, actions, predictions, over_limit=.1, lam=3):
    return candidate_table_policy_v1(np.stack(predictions), target, actions, .5, fractions(current,target), lam, over_limit)


def test_scores_clamp_shrinkage_and_choose_lowest_policy_loss():
    t, cur, actions = base()
    p1=cur.copy(); p1[3:4, 1:11] = 1  # C + 0.25, O +0 => L=-.25
    p2=cur.copy(); p2[3:5, 1:11] = 1  # C + 0.50, O+3/40 = .075, L=-.275
    p2[5, 1:4] = 1
    p3=cur.copy()                       # L=0
    rows, best, nsafe, no_safe = evaluate(t,cur,actions,[p1,p2,p3])
    assert (nsafe, no_safe) == (3, False)
    assert best['action_id'] == 'A02'
    assert best['delta_overall'] == .5
    assert best['delta_over'] == 3/40
    assert best['policy_loss'] == pytest.approx(-.275)
    assert apply_live_candidate_decision({}, best, nsafe, no_safe)['decision']=='continue'


def test_boundary_0p10_excluded_even_if_perfect_coverage():
    t,cur,actions=base()
    p1=cur.copy();p1[3:5,1:11]=1;p1[5,1:5]=1  # next over=4/40=.1
    p2=cur.copy();p2[3,1:3]=1           # L negative, safe
    p3=cur.copy();p3[5,1:15]=1          # over .35
    rows,best,n,no=evaluate(t,cur,actions,[p1,p2,p3])
    assert n==1 and not no
    assert rows[0]['is_safe'] is False
    assert best['action_id']=='A02'
    assert rows[0]['selected'] is False


def test_all_unsafe_no_fallback():
    t,cur,actions=base()
    pred=cur.copy();pred[5,1:5]=1
    rows,best,n,no=evaluate(t,cur,actions,[pred]*3)
    assert n==0 and no and best is None
    result=apply_live_candidate_decision({},best,n,no)
    assert result['decision']=='stop' and result['best_action'] is None


def test_all_safe_but_not_beneficial_stops():
    t,cur,actions=base()
    unchanged=cur.copy()
    shrunk=cur.copy();shrunk[1,:]=0
    over=cur.copy();over[5,1:2]=1
    rows,best,n,no=evaluate(t,cur,actions,[unchanged,shrunk,over])
    assert n==3
    assert best['policy_loss']==0
    assert rows[1]['delta_overall_raw']<0 and rows[1]['delta_overall']==0
    assert apply_live_candidate_decision({},best,n,no)['stop_reason']=='nonpositive_utility'


def test_exact_policy_tie_uses_action_registry_order():
    t,cur,actions=base()
    p=cur.copy();p[3,1:3]=1
    rows,best,n,no=evaluate(t,cur,actions,[p,p,p])
    assert best['candidate_index']==0
    assert sum(row['selected'] for row in rows)==1


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_nonfinite_predictions_fail_closed(bad):
    t,cur,actions=base()
    p=np.stack([cur.copy()]*3).astype(float);p[0,0,0]=bad
    with pytest.raises(ValueError,match='Nonfinite'):
        candidate_table_policy_v1(p,t,actions,.5,fractions(cur,t),3,.1)


def test_registry_count_mismatch_fails_closed():
    t,cur,actions=base()
    with pytest.raises(ValueError,match='shape/count mismatch'):
        evaluate(t,cur,actions[:2],[cur]*3)
