import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from runtime.selector import fractions,candidate_table
from runtime.action_registry import Action

def test_fractions():
    t=np.zeros((256,256),np.uint8); t[10:20,10:20]=1
    s=t.copy(); assert fractions(s,t)['overall_removal_fraction']==1.0

def test_safe_selection():
    t=np.zeros((256,256),np.float32); t[10:20,10:20]=1
    p1=np.zeros_like(t); p1[10:19,10:20]=1
    p2=np.zeros_like(t); p2[10:20,10:20]=1
    acts=[Action('A1',10,0,0),Action('A2',15,0,0)]
    rows,best,n,no=candidate_table(np.stack([p1,p2]),t,acts,0.5,1,1,0.05)
    assert best['action_id']=='A2' and n==2 and not no

def test_no_safe_fallback_min_over_then_cost():
    t=np.zeros((256,256),np.float32); t[10:20,10:20]=1
    # Both candidates exceed 5% over-removal relative to |T|=100.
    # A1 has 10 over pixels; A2 has 20 over pixels, so fallback must choose A1
    # regardless of the secondary shape cost.
    p1=t.copy(); p1[20,10:20]=1
    p2=t.copy(); p2[20:22,10:20]=1
    acts=[Action('A1',10,0,0),Action('A2',15,0,0)]
    rows,best,n,no=candidate_table(np.stack([p1,p2]),t,acts,0.5,1,1,0.05)
    assert n==0 and no
    assert best['action_id']=='A1'
