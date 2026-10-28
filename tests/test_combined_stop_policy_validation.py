"""No hardware dependencies; tests to prevent unsafe action selection."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
from combined_stop_policy_validation import policy,REASONS


def toy(c=.3,o=.05, cand_c=None,cand_o=None):
    if cand_c is None:cand_c=[.4,.5,.4]
    if cand_o is None:cand_o=[.06,.50,.08]
    current_c=np.array([[[c]]]); current_o=np.array([[[o]]]); nc=np.asarray(cand_c)[None,None,None,:];no=np.asarray(cand_o)[None,None,None,:]
    d={}
    for name in ['est','oracle']:
        d[name+'_cur_overall']=current_c;d[name+'_cur_over']=current_o
        d[name+'_cand_overall']=nc;d[name+'_cand_over']=no
    return d

class TestCombinedStopPolicy(unittest.TestCase):
    def reason(self,d,lam=3,comp=.9,olim=.15):
        return policy(d,'est',lam,comp,olim)['reason'][0,0,0]
    def test_first_stage_over(self):
        self.assertEqual(self.reason(toy(o=.20)),'current_over_limit')
    def test_first_stage_complete(self):
        self.assertEqual(self.reason(toy(c=.95)),'target_reached')
    def test_over_has_priority_when_both(self):
        self.assertEqual(self.reason(toy(c=.95,o=.30)),'current_over_limit')
    def test_no_safe_action_is_stop(self):
        self.assertEqual(self.reason(toy(cand_o=[.16,.20,.90])),'no_safe_action')
        self.assertEqual(int(policy(toy(cand_o=[.16,.20,.90]),'est',3,.9,.15)['chosen'][0,0,0]),-1)
    def test_do_not_pick_best_if_unsafe(self):
        res=policy(toy(cand_c=[.5,.95,.4],cand_o=[.07,.30,.09]),'est',1,.9,.15)
        self.assertTrue(bool(res['continue'][0,0,0]))
        self.assertEqual(int(res['chosen'][0,0,0]),0)
    def test_nonpositive_utility_stop(self):
        self.assertEqual(self.reason(toy(cand_c=[.301,.302,.303],cand_o=[.07,.08,.09])),'nonpositive_utility')
    def test_safe_negative_vs_bad_unsafe(self):
        d=toy(cand_c=[.301,.95,.31],cand_o=[.1,.5,.11])
        self.assertEqual(self.reason(d),'nonpositive_utility')
    def test_strict_over_limit(self):
        self.assertEqual(self.reason(toy(cand_o=[.15,.15,.15])),'no_safe_action')
    def test_positive_action(self):
        self.assertEqual(self.reason(toy()),'continue')

if __name__=='__main__':unittest.main()
