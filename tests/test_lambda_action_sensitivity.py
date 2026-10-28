import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'validation'))
from lambda_action_sensitivity import choose, choose_oracle, take, validate_cache


class LambdaSweepTests(unittest.TestCase):
    def test_tradeoff(self):
        # A: more overall but more over, B: less overall but almost zero over
        d={'est_dc':np.array([[[0.10,0.06]]]),'est_do':np.array([[[0.02,0.001]]]),
           'oracle_dc':np.array([[[0.10,0.06]]]),'oracle_do':np.array([[[0.02,0.001]]])}
        self.assertEqual(choose(d,0)[0,0],0)
        self.assertEqual(choose(d,1)[0,0],0)
        self.assertEqual(choose(d,3)[0,0],1)
        self.assertEqual(choose_oracle(d,3)[0,0],1)

    def test_normalization_invariant(self):
        d={'est_dc':np.array([[[10.,6.,1.]]]),'est_do':np.array([[[2.,0.1,0.]]])}
        a=choose(d,3)
        d2={k:v/2000. for k,v in d.items()}
        np.testing.assert_array_equal(a,choose(d2,3))

    def test_exact_ties_first_action(self):
        d={'est_dc':np.array([[[.1,.1,.1]]]),'est_do':np.array([[[.05,.05,.05]]])}
        self.assertEqual(choose(d,2)[0,0],0)

    def test_gather(self):
        z=np.array([[[2.,5.],[8.,11.]]])
        i=np.array([[1,0]])
        np.testing.assert_array_equal(take(z,i),np.array([[5.,8.]]))

    def test_cache_split_guard(self):
        d = self._mini_cache()
        with self.assertRaisesRegex(ValueError,'Cache split is'):
            validate_cache(d,{'split':'test','target_ids':['T1'],'action_ids':['A01','A02']},'val')
        self.assertEqual(validate_cache(d,{'split':'val','target_ids':['T1'],'action_ids':['A01','A02']},'val'),1)

    def test_nonmonotonic_cache_rejected(self):
        d=self._mini_cache()
        d['est_cand_over'][0,0,0,0] = -0.01
        with self.assertRaisesRegex(ValueError,'negative processed-area increment'):
            validate_cache(d,{'split':'val','target_ids':['T1'],'action_ids':['A01','A02']},'val')

    @staticmethod
    def _mini_cache():
        d={'modes':np.array(['metadata_only']), 'target_ids':np.array(['T1']),
           'action_ids':np.array(['A01','A02']), 'hole_uids':np.array(['h0001']),
           'current_pulses':np.array([5])}
        for k in ('est_cur_overall','est_cur_over','oracle_cur_overall','oracle_cur_over'):
            d[k]=np.zeros((1,1,1))
        for k in ('est_cand_overall','est_cand_over','oracle_cand_overall','oracle_cand_over','est_cost'):
            d[k]=np.zeros((1,1,1,2))
        return d


if __name__=='__main__':
    unittest.main()
