import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation'))
from preview_target_completion_overlays import fractions,resize_nearest,rgb_overlay

class TestOverlays(unittest.TestCase):
    def test_exact_metric(self):
        t=np.array([[1,1],[0,0]],dtype=bool)
        m=np.array([[1,0],[1,0]],dtype=bool)
        d=fractions(m,t)
        self.assertEqual(d['overall'],.5)
        self.assertEqual(d['over'],.5)
        self.assertAlmostEqual(d['iou'],1/3)
    def test_resize_no_translate(self):
        m=np.array([[0,1],[0,0]],dtype=np.uint8)
        a=resize_nearest(m,4)
        self.assertTrue(np.array_equal(a,[[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]]))
    def test_rgb_types(self):
        t=np.ones((2,2),bool);m=np.zeros((2,2),bool)
        self.assertEqual(rgb_overlay(t,m).shape,(2,2,3))

if __name__=='__main__':unittest.main()
