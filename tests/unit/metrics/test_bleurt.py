import unittest

from telescope.metrics.bleurt.metric import BLEURT


class TestBLEURT(unittest.TestCase):
    
    bleurt = BLEURT("bleurt-tiny-512")

    def test_score(self):
        
        cand = ['Hi world.', 'This is a Test.']
        ref = ['Hello world.', 'This is a test.']
        src = ['Bonjour le monde.', "C'est un test."]
        
        result = self.bleurt.score(src, cand, ref)
        expected_seg = [0.2378692328929901, 1.0849038362503052]
        expected_sys = 0.6613865345716476
        self.assertAlmostEqual(result.sys_score, expected_sys, places=4)
        for i in range(2):
            self.assertAlmostEqual(result.seg_scores[i], expected_seg[i], places=4)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, src)
        self.assertListEqual(result.cand, cand)

    def test_name_property(self):
        self.assertEqual(self.bleurt.name, "BLEURT")