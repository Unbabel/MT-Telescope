import unittest

from telescope.metrics.bertscore.metric import BERTScore


class TestBertScore(unittest.TestCase):
    
    bertscore = BERTScore("en")

    def test_score(self):
        
        cand = ['Hi world.', 'This is a Test.']
        ref = ['Hello world.', 'This is a test.']
        src = ['Bonjour le monde.', "C'est un test."]
        result = self.bertscore.score(src, cand, ref)
        expected_seg = [0.9965, 0.9759]
        expected_sys = 0.9862
        self.assertAlmostEqual(result.sys_score, expected_sys, places=4)
        for i in range(2):
            self.assertAlmostEqual(result.seg_scores[i], expected_seg[i], places=4)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, src)
        self.assertListEqual(result.cand, cand)

    def test_name_property(self):
        self.assertEqual(self.bertscore.name, "BERTScore")