import unittest

from telescope.metrics.chrf.metric import chrF


class TestchrF(unittest.TestCase):

        chrf = chrF()

    def test_score(self):
        
        cand = ['Hi world.', 'This is a Test.']
        ref = ['Hello world.', 'This is a test.']
        src = ['Bonjour le monde.', "C'est un test."]

        expected_sys = 0.528
        result = self.chrf.score(src, cand, ref)
        self.assertAlmostEqual(result.sys_score, expected_sys, places=2)
        self.assertIsNone(result.seg_scores)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, src)
        self.assertListEqual(result.cand, cand)
        
    def test_name_property(self):
        self.assertEqual(self.chrf.name, "chrF")