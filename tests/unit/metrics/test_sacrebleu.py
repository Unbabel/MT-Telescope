import unittest

from telescope.metrics.sacrebleu.metric import BLEU


class TestBLEU(unittest.TestCase):
    
    bleu = BLEU()

    def test_score(self):
        
        cand = ['Hi world.', 'This is a Test.']
        ref = ['Hello world.', 'This is a test.']
        src = ['Bonjour le monde.', "C'est un test."]

        expected_sys = 39.13
        result = self.bleu.score(src, cand, ref)
        self.assertAlmostEqual(result.sys_score, expected_sys, places=2)
        self.assertIsNone(result.seg_scores)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, src)
        self.assertListEqual(result.cand, cand)
    
    def test_name_property(self):
        self.assertEqual(self.bleu.name, "sacreBLEU")