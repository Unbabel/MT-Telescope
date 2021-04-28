import unittest

from telescope.metrics.sacrebleu.metric import sacreBLEU


class TestSacreBLEU(unittest.TestCase):

    bleu = sacreBLEU(language="en")

    def test_score(self):

        cand = ["Hi world.", "This is a Test."]
        ref = ["Hello world.", "This is a test."]
        src = ["Bonjour le monde.", "C'est un test."]  # Will be ignored

        expected_sys = 0.3913
        result = self.bleu.score(src, cand, ref)
        self.assertAlmostEqual(result.sys_score, expected_sys, places=3)
        self.assertFalse(result.seg_scores)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, src)
        self.assertListEqual(result.cand, cand)

    def test_name_property(self):
        self.assertEqual(self.bleu.name, "BLEU")
