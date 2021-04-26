import unittest

import torch
from telescope.metrics.prism.metric import Prism

torch.cuda.is_available = lambda: False


class TestPrism(unittest.TestCase):

    prism = Prism("en")

    def test_score(self):
        cand = ["Hi world.", "This is a Test."]
        ref = ["Hello world.", "This is a test."]
        src = ["Bonjour le monde.", "C'est un test."]

        result = self.prism.score(src, cand, ref)
        expected_seg = [-1.4878583, -0.5490748]
        expected_sys = -1.0184666

        self.assertAlmostEqual(result.sys_score, expected_sys, places=4)
        for i in range(2):
            self.assertAlmostEqual(result.seg_scores[i], expected_seg[i], places=4)

        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, src)
        self.assertListEqual(result.cand, cand)

    def test_name_property(self):
        self.assertEqual(self.prism.name, "Prism")

    def test_language_support(self):
        self.assertTrue(self.prism.language_support("en"))
        self.assertFalse(self.prism.language_support("is"))
        with self.assertRaises(Exception) as cm:
            Prism("is")
        the_exception = cm.exception
        self.assertEqual(str(the_exception), "is is not supported by Prism.")
