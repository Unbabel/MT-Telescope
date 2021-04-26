import unittest

from telescope.metrics.zero_edit.metric import ZeroEdit


class TestSacreBLEU(unittest.TestCase):

    zero_edit = ZeroEdit(language="en")

    def test_score(self):
        cand = ["Hi world.", "This is a Test."]
        ref = ["Hello world.", "This is a Test."]

        expected_sys = 1 / 2
        expected_seg = [0, 1]

        result = self.zero_edit.score(None, cand, ref)
        self.assertAlmostEqual(result.sys_score, expected_sys, places=2)
        self.assertListEqual(result.seg_scores, expected_seg)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.cand, cand)

    # def test_name_property(self):
    #    self.assertEqual(self.zero_edit.name, "ZeroEdit")

    def test_name_property(self):
        self.assertEqual(ZeroEdit.name, "ZeroEdit")
