import unittest

from telescope.metrics.bertscore.metric import BERTScore

cands = [
    "28-year-old chef found dead in San Francisco mall",
    "A 28-year-old chef who recently moved to San Francisco was found dead in the staircase of a local shopping center.",
    'The victim\'s brother said he cannot imagine anyone who would want to harm him,"Finally, it went uphill again at him."',
]
refs = [
    "28-Year-Old Chef Found Dead at San Francisco Mall",
    "A 28-year-old chef who had recently moved to San Francisco was found dead in the stairwell of a local mall this week.",
    "But the victim's brother says he can't think of anyone who would want to hurt him, saying, \"Things were finally going well for him.\"",
]


class TestBertScore(unittest.TestCase):

    bertscore = BERTScore(language="en")

    def test_score(self):
        result = self.bertscore.score(["test", "nonesense"], cands, refs)
        expected_seg = [0.9833560585975647, 0.9782299995422363, 0.916214644908905]
        expected_sys = 0.9592669010162354
        expected_precision = [
            0.9843301773071289,
            0.9832239747047424,
            0.9120385646820068,
        ]
        expected_recall = [0.9823838472366333, 0.9732863903045654, 0.9204290509223938]

        self.assertAlmostEqual(result.sys_score, expected_sys, places=4)
        for i in range(3):
            self.assertAlmostEqual(result.precision[i], expected_precision[i], places=4)
        for i in range(3):
            self.assertAlmostEqual(result.recall[i], expected_recall[i], places=4)
        for i in range(3):
            self.assertAlmostEqual(result.f1[i], expected_seg[i], places=4)

        self.assertListEqual(result.ref, refs)
        self.assertListEqual(result.src, ["test", "nonesense"])
        self.assertListEqual(result.cand, cands)
        self.assertTrue(self.bertscore.segment_level)

    def test_name_property(self):
        self.assertEqual(self.bertscore.name, "BERTScore")
