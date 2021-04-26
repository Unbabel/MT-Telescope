import os
import unittest

from telescope.metrics.gleu.metric import GLEU
from tests.data import DATA_PATH


class TestGLEU(unittest.TestCase):
    cand = [l.strip() for l in open(os.path.join(DATA_PATH, "hyp1_100.no")).readlines()]
    ref = [l.strip() for l in open(os.path.join(DATA_PATH, "ref_100.no")).readlines()]

    def test_name_property(self):
        self.assertEqual(GLEU(language="en", lowercase=True).name, "GLEU")

    def test_bleu_truecase(self):
        ref = [
            "It is a guide to action that ensures that the military "
            "will forever heed Party commands"
        ]
        hyp1 = [
            "It is a guide to action which ensures that the military "
            "always obeys the commands of the party"
        ]
        expected_result = 0.4393

        gleu = GLEU(language="en", lowercase=False, tokenize=False)
        result = gleu.score([], ref, hyp1)
        self.assertAlmostEqual(expected_result, result.sys_score, places=3)
