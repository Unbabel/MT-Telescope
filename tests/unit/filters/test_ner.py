import os
import unittest

from telescope.filters.ner import NERFilter
from telescope.testset import PairwiseTestset
from tests.data import DATA_PATH


class TestNERFilter(unittest.TestCase):

    hyp = [s.strip() for s in open(os.path.join(DATA_PATH, "hyp.en")).readlines()]
    alt = [s.strip() for s in open(os.path.join(DATA_PATH, "alt.hyp.en")).readlines()]
    src = [s.strip() for s in open(os.path.join(DATA_PATH, "src.de")).readlines()]
    ref = [s.strip() for s in open(os.path.join(DATA_PATH, "ref.en")).readlines()]
    testset = PairwiseTestset(
        src, hyp, alt, ref, "de-en", ["src.de", "hyp.en", "alt.hyp.en", "ref.en"]
    )

    def test_sucess_filter(self):
        filter = NERFilter(self.testset)
        new_testset = filter.apply_filter()
        self.assertEqual(len(new_testset), 1)
        self.assertTrue(len(new_testset) < len(self.testset))
        src, x, y, ref = new_testset[0]
        self.assertEqual(ref, "I love to live in Lisbon")

    def test_unsuported_language(self):
        self.testset.language_pair = "pt-ja"
        with self.assertRaises(Exception) as cm:
            NERFilter(self.testset)
        the_exception = cm.exception
        self.assertEqual(str(the_exception), "pt-ja is not supperted by Stanza NER.")
