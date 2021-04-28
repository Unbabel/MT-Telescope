import os
import unittest

from telescope.filters import DuplicatesFilter
from telescope.testset import PairwiseTestset
from tests.data import DATA_PATH


class TestDuplicatesFilter(unittest.TestCase):

    hyp = ["a", "b", "cd", "hello"]
    alt = ["A", "b", "cD", "hello"]
    src = ["A", "A", "cD", "A"]
    ref = ["a", "b", "cd", "hello"]

    testset = PairwiseTestset(
        src, hyp, alt, ref, "de-en", ["src.de", "hyp.en", "alt.hyp.en", "ref.en"]
    )

    def test_sucess_filter(self):
        filter = DuplicatesFilter(self.testset)
        orig_size = len(self.testset)
        self.testset.apply_filter(filter)
        self.assertEqual(len(self.testset), 2)
        self.assertTrue(len(self.testset) < orig_size)
        src, x, y, ref = self.testset[0]
        self.assertEqual(src, "A")
        src, x, y, ref = self.testset[1]
        self.assertEqual(src, "cD")
