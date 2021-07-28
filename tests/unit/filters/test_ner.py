# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
        orig_size = len(self.testset)
        self.testset.apply_filter(filter)
        self.assertEqual(len(self.testset), 1)
        self.assertTrue(len(self.testset) < orig_size)
        src, x, y, ref = self.testset[0]
        self.assertEqual(ref, "I love to live in Lisbon")

    def test_unsuported_language(self):
        self.testset.language_pair = "pt-ja"
        with self.assertRaises(Exception) as cm:
            NERFilter(self.testset)
        the_exception = cm.exception
        self.assertEqual(str(the_exception), "pt-ja is not supperted by Stanza NER.")
