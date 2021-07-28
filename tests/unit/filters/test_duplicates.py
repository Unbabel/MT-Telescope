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
import unittest

from telescope.filters import DuplicatesFilter
from telescope.testset import PairwiseTestset


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
