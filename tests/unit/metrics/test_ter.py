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

from telescope.metrics.ter.metric import TER


class TestTER(unittest.TestCase):

    ter = TER(language="en")

    def test_score(self):

        cand = ["aaaa bbbb cccc dddd", "aaaa bbbb cccc", "d e f g h a b c"]
        ref = ["aaaa bbbb cccc dddd", "dddd eeee ffff", "a b c d e f g h"]
        src = []  # Will be ignored

        expected_sys = (0 + 3 + 1) / 15
        result = self.ter.score(src, cand, ref)
        self.assertAlmostEqual(result.sys_score, expected_sys, places=2)
        self.assertFalse(result.seg_scores)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, src)
        self.assertListEqual(result.cand, cand)

    def test_name_property(self):
        self.assertEqual(self.ter.name, "TER")
