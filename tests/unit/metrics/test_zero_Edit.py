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
