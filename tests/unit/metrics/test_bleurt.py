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

from telescope.metrics.bleurt.metric import BLEURT


class TestBLEURT(unittest.TestCase):

    bleurt = BLEURT(language="en", model="bleurt-tiny-512")

    def test_score(self):
        cand = ["Hi world.", "This is a Test."]
        ref = ["Hello world.", "This is a test."]
        src = ["Bonjour le monde.", "C'est un test."]

        result = self.bleurt.score(src, cand, ref)
        expected_seg = [0.2378692328929901, 1.0849038362503052]
        expected_sys = 0.6613865345716476
        self.assertAlmostEqual(result.sys_score, expected_sys, places=4)
        for i in range(2):
            self.assertAlmostEqual(result.seg_scores[i], expected_seg[i], places=4)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, src)
        self.assertListEqual(result.cand, cand)

    def test_name_property(self):
        self.assertEqual(self.bleurt.name, "BLEURT")

    def test_language_support(self):
        self.assertTrue(self.bleurt.language_support("en"))
        self.assertFalse(self.bleurt.language_support("de"))
