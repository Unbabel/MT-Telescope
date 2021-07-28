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

import torch
from telescope.metrics import COMET

torch.cuda.is_available = lambda: False


class TestCOMET(unittest.TestCase):

    comet = COMET(modelname="wmt21-small-da-152012")

    def test_score(self):
        # README example!
        src = [
            "Dem Feuer konnte Einhalt geboten werden",
            "Schulen und Kindergärten wurden eröffnet.",
        ]
        cand = ["The fire could be stopped", "Schools and kindergartens were open"]
        ref = [
            "They were able to control the fire.",
            "Schools and kindergartens opened",
        ]
        expected_seg = [-0.14773446321487427, 1.0864498615264893]
        expected_sys = 0.4693576991558075
        result = self.comet.score(src, cand, ref)

        self.assertAlmostEqual(result.sys_score, expected_sys, places=4)
        for i in range(2):
            self.assertAlmostEqual(result.seg_scores[i], expected_seg[i], places=4)
        self.assertListEqual(result.ref, ref)
        self.assertListEqual(result.src, src)
        self.assertListEqual(result.cand, cand)

    def test_name_property(self):
        self.assertEqual(self.comet.name, "COMET")
