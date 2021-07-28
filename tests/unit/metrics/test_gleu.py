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
