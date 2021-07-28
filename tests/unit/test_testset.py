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

from telescope.testset import PairwiseTestset


class TestTestset(unittest.TestCase):
    
    testset = PairwiseTestset(
        src=['Bonjour le monde.', "C'est un test."],
        system_x=['Greetings world', 'This is an experiment.'],
        system_y=['Hi world.', 'This is a Test.'],
        ref=['Hello world.', 'This is a test.'],
        language_pair="en-fr",
        filenames=["src.txt", "google.txt", "unbabel.txt", "ref.txt"]
    )

    def test_length(self):
        self.assertEqual(len(self.testset), 2)

    def test_get_item(self):
        expected = (
            'Bonjour le monde.',
            'Greetings world',
            'Hi world.',
            'Hello world.'
        )
        self.assertTupleEqual(expected, self.testset[0])
    