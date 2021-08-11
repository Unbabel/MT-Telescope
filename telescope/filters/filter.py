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
import abc
from typing import List

from telescope.testset import Testset


class Filter:
    name = None

    def __init__(self, testset: Testset, *args):
        self.testset = testset

    @abc.abstractmethod
    def apply_filter(self) -> List[int]:
        """ Returns the indexes of elements to keep """
        return NotImplementedError
