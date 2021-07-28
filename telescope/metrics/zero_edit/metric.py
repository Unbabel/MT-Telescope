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
from typing import List

from telescope.metrics.metric import Metric
from telescope.metrics.zero_edit.result import ZeroEditResult


class ZeroEdit(Metric):

    name = "ZeroEdit"
    segment_level = True

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> ZeroEditResult:
        scores = []
        for h, r in zip(cand, ref):
            scores.append(1 if h == r else 0)

        return ZeroEditResult(
            sum(scores) / len(scores), scores, src, cand, ref, self.name, sum(scores)
        )
