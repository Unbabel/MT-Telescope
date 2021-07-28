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

from telescope.metrics.result import MetricResult


class BLEUResult(MetricResult):
    def __init__(
        self,
        sys_score: float,
        seg_scores: List[float],
        src: List[str],
        cand: List[str],
        ref: List[str],
        metric: str,
        precisions: List[float],
        brevity_penalty: float,
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref, metric)
        self.precisions = precisions
        self.brevity_penalty = brevity_penalty

    def __str__(self):
        return f"{self.metric}({self.sys_score}, Precisions = {self.precisions}, Brevity Penalty = {self.brevity_penalty})"
