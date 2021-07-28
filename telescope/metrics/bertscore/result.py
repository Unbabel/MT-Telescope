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


class BERTScoreResult(MetricResult):
    def __init__(
        self,
        sys_score: float,
        seg_scores: List[float],
        src: List[str],
        cand: List[str],
        ref: List[str],
        metric: str,
        precision: List[float],
        recall: List[float],
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref, metric)
        self.precision = precision
        self.recall = recall
        self.f1 = seg_scores

    def __str__(self):
        precision_score = sum(self.precision) / len(self.precision)
        recall_score = sum(self.recall) / len(self.recall)
        return "{}(F1 = {:.5f}, Precision = {:.5f}, Recall = {:.5f})".format(
            self.metric, self.sys_score, precision_score, recall_score
        )
