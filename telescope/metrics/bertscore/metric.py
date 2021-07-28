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

import bert_score
from telescope.metrics.bertscore.result import BERTScoreResult
from telescope.metrics.metric import Metric


class BERTScore(Metric):

    name = "BERTScore"
    segment_level = True

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> BERTScoreResult:
        precision, recall, f1 = bert_score.score(
            cands=cand,
            refs=ref,
            idf=False,
            batch_size=3,
            lang=self.language,
            rescale_with_baseline=False,
            verbose=True,
        )
        return BERTScoreResult(
            sum(f1.tolist()) / len(f1.tolist()),
            f1.tolist(),
            src,
            cand,
            ref,
            self.name,
            precision.tolist(),
            recall.tolist(),
        )
