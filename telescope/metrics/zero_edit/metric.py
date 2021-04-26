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
