from typing import List

from telescope.metrics.result import MetricResult


class PrismResult(MetricResult):
    def __init__(
        self,
        sys_score: float,
        seg_scores: List[float],
        src: List[str],
        cand: List[str],
        ref: List[str],
        metric: str,
        forward_score: float,
        reverse_score: float,
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref, metric)
        self.forward_score = forward_score
        self.reverse_score = reverse_score

    def __str__(self):
        return "{}({}, Forward={}, Reverse={})".format(
            self.metric, self.sys_score, self.forward_score, self.reverse_score
        )
