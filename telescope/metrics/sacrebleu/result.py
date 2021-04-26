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
