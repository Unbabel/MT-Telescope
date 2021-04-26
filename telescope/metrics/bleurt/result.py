from typing import List

from telescope.metrics.result import MetricResult


class BLEURTResult(MetricResult):
    def __init__(
        self,
        sys_score: float,
        seg_scores: List[float],
        src: List[str],
        cand: List[str],
        ref: List[str],
        metric: str,
        model: str,
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref, metric)
        self.model = model

    def __str__(self):
        return f"{self.metric}({self.sys_score}, Model={self.model})"
