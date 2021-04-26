from typing import List

from telescope.metrics.result import MetricResult


class ZeroEditResult(MetricResult):
    def __init__(
        self,
        sys_score: float,
        seg_scores: List[float],
        src: List[str],
        cand: List[str],
        ref: List[str],
        metric: str,
        exact_matches: float,
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref, metric)
        self.exact_matches = exact_matches

    def __str__(self):
        return f"{self.metric}({self.sys_score}, exact_matches = {self.exact_matches})"
