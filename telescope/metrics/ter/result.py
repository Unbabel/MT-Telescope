from typing import List

from telescope.metrics.result import MetricResult


class TERResult(MetricResult):
    def __init__(
        self,
        sys_score: float,
        seg_scores: List[float],
        src: List[str],
        cand: List[str],
        ref: List[str],
        metric: str,
        num_edits: float,
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref, metric)
        self.num_edits = num_edits

    def __str__(self):
        return f"{self.metric}({self.sys_score}, num_edits={self.num_edits})"
