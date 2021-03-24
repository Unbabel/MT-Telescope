from telescope.metrics.result import MetricResult


class PrismResult(MetricResult):
    def __init__(
        self, sys_score: int, seg_scores: list, src: list, cand: list, ref: list
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref)