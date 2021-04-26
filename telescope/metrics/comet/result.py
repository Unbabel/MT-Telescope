from telescope.metrics.result import MetricResult


class COMETResult(MetricResult):
    def __init__(
        self,
        sys_score: int,
        seg_scores: list,
        src: list,
        cand: list,
        ref: list,
        metric: str,
        model: str,
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref, metric)
        self.model = model

    def __str__(self):
        return f"{self.metric}({self.sys_score}, Model={self.model})"
