from telescope.metrics.result import MetricResult


class BLEUResult(MetricResult):
    def __init__(
        self, sys_score: int, src: list, cand: list, ref: list
    ) -> None:
        super().__init__(sys_score, None, src, cand, ref)
