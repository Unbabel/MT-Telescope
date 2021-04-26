from telescope.metrics.result import MetricResult


class chrFResult(MetricResult):
    def __str__(self):
        return f"{self.metric}({self.sys_score}, Beta = 2, ngram_order = 6)"
