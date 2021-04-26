from typing import List

from telescope.metrics.result import MetricResult


class BERTScoreResult(MetricResult):
    def __init__(
        self,
        sys_score: float,
        seg_scores: List[float],
        src: List[str],
        cand: List[str],
        ref: List[str],
        metric: str,
        precision: List[float],
        recall: List[float],
    ) -> None:
        super().__init__(sys_score, seg_scores, src, cand, ref, metric)
        self.precision = precision
        self.recall = recall
        self.f1 = seg_scores

    def __str__(self):
        precision_score = sum(self.precision) / len(self.precision)
        recall_score = sum(self.recall) / len(self.recall)
        return "{}(F1 = {:.5f}, Precision = {:.5f}, Recall = {:.5f})".format(
            self.metric, self.sys_score, precision_score, recall_score
        )
