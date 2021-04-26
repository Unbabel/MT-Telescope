import abc
from typing import List, Tuple
import pandas as pd
import numpy as np


class MetricResult(metaclass=abc.ABCMeta):
    def __init__(
        self,
        sys_score: int,
        seg_scores: List[float],
        src: List[str],
        cand: List[str],
        ref: List[str],
        metric: str,
    ) -> None:
        self.sys_score = sys_score
        self.seg_scores = seg_scores
        self.src = src
        self.ref = ref
        self.cand = cand
        self.metric = metric

    def segment_level(self):
        if not self.seg_scores:
            return False
        return True

    def __str__(self):
        return f"{self.metric}({self.sys_score})"


class PairwiseResult:
    def __init__(
        self,
        x_result: MetricResult,
        y_result: MetricResult,
    ) -> None:

        self.x_result = x_result
        self.y_result = y_result
        assert self.x_result.metric == self.y_result.metric
        assert self.x_result.src == self.y_result.src
        assert self.x_result.ref == self.y_result.ref
        self.metric = self.x_result.metric

    @property
    def src(self) -> List[str]:
        return self.x_result.src

    @property
    def ref(self) -> List[str]:
        return self.x_result.ref

    @property
    def system_x(self) -> List[str]:
        return self.x_result.cand

    @property
    def system_y(self) -> List[str]:
        return self.y_result.cand

    @staticmethod
    def results_to_dataframe(pairwise_results: list) -> pd.DataFrame:
        summary = {
            "x": [p_res.x_result.sys_score for p_res in pairwise_results],
            "y": [p_res.y_result.sys_score for p_res in pairwise_results],
        }
        df = pd.DataFrame.from_dict(summary)
        df.index = [p_res.metric for p_res in pairwise_results]
        return df

    @staticmethod
    def results_to_dict(pairwise_results: list) -> pd.DataFrame:
        return {
            p_res.metric: {"x": p_res.x_result.sys_score, "y": p_res.y_result.sys_score}
            for p_res in pairwise_results
        }


class BootstrapResult:
    def __init__(
        self,
        x_scores: List[float],
        y_scores: List[float],
        win_count: Tuple[int],
        metric: str,
    ):
        self.x_scores = x_scores
        self.y_scores = y_scores
        self.win_count = win_count
        self.metric = metric
        self.stats = {
            "x_wins (%)": win_count[0] / sum(win_count),
            "y_wins (%)": win_count[1] / sum(win_count),
            "ties (%)": win_count[2] / sum(win_count),
            "x-mean": np.mean(self.x_scores),
            "y-mean": np.mean(self.y_scores),
        }
