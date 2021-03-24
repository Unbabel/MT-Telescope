import abc
from typing import List, Tuple


class MetricResult(metaclass=abc.ABCMeta):
    def __init__(
        self,
        sys_score: int,
        seg_scores: List[float],
        src: List[str],
        cand: List[str],
        ref: List[str],
    ) -> None:
        self.sys_score = sys_score
        self.seg_scores = seg_scores
        self.src = src
        self.ref = ref
        self.cand = cand


class PairwiseResult:
    def __init__(
        self,
        x_result: MetricResult,
        y_result: MetricResult,
        metric: str,
        system_only: bool
    ) -> None:
        self.x_result = x_result
        self.y_result = y_result
        assert self.x_result.src == self.y_result.src
        assert self.x_result.ref == self.y_result.ref
        self.metric = metric
    
    @property
    def src(self):
        return self.x_result.src
    
    @property
    def ref(self):
        return self.x_result.ref

    @property
    def system_x(self):
        return self.x_result.cand

    @property
    def system_y(self):
        return self.y_result.cand


class BootstrapResult:
    def __init__(
        self,
        x_scores: List[float],
        y_scores: List[float],
        win_count: Tuple[int],
        num_samples: int,
        metric: str,
    ):
        self.x_scores = x_scores
        self.y_scores = y_scores
        self.win_count = win_count
        self.metric = metric
        self.x_stats = {
            "mean": np.mean(self.x_scores),
            "median": np.median(self.x_scores),
            "lower_bound": self.x_scores[int(num_samples * 0.025)],
            "upper_bound": self.x_scores[int(num_samples * 0.975)],
        }
        self.y_stats = {
            "mean": np.mean(self.y_scores),
            "median": np.median(self.y_scores),
            "lower_bound": self.y_scores[int(num_samples * 0.025)],
            "upper_bound": self.y_scores[int(num_samples * 0.975)],
        }