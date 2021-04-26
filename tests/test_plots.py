import unittest
import os

from telescope.plotting import (
    plot_segment_comparison,
    plot_pairwise_distributions,
    plot_bucket_comparison,
)
from telescope.metrics.result import PairwiseResult, MetricResult
from tests.data import DATA_PATH


class TestPlots(unittest.TestCase):

    result = PairwiseResult(
        x_result=MetricResult(
            sys_score=0.5,
            seg_scores=[0, 0.5, 1],
            src=["a", "b", "c"],
            cand=["a", "b", "c"],
            ref=["a", "b", "c"],
            metric="mock",
        ),
        y_result=MetricResult(
            sys_score=0.25,
            seg_scores=[0, 0.25, 0.5],
            src=["a", "b", "c"],
            cand=["a", "k", "c"],
            ref=["a", "b", "c"],
            metric="mock",
        ),
    )

    @classmethod
    def tearDownClass(cls):
        os.remove(DATA_PATH + "/segment-comparison.html")
        os.remove(DATA_PATH + "/scores-distribution.html")
        os.remove(DATA_PATH + "/bucket-analysis.png")

    def test_segment_comparison(self):
        plot_segment_comparison(self.result, DATA_PATH)
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "segment-comparison.html"))
        )

    def test_pairwise_distributions(self):
        plot_pairwise_distributions(self.result, DATA_PATH)
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "scores-distribution.html"))
        )

    def test_bucket_comparison(self):
        plot_bucket_comparison(self.result, DATA_PATH)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "bucket-analysis.png")))
