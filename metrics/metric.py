import abc

import numpy as np
import streamlit as st
from testset import PairedTestset

from metrics.result import BootstrapResult, PairedResult


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    @st.cache
    def score(self, sources, hypothesis, references, **kwargs):
        pass

    @st.cache
    def score_paired_testset(self, testset, **kwargs):
        with st.spinner(f"Running {self.name} for system X"):
            x_result = self.score(
                testset.sources, testset.system_x, testset.references, **kwargs
            )
        with st.spinner(f"Running {self.name} for system Y"):
            y_result = self.score(
                testset.sources, testset.system_y, testset.references, **kwargs
            )
        return PairedResult(x_result, y_result, self.name)

    @st.cache
    def paired_bootstrap(
        self, testset, num_samples=1000, sample_ratio=0.5, **kwargs
    ) -> None:
        def update_wins(x_score, y_score, wins):
            if y_score > x_score:
                wins[1] += 1
            elif y_score < x_score:
                wins[0] += 1
            else:
                wins[2] += 1
            return wins

        n = len(testset)
        ids = list(range(n))
        sample_size = int(n * sample_ratio)

        x_scores, y_scores = [], []
        wins = [0, 0, 0]
        for _ in range(num_samples):
            # Subsample the gold and system outputs (with replacement)
            reduced_ids = np.random.choice(ids, size=sample_size, replace=True)
            # Calculate accuracy on the reduced sample and save stats
            reduced_src = [testset[i][0] for i in reduced_ids]
            reduced_x = [testset[i][1] for i in reduced_ids]
            reduced_y = [testset[i][2] for i in reduced_ids]
            reduced_ref = [testset[i][3] for i in reduced_ids]

            paired_result = self.score_paired_testset(
                PairedTestset(reduced_src, reduced_x, reduced_y, reduced_ref), **kwargs
            )
            x_scores.append(paired_result.x_result.sys_score)
            y_scores.append(paired_result.y_result.sys_score)

            if paired_result.x_result.sys_score > paired_result.y_result.sys_score:
                wins[0] += 1
            elif paired_result.x_result.sys_score < paired_result.y_result.sys_score:
                wins[1] += 1
            else:
                wins[2] += 1

        return BootstrapResult(x_scores, y_scores, wins, num_samples, self.name)
