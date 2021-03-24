import abc

import numpy as np
import streamlit as st
from telescope.testset import Testset

from telescope.metrics.result import PairwiseResult, MetricResult, BootstrapResult
from typing import List, Union, Tuple


class Metric(metaclass=abc.ABCMeta):

    name = None

    @abc.abstractmethod
    def score(self, src: List[str], cand: List[str], ref: List[str]) -> MetricResult:
        pass
    
    @abc.abstractmethod
    @st.cache
    def streamlit_score(self, src: List[str], cand: List[str], ref: List[str]) -> Union[float, List[float]]:
        pass
    
    def pairwise_comparison(self, testset: Testset):
        with st.spinner(f'Running {self.name}...'):
            x_result = self.score(
                testset.src, testset.system_x, testset.ref
            )
            y_result = self.score(
                testset.src, testset.system_y, testset.ref
            )
            return PairwiseResult(x_result, y_result, self.name, self.system_only)

    def bootstrap_resampling(
        self, 
        testset: Testset,
        num_samples: int = 300, 
        sample_ratio: float = 0.5,
        pairwise_result: PairwiseResult = None
    ) -> BootstrapResult:
        """ 
        Bootstrap resampling for system-level metrics such as BLEU that have to recompute
        the system-level score for each partition 
        
        :param testset: Testset
        :param num_samples: Number of testset splits.
        :param sample_ratio: % of the testset to be used in each partition.
        :param pairwise_result: Precomputed scores between two systems.

        :return: BootstrapResult object
        """
        def update_wins(x_score: int, y_score: int , wins: Tuple[int]):
            if y_score > x_score:
                wins[1] += 1
            elif y_score < x_score:
                wins[0] += 1
            else:
                wins[2] += 1
            return wins

        def recompute_sys_scores():
            if self.system_only:
                pairwise_result = self.pairwise_comparison(
                    Testset(reduced_src, reduced_x, reduced_y, reduced_ref)
                )
                return (
                    pairwise_result.x_result.sys_score, 
                    pairwise_result.y_result.sys_score
                )
            elif pairwise_result is not None:
                reduces_x_scr = [pairwise_result.x_result.seg_scores[i] for i in reduced_ids]
                reduces_y_scr = [pairwise_result.x_result.seg_scores[i] for i in reduced_ids]
                return (
                    sum(reduces_x_scr)/len(reduces_x_scr), 
                    sum(reduces_y_scr)/len(reduces_y_scr), 
                )
            else:
                raise Exception("Bootstrap_resampling expects precomputed results for segment-level metrics.")


        n = len(testset)
        st.warning(
            f"Testset length is too short ({n}). Results are not be reliable, please upload a bigger testset."
        )
        ids = list(range(n))
        sample_size = int(n * sample_ratio)
        if sample_size < 500:
            st.warning((
                f"Proportion (P) of the initial sample results in small random samples of size {sample_size}."
                " Adjusting sample size to 500."
            ))
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

            x_result, y_result = recompute_sys_scores()
            x_scores.append(x_result)
            y_scores.append(y_result)

            if x_scores[-1] > y_scores[-1]:
                wins[0] += 1
            elif x_scores[-1] < y_scores[-1]:
                wins[1] += 1
            else:
                wins[2] += 1

        return BootstrapResult(x_scores, y_scores, wins, num_samples, self.name)
