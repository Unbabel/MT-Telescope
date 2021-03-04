import numpy as np
import sacrebleu
import streamlit as st

from comet.models import download_model
from metrics.metric import Metric
from metrics.result import BootstrapResult, MetricResult, PairedResult

COMET_MODELS = [
    "emnlp-base-da-ranker",
    "wmt-base-da-ranker-1719",
    "wmt-base-da-estimator-1718",
    "wmt-base-da-estimator-1719",
    "wmt-large-da-estimator-1718",
    "wmt-large-da-estimator-1719",
    "wmt-large-qe-estimator-1719",
    "wmt-large-hter-estimator",
    "wmt-base-hter-estimator",
]


class COMETResult(MetricResult):
    def __init__(
        self,
        sys_score: int,
        seg_scores: list,
        sources: list,
        hypothesis: list,
        references: list,
    ) -> None:
        super().__init__(sys_score, seg_scores, sources, hypothesis, references)


class COMET(Metric):
    def __init__(self, modelname: str = "wmt-large-da-estimator-1719"):
        self.name = "COMET"
        self.modelname = modelname
        with st.spinner("Loading COMET model..."):
            self.model = download_model(modelname)

        self.cache = {}

    @staticmethod
    def available_models() -> list:
        return COMET_MODELS

    def score(self, sources, hypothesis, references, **kwargs):
        new_idx = []
        for i, sample in enumerate(zip(sources, hypothesis, references)):
            if sample not in self.cache:
                new_idx.append(i)

        if len(new_idx) > 0:
            new_scores = self._score(
                [sources[idx] for idx in new_idx],
                [hypothesis[idx] for idx in new_idx],
                [references[idx] for idx in new_idx],
                kwargs["cuda"] if "cuda" in kwargs else False,
            )
            for i, idx in enumerate(new_idx):
                self.cache[
                    (sources[idx], hypothesis[idx], references[idx])
                ] = new_scores[i]

        scores = [
            self.cache[(s, h, r)] for s, h, r in zip(sources, hypothesis, references)
        ]
        return COMETResult(
            sum(scores) / len(scores), scores, sources, hypothesis, references
        )

    @st.cache
    def _score(self, sources, hypothesis, references, cuda):
        data = {"src": sources, "mt": hypothesis, "ref": references}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        _, scores = self.model.predict(data, cuda=cuda, show_progress=True)
        return scores

    @st.cache
    def paired_bootstrap(
        self,
        testset,
        num_samples=1000,
        sample_ratio=0.5,
        precomputed_result=None,
        **kwargs
    ) -> None:
        def update_wins(x_score, y_score, wins):
            if y_score > x_score:
                wins[1] += 1
            elif y_score < x_score:
                wins[0] += 1
            else:
                wins[2] += 1
            return wins

        # With COMET we only need to do this once. The system score is a simple average
        # of the segment level scores.
        if precomputed_result and isinstance(precomputed_result, PairedResult):
            paired_result = precomputed_result
        else:
            paired_result = self.score_paired_testset(testset, **kwargs)

        x_comet = paired_result.x_result.seg_scores
        y_comet = paired_result.y_result.seg_scores

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
            x_comet = [paired_result.x_result.seg_scores[i] for i in reduced_ids]
            y_comet = [paired_result.y_result.seg_scores[i] for i in reduced_ids]
            x_system = sum(x_comet) / len(reduced_ids)
            y_system = sum(y_comet) / len(reduced_ids)
            x_scores.append(x_system)
            y_scores.append(y_system)

            if x_system > y_system:
                wins[0] += 1
            elif x_system < y_system:
                wins[1] += 1
            else:
                wins[2] += 1

        return BootstrapResult(x_scores, y_scores, wins, num_samples, self.name)
