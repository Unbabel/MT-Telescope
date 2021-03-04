import sacrebleu
import streamlit as st

from metrics.metric import Metric
from metrics.result import MetricResult


class chrFResult(MetricResult):
    def __init__(
        self, sys_score: int, sources: list, hypothesis: list, references: list
    ) -> None:
        super().__init__(sys_score, None, sources, hypothesis, references)


class chrF(Metric):
    def __init__(self):
        self.name = "chrF"

    @st.cache
    def score(self, sources, hypothesis, references, **kwargs):
        chrf = sacrebleu.corpus_chrf(hypothesis, [references])
        return chrFResult(chrf.score, sources, hypothesis, references)
