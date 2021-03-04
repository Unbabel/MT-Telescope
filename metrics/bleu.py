import sacrebleu
import streamlit as st

from metrics.metric import Metric
from metrics.result import MetricResult


class BLEUResult(MetricResult):
    def __init__(
        self, sys_score: int, sources: list, hypothesis: list, references: list
    ) -> None:
        super().__init__(sys_score, None, sources, hypothesis, references)


class BLEU(Metric):
    def __init__(self):
        self.name = "BLEU"

    @st.cache
    def score(self, sources, hypothesis, references, **kwargs):
        bleu = sacrebleu.corpus_bleu(hypothesis, [references])
        return BLEUResult(bleu.score, sources, hypothesis, references)
