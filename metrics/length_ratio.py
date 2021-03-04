import sacrebleu
import streamlit as st

from metrics.metric import Metric
from metrics.result import MetricResult


class LengthRatioResult(MetricResult):
    def __init__(
        self, sys_score: int, sources: list, hypothesis: list, references: list
    ) -> None:
        super().__init__(sys_score, None, sources, hypothesis, references)


class LengthRatio(Metric):
    def __init__(self):
        self.name = "Length-Ratio"

    @st.cache
    def score(self, sources, hypothesis, references, **kwargs):
        bleu = sacrebleu.corpus_bleu(hypothesis, [references])
        return LengthRatioResult(
            bleu.sys_len / bleu.ref_len, sources, hypothesis, references
        )

    def paired_bootstrap(self, *args, **kwargs) -> None:
        pass
