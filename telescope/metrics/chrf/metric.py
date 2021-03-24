from typing import List

import sacrebleu
import streamlit as st
from telescope.metrics.chrf.result import chrFResult
from telescope.metrics.metric import Metric


class chrF(Metric):
    
    name = "chrF"

    def __init__(self, **kwargs):
        self.system_only = True

    @st.cache
    def streamlit_score(self, src: List[str], cand: List[str], ref: List[str]) -> float:
        chrf = sacrebleu.corpus_chrf(cand, [ref])
        return chrf.score

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> chrFResult:
        score = self.streamlit_score(src, cand, ref)
        return chrFResult(score, src, cand, ref)
