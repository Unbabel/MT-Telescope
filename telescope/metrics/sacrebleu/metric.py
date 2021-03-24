from typing import List

import streamlit as st
from telescope.metrics.metric import Metric
from telescope.metrics.sacrebleu.result import BLEUResult

import sacrebleu


class BLEU(Metric):
    name = "sacreBLEU"
    
    def __init__(self, **kwargs):
        self.system_only = True
    
    @st.cache
    def streamlit_score(self, src: List[str], cand: List[str], ref: List[str]) -> float:
        bleu = sacrebleu.corpus_bleu(cand, [ref])
        return bleu.score

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> BLEUResult:
        score = self.streamlit_score(src, cand, ref)
        return BLEUResult(score, src, cand, ref)
