from typing import List

import streamlit as st
from telescope.metrics.length_ratio.result import LengthRatioResult
from telescope.metrics.metric import Metric


class LengthRatio(Metric):
    name = "Length-ratio"
    
    def __init__(self):
        self.system_only = True

    @st.cache
    def streamlit_score(self, src: List[str], cand: List[str], ref: List[str]) -> float:
        cand_char_count, ref_char_count = 0, 0
        for h, r  in zip(cand, ref):
            cand_char_count += len(h)
            ref_char_count += len(r)
        return cand_char_count/ref_char_count

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> LengthRatioResult:
        score = self.streamlit_score(src, cand, ref)
        return LengthRatioResult(score, src, cand, ref)
