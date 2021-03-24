from typing import List

import bert_score
import streamlit as st
from telescope.metrics.bertscore.result import BERTScoreResult
from telescope.metrics.metric import Metric


class BERTScore(Metric):

    name = "BERTScore"

    def __init__(self, lang):
        self.system_only = False
        self.lang = lang

    @st.cache
    def streamlit_score(self, src: List[str], cand: List[str], ref: List[str]) -> List[float]:
        scores = bert_score.score(
            cands=cand,
            refs=ref,
            idf=True,
            batch_size=32,
            lang=self.lang,
            rescale_with_baseline=False,
            verbose=True,
            nthreads=4,
        )[2].tolist()
        return scores

        
    def score(self, src: List[str], cand: List[str], ref: List[str]) -> BERTScoreResult:
        scores = self.streamlit_score(src, cand, ref)
        return BERTScoreResult(
            sum(scores) / len(scores), scores, src, cand, ref
        )
