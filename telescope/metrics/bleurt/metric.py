import os

from telescope.metrics.bleurt.result import BLEURTResult
from telescope.metrics.metric import Metric
from telescope.utils import telescope_cache_folder
from torchnlp.download import download_file_maybe_extract

from bleurt import score

import streamlit as st


class BLEURT(Metric):
    name = "BLEURT"
    
    def __init__(self, model: str = "bleurt-base-128", **kwargs):
        self.model = model
        if not os.path.isdir(telescope_cache_folder() + model):
            download_file_maybe_extract(
                url=f"https://storage.googleapis.com/bleurt-oss/{model}.zip",
                directory=telescope_cache_folder(),
            )
        self.scorer = score.BleurtScorer(telescope_cache_folder() + model)
        self.system_only = False

    @st.cache
    def streamlit_score(self, src, cand, ref):
        return self.scorer.score(ref, cand)
        
    def score(self, src, cand, ref):
        scores = self.streamlit_score(src, cand, ref)
        return BLEURTResult(
            sum(scores) / len(scores), scores, src, cand, ref
        )
