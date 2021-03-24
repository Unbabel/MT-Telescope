from typing import List

import streamlit as st
import torch
from telescope.metrics.comet.result import COMETResult
from telescope.metrics.metric import Metric

from comet.models import download_model


class COMET(Metric):
    
    name = "COMET"

    def __init__(self, modelname: str = "wmt-large-da-estimator-1719",  **kwargs):
        self.modelname = modelname
        self.model = download_model(modelname)
        self.system_only = False

    @st.cache
    def streamlit_score(self, src: List[str], cand: List[str], ref: List[str]) -> List[float]:
        data = {"src": src, "mt": cand, "ref": ref}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        _, scores = self.model.predict(data, cuda=torch.cuda.is_available(), show_progress=True)
        return scores

    def score(self, src: List[str], cand: List[str], ref: List[str]) ->COMETResult:
        scores = self.streamlit_score(src, cand, ref)
        return COMETResult(
            sum(scores) / len(scores), scores, src, cand, ref
        )

