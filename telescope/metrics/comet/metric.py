from typing import List

import torch
from telescope.metrics.comet.result import COMETResult
from telescope.metrics.metric import Metric

from comet.models import download_model


class COMET(Metric):

    name = "COMET"
    system_only = False

    def __init__(
        self, language=None, modelname: str = "wmt-large-da-estimator-1719", **kwargs
    ):
        self.modelname = modelname
        self.model = download_model(modelname)

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> COMETResult:
        data = {"src": src, "mt": cand, "ref": ref}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        _, scores = self.model.predict(
            data, cuda=torch.cuda.is_available(), show_progress=True
        )
        return COMETResult(
            sum(scores) / len(scores), scores, src, cand, ref, self.name, self.modelname
        )
