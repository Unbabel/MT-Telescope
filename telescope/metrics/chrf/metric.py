from typing import List

import sacrebleu
from telescope.metrics.chrf.result import chrFResult
from telescope.metrics.metric import Metric


class chrF(Metric):

    name = "chrF"
    segment_level = False

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> chrFResult:
        chrf = sacrebleu.corpus_chrf(cand, [ref])
        return chrFResult(chrf.score, [], src, cand, ref, self.name)
