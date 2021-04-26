from typing import List

import sacrebleu
from telescope.metrics.metric import Metric
from telescope.metrics.ter.result import TERResult


class TER(Metric):

    name = "TER"
    segment_level = False

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> TERResult:
        ter = sacrebleu.corpus_ter(cand, [ref])
        return TERResult(ter.score, [], src, cand, ref, self.name, ter.num_edits)
