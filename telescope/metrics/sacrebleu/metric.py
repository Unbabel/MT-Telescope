from typing import List

from telescope.metrics.metric import Metric
from telescope.metrics.sacrebleu.result import BLEUResult

import sacrebleu


class sacreBLEU(Metric):

    name = "BLEU"
    segment_level = False

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> BLEUResult:
        bleu = sacrebleu.corpus_bleu(cand, [ref])
        return BLEUResult(
            bleu.score/100, [], src, cand, ref, self.name, bleu.precisions, bleu.bp
        )
