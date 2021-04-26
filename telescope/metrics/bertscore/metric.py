from typing import List

import bert_score
from telescope.metrics.bertscore.result import BERTScoreResult
from telescope.metrics.metric import Metric


class BERTScore(Metric):

    name = "BERTScore"
    segment_level = True

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> BERTScoreResult:
        precision, recall, f1 = bert_score.score(
            cands=cand,
            refs=ref,
            idf=False,
            batch_size=3,
            lang=self.language,
            rescale_with_baseline=False,
            verbose=True,
        )
        return BERTScoreResult(
            sum(f1.tolist()) / len(f1.tolist()),
            f1.tolist(),
            src,
            cand,
            ref,
            self.name,
            precision.tolist(),
            recall.tolist(),
        )
