# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import Counter
from itertools import chain
from typing import List

from sacrebleu import TOKENIZERS
from telescope.metrics.metric import Metric
from telescope.metrics.result import MetricResult


class GLEU(Metric):

    name = "GLEU"
    segment_level = True

    def __init__(self, language: str, lowercase: bool = False, tokenize: bool = True):
        super().__init__(language)
        if language == "zh":
            self.tokenizer = TOKENIZERS["zh"]()
        elif language == "ja":
            self.tokenizer = TOKENIZERS["ja-mecab"]()
        else:
            self.tokenizer = TOKENIZERS["13a"]()

        self.lowercase = lowercase
        self.tokenize = tokenize

    def score(self, src: List[str], cand: List[str], ref: List[str]) -> MetricResult:
        org_cand = cand
        org_ref = ref
        if self.tokenize:
            cand = [self.tokenizer(c.strip("\n")) for c in cand]
            ref = [self.tokenizer(r.strip("\n")) for r in ref]
        else:
            cand = [c.strip("\n").split(" ") for c in cand]
            ref = [r.strip("\n").split(" ") for r in ref]

        if self.lowercase:
            cand = [c.lower() for c in cand]
            ref = [r.lower() for r in ref]

        segment_gleu = [self.sentence_gleu(r, h) for r, h in zip(ref, cand)]
        corpus_gleu = sum(segment_gleu) / len(segment_gleu)
        cand = [" ".join(seg) for seg in cand]
        ref = [" ".join(seg) for seg in ref]
        return MetricResult(
            corpus_gleu, segment_gleu, src, org_cand, org_ref, self.name
        )

    def sentence_gleu(self, reference, hypothesis, min_len=1, max_len=4):
        references = [
            reference,
        ]
        return self.algorithm(
            [references], [hypothesis], min_len=min_len, max_len=max_len
        )

    def algorithm(self, list_of_references, hypotheses, min_len=1, max_len=4):
        """Original code from NLTK:
        https://www.nltk.org/_modules/nltk/translate/gleu_score.html

        """
        assert len(list_of_references) == len(
            hypotheses
        ), "The number of hypotheses and their reference(s) should be the same"

        # sum matches and max-token-lengths over all sentences
        corpus_n_match = 0
        corpus_n_all = 0

        for references, hypothesis in zip(list_of_references, hypotheses):
            hyp_ngrams = Counter(self.everygrams(hypothesis, min_len, max_len))
            tpfp = sum(hyp_ngrams.values())  # True positives + False positives.

            hyp_counts = []
            for reference in references:
                ref_ngrams = Counter(self.everygrams(reference, min_len, max_len))
                tpfn = sum(ref_ngrams.values())  # True positives + False negatives.

                overlap_ngrams = ref_ngrams & hyp_ngrams
                tp = sum(overlap_ngrams.values())  # True positives.

                # While GLEU is defined as the minimum of precision and
                # recall, we can reduce the number of division operations by one by
                # instead finding the maximum of the denominators for the precision
                # and recall formulae, since the numerators are the same:
                #     precision = tp / tpfp
                #     recall = tp / tpfn
                #     gleu_score = min(precision, recall) == tp / max(tpfp, tpfn)
                n_all = max(tpfp, tpfn)

                if n_all > 0:
                    hyp_counts.append((tp, n_all))

            # use the reference yielding the highest score
            if hyp_counts:
                n_match, n_all = max(hyp_counts, key=lambda hc: hc[0] / hc[1])
                corpus_n_match += n_match
                corpus_n_all += n_all

        # corner case: empty corpus or empty references---don't divide by zero!
        if corpus_n_all == 0:
            gleu_score = 0.0
        else:
            gleu_score = corpus_n_match / corpus_n_all

        return gleu_score

    def pad_sequence(
        self,
        sequence,
        n,
        pad_left=False,
        pad_right=False,
        left_pad_symbol=None,
        right_pad_symbol=None,
    ):
        sequence = iter(sequence)
        if pad_left:
            sequence = chain((left_pad_symbol,) * (n - 1), sequence)
        if pad_right:
            sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
        return sequence

    def ngrams(
        self,
        sequence,
        n,
        pad_left=False,
        pad_right=False,
        left_pad_symbol=None,
        right_pad_symbol=None,
    ):
        sequence = self.pad_sequence(
            sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
        )
        history = []
        while n > 1:
            history.append(next(sequence))
            n -= 1
        for item in sequence:
            history.append(item)
            yield tuple(history)
            del history[0]

    def everygrams(self, sequence, min_len=1, max_len=-1, **kwargs):
        if max_len == -1:
            max_len = len(sequence)
        for n in range(min_len, max_len + 1):
            for ng in self.ngrams(sequence, n, **kwargs):
                yield ng
