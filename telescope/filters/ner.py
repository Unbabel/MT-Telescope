import stanza
from telescope.filters.filter import Filter
from telescope.testset import Testset
from typing import List

STANZA_NER_LANGS = ["ar", "zh", "nl", "en", "fr", "de", "ru", "uk"]


class NERFilter(Filter):
    name = "named-entities"

    def __init__(self, testset: Testset, *args):
        super().__init__(testset)
        self.set_language()
        stanza.download(self.language)
        self.engine = stanza.Pipeline(lang=self.language, processors="tokenize,ner")

    def set_language(self) -> None:
        if self.testset.source_language in STANZA_NER_LANGS:
            self.language = self.testset.source_language
            self.segments = self.testset.src
        elif self.testset.target_language in STANZA_NER_LANGS:
            self.language = self.testset.target_language
            self.segments = self.testset.ref
        else:
            raise Exception(
                "{} is not supperted by Stanza NER.".format(self.testset.language_pair)
            )

    def apply_filter(self) -> List[int]:
        segments_with_ne = []
        for i, segment in enumerate(self.segments):
            doc = self.engine(segment)
            if doc.ents:
                segments_with_ne.append(i)

        return segments_with_ne
