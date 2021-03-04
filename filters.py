import abc

import pandas as pd
import stanza
import streamlit as st

from testset import PairedTestset


class TestsetFilter:
    def __init__(self, testset):
        self.testset = testset

    @abc.abstractmethod
    def apply_filter(self):
        pass


STANZA_NER_LANGS = ["ar", "zh", "nl", "en", "fr", "de", "ru", "uk"]


class NERFilter(TestsetFilter):
    def check_stanza_languages(self):
        if self.testset.src_lang in STANZA_NER_LANGS:
            return True
        elif self.testset.trg_lang in STANZA_NER_LANGS:
            return True
        else:
            st.warning(
                f"source and target languages not supported by stanza NER library."
            )
            return False

    @st.cache()
    def apply_filter(self):
        with st.spinner("Applying Named Entities filter..."):
            if not self.testset.src_lang in STANZA_NER_LANGS:
                language = self.testset.trg_lang
                segments = self.testset.references
            else:
                language = self.testset.src_lang
                segments = self.testset.sources

            stanza.download(language)
            nlp = stanza.Pipeline(lang=language, processors="tokenize,ner")

            segments_with_ne = []
            for i, segment in enumerate(segments):
                doc = nlp(segment)
                if doc.ents:
                    segments_with_ne.append(i)

            return PairedTestset(
                sources=[self.testset.sources[i] for i in segments_with_ne],
                system_x=[self.testset.system_x[i] for i in segments_with_ne],
                system_y=[self.testset.system_y[i] for i in segments_with_ne],
                references=[self.testset.references[i] for i in segments_with_ne],
                src_lang=self.testset.src_lang,
                trg_lang=self.testset.trg_lang,
            )


class GlossaryFilter(TestsetFilter):
    def __init__(self, testset, glossary):
        self.testset = testset
        self.glossary = glossary

    @st.cache()
    def apply_filter(self):
        gloss_idx = []
        for i, sentence in enumerate(self.testset.sources):
            for gloss_term in self.glossary:
                if gloss_term in sentence:
                    gloss_idx.append(i)

        return PairedTestset(
            sources=[self.testset.sources[i] for i in gloss_idx],
            system_x=[self.testset.system_x[i] for i in gloss_idx],
            system_y=[self.testset.system_y[i] for i in gloss_idx],
            references=[self.testset.references[i] for i in gloss_idx],
            src_lang=self.testset.src_lang,
            trg_lang=self.testset.trg_lang,
        )


class LengthFilter(TestsetFilter):
    def __init__(self, testset, min_value, max_value):
        self.testset = testset
        self.min_value = min_value
        self.max_value = max_value

    @st.cache()
    def apply_filter(self):
        with st.spinner("Applying length filter..."):
            dataframe = pd.DataFrame()
            dataframe["ref"] = self.testset.references
            dataframe["lengths"] = [len(ref) for ref in list(dataframe["ref"])]
            dataframe["bins"], retbins = pd.qcut(
                dataframe["lengths"].rank(method="first"),
                len(range(0, 100, 5)),
                labels=range(0, 100, 5),
                retbins=True,
            )
            length_buckets = list(dataframe["bins"])

            src = [
                self.testset.sources[i]
                for i in range(len(self.testset))
                if (
                    length_buckets[i] >= self.min_value
                    and length_buckets[i] <= self.max_value
                )
            ]
            ref = [
                self.testset.references[i]
                for i in range(len(self.testset))
                if (
                    length_buckets[i] >= self.min_value
                    and length_buckets[i] <= self.max_value
                )
            ]
            x = [
                self.testset.system_x[i]
                for i in range(len(self.testset))
                if (
                    length_buckets[i] >= self.min_value
                    and length_buckets[i] <= self.max_value
                )
            ]
            y = [
                self.testset.system_y[i]
                for i in range(len(self.testset))
                if (
                    length_buckets[i] >= self.min_value
                    and length_buckets[i] <= self.max_value
                )
            ]
            return PairedTestset(
                sources=src,
                system_x=x,
                system_y=y,
                references=ref,
                src_lang=self.testset.src_lang,
                trg_lang=self.testset.trg_lang,
            )


class DuplicatesFilter(TestsetFilter):
    def apply_filter(self):
        df = pd.DataFrame(
            list(
                zip(
                    self.testset.sources,
                    self.testset.references,
                    self.testset.system_x,
                    self.testset.system_y,
                )
            ),
            columns=["sources", "references", "system_x", "system_y"],
        )
        df.drop_duplicates(inplace=True)
        return PairedTestset(
            sources=list(df.sources),
            system_x=list(df.system_x),
            system_y=list(df.system_y),
            references=list(df.references),
            src_lang=self.testset.src_lang,
            trg_lang=self.testset.trg_lang,
        )
