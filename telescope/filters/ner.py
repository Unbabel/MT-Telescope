import stanza

import streamlit as st
from telescope.filters.filter import TestsetFilter
from telescope.testset import Testset


STANZA_NER_LANGS = ["ar", "zh", "nl", "en", "fr", "de", "ru", "uk"]

class NERFilter(TestsetFilter):
    name = "named-entities"
    
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
        
    def apply_filter(self):
        with st.spinner("Applying Named Entities filter..."):
            if not self.testset.src_lang in STANZA_NER_LANGS:
                language = self.testset.trg_lang
                segments = self.testset.ref
            else:
                language = self.testset.src_lang
                segments = self.testset.src

            stanza.download(language)
            nlp = stanza.Pipeline(lang=language, processors="tokenize,ner")

            segments_with_ne = []
            for i, segment in enumerate(segments):
                doc = nlp(segment)
                if doc.ents:
                    segments_with_ne.append(i)

            return Testset(
                src=[self.testset.src[i] for i in segments_with_ne],
                system_x=[self.testset.system_x[i] for i in segments_with_ne],
                system_y=[self.testset.system_y[i] for i in segments_with_ne],
                ref=[self.testset.ref[i] for i in segments_with_ne],
                src_lang=self.testset.src_lang,
                trg_lang=self.testset.trg_lang,
            )