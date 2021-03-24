from typing import List

import streamlit as st
from telescope.filters.filter import TestsetFilter
from telescope.testset import Testset
from telescope.utils import read_lines


class TerminologyFilter(TestsetFilter):
    name = "terminology"

    def __init__(self, testset: Testset, glossary: List[str]):
        self.testset = testset
        glossary_file = st.file_uploader("Upload a glossary file", type=["txt"])
        glossary = read_lines(glossary_file)


    def apply_filter(self):
        with st.spinner("Applying Named Entities filter..."):
            gloss_idx = []
            for i, sentence in enumerate(self.testset.sources):
                for gloss_term in self.glossary:
                    if gloss_term in sentence:
                        gloss_idx.append(i)

        return Testset(
            src=[self.testset.src[i] for i in gloss_idx],
            system_x=[self.testset.system_x[i] for i in gloss_idx],
            system_y=[self.testset.system_y[i] for i in gloss_idx],
            ref=[self.testset.ref[i] for i in gloss_idx],
            src_lang=self.testset.src_lang,
            trg_lang=self.testset.trg_lang,
        )
