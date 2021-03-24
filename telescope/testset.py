from io import StringIO

import pandas as pd
import streamlit as st
from typing import List, Tuple
from telescope.utils import read_lines


class Testset:
    def __init__(
        self,
        src: List[str],
        system_x: List[str],
        system_y: List[str],
        ref: List[str],
        src_lang: str = None,
        trg_lang: str = None,
    ) -> None:
        self.src = src
        self.ref = ref
        self.system_x = system_x
        self.system_y = system_y
        self.src_lang = src_lang
        self.trg_lang = trg_lang

        assert len(ref) == len(
            src
        ), "mismatch between references and sources ({} > {})".format(
            len(ref), len(src)
        )
        assert len(system_x) == len(
            system_y
        ), "mismatch between system x and system y ({} > {})".format(
            len(system_x), len(system_y)
        )
        assert len(system_x) == len(
            ref
        ), "mismatch between system x and references ({} > {})".format(
            len(system_x), len(ref)
        )

    @classmethod
    def read_data(cls):
        st.subheader("Upload Files for analysis:")
        left1, right1  = st.beta_columns(2)
        source_file = left1.file_uploader("Upload Sources", type=["txt"])
        sources = read_lines(source_file)

        ref_file = right1.file_uploader("Upload References", type=["txt"])
        references = read_lines(ref_file)

        left2, right2  = st.beta_columns(2)
        x_file = left2.file_uploader("Upload System X Translations", type=["txt"])
        x = read_lines(x_file)

        y_file = right2.file_uploader("Upload System Y Translations", type=["txt"])
        y = read_lines(y_file)

        if (
            (ref_file is not None)
            and (source_file is not None)
            and (y_file is not None)
            and (x_file is not None)
        ):

            st.success("Source, Refereces and Translations were successfully uploaded!")
            return cls(
                sources,
                x,
                y,
                references,
                source_file.name.split(".")[-2],
                ref_file.name.split(".")[-2],
            )

    def __len__(self) -> int:
        return len(self.ref)

    def __getitem__(self, i) -> Tuple[str]:
        return self.src[i], self.system_x[i], self.system_y[i], self.ref[i]
