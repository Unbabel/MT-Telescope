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
from typing import List, Tuple

import streamlit as st

from telescope.utils import read_lines


class Testset:
    def __init__(
        self,
        src: List[str],
        mt: List[str],
        ref: List[str],
        language_pair: str,
    ) -> None:
        self.src = src
        self.ref = ref
        self.mt = mt
        self.language_pair = language_pair

        assert len(ref) == len(
            src
        ), "mismatch between references and sources ({} > {})".format(
            len(ref), len(src)
        )
        assert len(mt) == len(
            ref
        ), "mismatch between MT and references ({} > {})".format(len(mt), len(ref))

    @property
    def source_language(self):
        return self.language_pair.split("-")[0]

    @property
    def target_language(self):
        return self.language_pair.split("-")[1]

    def __len__(self) -> int:
        return len(self.ref)

    def __getitem__(self, i) -> Tuple[str]:
        return self.src[i], self.mt[i], self.ref[i]

    def apply_filter(self, filter):
        to_keep = filter.apply_filter()
        self.src = [self.src[idx] for idx in to_keep]
        self.mt = [self.mt[idx] for idx in to_keep]
        self.ref = [self.ref[idx] for idx in to_keep]


class PairwiseTestset(Testset):
    def __init__(
        self,
        src: List[str],
        system_x: List[str],
        system_y: List[str],
        ref: List[str],
        language_pair: str,
        filenames: List[str],
    ) -> None:
        self.src = src
        self.ref = ref
        self.system_x = system_x
        self.system_y = system_y
        self.language_pair = language_pair
        self.filenames = filenames

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

    @staticmethod
    def hash_func(testset):
        return " ".join(testset.filenames)

    @classmethod
    def read_data(cls):
        st.subheader("Upload Files for analysis:")
        left1, right1 = st.beta_columns(2)
        source_file = left1.file_uploader("Upload Sources", type=["txt"])
        sources = read_lines(source_file)

        ref_file = right1.file_uploader("Upload References", type=["txt"])
        references = read_lines(ref_file)

        left2, right2 = st.beta_columns(2)
        x_file = left2.file_uploader("Upload System X Translations", type=["txt"])
        x = read_lines(x_file)

        y_file = right2.file_uploader("Upload System Y Translations", type=["txt"])
        y = read_lines(y_file)

        language_pair = st.text_input(
            "Please input the lanaguage pair of the files to analyse (e.g. 'en-ru'):",
            "",
        )

        if (
            (ref_file is not None)
            and (source_file is not None)
            and (y_file is not None)
            and (x_file is not None)
            and (language_pair != "")
        ):
            st.success(
                "Source, References, Translations and LP were successfully uploaded!"
            )
            return cls(
                sources,
                x,
                y,
                references,
                language_pair,
                [source_file.name, x_file.name, y_file.name, ref_file.name],
            )

    def __getitem__(self, i) -> Tuple[str]:
        return self.src[i], self.system_x[i], self.system_y[i], self.ref[i]

    def apply_filter(self, filter):
        to_keep = filter.apply_filter()
        self.src = [self.src[idx] for idx in to_keep]
        self.system_x = [self.system_x[idx] for idx in to_keep]
        self.system_y = [self.system_y[idx] for idx in to_keep]
        self.ref = [self.ref[idx] for idx in to_keep]
