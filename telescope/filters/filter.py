import abc
from typing import List

import streamlit as st
from telescope.testset import PairwiseTestset


class Filter:
    name = None

    def __init__(self, testset: PairwiseTestset, *args):
        self.testset = testset

    @abc.abstractmethod
    def apply_filter(self) -> PairwiseTestset:
        return NotImplementedError


class ComposedFilter(Filter):
    def __init__(self, testset: PairwiseTestset, filters: List[Filter]):
        super().__init__(testset)
        self.filters = filters

    def apply_filter(self) -> PairwiseTestset:
        for filter in self.filters:
            self.testset = filter.apply_filter()
            if st._is_running_with_streamlit:
                st.success(f"{filter.name} successfully applied...")

        # HACK
        # I'll add a new prefix to all testset filenames to "fool" streamlit cache
        filter_prefix = " ".join([f.name for f in self.filters])
        filenames = [filter_prefix + f for f in self.testset.filenames]
        return PairwiseTestset(
            src=self.testset.src,
            system_x=self.testset.system_x,
            system_y=self.testset.system_y,
            ref=self.testset.ref,
            language_pair=self.testset.language_pair,
            filenames=filenames,
        )
