import pandas as pd
import streamlit as st
from telescope.filters.filter import TestsetFilter
from telescope.testset import Testset


class LengthFilter(TestsetFilter):
    name = "length"
    
    def __init__(self, testset: Testset, min_value: float, max_value: float):
        self.testset = testset
        self.min_value = min_value
        self.max_value = max_value

    def apply_filter(self):
        with st.spinner("Applying length filter..."):
            dataframe = pd.DataFrame()
            dataframe["ref"] = self.testset.ref
            dataframe["lengths"] = [len(ref) for ref in list(dataframe["ref"])]
            dataframe["bins"], retbins = pd.qcut(
                dataframe["lengths"].rank(method="first"),
                len(range(0, 100, 5)),
                labels=range(0, 100, 5),
                retbins=True,
            )
            length_buckets = list(dataframe["bins"])

            src = [
                self.testset.src[i]
                for i in range(len(self.testset))
                if (
                    length_buckets[i] >= self.min_value
                    and length_buckets[i] <= self.max_value
                )
            ]
            ref = [
                self.testset.ref[i]
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
            return Testset(
                src=src,
                system_x=x,
                system_y=y,
                ref=ref,
                src_lang=self.testset.src_lang,
                trg_lang=self.testset.trg_lang,
            )
