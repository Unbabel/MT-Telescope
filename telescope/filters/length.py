import pandas as pd
from telescope.filters.filter import Filter
from telescope.testset import Testset
from typing import List


class LengthFilter(Filter):
    name = "length"

    def __init__(
        self, testset: Testset, min_value: float, max_value: float, *args
    ):
        super().__init__(testset)
        self.min_value = min_value
        self.max_value = max_value

    def apply_filter(self) -> List[int]:
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
        return [i for i in range(len(self.testset)) 
        if (length_buckets[i] >= self.min_value
            and length_buckets[i] <= self.max_value)
        ]
