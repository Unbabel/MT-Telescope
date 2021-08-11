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
from typing import List

import pandas as pd
from telescope.filters.filter import Filter
from telescope.testset import Testset


class LengthFilter(Filter):
    name = "length"

    def __init__(self, testset: Testset, min_value: float, max_value: float, *args):
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
        return [
            i
            for i in range(len(self.testset))
            if (
                length_buckets[i] >= self.min_value
                and length_buckets[i] <= self.max_value
            )
        ]
