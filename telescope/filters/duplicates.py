from typing import List

import pandas as pd
from telescope.filters.filter import TestsetFilter
from telescope.testset import Testset


class DuplicatesFilter(TestsetFilter):
    name = "duplicates"
    
    def __init__(self, testset: Testset):
        self.testset = testset

    def apply_filter(self):
        df = pd.DataFrame(
            list(
                zip(
                    self.testset.src,
                    self.testset.ref,
                    self.testset.system_x,
                    self.testset.system_y,
                )
            ),
            columns=["sources", "references", "system_x", "system_y"],
        )
        df.drop_duplicates(inplace=True)
        return Testset(
            src=list(df.sources),
            system_x=list(df.system_x),
            system_y=list(df.system_y),
            ref=list(df.references),
            src_lang=self.testset.src_lang,
            trg_lang=self.testset.trg_lang,
        )
