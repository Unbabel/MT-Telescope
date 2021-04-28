from telescope.filters.filter import Filter
from telescope.testset import Testset
from collections import Counter
from typing import List


class DuplicatesFilter(Filter):
    name = "duplicates"

    def __init__(self, testset: Testset, *args):
        self.testset = testset

    def apply_filter(self) -> List[int]:
        counter = Counter(self.testset.src)
        sources = []

        for i, item in enumerate(self.testset):
            src = item[0]
            if counter[src] == 0:
                continue
            # if counter > 1 we set it to 0 to skip the next time it appears
            if counter[src] > 1:
                counter[src] = 0
            
            sources.append(i)
        return sources