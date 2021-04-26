from telescope.filters.filter import Filter
from telescope.testset import PairwiseTestset
from collections import Counter


class DuplicatesFilter(Filter):
    name = "duplicates"

    def __init__(self, testset: PairwiseTestset, *args):
        self.testset = testset

    def apply_filter(self) -> PairwiseTestset:
        counter = Counter(self.testset.src)
        sources, system_x, system_y, references = [], [], [], []
        for src, x, y, ref in self.testset:
            if counter[src] == 0:
                continue
            # if counter > 1 we set it to 0 to skip the next time it appears
            if counter[src] > 1:
                counter[src] = 0

            sources.append(src)
            system_x.append(x)
            system_y.append(y)
            references.append(ref)

        return PairwiseTestset(
            src=sources,
            system_x=system_x,
            system_y=system_y,
            ref=references,
            language_pair=self.testset.language_pair,
            filenames=self.testset.filenames,
        )
