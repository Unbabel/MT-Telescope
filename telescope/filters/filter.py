import abc
from telescope.testset import Testset


class TestsetFilter:
    name = None
    
    def __init__(self, testset: Testset):
        self.testset = testset

    @abc.abstractmethod
    def apply_filter(self):
        pass