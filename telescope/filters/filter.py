import abc
from typing import List

import streamlit as st
from telescope.testset import Testset


class Filter:
    name = None

    def __init__(self, testset: Testset, *args):
        self.testset = testset

    @abc.abstractmethod
    def apply_filter(self) -> List[int]:
        """ Returns the indexes of elements to keep """
        return NotImplementedError

