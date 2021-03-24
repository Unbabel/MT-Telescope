from .duplicates import DuplicatesFilter
from .length import LengthFilter
from .ner import NERFilter
from .terminology import TerminologyFilter

FILTERS = [DuplicatesFilter, LengthFilter, NERFilter, TerminologyFilter]