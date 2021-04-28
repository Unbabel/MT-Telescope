from .ner import NERFilter
from .length import LengthFilter
from .duplicates import DuplicatesFilter

AVAILABLE_FILTERS = [NERFilter, LengthFilter, DuplicatesFilter]
