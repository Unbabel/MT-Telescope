from .bertscore import BERTScore
from .bleurt import BLEURT
from .chrf import chrF
from .comet import COMET
from .length_ratio import LengthRatio
from .prism import Prism
from .sacrebleu import BLEU

SEGMENT_METRICS = [COMET, BERTScore, BLEURT, Prism]
SYSTEM_METRICS = [BLEU, chrF, LengthRatio]
ALL_METRICS = SEGMENT_METRICS + SYSTEM_METRICS
