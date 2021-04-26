import os

from telescope.metrics.bleurt.result import BLEURTResult
from telescope.metrics.metric import Metric
from telescope.utils import telescope_cache_folder
from torchnlp.download import download_file_maybe_extract

from bleurt import score


class BLEURT(Metric):

    name = "BLEURT"
    segment_level = True

    def __init__(self, language, model: str = "bleurt-base-128"):
        super().__init__(language)
        # HACK TO SILENCE tensorflow and errors related to tf.FLAGS
        from silence_tensorflow import silence_tensorflow

        silence_tensorflow()
        import tensorflow.compat.v1 as tf

        flags = tf.flags
        flags.DEFINE_string("source", "", help="Source segments", required=False)
        flags.DEFINE_string("s", "", help="Source segments", required=False)
        flags.DEFINE_string("hypothesis", "", help="MT segments", required=False)
        flags.DEFINE_string("h", "", help="MT segments", required=False)
        flags.DEFINE_string("reference", "", help="Reference segments", required=False)
        flags.DEFINE_string("r", "", help="Reference segments", required=False)
        flags.DEFINE_string("language", "", help="Language", required=False)
        flags.DEFINE_string("l", "", help="Language", required=False)
        flags.DEFINE_string("metric", "", help="Metric to run.", required=False)
        flags.DEFINE_string("m", "", help="Metric to run.", required=False)

        self.model = model
        if not os.path.isdir(telescope_cache_folder() + model):
            download_file_maybe_extract(
                url=f"https://storage.googleapis.com/bleurt-oss/{model}.zip",
                directory=telescope_cache_folder(),
            )
        self.scorer = score.BleurtScorer(telescope_cache_folder() + model)
        self.system_only = False

    @classmethod
    def language_support(self, language):
        return language == "en"

    def score(self, src, cand, ref):
        scores = self.scorer.score(ref, cand)
        return BLEURTResult(
            sum(scores) / len(scores), scores, src, cand, ref, self.name, self.model
        )
