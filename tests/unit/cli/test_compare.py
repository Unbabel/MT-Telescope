import os
import unittest

from click.testing import CliRunner
from telescope.cli import compare
from tests.data import DATA_PATH


class TestCompareCli(unittest.TestCase):

    system_x = os.path.join(DATA_PATH, "OnlineA.txt")
    system_y = os.path.join(DATA_PATH, "OnlineB.txt")
    src = os.path.join(DATA_PATH, "src_400.ru.txt")
    ref = os.path.join(DATA_PATH, "ref_400.en.txt")

    def setUp(self):
        self.runner = CliRunner()

    def test_with_seg_metric(self):
        args = [
            "-s",
            self.src,
            "-x",
            self.system_x,
            "-y",
            self.system_y,
            "-r",
            self.ref,
            "-l",
            "en",
            "-m",
            "chrF",
            "--seg_metric",
            "GLEU"
        ]
        result = self.runner.invoke(compare, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

    def test_with_bootstrap(self):
        args = [
            "-s",
            self.src,
            "-x",
            self.system_x,
            "-y",
            self.system_y,
            "-r",
            self.ref,
            "-l",
            "en",
            "-m",
            "chrF",
            "--bootstrap",
            "--num_splits",
            10,
            "--sample_ratio",
            0.3
        ]
        result = self.runner.invoke(compare, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

    def test_ner_filter(self):
        args = [
            "-s",
            self.src,
            "-x",
            self.system_x,
            "-y",
            self.system_y,
            "-r",
            self.ref,
            "-l",
            "en",
            "-f",
            "named-entities",
            "-m",
            "chrF",
            "--seg_metric",
            "GLEU"
        ]
        result = self.runner.invoke(compare, args, catch_exceptions=False)
        self.assertIn("Filters Successfully applied. Corpus reduced in", result.stdout)
        self.assertEqual(result.exit_code, 0)

    def test_with_output(self):
        args = [
            "-s",
            self.src,
            "-x",
            self.system_x,
            "-y",
            self.system_y,
            "-r",
            self.ref,
            "-l",
            "en",
            "-m",
            "chrF",
            "--seg_metric",
            "GLEU",
            "--output_folder",
            DATA_PATH
        ]
        result = self.runner.invoke(compare, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.isfile(os.path.join(DATA_PATH, "bucket-analysis.png")))
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "scores-distribution.html"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "segment-comparison.html"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(DATA_PATH, "results.json"))
        )
        os.remove(DATA_PATH + "/segment-comparison.html")
        os.remove(DATA_PATH + "/scores-distribution.html")
        os.remove(DATA_PATH + "/bucket-analysis.png")
        os.remove(DATA_PATH + "/results.json")