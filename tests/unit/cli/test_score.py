import os
import unittest

from click.testing import CliRunner
from telescope.cli import score
from tests.data import DATA_PATH


class TestScoreCli(unittest.TestCase):

    hyp = os.path.join(DATA_PATH, "hyp.en")
    src = os.path.join(DATA_PATH, "src.de")
    ref = os.path.join(DATA_PATH, "ref.en")

    def setUp(self):
        self.runner = CliRunner()

    def test_correct_cli_with_chrf(self):
        args = [
            "-s",
            self.src,
            "-h",
            self.hyp,
            "-r",
            self.ref,
            "-l",
            "en",
            "-m",
            "chrF",
        ]
        result = self.runner.invoke(score, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

    # TODO: conflict with flags
    # def test_correct_cli_with_bleurt(self):
    #    args = ["-s", self.src, "-h", self.hyp, "-r", self.ref, "-l", "en", "-m", "BLEURT"]
    #    result = self.runner.invoke(score, args, catch_exceptions=False)
    #    self.assertEqual(result.exit_code, 0)

    def test_unsupported_language(self):
        args = [
            "-s",
            self.src,
            "-h",
            self.hyp,
            "-r",
            self.ref,
            "-l",
            "de",
            "-m",
            "BLEURT",
        ]
        result = self.runner.invoke(score, args, catch_exceptions=False)
        expected_stdout = "Error: BLEURT does not support 'de'\n"
        self.assertEqual(result.stdout, expected_stdout)
        self.assertEqual(result.exit_code, 1)

    def test_correct_cli_with_bertscore(self):
        args = [
            "-s",
            self.src,
            "-h",
            self.hyp,
            "-r",
            self.ref,
            "-l",
            "en",
            "-m",
            "BERTScore",
        ]
        result = self.runner.invoke(score, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

    def test_correct_cli_with_sacrebleu(self):
        args = [
            "-s",
            self.src,
            "-h",
            self.hyp,
            "-r",
            self.ref,
            "-l",
            "en",
            "-m",
            "sacreBLEU",
        ]
        result = self.runner.invoke(score, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

    def test_correct_cli_with_zeroedit(self):
        args = [
            "-s",
            self.src,
            "-h",
            self.hyp,
            "-r",
            self.ref,
            "-l",
            "en",
            "-m",
            "ZeroEdit",
        ]
        result = self.runner.invoke(score, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

    def test_correct_cli_with_bleu(self):
        args = [
            "-s",
            self.src,
            "-h",
            self.hyp,
            "-r",
            self.ref,
            "-l",
            "en",
            "-m",
            "BLEU",
        ]
        result = self.runner.invoke(score, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)

    def test_correct_cli_with_gleu(self):
        args = [
            "-s",
            self.src,
            "-h",
            self.hyp,
            "-r",
            self.ref,
            "-l",
            "en",
            "-m",
            "GLEU",
        ]
        result = self.runner.invoke(score, args, catch_exceptions=False)
        self.assertEqual(result.exit_code, 0)
