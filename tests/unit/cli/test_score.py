# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
