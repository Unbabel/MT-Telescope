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
r"""
MT-Telescope command line interface (CLI)
==============
Main commands:
    - score     Used to download Machine Translation metrics.
"""
from typing import List, Union, Tuple
import os
import click
import json

from telescope.metrics import AVAILABLE_METRICS, PairwiseResult
from telescope.testset import PairwiseTestset
from telescope.filters import AVAILABLE_FILTERS
from telescope.plotting import (
    plot_segment_comparison,
    plot_pairwise_distributions,
    plot_bucket_comparison,
)

available_metrics = {m.name: m for m in AVAILABLE_METRICS}
available_filters = {f.name: f for f in AVAILABLE_FILTERS if f.name != "length"}


def readlines(ctx, param, file: click.File) -> List[str]:
    return [l.strip() for l in file.readlines()]


def output_folder_exists(ctx, param, output_folder):
    if output_folder != "" and not os.path.exists(output_folder):
        raise click.BadParameter(f"{output_folder} does not exist!")
    return output_folder


@click.group()
def telescope():
    pass


@telescope.command()
@click.option(
    "--source",
    "-s",
    required=True,
    help="Source segments.",
    type=click.File(),
)
@click.option(
    "--system_x",
    "-x",
    required=True,
    help="System X MT outputs.",
    type=click.File(),
)
@click.option(
    "--system_y",
    "-y",
    required=True,
    help="System Y MT outputs.",
    type=click.File(),
)
@click.option(
    "--reference",
    "-r",
    required=True,
    help="Reference segments.",
    type=click.File(),
)
@click.option(
    "--language",
    "-l",
    required=True,
    help="Language of the evaluated text.",
)
@click.option(
    "--metric",
    "-m",
    type=click.Choice(list(available_metrics.keys())),
    required=True,
    multiple=True,
    help="MT metric to run.",
)
@click.option(
    "--filter",
    "-f",
    type=click.Choice(list(available_filters.keys())),
    required=False,
    default=[],
    multiple=True,
    help="MT metric to run.",
)
@click.option(
    "--seg_metric",
    type=click.Choice([m.name for m in available_metrics.values() if m.segment_level]),
    required=False,
    default="COMET",
    help="Segment-level metric to use for segment-level analysis.",
)
@click.option(
    "--output_folder",
    "-o",
    required=False,
    default="",
    callback=output_folder_exists,
    type=str,
    help="Folder you wish to use to save plots.",
)
@click.option("--bootstrap", is_flag=True)
@click.option(
    "--num_splits",
    required=False,
    default=300,
    type=int,
    help="Number of random partitions used in Bootstrap resampling.",
)
@click.option(
    "--sample_ratio",
    required=False,
    default=0.5,
    type=float,
    help="Folder you wish to use to save plots.",
)
def compare(
    source: click.File,
    system_x: click.File,
    system_y: click.File,
    reference: click.File,
    language: str,
    metric: Union[Tuple[str], str],
    filter: Union[Tuple[str], str],
    seg_metric: str,
    output_folder: str,
    bootstrap: bool,
    num_splits: int,
    sample_ratio: float,
):
    testset = PairwiseTestset(
        src=[l.strip() for l in source.readlines()],
        system_x=[l.strip() for l in system_x.readlines()],
        system_y=[l.strip() for l in system_y.readlines()],
        ref=[l.strip() for l in reference.readlines()],
        language_pair="X-" + language,
        filenames=[source.name, system_x.name, system_y.name, reference.name],
    )
    corpus_size = len(testset)
    if filter:
        filters = [available_filters[f](testset) for f in filter]
        for filter in filters:
            testset.apply_filter(filter)

        click.secho(
            "Filters Successfully applied. Corpus reduced in {:.2f}%.".format(
                (1 - (len(testset) / corpus_size)) * 100
            ),
            fg="green",
        )

    if seg_metric not in metric:
        metric = tuple(
            [
                seg_metric,
            ]
            + list(metric)
        )
    else:
        # Put COMET in first place
        metric = list(metric)
        metric.remove(seg_metric)
        metric = tuple(
            [
                seg_metric,
            ]
            + metric
        )

    results = {
        m: available_metrics[m](language=testset.target_language).pairwise_comparison(
            testset
        )
        for m in metric
    }

    # results_dict = PairwiseResult.results_to_dict(list(results.values()))
    results_df = PairwiseResult.results_to_dataframe(list(results.values()))
    if bootstrap:
        bootstrap_results = []
        for m in metric:
            bootstrap_result = available_metrics[m].bootstrap_resampling(
                testset, num_splits, sample_ratio, results[m]
            )
            bootstrap_results.append(
                available_metrics[m]
                .bootstrap_resampling(testset, num_splits, sample_ratio, results[m])
                .stats
            )
        bootstrap_results = {
            k: [dic[k] for dic in bootstrap_results] for k in bootstrap_results[0]
        }
        for k, v in bootstrap_results.items():
            results_df[k] = v

    click.secho(str(results_df), fg="yellow")
    if output_folder != "":
        if not output_folder.endswith("/"):
            output_folder += "/"
        results_df.to_json(output_folder + "results.json", orient="index", indent=4)
        plot_segment_comparison(results[seg_metric], output_folder)
        plot_pairwise_distributions(results[seg_metric], output_folder)
        plot_bucket_comparison(results[seg_metric], output_folder)


@telescope.command()
@click.option(
    "--source",
    "-s",
    required=True,
    help="Source segments.",
    type=click.File(),
    callback=readlines,
)
@click.option(
    "--hypothesis",
    "-h",
    required=True,
    help="MT outputs.",
    type=click.File(),
    callback=readlines,
)
@click.option(
    "--reference",
    "-r",
    required=True,
    help="Reference segments.",
    type=click.File(),
    callback=readlines,
)
@click.option(
    "--language",
    "-l",
    required=True,
    help="Language of the evaluated text.",
)
@click.option(
    "--metric",
    "-m",
    type=click.Choice(list(available_metrics.keys())),
    required=True,
    multiple=True,
    help="MT metric to run.",
)
def score(
    source: List[str],
    hypothesis: List[str],
    reference: List[str],
    language: str,
    metric: Union[Tuple[str], str],
):
    metrics = metric
    for metric in metrics:
        if not available_metrics[metric].language_support(language):
            raise click.ClickException(f"{metric} does not support '{language}'")
    results = []
    for metric in metrics:
        metric = available_metrics[metric](language)
        results.append(metric.score(source, hypothesis, reference))

    for result in results:
        click.secho(str(result), fg="yellow")


@telescope.command()
@click.pass_context
def streamlit(ctx):
    file_path = os.path.realpath(__file__)
    script_path = "/".join(file_path.split("/")[:-1]) + "/app.py"
    os.system("streamlit run " + script_path)
